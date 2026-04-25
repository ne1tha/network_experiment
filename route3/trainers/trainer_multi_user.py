from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from evaluators import MultiUserComparison, MultiUserComparisonEvaluator, ReconstructionMetrics, compute_reconstruction_metrics
from optim import move_to_device, normalize_runtime_value, select_torch_device

from .progressive_stage import ProgressiveStageController, ProgressiveTrainingConfig


@dataclass(frozen=True)
class MultiUserTrainConfig:
    learning_rate: float = 1e-4
    discriminator_learning_rate: float | None = None
    weight_decay: float = 0.0
    distillation_weight: float = 1.0
    enhancement_weight: float = 1.0
    max_grad_norm: float | None = None
    train_decode_stochastic: bool = False
    val_decode_stochastic: bool = False
    device: str = "auto"
    total_epochs: int = 1
    phase1_epochs: int = 0
    phase2_epochs: int = 0
    perceptual_weight_max: float = 1.0
    perceptual_loss_scale: float = 1.0
    adversarial_weight: float = 0.1
    adversarial_ramp_epochs: int = 0
    decoder_ms_ssim_weight: float = 0.0
    decoder_l1_weight: float = 1.0
    decoder_mse_weight: float = 1.0
    decoder_residual_weight: float = 0.25


@dataclass(frozen=True)
class MultiUserTrainStepOutput:
    total_loss: float
    terms: dict[str, float]
    metrics: ReconstructionMetrics
    global_step: int


@dataclass(frozen=True)
class MultiUserValidationOutput:
    terms: dict[str, float]
    metrics: ReconstructionMetrics
    base_metrics: ReconstructionMetrics | None
    final_metrics: ReconstructionMetrics | None
    comparison: MultiUserComparison
    enhancement_status: dict[str, Any] | None


class MultiUserTrainer:
    """Route-3 multi-user trainer with the same progressive stage contract as single-user training."""

    def __init__(
        self,
        model: nn.Module,
        config: MultiUserTrainConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        discriminator_optimizer: torch.optim.Optimizer | None = None,
        comparison_evaluator: MultiUserComparisonEvaluator | None = None,
    ):
        self.model = model
        self.config = config or MultiUserTrainConfig()
        self.device = select_torch_device(self.config.device)
        self.model.to(self.device)
        discriminator_params = []
        enhancer = getattr(self.model.single_user_model, "enhancer", None)
        if enhancer is not None and enhancer.discriminator is not None:
            discriminator_params = [param for param in enhancer.discriminator.parameters() if param.requires_grad]
        discriminator_param_ids = {id(param) for param in discriminator_params}
        generator_params = [
            param
            for param in model.parameters()
            if param.requires_grad and id(param) not in discriminator_param_ids
        ]
        self.optimizer = optimizer or torch.optim.Adam(
            generator_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.discriminator_optimizer = discriminator_optimizer
        if discriminator_params:
            self.discriminator_optimizer = discriminator_optimizer or torch.optim.Adam(
                discriminator_params,
                lr=self.config.discriminator_learning_rate
                if self.config.discriminator_learning_rate is not None
                else self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        self.global_step = 0
        self.validation_step_count = 0
        self.current_epoch = 1
        self.comparison_evaluator = comparison_evaluator or MultiUserComparisonEvaluator()
        self.stage_controller = ProgressiveStageController(
            ProgressiveTrainingConfig(
                total_epochs=self.config.total_epochs,
                phase1_epochs=self.config.phase1_epochs,
                phase2_epochs=self.config.phase2_epochs,
                perceptual_weight_max=self.config.perceptual_weight_max,
                adversarial_weight=self.config.adversarial_weight,
                adversarial_enabled=self._adversarial_ready(),
                adversarial_ramp_epochs=self.config.adversarial_ramp_epochs,
            )
        )
        self.active_stage = self.stage_controller.stage_at_epoch(self.current_epoch)
        self._apply_loss_scales()
        self._apply_stage_settings()

    def _adversarial_ready(self) -> bool:
        enhancer = getattr(self.model.single_user_model, "enhancer", None)
        return enhancer is not None and enhancer.discriminator is not None

    def _perceptual_ready(self) -> bool:
        enhancer = getattr(self.model.single_user_model, "enhancer", None)
        return enhancer is not None and enhancer.perceptual_loss is not None

    def _apply_stage_settings(self) -> None:
        enhancer = getattr(self.model.single_user_model, "enhancer", None)
        if enhancer is None:
            return
        enhancer.perceptual_weight = self.active_stage.perceptual_weight
        enhancer.adversarial_weight = self.active_stage.adversarial_weight

    def _apply_loss_scales(self) -> None:
        enhancer = getattr(self.model.single_user_model, "enhancer", None)
        if enhancer is None or enhancer.perceptual_loss is None:
            return
        enhancer.perceptual_loss.loss_scale = self.config.perceptual_loss_scale

    def current_stage_summary(self) -> dict[str, Any]:
        return self.active_stage.asdict()

    def _compose_decoder_total_loss(self, output) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if output.shared_reconstruction is None or output.shared_decoder_losses is None:
            raise RuntimeError("Multi-user training requires shared reconstruction and decoder losses.")

        terms = dict(output.shared_decoder_losses.terms)
        zero = output.shared_reconstruction.x_hat.new_tensor(0.0)
        structural_total = (
            self.config.decoder_ms_ssim_weight * terms.get("decoder_ms_ssim", zero)
            + self.config.decoder_l1_weight * terms.get("decoder_l1", zero)
            + self.config.decoder_mse_weight * terms.get("decoder_mse", zero)
            + self.config.decoder_residual_weight * terms.get("decoder_residual", zero)
        )
        terms["decoder_structural_total"] = structural_total
        return structural_total, terms

    def _compose_total_loss(self, output) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total, terms = self._compose_decoder_total_loss(output)

        if output.single_user.distillation is not None:
            distill_term = output.single_user.distillation.total_loss
            terms["distillation_total"] = distill_term
            total = total + self.config.distillation_weight * distill_term

        if output.shared_enhancement_losses is not None:
            enhancement = output.shared_enhancement_losses.generator_total_loss
            terms.update(output.shared_enhancement_losses.generator_terms)
            terms["enhancement_g_total"] = enhancement
            if output.shared_enhancement_losses.discriminator_loss is not None:
                terms["enhancement_d_total"] = output.shared_enhancement_losses.discriminator_loss
            zero = total.new_tensor(0.0)
            terms["enhancement_perceptual_weighted"] = self.config.enhancement_weight * terms.get("perceptual_vgg_effective", zero)
            terms["enhancement_adversarial_weighted"] = self.config.enhancement_weight * terms.get("adversarial_g_effective", zero)
            terms["enhancement_weighted_total"] = self.config.enhancement_weight * enhancement
            total = total + terms["enhancement_weighted_total"]

        terms["multi_user_train_total"] = total
        return total, terms

    @staticmethod
    def _to_float_terms(terms: dict[str, torch.Tensor]) -> dict[str, float]:
        return {name: float(value.detach().item()) for name, value in terms.items()}

    def _set_discriminator_grad(self, enabled: bool) -> None:
        enhancer = getattr(self.model.single_user_model, "enhancer", None)
        if enhancer is not None and enhancer.discriminator is not None:
            enhancer.discriminator.requires_grad_(enabled)

    def set_epoch(self, epoch: int) -> None:
        if epoch <= 0:
            raise ValueError(f"epoch must be positive, got {epoch}")
        self.current_epoch = int(epoch)
        self.active_stage = self.stage_controller.stage_at_epoch(self.current_epoch)
        self._apply_stage_settings()

    def adversarial_active(self) -> bool:
        return self.active_stage.adversarial_weight > 0.0 and self._adversarial_ready()

    def perceptual_active(self) -> bool:
        return self.active_stage.perceptual_weight > 0.0 and self._perceptual_ready()

    def enhancement_active(self) -> bool:
        return self.perceptual_active() or self.adversarial_active()

    @staticmethod
    def _sample_decoder_noise(x: torch.Tensor, stochastic: bool) -> torch.Tensor | None:
        if not stochastic:
            return None
        return torch.randn_like(x)

    def train_step(
        self,
        x: torch.Tensor,
        snr_db: torch.Tensor | float,
        sem_rate_ratio: torch.Tensor | float,
        det_rate_ratio: torch.Tensor | float,
        semantic_bandwidth_budget: float,
    ) -> MultiUserTrainStepOutput:
        self.model.train()
        x = move_to_device(x, self.device)
        noise_image = self._sample_decoder_noise(x, self.config.train_decode_stochastic)
        enhancer = getattr(self.model.single_user_model, "enhancer", None)
        adversarial_active = self.adversarial_active()
        perceptual_active = self.perceptual_active()
        enhancement_active = perceptual_active or adversarial_active
        train_discriminator = self.active_stage.train_discriminator and self._adversarial_ready()
        discriminator_loss_value = None
        if train_discriminator and self.discriminator_optimizer is not None and enhancer is not None:
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                d_output = self.model(
                    x,
                    snr_db=normalize_runtime_value(snr_db, self.device),
                    sem_rate_ratio=normalize_runtime_value(sem_rate_ratio, self.device),
                    det_rate_ratio=normalize_runtime_value(det_rate_ratio, self.device),
                    semantic_bandwidth_budget=semantic_bandwidth_budget,
                    decode_stochastic=self.config.train_decode_stochastic,
                    noise_image=noise_image,
                    run_enhancement=False,
                    compute_enhancement_discriminator_loss=False,
                    enhancement_enable_perceptual=perceptual_active,
                    enhancement_enable_adversarial=adversarial_active,
                )
                d_candidate = (
                    d_output.base_shared_reconstruction
                    if d_output.base_shared_reconstruction is not None
                    else d_output.shared_reconstruction
                )
                if d_candidate is None:
                    raise RuntimeError("Multi-user discriminator step requires a shared base reconstruction.")
                d_refined_candidate = d_candidate
                if enhancement_active:
                    d_refined_candidate = enhancer.refine_reconstruction(base_reconstruction=d_candidate)
            discriminator_loss = enhancer.discriminator_step(
                x_source=d_candidate.x_hat,
                x_gt=x,
                x_pred=d_refined_candidate.x_hat,
            )
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
            discriminator_loss_value = discriminator_loss.detach()

        self._set_discriminator_grad(False)
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(
            x,
            snr_db=normalize_runtime_value(snr_db, self.device),
            sem_rate_ratio=normalize_runtime_value(sem_rate_ratio, self.device),
            det_rate_ratio=normalize_runtime_value(det_rate_ratio, self.device),
            semantic_bandwidth_budget=semantic_bandwidth_budget,
            decode_stochastic=self.config.train_decode_stochastic,
            noise_image=noise_image,
            run_enhancement=enhancement_active,
            compute_enhancement_discriminator_loss=False,
            enhancement_enable_perceptual=perceptual_active,
            enhancement_enable_adversarial=adversarial_active,
        )
        total, terms = self._compose_total_loss(output)
        if discriminator_loss_value is not None:
            terms["enhancement_d_total"] = discriminator_loss_value
        total.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()
        self._set_discriminator_grad(True)

        self.global_step += 1
        metrics = compute_reconstruction_metrics(x.detach(), output.shared_reconstruction.x_hat.detach())
        return MultiUserTrainStepOutput(
            total_loss=float(total.detach().item()),
            terms=self._to_float_terms(terms),
            metrics=metrics,
            global_step=self.global_step,
        )

    @torch.no_grad()
    def validate_step(
        self,
        x: torch.Tensor,
        snr_db: torch.Tensor | float,
        sem_rate_ratio: torch.Tensor | float,
        det_rate_ratio: torch.Tensor | float,
        semantic_bandwidth_budget: float,
    ) -> MultiUserValidationOutput:
        self.model.eval()
        x = move_to_device(x, self.device)
        noise_image = self._sample_decoder_noise(x, self.config.val_decode_stochastic)
        adversarial_active = self.adversarial_active()
        perceptual_active = self.perceptual_active()
        enhancement_active = perceptual_active or adversarial_active
        output = self.model(
            x,
            snr_db=normalize_runtime_value(snr_db, self.device),
            sem_rate_ratio=normalize_runtime_value(sem_rate_ratio, self.device),
            det_rate_ratio=normalize_runtime_value(det_rate_ratio, self.device),
            semantic_bandwidth_budget=semantic_bandwidth_budget,
            decode_stochastic=self.config.val_decode_stochastic,
            noise_image=noise_image,
            run_enhancement=enhancement_active,
            compute_enhancement_discriminator_loss=adversarial_active,
            enhancement_enable_perceptual=perceptual_active,
            enhancement_enable_adversarial=adversarial_active,
        )
        total, terms = self._compose_total_loss(output)
        terms["multi_user_validation_total"] = total
        self.validation_step_count += 1

        enhancement_status = None
        if output.shared_enhancement_status is not None:
            enhancement_status = asdict(output.shared_enhancement_status)
        base_metrics = None
        if output.base_shared_reconstruction is not None:
            base_metrics = compute_reconstruction_metrics(x, output.base_shared_reconstruction.x_hat)
        final_metrics = None
        if output.final_shared_reconstruction is not None:
            final_metrics = compute_reconstruction_metrics(x, output.final_shared_reconstruction.x_hat)

        comparison = self.comparison_evaluator.evaluate_batch(x, output)
        metrics = compute_reconstruction_metrics(x, output.shared_reconstruction.x_hat)
        return MultiUserValidationOutput(
            terms=self._to_float_terms(terms),
            metrics=metrics,
            base_metrics=base_metrics,
            final_metrics=final_metrics,
            comparison=comparison,
            enhancement_status=enhancement_status,
        )

    def save_checkpoint(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict() if self.discriminator_optimizer is not None else None,
            "trainer": {
                "global_step": self.global_step,
                "validation_step_count": self.validation_step_count,
                "config": asdict(self.config),
            },
            "extra_state": extra_state or {},
        }
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> dict[str, Any]:
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"], strict=strict)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.discriminator_optimizer is not None and checkpoint.get("discriminator_optimizer") is not None:
            self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        trainer_state = checkpoint.get("trainer", {})
        self.global_step = int(trainer_state.get("global_step", 0))
        self.validation_step_count = int(trainer_state.get("validation_step_count", 0))
        return checkpoint.get("extra_state", {})

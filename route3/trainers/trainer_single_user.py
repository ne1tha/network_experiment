from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from evaluators import (
    AblationComparison,
    ReconstructionMetrics,
    SingleUserAblationRunner,
    SingleUserBudgetMetrics,
    compute_reconstruction_metrics,
    summarize_single_user_budget,
)
from optim import move_to_device, normalize_runtime_value, select_torch_device

from .progressive_stage import ActiveStage, ProgressiveStageController, ProgressiveTrainingConfig


@dataclass(frozen=True)
class SingleUserTrainConfig:
    learning_rate: float = 1e-4
    discriminator_learning_rate: float | None = None
    weight_decay: float = 0.0
    distillation_weight: float = 1.0
    enhancement_weight: float = 1.0
    max_grad_norm: float | None = None
    train_decode_stochastic: bool = False
    val_decode_stochastic: bool = False
    device: str = "auto"
    operating_mode: str = "open_quality"
    target_effective_cbr: float | None = None
    target_effective_cbr_tolerance: float = 0.05
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
    base_decoder_aux_weight: float = 0.5
    final_decoder_weight: float = 1.0
    rate_regularization_weight: float = 0.0
    refinement_consistency_weight: float = 1.0
    refinement_delta_weight: float = 0.1


@dataclass(frozen=True)
class TrainStepOutput:
    total_loss: float
    terms: dict[str, float]
    metrics: ReconstructionMetrics
    budget_metrics: SingleUserBudgetMetrics | None
    global_step: int


@dataclass(frozen=True)
class ValidationStepOutput:
    terms: dict[str, float]
    metrics: ReconstructionMetrics
    base_metrics: ReconstructionMetrics | None
    final_metrics: ReconstructionMetrics | None
    budget_metrics: SingleUserBudgetMetrics | None
    enhancement_status: dict[str, Any] | None


class SingleUserTrainer:
    """Route-3 single-user training loop with progressive structural/perceptual/GAN stages."""

    def __init__(
        self,
        model: nn.Module,
        config: SingleUserTrainConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        discriminator_optimizer: torch.optim.Optimizer | None = None,
        ablation_runner: SingleUserAblationRunner | None = None,
    ):
        self.model = model
        self.config = config or SingleUserTrainConfig()
        self.device = select_torch_device(self.config.device)
        self.model.to(self.device)
        discriminator_params = []
        if getattr(self.model, "enhancer", None) is not None and self.model.enhancer.discriminator is not None:
            discriminator_params = [param for param in self.model.enhancer.discriminator.parameters() if param.requires_grad]
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
        self.ablation_runner = ablation_runner or SingleUserAblationRunner(model)
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
        enhancer = getattr(self.model, "enhancer", None)
        return enhancer is not None and enhancer.discriminator is not None

    def _perceptual_ready(self) -> bool:
        enhancer = getattr(self.model, "enhancer", None)
        return enhancer is not None and enhancer.perceptual_loss is not None

    def _apply_stage_settings(self) -> None:
        enhancer = getattr(self.model, "enhancer", None)
        if enhancer is None:
            return
        enhancer.perceptual_weight = self.active_stage.perceptual_weight
        enhancer.adversarial_weight = self.active_stage.adversarial_weight

    def _apply_loss_scales(self) -> None:
        enhancer = getattr(self.model, "enhancer", None)
        if enhancer is None or enhancer.perceptual_loss is None:
            return
        enhancer.perceptual_loss.loss_scale = self.config.perceptual_loss_scale

    def current_stage_summary(self) -> dict[str, Any]:
        return self.active_stage.asdict()

    @staticmethod
    def _tail_energy_ratio(x: torch.Tensor, spatial_mask: torch.Tensor) -> torch.Tensor:
        inactive_mask = 1.0 - spatial_mask
        total_energy = x.pow(2).mean().clamp_min(1e-8)
        tail_energy = (x * inactive_mask).pow(2).mean()
        return tail_energy / total_energy

    def _compose_rate_regularization(self, output) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if output.reconstruction is None:
            raise RuntimeError("Single-user rate regularization requires a reconstruction output.")

        zero = output.reconstruction.x_hat.new_tensor(0.0)
        semantic_tail_energy_ratio = self._tail_energy_ratio(output.semantic.modulated, output.semantic.mask.spatial_mask)
        detail_tail_energy_ratio = self._tail_energy_ratio(output.detail.modulated, output.detail.mask.spatial_mask)
        rate_regularization_total = 0.5 * (semantic_tail_energy_ratio + detail_tail_energy_ratio)
        weighted_rate_regularization = self.config.rate_regularization_weight * rate_regularization_total
        return weighted_rate_regularization, {
            "semantic_tail_energy_ratio": semantic_tail_energy_ratio,
            "detail_tail_energy_ratio": detail_tail_energy_ratio,
            "rate_regularization_total": rate_regularization_total,
            "rate_regularization_weighted_total": weighted_rate_regularization if self.config.rate_regularization_weight > 0.0 else zero,
        }

    def _summarize_budget(self, x: torch.Tensor, output) -> SingleUserBudgetMetrics | None:
        if output.reconstruction is None:
            return None
        return summarize_single_user_budget(
            input_image=x,
            output=output,
            operating_mode=self.config.operating_mode,
            target_effective_cbr=self.config.target_effective_cbr,
            target_effective_cbr_tolerance=self.config.target_effective_cbr_tolerance,
        )

    @staticmethod
    def _append_budget_terms(
        terms: dict[str, torch.Tensor],
        budget_metrics: SingleUserBudgetMetrics | None,
        reference_tensor: torch.Tensor,
    ) -> None:
        if budget_metrics is None:
            return
        terms["budget_effective_cbr"] = reference_tensor.new_tensor(budget_metrics.effective_cbr)
        terms["budget_semantic_active_channels_mean"] = reference_tensor.new_tensor(
            budget_metrics.semantic_active_channels_mean
        )
        terms["budget_detail_active_channels_mean"] = reference_tensor.new_tensor(
            budget_metrics.detail_active_channels_mean
        )
        terms["budget_semantic_rate_ratio_mean"] = reference_tensor.new_tensor(
            budget_metrics.semantic_rate_ratio_mean
        )
        terms["budget_detail_rate_ratio_mean"] = reference_tensor.new_tensor(
            budget_metrics.detail_rate_ratio_mean
        )
        if budget_metrics.target_effective_cbr is not None:
            terms["budget_target_effective_cbr"] = reference_tensor.new_tensor(budget_metrics.target_effective_cbr)
        if budget_metrics.cbr_absolute_gap is not None:
            terms["budget_cbr_absolute_gap"] = reference_tensor.new_tensor(budget_metrics.cbr_absolute_gap)
        if budget_metrics.cbr_relative_gap is not None:
            terms["budget_cbr_relative_gap"] = reference_tensor.new_tensor(budget_metrics.cbr_relative_gap)
        if budget_metrics.within_target_tolerance is not None:
            terms["budget_within_target_tolerance"] = reference_tensor.new_tensor(
                1.0 if budget_metrics.within_target_tolerance else 0.0
            )

    def _compose_structural_terms(
        self,
        decoder_losses,
        *,
        key_prefix: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reference = decoder_losses.total_loss
        terms = dict(decoder_losses.terms)
        zero = reference.new_tensor(0.0)
        structural_total = (
            self.config.decoder_ms_ssim_weight * terms.get("decoder_ms_ssim", zero)
            + self.config.decoder_l1_weight * terms.get("decoder_l1", zero)
            + self.config.decoder_mse_weight * terms.get("decoder_mse", zero)
            + self.config.decoder_residual_weight * terms.get("decoder_residual", zero)
        )
        if key_prefix is None:
            terms["decoder_structural_total"] = structural_total
            return structural_total, terms

        prefixed_terms = {f"{key_prefix}_{name}": value for name, value in terms.items()}
        prefixed_terms[f"{key_prefix}_structural_total"] = structural_total
        return structural_total, prefixed_terms

    def _compose_decoder_total_loss(self, output) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if output.reconstruction is None or output.decoder_losses is None:
            raise RuntimeError("Single-user training requires the main decoder and decoder loss to be enabled.")

        structural_total, terms = self._compose_structural_terms(output.decoder_losses)
        terms["decoder_weighted_total"] = structural_total

        has_distinct_refined_output = (
            output.base_reconstruction is not None
            and output.final_reconstruction is not None
            and output.final_reconstruction is not output.base_reconstruction
        )
        if not has_distinct_refined_output:
            return structural_total, terms

        if output.base_decoder_losses is None or output.final_decoder_losses is None:
            raise RuntimeError("Refined single-user training requires both base and final decoder losses.")

        base_structural_total, base_terms = self._compose_structural_terms(
            output.base_decoder_losses,
            key_prefix="base_decoder",
        )
        final_structural_total, final_terms = self._compose_structural_terms(
            output.final_decoder_losses,
            key_prefix="final_decoder",
        )
        terms.update(base_terms)
        terms.update(final_terms)
        terms["base_decoder_aux_weighted_total"] = self.config.base_decoder_aux_weight * base_structural_total
        terms["final_decoder_weighted_total"] = self.config.final_decoder_weight * final_structural_total
        terms["decoder_weighted_total"] = terms["final_decoder_weighted_total"]
        total = terms["base_decoder_aux_weighted_total"] + terms["final_decoder_weighted_total"]
        return total, terms

    def _compose_total_loss(self, output) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total, terms = self._compose_decoder_total_loss(output)

        if output.distillation is not None:
            distill_term = output.distillation.total_loss
            terms["distillation_total"] = distill_term
            total = total + self.config.distillation_weight * distill_term

        if output.enhancement_losses is not None:
            enhancement_term = output.enhancement_losses.generator_total_loss
            terms.update(output.enhancement_losses.generator_terms)
            terms["enhancement_g_total"] = enhancement_term
            if output.enhancement_losses.discriminator_loss is not None:
                terms["enhancement_d_total"] = output.enhancement_losses.discriminator_loss
            zero = total.new_tensor(0.0)
            enhancement_schedule_scale = total.new_tensor(self._enhancement_schedule_scale())
            refinement_consistency = (
                self.config.enhancement_weight
                * enhancement_schedule_scale
                * self.config.refinement_consistency_weight
                * terms.get("refinement_l1", zero)
            )
            refinement_delta = (
                self.config.enhancement_weight
                * enhancement_schedule_scale
                * self.config.refinement_delta_weight
                * terms.get("refinement_delta_l1", zero)
            )
            terms["enhancement_perceptual_weighted"] = self.config.enhancement_weight * terms.get("perceptual_vgg_effective", zero)
            terms["enhancement_adversarial_weighted"] = self.config.enhancement_weight * terms.get("adversarial_g_effective", zero)
            terms["enhancement_schedule_scale"] = enhancement_schedule_scale
            terms["enhancement_refinement_consistency_weighted"] = refinement_consistency
            terms["enhancement_refinement_delta_weighted"] = refinement_delta
            terms["enhancement_regularized_total"] = enhancement_term + refinement_consistency + refinement_delta
            terms["enhancement_weighted_total"] = self.config.enhancement_weight * enhancement_term
            total = total + terms["enhancement_weighted_total"] + refinement_consistency + refinement_delta

        if self.config.rate_regularization_weight > 0.0:
            weighted_rate_regularization, rate_terms = self._compose_rate_regularization(output)
            terms.update(rate_terms)
            total = total + weighted_rate_regularization

        terms["train_total"] = total
        return total, terms

    @staticmethod
    def _to_float_terms(terms: dict[str, torch.Tensor]) -> dict[str, float]:
        return {name: float(value.detach().item()) for name, value in terms.items()}

    def _set_discriminator_grad(self, enabled: bool) -> None:
        enhancer = getattr(self.model, "enhancer", None)
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

    def _enhancement_schedule_scale(self) -> float:
        if not self.enhancement_active():
            return 0.0
        if self.config.perceptual_weight_max > 0.0:
            return min(max(self.active_stage.perceptual_weight / self.config.perceptual_weight_max, 0.0), 1.0)
        return 1.0

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
    ) -> TrainStepOutput:
        self.model.train()
        x = move_to_device(x, self.device)
        noise_image = self._sample_decoder_noise(x, self.config.train_decode_stochastic)
        adversarial_active = self.adversarial_active()
        perceptual_active = self.perceptual_active()
        enhancement_active = perceptual_active or adversarial_active
        train_discriminator = self.active_stage.train_discriminator and self._adversarial_ready()
        discriminator_loss_value = None
        if train_discriminator and self.discriminator_optimizer is not None and getattr(self.model, "enhancer", None) is not None:
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                d_output = self.model(
                    x,
                    snr_db=normalize_runtime_value(snr_db, self.device),
                    sem_rate_ratio=normalize_runtime_value(sem_rate_ratio, self.device),
                    det_rate_ratio=normalize_runtime_value(det_rate_ratio, self.device),
                    decode_stochastic=self.config.train_decode_stochastic,
                    noise_image=noise_image,
                    run_enhancement=False,
                    compute_enhancement_discriminator_loss=False,
                    enhancement_enable_perceptual=perceptual_active,
                    enhancement_enable_adversarial=adversarial_active,
                )
                d_base_candidate = (
                    d_output.base_reconstruction
                    if d_output.base_reconstruction is not None
                    else d_output.reconstruction
                )
                if d_base_candidate is None:
                    raise RuntimeError("Single-user discriminator step requires a base reconstruction.")
                d_refined_candidate = d_base_candidate
                if enhancement_active:
                    d_refined_candidate = self.model.enhancer.refine_reconstruction(base_reconstruction=d_base_candidate)
            discriminator_loss = self.model.enhancer.discriminator_step(
                x_source=d_base_candidate.x_hat,
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
        budget_metrics = self._summarize_budget(x, output)
        self._append_budget_terms(terms, budget_metrics, output.reconstruction.x_hat)
        total.backward()

        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()
        self._set_discriminator_grad(True)

        self.global_step += 1
        metrics = compute_reconstruction_metrics(x.detach(), output.reconstruction.x_hat.detach())
        return TrainStepOutput(
            total_loss=float(total.detach().item()),
            terms=self._to_float_terms(terms),
            metrics=metrics,
            budget_metrics=budget_metrics,
            global_step=self.global_step,
        )

    @torch.no_grad()
    def validate_step(
        self,
        x: torch.Tensor,
        snr_db: torch.Tensor | float,
        sem_rate_ratio: torch.Tensor | float,
        det_rate_ratio: torch.Tensor | float,
    ) -> ValidationStepOutput:
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
            decode_stochastic=self.config.val_decode_stochastic,
            noise_image=noise_image,
            run_enhancement=enhancement_active,
            compute_enhancement_discriminator_loss=adversarial_active,
            enhancement_enable_perceptual=perceptual_active,
            enhancement_enable_adversarial=adversarial_active,
        )
        total, terms = self._compose_total_loss(output)
        terms["validation_total"] = total
        self.validation_step_count += 1
        enhancement_status = None
        if output.enhancement_status is not None:
            enhancement_status = asdict(output.enhancement_status)
        base_metrics = None
        if output.base_reconstruction is not None:
            base_metrics = compute_reconstruction_metrics(x, output.base_reconstruction.x_hat)
        final_metrics = None
        if output.final_reconstruction is not None:
            final_metrics = compute_reconstruction_metrics(x, output.final_reconstruction.x_hat)
        budget_metrics = self._summarize_budget(x, output)
        self._append_budget_terms(terms, budget_metrics, output.reconstruction.x_hat)
        return ValidationStepOutput(
            terms=self._to_float_terms(terms),
            metrics=compute_reconstruction_metrics(x, output.reconstruction.x_hat),
            base_metrics=base_metrics,
            final_metrics=final_metrics,
            budget_metrics=budget_metrics,
            enhancement_status=enhancement_status,
        )

    def run_ablations(
        self,
        x: torch.Tensor,
        snr_db: torch.Tensor | float,
        sem_rate_ratio: torch.Tensor | float,
        det_rate_ratio: torch.Tensor | float,
    ) -> AblationComparison:
        return self.ablation_runner.run_batch(
            x=move_to_device(x, self.device),
            snr_db=normalize_runtime_value(snr_db, self.device),
            sem_rate_ratio=normalize_runtime_value(sem_rate_ratio, self.device),
            det_rate_ratio=normalize_runtime_value(det_rate_ratio, self.device),
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

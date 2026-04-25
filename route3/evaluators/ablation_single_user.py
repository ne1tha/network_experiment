from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .single_user_metrics import ReconstructionMetrics, compute_reconstruction_metrics


class DeterministicDecoderBaseline(nn.Module):
    """Simple deterministic baseline used only for ablation infrastructure."""

    def __init__(self, semantic_channels: int = 192, detail_channels: int = 128, hidden_channels: int = 96):
        super().__init__()
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.detail_proj = nn.Sequential(
            nn.Conv2d(detail_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, rx_sem: torch.Tensor, rx_det: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        sem = self.semantic_proj(rx_sem)
        sem = F.interpolate(sem, size=rx_det.shape[-2:], mode="bilinear", align_corners=False)
        det = self.detail_proj(rx_det)
        x = torch.cat([sem, det], dim=1)
        x = self.head(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x.clamp(0.0, 1.0)


@dataclass(frozen=True)
class AblationComparison:
    full_model: ReconstructionMetrics
    base_model: ReconstructionMetrics | None
    no_detail_branch: ReconstructionMetrics
    no_distillation_objective: float | None
    with_distillation_objective: float | None
    deterministic_baseline: ReconstructionMetrics
    refinement_gain_psnr: float | None
    requires_retraining: bool


class SingleUserAblationRunner:
    """Executable single-user ablation runner for route 3 infrastructure."""

    def __init__(self, model: nn.Module, deterministic_baseline: nn.Module | None = None):
        self.model = model
        self.deterministic_baseline = deterministic_baseline or DeterministicDecoderBaseline()

    @torch.no_grad()
    def run_batch(
        self,
        x: torch.Tensor,
        snr_db: torch.Tensor | float,
        sem_rate_ratio: torch.Tensor | float,
        det_rate_ratio: torch.Tensor | float,
    ) -> AblationComparison:
        if getattr(self.model, "decoder", None) is None:
            raise RuntimeError("Ablation runner requires the main decoder to be enabled.")

        was_training = self.model.training
        baseline_was_training = self.deterministic_baseline.training
        self.model.eval()
        self.deterministic_baseline.to(device=x.device)
        self.deterministic_baseline.eval()
        try:
            output = self.model(
                x,
                snr_db=snr_db,
                sem_rate_ratio=sem_rate_ratio,
                det_rate_ratio=det_rate_ratio,
                decode_stochastic=False,
            )
            if output.reconstruction is None:
                raise RuntimeError("Main decoder did not produce a reconstruction for ablation.")

            full_metrics = compute_reconstruction_metrics(x, output.reconstruction.x_hat)
            base_metrics = None
            refinement_gain_psnr = None
            if output.base_reconstruction is not None:
                base_metrics = compute_reconstruction_metrics(x, output.base_reconstruction.x_hat)
                refinement_gain_psnr = full_metrics.psnr - base_metrics.psnr
            elif output.final_reconstruction is not None:
                base_metrics = compute_reconstruction_metrics(x, output.final_reconstruction.x_hat)
                refinement_gain_psnr = 0.0

            no_detail_noise = None
            if output.base_reconstruction is not None:
                no_detail_noise = output.base_reconstruction.noisy_input
            elif output.reconstruction is not None:
                no_detail_noise = output.reconstruction.noisy_input
            if no_detail_noise is None:
                raise RuntimeError("Ablation runner requires decoder noisy_input for no-detail reconstruction.")
            no_detail_reconstruction = self.model.decoder(
                rx_sem=output.semantic.rx,
                rx_det=torch.zeros_like(output.detail.rx),
                sem_state=output.semantic.channel_state,
                det_state=output.detail.channel_state,
                sem_rate_ratio=sem_rate_ratio,
                det_rate_ratio=det_rate_ratio,
                output_size=output.encoder.input_size,
                stochastic=False,
                decode_timestep=1.0,
                noise_image=torch.zeros_like(no_detail_noise),
            ).x_hat
            no_detail_metrics = compute_reconstruction_metrics(x, no_detail_reconstruction)

            deterministic_reconstruction = self.deterministic_baseline(
                rx_sem=output.semantic.rx,
                rx_det=output.detail.rx,
                output_size=output.encoder.input_size,
            )
            deterministic_metrics = compute_reconstruction_metrics(x, deterministic_reconstruction)

            with_distill = None
            no_distill = None
            if output.distillation is not None and output.decoder_losses is not None:
                with_distill = (output.decoder_losses.total_loss + output.distillation.total_loss).item()
                no_distill = output.decoder_losses.total_loss.item()
            elif output.decoder_losses is not None:
                no_distill = output.decoder_losses.total_loss.item()

            return AblationComparison(
                full_model=full_metrics,
                base_model=base_metrics,
                no_detail_branch=no_detail_metrics,
                no_distillation_objective=no_distill,
                with_distillation_objective=with_distill,
                deterministic_baseline=deterministic_metrics,
                refinement_gain_psnr=refinement_gain_psnr,
                requires_retraining=True,
            )
        finally:
            if was_training:
                self.model.train()
            if baseline_was_training:
                self.deterministic_baseline.train()

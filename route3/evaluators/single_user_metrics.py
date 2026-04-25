from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReconstructionMetrics:
    mse: float
    psnr: float
    mean_abs_error: float


def compute_reconstruction_metrics(x_gt: torch.Tensor, x_hat: torch.Tensor) -> ReconstructionMetrics:
    if x_gt.shape != x_hat.shape:
        raise ValueError(f"Metric computation expects matching image shapes, got {x_gt.shape} and {x_hat.shape}.")

    mse = F.mse_loss(x_hat, x_gt).item()
    mae = F.l1_loss(x_hat, x_gt).item()
    psnr = -10.0 * torch.log10(torch.tensor(max(mse, 1e-12), dtype=x_gt.dtype)).item()
    return ReconstructionMetrics(mse=mse, psnr=psnr, mean_abs_error=mae)


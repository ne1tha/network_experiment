from __future__ import annotations

from dataclasses import dataclass

import lpips
import torch
import torch.nn as nn

from route2_swinjscc_gan.common.checks import require_same_shape
from route2_swinjscc_gan.losses.reconstruction import MSSSIMLoss


def compute_psnr(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    require_same_shape("reconstructed", reconstructed, "target", target)
    mse = torch.mean((reconstructed * 255.0 - target * 255.0) ** 2, dim=(1, 2, 3))
    psnr = torch.where(
        mse > 0,
        10.0 * torch.log10((255.0**2) / mse),
        torch.full_like(mse, float("inf")),
    )
    return psnr


class LPIPSMetric(nn.Module):
    def __init__(self, network: str = "vgg") -> None:
        super().__init__()
        self.model = lpips.LPIPS(net=network)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        require_same_shape("reconstructed", reconstructed, "target", target)
        reconstructed_scaled = reconstructed * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        return self.model(reconstructed_scaled, target_scaled).view(-1)


@dataclass
class BatchMetrics:
    psnr: float
    ms_ssim: float
    lpips: float


class ImageQualityMetricSuite(nn.Module):
    def __init__(self, *, lpips_network: str = "vgg") -> None:
        super().__init__()
        self.ms_ssim_loss = MSSSIMLoss()
        self.lpips_metric = LPIPSMetric(network=lpips_network)

    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor) -> BatchMetrics:
        psnr = compute_psnr(reconstructed, target).mean().item()
        ms_ssim = (1.0 - self.ms_ssim_loss(reconstructed, target)).item()
        lpips_value = self.lpips_metric(reconstructed, target).mean().item()
        return BatchMetrics(psnr=psnr, ms_ssim=ms_ssim, lpips=lpips_value)


from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reconstruction_structural import MSSSIMLoss


@dataclass(frozen=True)
class DecoderLossOutput:
    total_loss: torch.Tensor
    terms: dict[str, torch.Tensor]


class ConditionalSSDDDecoderLoss(nn.Module):
    """Decoder-specific loss block for the route-3 SSDD-style decoder."""

    def __init__(
        self,
        ms_ssim_weight: float = 0.0,
        l1_weight: float = 1.0,
        mse_weight: float = 1.0,
        residual_weight: float = 0.25,
        ms_ssim_loss: MSSSIMLoss | None = None,
    ):
        super().__init__()
        if min(ms_ssim_weight, l1_weight, mse_weight, residual_weight) < 0:
            raise ValueError("Decoder loss weights must be non-negative.")
        self.ms_ssim_weight = ms_ssim_weight
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.residual_weight = residual_weight
        self.ms_ssim_loss = ms_ssim_loss or MSSSIMLoss()

    def forward(
        self,
        x_gt: torch.Tensor,
        x_hat: torch.Tensor,
        noisy_input: torch.Tensor | None = None,
        predicted_residual: torch.Tensor | None = None,
    ) -> DecoderLossOutput:
        if x_gt.shape != x_hat.shape:
            raise ValueError(f"Decoder loss expects matching image shapes, got {x_gt.shape} and {x_hat.shape}.")

        terms = {
            "decoder_ms_ssim": self.ms_ssim_loss(x_hat, x_gt),
            "decoder_l1": F.l1_loss(x_hat, x_gt),
            "decoder_mse": F.mse_loss(x_hat, x_gt),
        }

        if predicted_residual is not None:
            if noisy_input is None:
                raise ValueError("predicted_residual requires noisy_input so the target residual can be defined.")
            if noisy_input.shape != x_gt.shape or predicted_residual.shape != x_gt.shape:
                raise ValueError("Residual supervision expects noisy_input and predicted_residual to match image shape.")
            target_residual = x_gt - noisy_input
            terms["decoder_residual"] = F.mse_loss(predicted_residual, target_residual)
        else:
            terms["decoder_residual"] = x_hat.new_tensor(0.0)

        total = (
            self.ms_ssim_weight * terms["decoder_ms_ssim"]
            + self.l1_weight * terms["decoder_l1"]
            + self.mse_weight * terms["decoder_mse"]
            + self.residual_weight * terms["decoder_residual"]
        )
        return DecoderLossOutput(total_loss=total, terms=terms)

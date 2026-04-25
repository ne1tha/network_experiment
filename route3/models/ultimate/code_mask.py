from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CodeMaskOutput:
    masked: torch.Tensor
    channel_mask: torch.Tensor
    spatial_mask: torch.Tensor
    active_channels: torch.Tensor
    rate_ratio: torch.Tensor


class CodeMaskModule(nn.Module):
    """Top-k channel mask driven by rate-conditioned channel scores."""

    def __init__(self, enforce_pruning: bool = True):
        super().__init__()
        self.enforce_pruning = enforce_pruning

    def forward(self, x: torch.Tensor, channel_scores: torch.Tensor, rate_ratio: torch.Tensor | float) -> CodeMaskOutput:
        if x.ndim != 4:
            raise ValueError(f"CodeMaskModule expects a 4D tensor, got {x.ndim}D.")
        if channel_scores.ndim != 2:
            raise ValueError(f"channel_scores must be 2D, got {channel_scores.ndim}D.")

        batch_size, channels, _, _ = x.shape
        if channel_scores.shape != (batch_size, channels):
            raise ValueError(
                f"channel_scores shape must be {(batch_size, channels)}, got {tuple(channel_scores.shape)}."
            )

        if not torch.is_tensor(rate_ratio):
            rate_ratio = torch.tensor([rate_ratio], device=x.device, dtype=x.dtype)
        rate_ratio = rate_ratio.to(device=x.device, dtype=x.dtype).reshape(-1)
        if rate_ratio.numel() == 1:
            rate_ratio = rate_ratio.expand(batch_size)
        if rate_ratio.numel() != batch_size:
            raise ValueError(f"Expected scalar or batch-sized rate ratio for batch {batch_size}, got {rate_ratio.numel()}.")
        if torch.any(rate_ratio <= 0) or torch.any(rate_ratio > 1):
            raise ValueError(f"rate_ratio must be in (0, 1], got {rate_ratio}.")

        active_channels = torch.clamp(torch.round(rate_ratio * channels).to(dtype=torch.int64), min=1, max=channels)

        mask = torch.zeros((batch_size, channels), device=x.device, dtype=x.dtype)
        for batch_idx in range(batch_size):
            k = int(active_channels[batch_idx].item())
            topk = torch.topk(channel_scores[batch_idx], k=k, dim=0, largest=True, sorted=False)
            mask[batch_idx, topk.indices] = 1.0

        if self.enforce_pruning and torch.any(rate_ratio < 1.0):
            expected_pruned = active_channels < channels
            actual_pruned = mask.sum(dim=1).to(dtype=torch.int64) < channels
            if not torch.equal(expected_pruned, actual_pruned):
                raise RuntimeError("Code mask did not actually prune channels for a reduced rate request.")

        spatial_mask = mask.unsqueeze(-1).unsqueeze(-1)
        masked = x * spatial_mask
        return CodeMaskOutput(
            masked=masked,
            channel_mask=mask,
            spatial_mask=spatial_mask,
            active_channels=active_channels,
            rate_ratio=rate_ratio,
        )


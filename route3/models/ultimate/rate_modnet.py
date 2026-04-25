from __future__ import annotations

import torch
import torch.nn as nn


class RateModNet(nn.Module):
    """Rate-conditioned channel scoring module."""

    def __init__(self, channels: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(channels, 32)
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, channels),
        )

    def forward(self, x: torch.Tensor, rate_ratio: torch.Tensor | float) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"RateModNet expects a 4D tensor, got {x.ndim}D.")

        batch_size = x.shape[0]
        if not torch.is_tensor(rate_ratio):
            rate_ratio = torch.tensor([rate_ratio], device=x.device, dtype=x.dtype)
        rate_ratio = rate_ratio.to(device=x.device, dtype=x.dtype).reshape(-1)
        if rate_ratio.numel() == 1:
            rate_ratio = rate_ratio.expand(batch_size)
        if rate_ratio.numel() != batch_size:
            raise ValueError(f"Expected scalar or batch-sized rate ratio for batch {batch_size}, got {rate_ratio.numel()}.")
        if torch.any(rate_ratio <= 0) or torch.any(rate_ratio > 1):
            raise ValueError(f"rate_ratio must be in (0, 1], got {rate_ratio}.")

        content_score = x.mean(dim=(-2, -1))
        rate_score = self.net(rate_ratio.unsqueeze(-1))
        return torch.sigmoid(content_score + rate_score)


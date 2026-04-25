from __future__ import annotations

import torch
import torch.nn as nn


class ChannelModNet(nn.Module):
    """SNR-conditioned channel-wise modulation block."""

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
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"ChannelModNet expects a 4D tensor, got {x.ndim}D.")

        batch_size = x.shape[0]
        if not torch.is_tensor(snr_db):
            snr_db = torch.tensor([snr_db], device=x.device, dtype=x.dtype)
        snr_db = snr_db.to(device=x.device, dtype=x.dtype).reshape(-1)
        if snr_db.numel() == 1:
            snr_db = snr_db.expand(batch_size)
        if snr_db.numel() != batch_size:
            raise ValueError(f"Expected scalar or batch-sized SNR for batch {batch_size}, got {snr_db.numel()}.")

        gates = self.net(snr_db.unsqueeze(-1)).view(batch_size, self.channels, 1, 1)
        return x * gates, gates


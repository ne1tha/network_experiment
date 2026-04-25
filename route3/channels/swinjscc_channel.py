from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


ChannelType = Literal["awgn", "rayleigh"]


@dataclass(frozen=True)
class ChannelState:
    channel_type: ChannelType
    snr_db: torch.Tensor
    noise_std: torch.Tensor
    tx_power: torch.Tensor
    fading_gain: torch.Tensor | None = None


class DifferentiableChannelSimulator(nn.Module):
    """Differentiable AWGN / Rayleigh channel on real-valued feature tensors."""

    def __init__(self, channel_type: ChannelType = "awgn"):
        super().__init__()
        if channel_type not in {"awgn", "rayleigh"}:
            raise ValueError(f"Unsupported channel type: {channel_type}.")
        self.channel_type = channel_type

    @staticmethod
    def _expand_batch_scalar(value: torch.Tensor | float, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(value):
            value = torch.tensor([value], device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype).reshape(-1)
        if value.numel() == 1:
            return value.expand(batch_size)
        if value.numel() != batch_size:
            raise ValueError(f"Expected scalar or batch-sized tensor of length {batch_size}, got {value.numel()}.")
        return value

    @staticmethod
    def _to_iq_symbols(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        if x.ndim != 4:
            raise ValueError(f"Channel input must be 4D, got {x.ndim}D.")
        batch_size = x.shape[0]
        flattened = x.reshape(batch_size, -1)
        if flattened.shape[1] % 2 != 0:
            raise ValueError(
                "Channel input must contain an even number of real values per sample so it can be packed into complex symbols; "
                f"got {flattened.shape[1]}."
            )
        half = flattened.shape[1] // 2
        return flattened[:, :half], flattened[:, half:], x.shape

    @staticmethod
    def _from_iq_symbols(real: torch.Tensor, imag: torch.Tensor, output_shape: tuple[int, ...]) -> torch.Tensor:
        real = torch.cat([real, imag], dim=1)
        return real.reshape(output_shape)

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor | float) -> tuple[torch.Tensor, ChannelState]:
        batch_size = x.shape[0]
        snr_db = self._expand_batch_scalar(snr_db, batch_size, device=x.device, dtype=x.dtype)
        real_symbols, imag_symbols, output_shape = self._to_iq_symbols(x)

        power = (real_symbols.pow(2) + imag_symbols.pow(2)).mean(dim=1, keepdim=True).clamp_min(1e-8)
        power_scale = power.sqrt()
        norm_real = real_symbols / power_scale
        norm_imag = imag_symbols / power_scale

        noise_std = torch.sqrt(1.0 / (2.0 * torch.pow(10.0, snr_db / 10.0))).unsqueeze(1)
        noise_real = torch.randn_like(norm_real) * noise_std
        noise_imag = torch.randn_like(norm_imag) * noise_std

        fading_gain = None
        if self.channel_type == "rayleigh":
            fading_gain = torch.sqrt(
                torch.randn_like(norm_real).pow(2) + torch.randn_like(norm_real).pow(2)
            ) / torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))
            recv_real = norm_real * fading_gain + noise_real
            recv_imag = norm_imag * fading_gain + noise_imag
        else:
            recv_real = norm_real + noise_real
            recv_imag = norm_imag + noise_imag

        recv_real = recv_real * power_scale
        recv_imag = recv_imag * power_scale
        rx = self._from_iq_symbols(recv_real, recv_imag, output_shape)
        state = ChannelState(
            channel_type=self.channel_type,
            snr_db=snr_db,
            noise_std=noise_std.squeeze(1),
            tx_power=power.squeeze(1),
            fading_gain=fading_gain,
        )
        return rx, state

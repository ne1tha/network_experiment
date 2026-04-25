from __future__ import annotations

import math

import torch
import torch.nn as nn

from route2_swinjscc_gan.common.checks import require, require_finite


def _pack_complex(feature: torch.Tensor) -> torch.Tensor:
    flat = feature.reshape(-1)
    require(flat.numel() % 2 == 0, "Channel feature length must be even to form complex symbols.")
    half = flat.numel() // 2
    return flat[:half] + 1j * flat[half:]


def _unpack_complex(symbols: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    feature = torch.cat([symbols.real, symbols.imag], dim=0)
    return feature.reshape(shape)


def _normalize_feature_power(feature: torch.Tensor, power: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    current_power = torch.mean(feature**2) * 2.0
    require(current_power.item() > 0.0, "Feature power must be positive before channel normalization.")
    normalized = math.sqrt(power) * feature / torch.sqrt(current_power)
    return normalized, current_power


class AWGNChannel(nn.Module):
    def forward(self, feature: torch.Tensor, snr_db: int | float, avg_pwr: torch.Tensor | None = None) -> torch.Tensor:
        require_finite(feature, "channel_input")
        if avg_pwr is not None:
            tx_feature = feature / torch.sqrt(avg_pwr * 2.0)
            restored_scale = torch.sqrt(avg_pwr * 2.0)
        else:
            tx_feature, original_power = _normalize_feature_power(feature, power=1.0)
            restored_scale = torch.sqrt(original_power)

        channel_in = _pack_complex(tx_feature)
        sigma = math.sqrt(1.0 / (2.0 * (10.0 ** (float(snr_db) / 10.0))))
        noise = torch.complex(
            torch.normal(mean=0.0, std=sigma, size=channel_in.shape, device=channel_in.device),
            torch.normal(mean=0.0, std=sigma, size=channel_in.shape, device=channel_in.device),
        )
        channel_out = channel_in + noise
        reconstructed = _unpack_complex(channel_out, feature.shape) * restored_scale
        require_finite(reconstructed, "awgn_channel_output")
        return reconstructed


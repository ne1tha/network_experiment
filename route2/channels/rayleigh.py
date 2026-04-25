from __future__ import annotations

import math

import torch
import torch.nn as nn

from route2_swinjscc_gan.channels.awgn import _normalize_feature_power, _pack_complex, _unpack_complex
from route2_swinjscc_gan.common.checks import require_finite


class RayleighFadingChannel(nn.Module):
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
        fading = torch.sqrt(
            torch.normal(mean=0.0, std=1.0, size=channel_in.shape, device=channel_in.device) ** 2
            + torch.normal(mean=0.0, std=1.0, size=channel_in.shape, device=channel_in.device) ** 2
        ) / math.sqrt(2.0)
        noise = torch.complex(
            torch.normal(mean=0.0, std=sigma, size=channel_in.shape, device=channel_in.device),
            torch.normal(mean=0.0, std=sigma, size=channel_in.shape, device=channel_in.device),
        )
        channel_out = channel_in * fading + noise
        reconstructed = _unpack_complex(channel_out, feature.shape) * restored_scale
        require_finite(reconstructed, "rayleigh_channel_output")
        return reconstructed

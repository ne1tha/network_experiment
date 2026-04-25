from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch
import torch.nn as nn

from route2_swinjscc_gan.channels import build_channel
from route2_swinjscc_gan.common.checks import (
    require,
    require_finite,
    require_in,
    require_non_empty_sequence,
    require_same_shape,
)
from route2_swinjscc_gan.models.swinjscc_gan.third_party_bridge import load_third_party_swinjscc


SUPPORTED_VARIANTS = {
    "SwinJSCC_w/o_SAandRA",
    "SwinJSCC_w/_SA",
    "SwinJSCC_w/_RA",
    "SwinJSCC_w/_SAandRA",
}

SUPPORTED_CHANNELS = {"awgn", "rayleigh"}


@dataclass(frozen=True)
class SwinJSCCGeneratorConfig:
    model_variant: str
    encoder_kwargs: dict[str, Any]
    decoder_kwargs: dict[str, Any]
    multiple_snr: tuple[int, ...]
    channel_numbers: tuple[int, ...]
    channel_type: str = "rayleigh"
    pass_channel: bool = True
    downsample: int = 4
    device: str = "cuda"

    def __post_init__(self) -> None:
        require_in("model_variant", self.model_variant, SUPPORTED_VARIANTS)
        require_in("channel_type", self.channel_type, SUPPORTED_CHANNELS)
        require_non_empty_sequence("multiple_snr", self.multiple_snr)
        require_non_empty_sequence("channel_numbers", self.channel_numbers)


@dataclass(frozen=True)
class GeneratorOutput:
    reconstruction: torch.Tensor
    cbr: float
    snr: int
    rate: int
    noisy_feature: torch.Tensor
    mask: torch.Tensor | None


class SwinJSCCGenerator(nn.Module):
    """Route 2 generator wrapper around the original SwinJSCC backbone."""

    def __init__(self, config: SwinJSCCGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        modules = load_third_party_swinjscc()

        self.encoder = modules.encoder_module.create_encoder(**config.encoder_kwargs)
        self.decoder = modules.decoder_module.create_decoder(**config.decoder_kwargs)
        self.channel = build_channel(config.channel_type)
        configured_size = config.encoder_kwargs.get("img_size", (0, 0))
        if isinstance(configured_size, tuple) and len(configured_size) == 2:
            self.height, self.width = int(configured_size[0]), int(configured_size[1])
        else:
            self.height = 0
            self.width = 0

    def _required_spatial_multiple(self) -> int:
        patch_size = self.config.encoder_kwargs.get("patch_size", 2)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        window_size = int(self.config.encoder_kwargs.get("window_size", 8))
        num_layers = len(self.config.encoder_kwargs.get("depths", ()))
        require(num_layers > 0, "Encoder depths must not be empty.")
        return int(patch_size) * window_size * (2 ** (num_layers - 1))

    def _choose_snr(self, snr: int | None) -> int:
        if snr is None:
            return int(random.choice(self.config.multiple_snr))
        if snr not in self.config.multiple_snr:
            raise ValueError(f"SNR {snr} is not part of configured training range {self.config.multiple_snr}.")
        return int(snr)

    def _choose_rate(self, rate: int | None) -> int:
        if rate is None:
            return int(random.choice(self.config.channel_numbers))
        if rate not in self.config.channel_numbers:
            raise ValueError(f"Rate {rate} is not part of configured rate set {self.config.channel_numbers}.")
        return int(rate)

    def sample_training_conditions(self, *, snr: int | None = None, rate: int | None = None) -> tuple[int, int]:
        return self._choose_snr(snr), self._choose_rate(rate)

    def _update_resolution(self, height: int, width: int) -> None:
        if height == self.height and width == self.width:
            return
        self.encoder.update_resolution(height, width)
        self.decoder.update_resolution(height // (2 ** self.config.downsample), width // (2 ** self.config.downsample))
        self.height = height
        self.width = width

    def _pass_channel(self, feature: torch.Tensor, snr: int, avg_pwr: torch.Tensor | None) -> torch.Tensor:
        if not self.config.pass_channel:
            return feature
        if avg_pwr is None:
            return self.channel.forward(feature, snr)
        return self.channel.forward(feature, snr, avg_pwr)

    def forward(self, input_image: torch.Tensor, *, snr: int | None = None, rate: int | None = None) -> GeneratorOutput:
        require(input_image.ndim == 4, f"Generator expects NCHW input, got shape {tuple(input_image.shape)}.")
        batch_size, channels, height, width = input_image.shape
        require(channels == 3, f"Generator expects RGB images, got shape {tuple(input_image.shape)}.")

        required_multiple = self._required_spatial_multiple()
        require(
            height % required_multiple == 0 and width % required_multiple == 0,
            f"Input resolution {(height, width)} must be divisible by {required_multiple}.",
        )

        selected_snr = self._choose_snr(snr)
        selected_rate = self._choose_rate(rate)
        self._update_resolution(height, width)

        variant = self.config.model_variant
        if variant in {"SwinJSCC_w/o_SAandRA", "SwinJSCC_w/_SA"}:
            feature = self.encoder(input_image, selected_snr, selected_rate, variant)
            cbr = feature.numel() / (2.0 * input_image.numel())
            mask = None
            noisy_feature = self._pass_channel(feature, selected_snr, avg_pwr=None)
        else:
            feature, mask = self.encoder(input_image, selected_snr, selected_rate, variant)
            require(mask is not None, "Rate-adaptive generator path must produce a binary mask.")
            require(mask.shape[0] == batch_size, "Rate-adaptive mask batch size mismatch.")
            require(mask.sum().item() > 0, "Rate-adaptive mask selected zero channels.")
            avg_pwr = torch.sum(feature**2) / mask.sum()
            noisy_feature = self._pass_channel(feature, selected_snr, avg_pwr=avg_pwr)
            noisy_feature = noisy_feature * mask
            cbr = selected_rate / float(2 * 3 * (2 ** (self.config.downsample * 2)))

        reconstruction = self.decoder(noisy_feature, selected_snr, variant).clamp(0.0, 1.0)
        require_same_shape("reconstruction", reconstruction, "input_image", input_image)
        require_finite(reconstruction, "generator_reconstruction")

        return GeneratorOutput(
            reconstruction=reconstruction,
            cbr=float(cbr),
            snr=selected_snr,
            rate=selected_rate,
            noisy_feature=noisy_feature,
            mask=mask,
        )

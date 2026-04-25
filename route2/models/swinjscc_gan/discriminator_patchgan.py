from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn

from route2_swinjscc_gan.common.checks import require, require_finite, require_in, require_positive_int, require_same_shape


SUPPORTED_PATCHGAN_KINDS = {"legacy_patchgan", "conditional_multiscale_v1"}
SUPPORTED_NORM_TYPES = {"none", "instance", "batch"}


@dataclass(frozen=True)
class PatchGANConfig:
    kind: str = "legacy_patchgan"
    image_channels: int = 3
    base_channels: int = 64
    max_channels: int = 512
    num_downsampling_layers: int = 4
    negative_slope: float = 0.2
    norm_type: str = "instance"
    use_spectral_norm: bool = False

    def __post_init__(self) -> None:
        require_in("kind", self.kind, SUPPORTED_PATCHGAN_KINDS)
        require_positive_int("image_channels", self.image_channels)
        require_positive_int("base_channels", self.base_channels)
        require_positive_int("max_channels", self.max_channels)
        require_positive_int("num_downsampling_layers", self.num_downsampling_layers)
        require_in("norm_type", self.norm_type, SUPPORTED_NORM_TYPES)
        if self.kind == "conditional_multiscale_v1":
            require(
                self.num_downsampling_layers >= 2,
                "Conditional multi-scale PatchGAN requires at least two downsampling stages.",
            )


def build_norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    return nn.BatchNorm2d(channels)


def build_conv(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int,
    padding: int,
    bias: bool,
    use_spectral_norm: bool,
) -> nn.Conv2d:
    layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    if use_spectral_norm:
        layer = nn.utils.spectral_norm(layer)
    return layer


class PatchGANDiscriminator(nn.Module):
    """Route 2 PatchGAN family.

    `legacy_patchgan` preserves the old unconditional discriminator so historical
    checkpoints remain loadable. `conditional_multiscale_v1` is the new Route 2
    discriminator: it judges `(source, candidate)` pairs and emits two patch maps
    at different receptive-field scales.
    """

    def __init__(self, config: PatchGANConfig | None = None) -> None:
        super().__init__()
        self.config = config or PatchGANConfig()

        layers: list[nn.Module] = []
        feature_channels: list[int] = []
        in_channels = self._expected_input_channels()
        out_channels = self.config.base_channels

        for _ in range(self.config.num_downsampling_layers):
            block = nn.Sequential(
                build_conv(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=self.config.norm_type == "none",
                    use_spectral_norm=self.config.use_spectral_norm,
                ),
                build_norm(self.config.norm_type, out_channels),
                nn.LeakyReLU(self.config.negative_slope, inplace=True),
            )
            layers.append(block)
            feature_channels.append(out_channels)
            in_channels = out_channels
            out_channels = min(out_channels * 2, self.config.max_channels)

        self.downsample_blocks = nn.ModuleList(layers)
        if self.config.kind == "conditional_multiscale_v1":
            self.head_feature_indices = (-2, -1)
        else:
            self.head_feature_indices = (-1,)
        self.output_heads = nn.ModuleList(
            [
                build_conv(
                    feature_channels[index],
                    1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=True,
                    use_spectral_norm=self.config.use_spectral_norm,
                )
                for index in self.head_feature_indices
            ]
        )

    def _expected_input_channels(self) -> int:
        if self.config.kind == "conditional_multiscale_v1":
            return self.config.image_channels * 2
        return self.config.image_channels

    def _compose_input(self, source: torch.Tensor, candidate: torch.Tensor | None) -> torch.Tensor:
        require(source.ndim == 4, f"PatchGAN expects an NCHW tensor, got shape {tuple(source.shape)}.")
        require(
            source.shape[1] == self.config.image_channels,
            f"PatchGAN expects source tensors with {self.config.image_channels} channels, got {tuple(source.shape)}.",
        )
        if self.config.kind == "legacy_patchgan":
            image = candidate if candidate is not None else source
            require(image.ndim == 4, f"PatchGAN expects an NCHW tensor, got shape {tuple(image.shape)}.")
            require(
                image.shape[1] == self.config.image_channels,
                f"PatchGAN expects {self.config.image_channels} channels, got shape {tuple(image.shape)}.",
            )
            return image

        paired_candidate = source if candidate is None else candidate
        require_same_shape("candidate", paired_candidate, "source", source)
        return torch.cat([source, paired_candidate], dim=1)

    def forward(self, source: torch.Tensor, candidate: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, ...]:
        image = self._compose_input(source, candidate)
        require(
            image.shape[-2] >= 2 ** self.config.num_downsampling_layers
            and image.shape[-1] >= 2 ** self.config.num_downsampling_layers,
            "Input image is too small for the configured PatchGAN depth.",
        )

        feature_maps: list[torch.Tensor] = []
        feature_map = image
        for block in self.downsample_blocks:
            feature_map = block(feature_map)
            feature_maps.append(feature_map)

        logits = tuple(head(feature_maps[index]) for head, index in zip(self.output_heads, self.head_feature_indices))
        for logit in logits:
            require(logit.shape[-2] > 1 and logit.shape[-1] > 1, "PatchGAN output must remain a patch map.")
            require_finite(logit, "patchgan_logits")
        if len(logits) == 1:
            return logits[0]
        return logits

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):  # type: ignore[override]
        remapped = OrderedDict(state_dict)
        if self.config.kind == "legacy_patchgan":
            legacy_weight = remapped.pop("output_conv.weight", None)
            legacy_bias = remapped.pop("output_conv.bias", None)
            if legacy_weight is not None and "output_heads.0.weight" not in remapped:
                remapped["output_heads.0.weight"] = legacy_weight
            if legacy_bias is not None and "output_heads.0.bias" not in remapped:
                remapped["output_heads.0.bias"] = legacy_bias
        return super().load_state_dict(remapped, strict=strict)

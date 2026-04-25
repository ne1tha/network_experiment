from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

try:
    from torchvision import models as tv_models
except Exception:  # pragma: no cover
    tv_models = None

from .decoder_ssdd import DecoderOutput


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


@dataclass(frozen=True)
class EnhancementStatus:
    perceptual_enabled: bool
    adversarial_enabled: bool
    vgg_backbone_ready: bool
    discriminator_ready: bool
    vgg_source: str


@dataclass(frozen=True)
class EnhancementLossOutput:
    refined_reconstruction: DecoderOutput
    generator_total_loss: torch.Tensor
    generator_terms: dict[str, torch.Tensor]
    discriminator_loss: torch.Tensor | None
    status: EnhancementStatus


class RefinementResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResidualImageRefiner(nn.Module):
    """Predict a residual correction using both the decoded image and decoder context."""

    def __init__(
        self,
        image_channels: int = 3,
        hidden_channels: int = 64,
        conditioning_channels: int = 128,
        context_dim: int = 512,
        num_blocks: int = 4,
    ):
        super().__init__()
        if image_channels <= 0:
            raise ValueError(f"image_channels must be positive, got {image_channels}")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        if conditioning_channels <= 0:
            raise ValueError(f"conditioning_channels must be positive, got {conditioning_channels}")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        norm_groups = 8 if hidden_channels % 8 == 0 else 1
        self.image_channels = image_channels
        self.image_stem = nn.Sequential(
            nn.Conv2d(image_channels * 3, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.SiLU(),
        )
        self.condition_adapters = nn.ModuleDict(
            {
                "conditioning": nn.Sequential(
                    nn.Conv2d(conditioning_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(norm_groups, hidden_channels),
                    nn.SiLU(),
                ),
                "semantic": nn.Sequential(
                    nn.Conv2d(conditioning_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(norm_groups, hidden_channels),
                    nn.SiLU(),
                ),
                "detail": nn.Sequential(
                    nn.Conv2d(conditioning_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(norm_groups, hidden_channels),
                    nn.SiLU(),
                ),
                "bottleneck": nn.Sequential(
                    nn.Conv2d(conditioning_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(norm_groups, hidden_channels),
                    nn.SiLU(),
                ),
            }
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.SiLU(),
        )
        self.context_proj = nn.Linear(context_dim, hidden_channels * 2)
        self.blocks = nn.Sequential(*[RefinementResBlock(hidden_channels) for _ in range(num_blocks)])
        self.head = nn.Conv2d(hidden_channels, image_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _resize_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-2:] == reference.shape[-2:]:
            return tensor
        return F.interpolate(tensor, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        base_image: torch.Tensor,
        *,
        predicted_residual: torch.Tensor | None = None,
        noisy_input: torch.Tensor | None = None,
        conditioning_map: torch.Tensor | None = None,
        semantic_condition_map: torch.Tensor | None = None,
        detail_condition_map: torch.Tensor | None = None,
        bottleneck_condition_map: torch.Tensor | None = None,
        context_vector: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(base_image)
        image_input = torch.cat(
            [
                base_image,
                predicted_residual if predicted_residual is not None else zeros,
                noisy_input if noisy_input is not None else zeros,
            ],
            dim=1,
        )
        hidden = self.image_stem(image_input)
        condition_hidden = None
        for name, tensor in (
            ("conditioning", conditioning_map),
            ("semantic", semantic_condition_map),
            ("detail", detail_condition_map),
            ("bottleneck", bottleneck_condition_map),
        ):
            if tensor is None:
                continue
            adapted = self.condition_adapters[name](self._resize_like(tensor, base_image))
            condition_hidden = adapted if condition_hidden is None else condition_hidden + adapted
        if condition_hidden is not None:
            hidden = self.fuse(torch.cat([hidden, condition_hidden], dim=1))
        if context_vector is not None:
            context_vector = context_vector.to(device=base_image.device, dtype=base_image.dtype)
            scale, shift = self.context_proj(context_vector).chunk(2, dim=1)
            hidden = hidden * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        hidden = self.blocks(hidden)
        refinement_delta = self.head(hidden)
        refined = (base_image + refinement_delta).clamp(0.0, 1.0)
        return refined, refinement_delta


SUPPORTED_PATCHGAN_KINDS = {"legacy_patchgan", "conditional_multiscale_v1", "conditional_dual_branch_v2"}
SUPPORTED_NORM_TYPES = {"none", "instance", "batch", "group"}


@dataclass(frozen=True)
class PatchGANConfig:
    kind: str = "conditional_dual_branch_v2"
    image_channels: int = 3
    base_channels: int = 64
    max_channels: int = 512
    num_downsampling_layers: int = 4
    negative_slope: float = 0.2
    norm_type: str = "none"
    use_spectral_norm: bool = True

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_PATCHGAN_KINDS:
            raise ValueError(f"Unsupported PatchGAN kind: {self.kind}")
        if self.image_channels <= 0:
            raise ValueError(f"image_channels must be positive, got {self.image_channels}")
        if self.base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {self.base_channels}")
        if self.max_channels <= 0:
            raise ValueError(f"max_channels must be positive, got {self.max_channels}")
        if self.num_downsampling_layers <= 0:
            raise ValueError(
                f"num_downsampling_layers must be positive, got {self.num_downsampling_layers}"
            )
        if self.kind in {"conditional_multiscale_v1", "conditional_dual_branch_v2"} and self.num_downsampling_layers < 2:
            raise ValueError(f"{self.kind} requires at least two downsampling layers.")
        if self.norm_type not in SUPPORTED_NORM_TYPES:
            raise ValueError(f"Unsupported PatchGAN norm_type: {self.norm_type}")


def _build_norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    return nn.GroupNorm(1, channels)


def _build_conv(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int,
    padding: int,
    bias: bool,
    use_spectral_norm: bool,
) -> nn.Module:
    layer: nn.Module = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    if use_spectral_norm:
        layer = spectral_norm(layer)
    return layer


def _as_logit_maps(logits: torch.Tensor | Sequence[torch.Tensor], *, label: str) -> tuple[torch.Tensor, ...]:
    if isinstance(logits, torch.Tensor):
        logit_maps = (logits,)
    else:
        logit_maps = tuple(logits)

    if not logit_maps:
        raise ValueError(f"{label} must contain at least one patch-logit tensor.")

    for index, logit in enumerate(logit_maps):
        if not isinstance(logit, torch.Tensor):
            raise TypeError(f"{label}[{index}] must be a tensor.")
        if logit.ndim < 3:
            raise ValueError(f"{label}[{index}] must be patch-level logits, got shape {tuple(logit.shape)}.")
        if not torch.isfinite(logit).all():
            raise ValueError(f"{label}[{index}] contains non-finite values.")
    return logit_maps


def _generator_hinge_loss(fake_logits: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    fake_logit_maps = _as_logit_maps(fake_logits, label="fake_logits")
    return sum(-logit.mean() for logit in fake_logit_maps) / float(len(fake_logit_maps))


def _discriminator_hinge_loss(
    real_logits: torch.Tensor | Sequence[torch.Tensor],
    fake_logits: torch.Tensor | Sequence[torch.Tensor],
) -> torch.Tensor:
    real_logit_maps = _as_logit_maps(real_logits, label="real_logits")
    fake_logit_maps = _as_logit_maps(fake_logits, label="fake_logits")
    if len(real_logit_maps) != len(fake_logit_maps):
        raise ValueError("Real and fake PatchGAN outputs must expose the same number of scales.")

    losses: list[torch.Tensor] = []
    for index, (real_logit, fake_logit) in enumerate(zip(real_logit_maps, fake_logit_maps)):
        if real_logit.shape != fake_logit.shape:
            raise ValueError(
                "Real and fake PatchGAN logits must share the same shape at "
                f"scale {index}, got {tuple(real_logit.shape)} vs {tuple(fake_logit.shape)}."
            )
        loss_real = F.relu(1.0 - real_logit).mean()
        loss_fake = F.relu(1.0 + fake_logit).mean()
        losses.append(0.5 * (loss_real + loss_fake))
    return sum(losses) / float(len(losses))


class PatchGANDiscriminator(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 4,
        *,
        kind: str = "conditional_dual_branch_v2",
        max_channels: int = 512,
        negative_slope: float = 0.2,
        norm_type: str | None = None,
        use_spectral_norm: bool | None = None,
        config: PatchGANConfig | None = None,
    ):
        super().__init__()
        if config is None:
            resolved_use_spectral_norm = True if use_spectral_norm is None else bool(use_spectral_norm)
            resolved_norm_type = "none" if norm_type is None else norm_type
            config = PatchGANConfig(
                kind=kind,
                image_channels=input_channels,
                base_channels=base_channels,
                max_channels=max_channels,
                num_downsampling_layers=n_layers,
                negative_slope=negative_slope,
                norm_type=resolved_norm_type,
                use_spectral_norm=resolved_use_spectral_norm,
            )
        self.config = config

        if self.config.kind == "conditional_dual_branch_v2":
            feature_channels = self._feature_channels()
            self.source_downsample_blocks = self._build_downsample_blocks(input_channels=self.config.image_channels)
            self.candidate_downsample_blocks = self._build_downsample_blocks(input_channels=self.config.image_channels)
            self.head_feature_indices = (-2, -1)
            self.fusion_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        _build_conv(
                            feature_channels[index] * 4,
                            feature_channels[index],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=self.config.norm_type == "none",
                            use_spectral_norm=self.config.use_spectral_norm,
                        ),
                        _build_norm(self.config.norm_type, feature_channels[index]),
                        nn.LeakyReLU(self.config.negative_slope, inplace=True),
                    )
                    for index in self.head_feature_indices
                ]
            )
            self.output_heads = nn.ModuleList(
                [
                    _build_conv(
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
        else:
            feature_channels = self._feature_channels()
            self.downsample_blocks = self._build_downsample_blocks(input_channels=self._expected_input_channels())
            self.head_feature_indices = (-2, -1) if self.config.kind == "conditional_multiscale_v1" else (-1,)
            self.output_heads = nn.ModuleList(
                [
                    _build_conv(
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

    def _feature_channels(self) -> list[int]:
        channels: list[int] = []
        out_channels = self.config.base_channels
        for _ in range(self.config.num_downsampling_layers):
            channels.append(out_channels)
            out_channels = min(out_channels * 2, self.config.max_channels)
        return channels

    def _build_downsample_blocks(self, *, input_channels: int) -> nn.ModuleList:
        blocks: list[nn.Module] = []
        in_channels = input_channels
        out_channels = self.config.base_channels
        for _ in range(self.config.num_downsampling_layers):
            blocks.append(
                nn.Sequential(
                    _build_conv(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=self.config.norm_type == "none",
                        use_spectral_norm=self.config.use_spectral_norm,
                    ),
                    _build_norm(self.config.norm_type, out_channels),
                    nn.LeakyReLU(self.config.negative_slope, inplace=True),
                )
            )
            in_channels = out_channels
            out_channels = min(out_channels * 2, self.config.max_channels)
        return nn.ModuleList(blocks)

    def _expected_input_channels(self) -> int:
        if self.config.kind == "conditional_multiscale_v1":
            return self.config.image_channels * 2
        return self.config.image_channels

    def _compose_input(self, source: torch.Tensor, candidate: torch.Tensor | None) -> torch.Tensor:
        if source.ndim != 4:
            raise ValueError(f"PatchGANDiscriminator expects BCHW input, got shape {tuple(source.shape)}.")
        if source.shape[1] != self.config.image_channels:
            raise ValueError(
                f"PatchGANDiscriminator expects {self.config.image_channels} source channels, "
                f"got shape {tuple(source.shape)}."
            )

        if self.config.kind == "legacy_patchgan":
            image = source if candidate is None else candidate
            if image.ndim != 4:
                raise ValueError(f"PatchGANDiscriminator expects BCHW input, got shape {tuple(image.shape)}.")
            if image.shape[1] != self.config.image_channels:
                raise ValueError(
                    f"PatchGANDiscriminator expects {self.config.image_channels} channels, got {tuple(image.shape)}."
                )
            return image.clamp(0.0, 1.0)

        paired_candidate = source if candidate is None else candidate
        if paired_candidate.shape != source.shape:
            raise ValueError(
                "Conditional PatchGAN requires source and candidate to share the same shape, got "
                f"{tuple(source.shape)} vs {tuple(paired_candidate.shape)}."
            )
        source = source.clamp(0.0, 1.0)
        paired_candidate = paired_candidate.clamp(0.0, 1.0)
        if self.config.kind == "conditional_multiscale_v1":
            return torch.cat([source, paired_candidate], dim=1)
        return source, paired_candidate

    @staticmethod
    def _fuse_branch_features(source_feature: torch.Tensor, candidate_feature: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                source_feature,
                candidate_feature,
                torch.abs(source_feature - candidate_feature),
                source_feature * candidate_feature,
            ],
            dim=1,
        )

    def forward(
        self,
        source: torch.Tensor,
        candidate: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        image = self._compose_input(source, candidate)
        min_side = 2 ** (self.config.num_downsampling_layers + 1)
        sample = image[0] if isinstance(image, tuple) else image
        if sample.shape[-2] < min_side or sample.shape[-1] < min_side:
            raise ValueError(
                f"Input image is too small for PatchGAN depth {self.config.num_downsampling_layers}; "
                f"expected both spatial dims >= {min_side}, got {tuple(sample.shape[-2:])}."
            )

        if self.config.kind == "conditional_dual_branch_v2":
            source_image, candidate_image = image
            source_feature_maps: list[torch.Tensor] = []
            candidate_feature_maps: list[torch.Tensor] = []
            source_feature = source_image
            candidate_feature = candidate_image
            for source_block, candidate_block in zip(self.source_downsample_blocks, self.candidate_downsample_blocks):
                source_feature = source_block(source_feature)
                candidate_feature = candidate_block(candidate_feature)
                source_feature_maps.append(source_feature)
                candidate_feature_maps.append(candidate_feature)

            logits = tuple(
                head(
                    fusion_block(
                        self._fuse_branch_features(
                            source_feature_maps[index],
                            candidate_feature_maps[index],
                        )
                    )
                )
                for fusion_block, head, index in zip(self.fusion_blocks, self.output_heads, self.head_feature_indices)
            )
        else:
            feature_maps: list[torch.Tensor] = []
            feature_map = image
            for block in self.downsample_blocks:
                feature_map = block(feature_map)
                feature_maps.append(feature_map)

            logits = tuple(head(feature_maps[index]) for head, index in zip(self.output_heads, self.head_feature_indices))
        for index, logit in enumerate(logits):
            if logit.shape[-2] <= 1 or logit.shape[-1] <= 1:
                raise ValueError(
                    f"PatchGAN head {index} collapsed to a non-patch output with shape {tuple(logit.shape)}."
                )
            if not torch.isfinite(logit).all():
                raise ValueError(f"PatchGAN head {index} produced non-finite logits.")

        if len(logits) == 1:
            return logits[0]
        return logits


class VGGFeatureExtractor(nn.Module):
    """Frozen VGG feature extractor for perceptual loss."""

    def __init__(
        self,
        layers: Sequence[int] = (3, 8, 17, 26),
        checkpoint_path: str | Path | None = None,
        feature_extractor: nn.Module | None = None,
        allow_untrained: bool = False,
    ):
        super().__init__()
        self.layers = tuple(layers)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
        self.allow_untrained = allow_untrained

        if feature_extractor is not None:
            self.features = feature_extractor
            self.source = "injected"
        else:
            self.features, self.source = self._build_vgg_features(checkpoint_path, allow_untrained)

        self.features.eval()
        self.features.requires_grad_(False)
        self.max_layer = max(self.layers)

    @staticmethod
    def _build_vgg_features(checkpoint_path: str | Path | None, allow_untrained: bool) -> tuple[nn.Module, str]:
        if tv_models is None:
            raise RuntimeError("torchvision is unavailable, so VGG perceptual loss cannot be initialized.")

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"VGG checkpoint not found: {checkpoint_path}")
            model = tv_models.vgg19(weights=None)
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            return model.features, str(checkpoint_path)

        if allow_untrained:
            model = tv_models.vgg19(weights=None)
            return model.features, "torchvision:vgg19-random-init"

        raise RuntimeError(
            "VGG perceptual loss requires an explicit checkpoint, an injected extractor, "
            "or allow_untrained=True. Route 3 will not silently proceed without a perceptual backbone."
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"VGGFeatureExtractor expects BCHW RGB input, got {tuple(x.shape)}.")
        x = _normalize_imagenet(x)
        outputs = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.layers:
                outputs.append(x)
            if idx >= self.max_layer:
                break
        if len(outputs) != len(self.layers):
            raise RuntimeError(
                f"Requested VGG layers {self.layers} but only extracted {len(outputs)} feature maps."
            )
        return tuple(outputs)


class VGGPerceptualLoss(nn.Module):
    def __init__(
        self,
        feature_extractor: VGGFeatureExtractor,
        layer_weights: Sequence[float] | None = None,
        loss_scale: float = 1.0,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        if layer_weights is None:
            layer_weights = tuple(1.0 for _ in feature_extractor.layers)
        if len(layer_weights) != len(feature_extractor.layers):
            raise ValueError(
                f"Expected {len(feature_extractor.layers)} layer weights, got {len(layer_weights)}."
            )
        if loss_scale < 0.0:
            raise ValueError(f"loss_scale must be non-negative, got {loss_scale}.")
        self.register_buffer("layer_weights", torch.tensor(layer_weights, dtype=torch.float32), persistent=False)
        self.loss_scale = float(loss_scale)

    def raw_loss(self, x_gt: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        gt_features = self.feature_extractor(x_gt)
        pred_features = self.feature_extractor(x_pred)
        total = x_gt.new_tensor(0.0)
        for idx, (gt_feat, pred_feat) in enumerate(zip(gt_features, pred_features)):
            weight = self.layer_weights[idx].to(device=x_gt.device, dtype=x_gt.dtype)
            total = total + weight * F.l1_loss(pred_feat, gt_feat)
        return total

    def forward(self, x_gt: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        return self.raw_loss(x_gt, x_pred) * self.loss_scale


class PerceptualAdversarialEnhancer(nn.Module):
    """Training-time VGG perceptual + PatchGAN objective for route 3 phase 5."""

    def __init__(
        self,
        perceptual_loss: VGGPerceptualLoss | None = None,
        discriminator: PatchGANDiscriminator | None = None,
        refiner: nn.Module | None = None,
        perceptual_weight: float = 1.0,
        adversarial_weight: float = 0.1,
    ):
        super().__init__()
        self.perceptual_loss = perceptual_loss
        self.discriminator = discriminator
        self.refiner = refiner or ResidualImageRefiner()
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight

    def status(
        self,
        *,
        enable_perceptual: bool | None = None,
        enable_adversarial: bool | None = None,
    ) -> EnhancementStatus:
        perceptual_ready = self.perceptual_loss is not None
        adversarial_ready = self.discriminator is not None
        perceptual_enabled = perceptual_ready if enable_perceptual is None else bool(enable_perceptual) and perceptual_ready
        adversarial_enabled = adversarial_ready if enable_adversarial is None else bool(enable_adversarial) and adversarial_ready
        vgg_source = "disabled"
        if perceptual_ready:
            vgg_source = self.perceptual_loss.feature_extractor.source
        return EnhancementStatus(
            perceptual_enabled=perceptual_enabled,
            adversarial_enabled=adversarial_enabled,
            vgg_backbone_ready=perceptual_ready,
            discriminator_ready=adversarial_ready,
            vgg_source=vgg_source,
        )

    def _discriminator_forward(
        self,
        source: torch.Tensor,
        candidate: torch.Tensor | None,
        *,
        update_stats: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self.discriminator is None:
            raise RuntimeError("Discriminator forward was requested but no PatchGAN discriminator is enabled.")
        if update_stats or not self.discriminator.training:
            return self.discriminator(source, candidate)

        previous_training = self.discriminator.training
        self.discriminator.eval()
        try:
            return self.discriminator(source, candidate)
        finally:
            self.discriminator.train(previous_training)

    def refine_reconstruction(
        self,
        *,
        x_pred: torch.Tensor | None = None,
        base_reconstruction: DecoderOutput | None = None,
    ) -> DecoderOutput:
        if base_reconstruction is None:
            if x_pred is None:
                raise ValueError("Enhancement refinement requires either x_pred or base_reconstruction.")
            base_reconstruction = DecoderOutput(x_hat=x_pred, output_kind="base")

        base_x = base_reconstruction.x_hat
        refined_x, refinement_delta = self.refiner(
            base_x,
            predicted_residual=base_reconstruction.predicted_residual,
            noisy_input=base_reconstruction.noisy_input,
            conditioning_map=base_reconstruction.conditioning_map,
            semantic_condition_map=base_reconstruction.semantic_condition_map,
            detail_condition_map=base_reconstruction.detail_condition_map,
            bottleneck_condition_map=base_reconstruction.bottleneck_condition_map,
            context_vector=base_reconstruction.context_vector,
        )
        return DecoderOutput(
            x_hat=refined_x,
            predicted_residual=base_reconstruction.predicted_residual,
            noisy_input=base_reconstruction.noisy_input,
            conditioning_map=base_reconstruction.conditioning_map,
            semantic_condition_map=base_reconstruction.semantic_condition_map,
            detail_condition_map=base_reconstruction.detail_condition_map,
            bottleneck_condition_map=base_reconstruction.bottleneck_condition_map,
            context_vector=base_reconstruction.context_vector,
            decode_steps=base_reconstruction.decode_steps,
            stochastic=base_reconstruction.stochastic,
            base_x_hat=base_x,
            refinement_delta=refinement_delta,
            output_kind="final",
        )

    def forward(
        self,
        x_gt: torch.Tensor,
        x_pred: torch.Tensor | None = None,
        *,
        base_reconstruction: DecoderOutput | None = None,
        compute_discriminator_loss: bool = True,
        freeze_discriminator_stats: bool = False,
        enable_perceptual: bool | None = None,
        enable_adversarial: bool | None = None,
    ) -> EnhancementLossOutput:
        refined_reconstruction = self.refine_reconstruction(
            x_pred=x_pred,
            base_reconstruction=base_reconstruction,
        )
        x_pred = refined_reconstruction.base_x_hat
        if x_pred is None:
            raise RuntimeError("Refined reconstruction must retain the base decoder image.")
        source_condition = x_pred.detach()
        refined_x = refined_reconstruction.x_hat
        refinement_delta = refined_reconstruction.refinement_delta
        if refinement_delta is None:
            raise RuntimeError("Refined reconstruction must retain the refinement residual.")

        status = self.status(
            enable_perceptual=enable_perceptual,
            enable_adversarial=enable_adversarial,
        )
        if not status.perceptual_enabled and not status.adversarial_enabled:
            raise RuntimeError("PerceptualAdversarialEnhancer has no active perceptual or adversarial objective.")

        generator_terms: dict[str, torch.Tensor] = {}
        total = x_gt.new_tensor(0.0)

        if status.perceptual_enabled and self.perceptual_loss is not None:
            perceptual_raw = self.perceptual_loss.raw_loss(x_gt, refined_x)
            perceptual = perceptual_raw * self.perceptual_loss.loss_scale
            perceptual_effective = self.perceptual_weight * perceptual
            generator_terms["perceptual_vgg_raw"] = perceptual_raw
            generator_terms["perceptual_vgg"] = perceptual
            generator_terms["perceptual_vgg_effective"] = perceptual_effective
            total = total + perceptual_effective
        else:
            generator_terms["perceptual_vgg_raw"] = x_gt.new_tensor(0.0)
            generator_terms["perceptual_vgg"] = x_gt.new_tensor(0.0)
            generator_terms["perceptual_vgg_effective"] = x_gt.new_tensor(0.0)

        discriminator_loss = None
        if status.adversarial_enabled and self.discriminator is not None:
            logits_fake_for_g = self._discriminator_forward(
                source_condition,
                refined_x,
                update_stats=not freeze_discriminator_stats,
            )
            generator_gan = _generator_hinge_loss(logits_fake_for_g)
            generator_terms["adversarial_g"] = generator_gan
            generator_terms["adversarial_g_effective"] = self.adversarial_weight * generator_gan
            total = total + generator_terms["adversarial_g_effective"]

            if compute_discriminator_loss:
                logits_real = self._discriminator_forward(
                    source_condition,
                    x_gt.detach(),
                    update_stats=not freeze_discriminator_stats,
                )
                logits_fake = self._discriminator_forward(
                    source_condition,
                    refined_x.detach(),
                    update_stats=not freeze_discriminator_stats,
                )
                discriminator_loss = _discriminator_hinge_loss(logits_real, logits_fake)
        else:
            generator_terms["adversarial_g"] = x_gt.new_tensor(0.0)
            generator_terms["adversarial_g_effective"] = x_gt.new_tensor(0.0)

        generator_terms["refinement_l1"] = F.l1_loss(refined_x, x_pred)
        generator_terms["refinement_delta_l1"] = refinement_delta.abs().mean()

        return EnhancementLossOutput(
            refined_reconstruction=refined_reconstruction,
            generator_total_loss=total,
            generator_terms=generator_terms,
            discriminator_loss=discriminator_loss,
            status=status,
        )

    def generator_step(
        self,
        x_gt: torch.Tensor,
        x_pred: torch.Tensor | None = None,
        *,
        base_reconstruction: DecoderOutput | None = None,
        compute_discriminator_loss: bool = False,
        enable_perceptual: bool | None = None,
        enable_adversarial: bool | None = None,
    ) -> EnhancementLossOutput:
        return self.forward(
            x_gt=x_gt,
            x_pred=x_pred,
            base_reconstruction=base_reconstruction,
            compute_discriminator_loss=compute_discriminator_loss,
            freeze_discriminator_stats=True,
            enable_perceptual=enable_perceptual,
            enable_adversarial=enable_adversarial,
        )

    def discriminator_step(
        self,
        *,
        x_source: torch.Tensor,
        x_gt: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> torch.Tensor:
        if self.discriminator is None:
            raise RuntimeError("Discriminator step was requested but no PatchGAN discriminator is enabled.")
        logits_real = self.discriminator(x_source.detach(), x_gt.detach())
        logits_fake = self.discriminator(x_source.detach(), x_pred.detach())
        return _discriminator_hinge_loss(logits_real, logits_fake)

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch.nn as nn

from route2_swinjscc_gan.common.checks import require, require_non_negative_int, require_positive_int
from route2_swinjscc_gan.models.swinjscc_gan.discriminator_patchgan import PatchGANConfig
from route2_swinjscc_gan.models.swinjscc_gan.generator import SwinJSCCGeneratorConfig


@dataclass(frozen=True)
class DataConfig:
    train_roots: tuple[str, ...] = ()
    val_roots: tuple[str, ...] = ()
    test_roots: tuple[str, ...] = ()
    crop_size: int = 256
    batch_size: int = 16
    num_workers: int = 4
    eval_divisible_by: int = 128
    allow_eval_size_adjustment: bool = False
    val_split_ratio: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        require_positive_int("crop_size", self.crop_size)
        require_positive_int("batch_size", self.batch_size)
        require_non_negative_int("num_workers", self.num_workers)
        require_positive_int("eval_divisible_by", self.eval_divisible_by)
        require(0.0 < self.val_split_ratio < 1.0, "`val_split_ratio` must be between 0 and 1.")


@dataclass(frozen=True)
class ModelConfig:
    model_variant: str = "SwinJSCC_w/_SAandRA"
    model_size: str = "base"
    image_size: int = 256
    channel_type: str = "rayleigh"
    multiple_snr: tuple[int, ...] = (1, 4, 7, 10, 13)
    channel_numbers: tuple[int, ...] = (32, 64, 96, 128, 192)
    pass_channel: bool = True
    device: str = "cuda"
    vgg_weights_path: str | None = None

    def __post_init__(self) -> None:
        require(self.model_size in {"small", "base", "large"}, "Model size must be `small`, `base`, or `large`.")
        require_positive_int("image_size", self.image_size)
        require(len(self.multiple_snr) > 0, "`multiple_snr` must not be empty.")
        require(len(self.channel_numbers) > 0, "`channel_numbers` must not be empty.")

    def _encoder_decoder_dims(self) -> tuple[list[int], list[int], list[int], list[int]]:
        if self.model_size == "small":
            return [128, 192, 256, 320], [2, 2, 2, 2], [4, 6, 8, 10], [320, 256, 192, 128]
        if self.model_size == "base":
            return [128, 192, 256, 320], [2, 2, 6, 2], [4, 6, 8, 10], [320, 256, 192, 128]
        return [128, 192, 256, 320], [2, 2, 18, 2], [4, 6, 8, 10], [320, 256, 192, 128]

    def _channel_number_for_backbone(self) -> int | None:
        if self.model_variant in {"SwinJSCC_w/o_SAandRA", "SwinJSCC_w/_SA"}:
            return int(self.channel_numbers[0])
        return None

    def build_generator_config(self) -> SwinJSCCGeneratorConfig:
        encoder_embed_dims, encoder_depths, encoder_heads, decoder_embed_dims = self._encoder_decoder_dims()
        decoder_depths = list(reversed(encoder_depths))
        decoder_heads = list(reversed(encoder_heads))
        channel_number = self._channel_number_for_backbone()

        encoder_kwargs: dict[str, Any] = dict(
            model=self.model_variant,
            img_size=(self.image_size, self.image_size),
            patch_size=2,
            in_chans=3,
            embed_dims=encoder_embed_dims,
            depths=encoder_depths,
            num_heads=encoder_heads,
            C=channel_number,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        decoder_kwargs: dict[str, Any] = dict(
            model=self.model_variant,
            img_size=(self.image_size, self.image_size),
            embed_dims=decoder_embed_dims,
            depths=decoder_depths,
            num_heads=decoder_heads,
            C=channel_number,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )

        return SwinJSCCGeneratorConfig(
            model_variant=self.model_variant,
            encoder_kwargs=encoder_kwargs,
            decoder_kwargs=decoder_kwargs,
            multiple_snr=self.multiple_snr,
            channel_numbers=self.channel_numbers,
            channel_type=self.channel_type,
            pass_channel=self.pass_channel,
            downsample=4,
            device=self.device,
        )


@dataclass(frozen=True)
class OptimizerConfig:
    generator_lr: float = 1e-4
    discriminator_lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    min_lr: float = 1e-6
    warmup_epochs: int = 5

    def __post_init__(self) -> None:
        require(self.generator_lr > 0.0, "`generator_lr` must be positive.")
        require(self.discriminator_lr > 0.0, "`discriminator_lr` must be positive.")
        require(self.min_lr > 0.0, "`min_lr` must be positive.")
        require_positive_int("warmup_epochs", self.warmup_epochs)


@dataclass(frozen=True)
class AdversarialAblationConfig:
    enabled: bool = True
    loss_mode: str = "hinge"
    weight: float = 0.05
    ramp_epochs: int = 0
    discriminator_lr_scale: float = 1.0

    def __post_init__(self) -> None:
        require(self.loss_mode in {"hinge", "bce"}, "`loss_mode` must be `hinge` or `bce`.")
        require(self.weight >= 0.0, "`weight` must be non-negative.")
        require_non_negative_int("ramp_epochs", self.ramp_epochs)
        require(self.discriminator_lr_scale > 0.0, "`discriminator_lr_scale` must be positive.")


@dataclass(frozen=True)
class TrainingConfig:
    total_epochs: int = 300
    phase1_epochs: int = 100
    phase2_epochs: int = 100
    save_every_epochs: int = 10
    eval_every_epochs: int = 10
    log_every_steps: int = 50
    output_dir: str = "route2_swinjscc_gan/artifacts/default_run"
    checkpoint_path: str | None = None
    init_checkpoint_path: str | None = None
    max_steps: int | None = None

    def __post_init__(self) -> None:
        require_positive_int("total_epochs", self.total_epochs)
        require_non_negative_int("phase1_epochs", self.phase1_epochs)
        require_non_negative_int("phase2_epochs", self.phase2_epochs)
        require_positive_int("save_every_epochs", self.save_every_epochs)
        require_positive_int("eval_every_epochs", self.eval_every_epochs)
        require_positive_int("log_every_steps", self.log_every_steps)
        require(self.phase1_epochs + self.phase2_epochs <= self.total_epochs, "Training phases exceed total epochs.")
        require(
            not (self.checkpoint_path is not None and self.init_checkpoint_path is not None),
            "Use either `checkpoint_path` for resume or `init_checkpoint_path` for weight initialization, not both.",
        )
        if self.max_steps is not None:
            require_positive_int("max_steps", self.max_steps)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


@dataclass(frozen=True)
class EvaluationConfig:
    snr: int = 10
    rate: int = 96
    save_images: bool = True
    max_saved_images: int = 8
    lpips_network: str = "vgg"

    def __post_init__(self) -> None:
        require_positive_int("max_saved_images", self.max_saved_images)
        require(self.lpips_network in {"alex", "vgg"}, "`lpips_network` must be `alex` or `vgg`.")


@dataclass(frozen=True)
class Route2ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    discriminator: PatchGANConfig = field(
        default_factory=lambda: PatchGANConfig(
            kind="legacy_patchgan",
            image_channels=3,
            base_channels=64,
            max_channels=512,
            num_downsampling_layers=4,
            negative_slope=0.2,
            norm_type="instance",
            use_spectral_norm=False,
        )
    )
    adversarial: AdversarialAblationConfig = field(default_factory=AdversarialAblationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def build_default_experiment_config() -> Route2ExperimentConfig:
    return Route2ExperimentConfig()

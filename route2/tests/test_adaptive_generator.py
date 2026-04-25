from __future__ import annotations

import torch
import torch.nn as nn

from route2_swinjscc_gan.models.swinjscc_gan.generator import SwinJSCCGenerator, SwinJSCCGeneratorConfig


def _build_generator_config(model_variant: str) -> SwinJSCCGeneratorConfig:
    channel_number = 24 if model_variant in {"SwinJSCC_w/o_SAandRA", "SwinJSCC_w/_SA"} else None
    encoder_kwargs = dict(
        model=model_variant,
        img_size=(128, 128),
        patch_size=2,
        in_chans=3,
        embed_dims=[32, 48, 64, 96],
        depths=[2, 2, 2, 2],
        num_heads=[2, 2, 4, 6],
        C=channel_number,
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )
    decoder_kwargs = dict(
        model=model_variant,
        img_size=(128, 128),
        embed_dims=[96, 64, 48, 32],
        depths=[2, 2, 2, 2],
        num_heads=[6, 4, 2, 2],
        C=channel_number,
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )
    return SwinJSCCGeneratorConfig(
        model_variant=model_variant,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        multiple_snr=(10,),
        channel_numbers=(24,),
        channel_type="awgn",
        pass_channel=True,
        downsample=4,
        device="cpu",
    )


def test_rate_adaptive_generator_runs_on_cpu() -> None:
    generator = SwinJSCCGenerator(_build_generator_config("SwinJSCC_w/_RA"))
    image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        output = generator(image, snr=10, rate=24)
    assert output.reconstruction.shape == image.shape
    assert output.mask is not None
    assert output.mask.shape[0] == 1
    assert torch.isfinite(output.reconstruction).all()


def test_joint_sa_ra_generator_runs_on_cpu() -> None:
    generator = SwinJSCCGenerator(_build_generator_config("SwinJSCC_w/_SAandRA"))
    image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        output = generator(image, snr=10, rate=24)
    assert output.reconstruction.shape == image.shape
    assert output.mask is not None
    assert output.mask.shape[0] == 1
    assert torch.isfinite(output.reconstruction).all()

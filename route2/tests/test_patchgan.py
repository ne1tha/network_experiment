import torch

from route2_swinjscc_gan.models.swinjscc_gan.discriminator_patchgan import PatchGANConfig, PatchGANDiscriminator


def test_legacy_patchgan_outputs_patch_map() -> None:
    model = PatchGANDiscriminator()
    image = torch.randn(2, 3, 256, 256)
    logits = model(image)
    assert logits.shape[0] == 2
    assert logits.shape[1] == 1
    assert logits.shape[-2] > 1
    assert logits.shape[-1] > 1


def test_conditional_multiscale_patchgan_outputs_two_patch_maps() -> None:
    model = PatchGANDiscriminator(
        PatchGANConfig(
            kind="conditional_multiscale_v1",
            image_channels=3,
            norm_type="none",
            use_spectral_norm=True,
        )
    )
    source = torch.randn(2, 3, 256, 256)
    reconstruction = torch.randn(2, 3, 256, 256)

    logits = model(source, reconstruction)

    assert isinstance(logits, tuple)
    assert len(logits) == 2
    fine_logits, coarse_logits = logits
    assert fine_logits.shape[0] == 2
    assert fine_logits.shape[1] == 1
    assert coarse_logits.shape[0] == 2
    assert coarse_logits.shape[1] == 1
    assert fine_logits.shape[-2] > coarse_logits.shape[-2]
    assert fine_logits.shape[-1] > coarse_logits.shape[-1]


def test_legacy_patchgan_loads_old_output_conv_checkpoint_keys() -> None:
    model = PatchGANDiscriminator()
    state_dict = model.state_dict()
    legacy_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("output_heads.0.")
    }
    legacy_state_dict["output_conv.weight"] = state_dict["output_heads.0.weight"]
    legacy_state_dict["output_conv.bias"] = state_dict["output_heads.0.bias"]

    model.load_state_dict(legacy_state_dict, strict=True)

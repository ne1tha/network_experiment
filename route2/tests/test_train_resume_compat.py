from __future__ import annotations

import pytest

from route2_swinjscc_gan.models.swinjscc_gan.discriminator_patchgan import PatchGANConfig
from route2_swinjscc_gan.scripts.train import _checkpoint_discriminator_config, _should_resume_discriminator_state


def test_checkpoint_discriminator_config_defaults_to_legacy_when_missing() -> None:
    config = _checkpoint_discriminator_config({})

    assert config.kind == "legacy_patchgan"
    assert config.norm_type == "instance"
    assert config.use_spectral_norm is False


def test_should_resume_discriminator_state_accepts_matching_configs() -> None:
    current = PatchGANConfig(
        kind="conditional_multiscale_v1",
        image_channels=3,
        norm_type="none",
        use_spectral_norm=True,
    )

    assert _should_resume_discriminator_state(current, current) is True


def test_should_resume_discriminator_state_skips_legacy_checkpoint_for_new_discriminator() -> None:
    current = PatchGANConfig(
        kind="conditional_multiscale_v1",
        image_channels=3,
        norm_type="none",
        use_spectral_norm=True,
    )
    legacy = PatchGANConfig()

    assert _should_resume_discriminator_state(current, legacy) is False


def test_should_resume_discriminator_state_rejects_new_checkpoint_mismatch() -> None:
    current = PatchGANConfig()
    checkpoint = PatchGANConfig(
        kind="conditional_multiscale_v1",
        image_channels=3,
        norm_type="none",
        use_spectral_norm=True,
    )

    with pytest.raises(ValueError, match="does not match"):
        _should_resume_discriminator_state(current, checkpoint)

import pytest

from route2_swinjscc_gan.losses.perceptual_vgg import VGGPerceptualLoss


def test_vgg_loss_fails_fast_without_torchvision() -> None:
    try:
        import torchvision  # type: ignore # pragma: no cover
    except ModuleNotFoundError:
        with pytest.raises(RuntimeError, match="torchvision"):
            VGGPerceptualLoss()
        return

    # If torchvision is available in the environment, construction should proceed to the
    # pretrained-weight path instead of failing at import time.
    VGGPerceptualLoss


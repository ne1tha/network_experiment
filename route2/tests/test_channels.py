import torch

from route2_swinjscc_gan.channels.awgn import AWGNChannel
from route2_swinjscc_gan.channels.rayleigh import RayleighFadingChannel


def test_awgn_channel_preserves_shape_and_finiteness() -> None:
    channel = AWGNChannel()
    feature = torch.randn(2, 64, 24)
    output = channel(feature, snr_db=10)
    assert output.shape == feature.shape
    assert torch.isfinite(output).all()


def test_rayleigh_channel_preserves_shape_and_finiteness() -> None:
    channel = RayleighFadingChannel()
    feature = torch.randn(2, 64, 24)
    output = channel(feature, snr_db=10)
    assert output.shape == feature.shape
    assert torch.isfinite(output).all()

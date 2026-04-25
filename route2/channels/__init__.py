"""Channel models for Route 2."""

from route2_swinjscc_gan.channels.awgn import AWGNChannel
from route2_swinjscc_gan.channels.rayleigh import RayleighFadingChannel


def build_channel(channel_type: str):
    if channel_type == "awgn":
        return AWGNChannel()
    if channel_type == "rayleigh":
        return RayleighFadingChannel()
    raise ValueError(f"Unsupported channel type: {channel_type}")


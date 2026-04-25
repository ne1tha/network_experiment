SUPPORTED_CHANNELS = {"awgn", "rayleigh"}


def validate_channel_type(channel_type: str) -> str:
    if channel_type not in SUPPORTED_CHANNELS:
        supported = ", ".join(sorted(SUPPORTED_CHANNELS))
        raise ValueError(
            f"Unsupported channel type {channel_type!r}. "
            f"Route 1 only allows: {supported}."
        )
    return channel_type

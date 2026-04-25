from __future__ import annotations

import torch

from route2_swinjscc_gan.common.checks import require


def resolve_runtime_device(requested_device: str | None) -> str:
    raw = "auto" if requested_device is None else str(requested_device).strip().lower()
    if raw in {"", "auto"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        parsed = torch.device(raw)
    except RuntimeError as exc:
        raise ValueError(
            f"Invalid device `{requested_device}`. Expected `cpu`, `cuda`, `cuda:N`, or `auto`."
        ) from exc

    if parsed.type == "cpu":
        return "cpu"
    if parsed.type != "cuda":
        raise ValueError(
            f"Unsupported device `{requested_device}`. Expected `cpu`, `cuda`, `cuda:N`, or `auto`."
        )

    require(
        torch.cuda.is_available(),
        f"CUDA device `{requested_device}` was requested, but torch.cuda.is_available() is False.",
    )
    index = 0 if parsed.index is None else int(parsed.index)
    device_count = torch.cuda.device_count()
    require(
        index < device_count,
        f"CUDA device `{requested_device}` was requested, but only {device_count} CUDA device(s) are visible.",
    )
    return f"cuda:{index}"


def prepare_runtime_device(requested_device: str | None) -> torch.device:
    resolved = resolve_runtime_device(requested_device)
    device = torch.device(resolved)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def describe_cuda_environment() -> dict[str, object]:
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    devices = [torch.cuda.get_device_name(index) for index in range(count)] if available else []
    return {
        "cuda_available": available,
        "device_count": count,
        "devices": devices,
    }

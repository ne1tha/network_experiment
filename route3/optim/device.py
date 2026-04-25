from __future__ import annotations

from typing import Any

import torch


def select_torch_device(requested: str = "auto") -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in the current runtime.")
        return torch.device(requested)
    raise ValueError(f"Unsupported device request: {requested}")


def normalize_runtime_value(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    return value


def move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device=device)
    if isinstance(batch, dict):
        return {key: move_to_device(val, device) for key, val in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(item, device) for item in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(item, device) for item in batch)
    return batch

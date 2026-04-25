from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / self.count


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def ensure_run_dirs(workdir: Path, models_dir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "samples").mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)


def build_logger(name: str, *, log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_serialize(payload), handle, ensure_ascii=False, indent=2)


def save_config_snapshot(experiment: Any, output_path: Path) -> None:
    save_json({"saved_at": datetime.utcnow().isoformat() + "Z", "experiment": experiment}, output_path)


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    global_step: int,
    experiment: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "experiment": _serialize(experiment),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    global_step: int,
    experiment: Any,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        experiment=experiment,
    )
    torch.save(payload, output_path)


def _load_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


def _checkpoint_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "saved_at": payload.get("saved_at"),
        "experiment": payload.get("experiment"),
    }


def load_checkpoint_compatible(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    payload = _load_checkpoint_payload(checkpoint_path, device)
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported state_dict payload type: {type(state_dict)!r}")

    model_state = model.state_dict()
    compatible_state: dict[str, Any] = {}
    unexpected_keys: list[str] = []
    shape_mismatch_keys: list[str] = []

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            shape_mismatch_keys.append(key)
            continue
        compatible_state[key] = value

    if not compatible_state:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not share any compatible parameters with the target model."
        )

    merged_state = dict(model_state)
    merged_state.update(compatible_state)
    model.load_state_dict(merged_state, strict=True)

    missing_keys = [key for key in model_state.keys() if key not in compatible_state]
    metadata = _checkpoint_metadata(payload)
    metadata.update(
        {
            "load_mode": "compatible",
            "loaded_key_count": len(compatible_state),
            "missing_key_count": len(missing_keys),
            "unexpected_key_count": len(unexpected_keys),
            "shape_mismatch_count": len(shape_mismatch_keys),
            "missing_key_examples": missing_keys[:10],
            "unexpected_key_examples": unexpected_keys[:10],
            "shape_mismatch_examples": shape_mismatch_keys[:10],
        }
    )
    return metadata


def load_checkpoint_strict(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    require_optimizer: bool = False,
) -> dict[str, Any]:
    payload = _load_checkpoint_payload(checkpoint_path, device)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict, strict=True)

    if optimizer is not None:
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state is None and require_optimizer:
            raise ValueError(
                f"Checkpoint {checkpoint_path} does not contain optimizer_state_dict."
            )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    return _checkpoint_metadata(payload)


def build_ms_ssim_metric(ms_ssim_cls: type[torch.nn.Module], device: torch.device, trainset: str) -> torch.nn.Module:
    if trainset == "CIFAR10":
        metric = ms_ssim_cls(window_size=3, data_range=1.0, levels=4, channel=3)
    else:
        metric = ms_ssim_cls(data_range=1.0, levels=4, channel=3)
    return metric.to(device)


def compute_psnr_from_mse(mse_255: float) -> float:
    if mse_255 < 0:
        raise ValueError(f"MSE must be non-negative, got {mse_255}.")
    if mse_255 == 0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor((255.0 * 255.0) / mse_255)).item())


def compute_ms_ssim_value(
    metric: torch.nn.Module,
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
) -> float:
    loss = metric(inputs, reconstructions.clamp(0.0, 1.0)).mean()
    value = 1.0 - float(loss.item())
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"MS-SSIM metric became non-finite: {value}")
    return value

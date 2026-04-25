from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from configs.route1_reference import build_div2k_reference_config


PLACEHOLDER_MARKERS = {
    "<required>",
    "<fill-me>",
    "/abs/path/to/train",
    "/abs/path/to/test",
    "/abs/path/to/DIV2K_train_HR",
    "/abs/path/to/Kodak",
}


@dataclass(slots=True)
class DatasetManifest:
    path: Path
    name: str
    train_dirs: list[Path]
    test_dirs: list[Path]
    allow_eval_size_adjustment: bool = False


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload)!r}.")
    return payload


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _validate_non_placeholder(paths: list[Path], *, label: str) -> None:
    if not paths:
        raise ValueError(f"{label} must not be empty.")
    for path in paths:
        text = str(path)
        if any(marker in text for marker in PLACEHOLDER_MARKERS):
            raise ValueError(f"{label} contains placeholder path {text!r}.")


def load_dataset_manifest(path: Path) -> DatasetManifest:
    payload = _load_json(path)
    base_dir = path.parent

    train_dirs_raw = payload.get("train_dirs")
    test_dirs_raw = payload.get("test_dirs")
    if not isinstance(train_dirs_raw, list) or not all(isinstance(item, str) for item in train_dirs_raw):
        raise ValueError(f"{path} must define train_dirs as a string list.")
    if not isinstance(test_dirs_raw, list) or not all(isinstance(item, str) for item in test_dirs_raw):
        raise ValueError(f"{path} must define test_dirs as a string list.")

    train_dirs = [_resolve_path(item, base_dir=base_dir) for item in train_dirs_raw]
    test_dirs = [_resolve_path(item, base_dir=base_dir) for item in test_dirs_raw]
    _validate_non_placeholder(train_dirs, label="train_dirs")
    _validate_non_placeholder(test_dirs, label="test_dirs")

    name = payload.get("name", path.stem)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path} must define a non-empty manifest name.")

    allow_eval_size_adjustment = bool(payload.get("allow_eval_size_adjustment", False))
    return DatasetManifest(
        path=path,
        name=name,
        train_dirs=train_dirs,
        test_dirs=test_dirs,
        allow_eval_size_adjustment=allow_eval_size_adjustment,
    )


def load_reference_experiment(config_path: Path):
    payload = _load_json(config_path)
    base_dir = config_path.parent

    dataset_manifest_value = payload.get("dataset_manifest")
    if not isinstance(dataset_manifest_value, str) or not dataset_manifest_value.strip():
        raise ValueError(f"{config_path} must define dataset_manifest.")
    dataset_manifest = load_dataset_manifest(_resolve_path(dataset_manifest_value, base_dir=base_dir))

    workspace_root_value = payload.get("workspace_root")
    if workspace_root_value is None:
        resolved = config_path.resolve()
        if resolved.parent.name == "experiments" and resolved.parent.parent.name == "configs":
            workspace_root = resolved.parents[2]
        else:
            workspace_root = resolved.parents[1]
    elif isinstance(workspace_root_value, str):
        workspace_root = _resolve_path(workspace_root_value, base_dir=base_dir)
    else:
        raise ValueError(f"{config_path} workspace_root must be a string when provided.")

    experiment = build_div2k_reference_config(
        workspace_root=workspace_root,
        train_dirs=dataset_manifest.train_dirs,
        test_dirs=dataset_manifest.test_dirs,
        testset=str(payload.get("testset", "kodak")),
        model=str(payload.get("model", "SwinJSCC_w/_SAandRA")),
        channel_type=str(payload.get("channel_type", "awgn")),
        channels_csv=str(payload.get("channels_csv", "32,64,96,128,192")),
        snrs_csv=str(payload.get("snrs_csv", "1,4,7,10,13")),
        model_size=str(payload.get("model_size", "base")),
        run_name=str(payload.get("run_name", config_path.stem)),
        training=bool(payload.get("training", False)),
        distortion_metric=str(payload.get("distortion_metric", "MSE")),
        batch_size=int(payload.get("batch_size", 16)),
        num_workers=int(payload.get("num_workers", 0)),
        pin_memory=bool(payload.get("pin_memory", False)),
        total_epochs=int(payload.get("total_epochs", 1)),
        learning_rate=float(payload.get("learning_rate", 1e-4)),
        save_model_freq=int(payload.get("save_model_freq", 1)),
        print_step=int(payload.get("print_step", 10)),
        device=str(payload.get("device", "cpu")),
        checkpoint_path=(
            _resolve_path(payload["checkpoint_path"], base_dir=base_dir)
            if payload.get("checkpoint_path")
            else None
        ),
        checkpoint_load_mode=str(payload.get("checkpoint_load_mode", "strict")),
        resume_path=(
            _resolve_path(payload["resume_path"], base_dir=base_dir)
            if payload.get("resume_path")
            else None
        ),
        save_logs=bool(payload.get("save_logs", True)),
        max_train_steps=(
            int(payload["max_train_steps"]) if payload.get("max_train_steps") is not None else None
        ),
        max_eval_samples=(
            int(payload["max_eval_samples"]) if payload.get("max_eval_samples") is not None else None
        ),
        allow_eval_size_adjustment=bool(
            payload.get("allow_eval_size_adjustment", dataset_manifest.allow_eval_size_adjustment)
        ),
    )
    return experiment, dataset_manifest

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

from route2_swinjscc_gan.configs.defaults import Route2ExperimentConfig, build_default_experiment_config


PLACEHOLDER_MARKERS = {
    "<required>",
    "<fill-me>",
    "/abs/path/to/DIV2K_train_HR",
    "/abs/path/to/DIV2K_valid_HR",
    "/abs/path/to/Kodak",
}


@dataclass(slots=True)
class DatasetManifest:
    path: Path
    name: str
    train_roots: list[Path]
    val_roots: list[Path]
    test_roots: list[Path]
    allow_eval_size_adjustment: bool = False


@dataclass(slots=True)
class LoadedRoute2Experiment:
    path: Path
    name: str
    dataset_manifest: DatasetManifest
    config: Route2ExperimentConfig
    checkpoint: Path | None = None


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


def _resolve_path_list(values: list[str], *, base_dir: Path) -> list[Path]:
    return [_resolve_path(item, base_dir=base_dir) for item in values]


def _validate_non_placeholder(paths: list[Path], *, label: str, allow_empty: bool = False) -> None:
    if not allow_empty and not paths:
        raise ValueError(f"{label} must not be empty.")
    for path in paths:
        text = str(path)
        if any(marker in text for marker in PLACEHOLDER_MARKERS):
            raise ValueError(f"{label} contains placeholder path {text!r}.")


def _parse_int_sequence(value: object, *, label: str) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{label} must not be empty.")
        return tuple(int(item) for item in value)
    if isinstance(value, str):
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        if not parsed:
            raise ValueError(f"{label} must not be empty.")
        return parsed
    raise ValueError(f"{label} must be a string or integer list, got {type(value)!r}.")


def _default_workspace_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name == "experiments" and resolved.parent.parent.name == "configs":
        return resolved.parents[2]
    return resolved.parents[1]


def _parse_optional_mapping(value: object, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object when provided.")
    return dict(value)


def load_dataset_manifest(path: Path) -> DatasetManifest:
    payload = _load_json(path)
    base_dir = path.parent

    train_roots_raw = payload.get("train_roots")
    test_roots_raw = payload.get("test_roots")
    val_roots_raw = payload.get("val_roots", [])
    if not isinstance(train_roots_raw, list) or not all(isinstance(item, str) for item in train_roots_raw):
        raise ValueError(f"{path} must define train_roots as a string list.")
    if not isinstance(test_roots_raw, list) or not all(isinstance(item, str) for item in test_roots_raw):
        raise ValueError(f"{path} must define test_roots as a string list.")
    if not isinstance(val_roots_raw, list) or not all(isinstance(item, str) for item in val_roots_raw):
        raise ValueError(f"{path} val_roots must be a string list when provided.")

    train_roots = _resolve_path_list(train_roots_raw, base_dir=base_dir)
    val_roots = _resolve_path_list(val_roots_raw, base_dir=base_dir)
    test_roots = _resolve_path_list(test_roots_raw, base_dir=base_dir)
    _validate_non_placeholder(train_roots, label="train_roots")
    _validate_non_placeholder(val_roots, label="val_roots", allow_empty=True)
    _validate_non_placeholder(test_roots, label="test_roots")

    name = payload.get("name", path.stem)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path} must define a non-empty manifest name.")

    return DatasetManifest(
        path=path.resolve(),
        name=name,
        train_roots=train_roots,
        val_roots=val_roots,
        test_roots=test_roots,
        allow_eval_size_adjustment=bool(payload.get("allow_eval_size_adjustment", False)),
    )


def load_experiment_config(path: Path, *, require_checkpoint: bool = False) -> LoadedRoute2Experiment:
    payload = _load_json(path)
    base_dir = path.parent
    default = build_default_experiment_config()
    optimizer_payload = _parse_optional_mapping(payload.get("optimizer"), label="optimizer")
    discriminator_payload = _parse_optional_mapping(payload.get("discriminator"), label="discriminator")
    adversarial_payload = _parse_optional_mapping(payload.get("adversarial"), label="adversarial")

    dataset_manifest_value = payload.get("dataset_manifest")
    if not isinstance(dataset_manifest_value, str) or not dataset_manifest_value.strip():
        raise ValueError(f"{path} must define dataset_manifest.")
    dataset_manifest = load_dataset_manifest(_resolve_path(dataset_manifest_value, base_dir=base_dir))

    name = payload.get("name", path.stem)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path} must define a non-empty experiment name.")

    image_size = int(payload.get("image_size", default.model.image_size))
    crop_size = int(payload.get("crop_size", image_size))
    train_roots = tuple(str(item) for item in dataset_manifest.train_roots)
    val_roots = tuple(str(item) for item in dataset_manifest.val_roots)
    test_roots = tuple(str(item) for item in dataset_manifest.test_roots)

    data = replace(
        default.data,
        train_roots=train_roots,
        val_roots=val_roots,
        test_roots=test_roots,
        crop_size=crop_size,
        batch_size=int(payload.get("batch_size", default.data.batch_size)),
        num_workers=int(payload.get("num_workers", default.data.num_workers)),
        eval_divisible_by=int(payload.get("eval_divisible_by", default.data.eval_divisible_by)),
        allow_eval_size_adjustment=bool(
            payload.get("allow_eval_size_adjustment", dataset_manifest.allow_eval_size_adjustment)
        ),
        val_split_ratio=float(payload.get("val_split_ratio", default.data.val_split_ratio)),
        seed=int(payload.get("seed", default.data.seed)),
    )

    vgg_weights_path_raw = payload.get("vgg_weights_path")
    vgg_weights_path = None if vgg_weights_path_raw in (None, "") else str(
        _resolve_path(str(vgg_weights_path_raw), base_dir=base_dir)
    )
    if vgg_weights_path is not None and not Path(vgg_weights_path).exists():
        raise FileNotFoundError(f"VGG weights path does not exist: {vgg_weights_path}")

    model = replace(
        default.model,
        model_variant=str(payload.get("model_variant", default.model.model_variant)),
        model_size=str(payload.get("model_size", default.model.model_size)),
        image_size=image_size,
        channel_type=str(payload.get("channel_type", default.model.channel_type)),
        multiple_snr=_parse_int_sequence(payload.get("multiple_snr", default.model.multiple_snr), label="multiple_snr"),
        channel_numbers=_parse_int_sequence(
            payload.get("channel_numbers", default.model.channel_numbers),
            label="channel_numbers",
        ),
        pass_channel=bool(payload.get("pass_channel", default.model.pass_channel)),
        device=str(payload.get("device", default.model.device)),
        vgg_weights_path=vgg_weights_path,
    )

    workspace_root = _default_workspace_root(path)
    output_dir_raw = payload.get("output_dir")
    if output_dir_raw is None:
        output_dir = workspace_root / "artifacts" / "runs" / name
    else:
        output_dir = _resolve_path(str(output_dir_raw), base_dir=base_dir)
    training = replace(
        default.training,
        total_epochs=int(payload.get("total_epochs", default.training.total_epochs)),
        phase1_epochs=int(payload.get("phase1_epochs", default.training.phase1_epochs)),
        phase2_epochs=int(payload.get("phase2_epochs", default.training.phase2_epochs)),
        save_every_epochs=int(payload.get("save_every_epochs", default.training.save_every_epochs)),
        eval_every_epochs=int(payload.get("eval_every_epochs", default.training.eval_every_epochs)),
        log_every_steps=int(payload.get("log_every_steps", default.training.log_every_steps)),
        output_dir=str(output_dir),
        checkpoint_path=(
            str(_resolve_path(str(payload["checkpoint_path"]), base_dir=base_dir))
            if payload.get("checkpoint_path") not in (None, "")
            else default.training.checkpoint_path
        ),
        init_checkpoint_path=(
            str(_resolve_path(str(payload["init_checkpoint_path"]), base_dir=base_dir))
            if payload.get("init_checkpoint_path") not in (None, "")
            else default.training.init_checkpoint_path
        ),
        max_steps=int(payload["max_steps"]) if payload.get("max_steps") is not None else None,
    )
    if training.checkpoint_path is not None and not Path(training.checkpoint_path).exists():
        raise FileNotFoundError(f"Training checkpoint does not exist: {training.checkpoint_path}")
    if training.init_checkpoint_path is not None and not Path(training.init_checkpoint_path).exists():
        raise FileNotFoundError(f"Training init checkpoint does not exist: {training.init_checkpoint_path}")

    optimizer = replace(
        default.optimizer,
        generator_lr=float(optimizer_payload.get("generator_lr", default.optimizer.generator_lr)),
        discriminator_lr=float(optimizer_payload.get("discriminator_lr", default.optimizer.discriminator_lr)),
        betas=tuple(optimizer_payload.get("betas", default.optimizer.betas)),
        weight_decay=float(optimizer_payload.get("weight_decay", default.optimizer.weight_decay)),
        min_lr=float(optimizer_payload.get("min_lr", default.optimizer.min_lr)),
        warmup_epochs=int(optimizer_payload.get("warmup_epochs", default.optimizer.warmup_epochs)),
    )

    discriminator = replace(
        default.discriminator,
        kind=str(discriminator_payload.get("kind", default.discriminator.kind)),
        image_channels=int(discriminator_payload.get("image_channels", default.discriminator.image_channels)),
        base_channels=int(discriminator_payload.get("base_channels", default.discriminator.base_channels)),
        max_channels=int(discriminator_payload.get("max_channels", default.discriminator.max_channels)),
        num_downsampling_layers=int(
            discriminator_payload.get(
                "num_downsampling_layers",
                default.discriminator.num_downsampling_layers,
            )
        ),
        negative_slope=float(discriminator_payload.get("negative_slope", default.discriminator.negative_slope)),
        norm_type=str(discriminator_payload.get("norm_type", default.discriminator.norm_type)),
        use_spectral_norm=bool(
            discriminator_payload.get("use_spectral_norm", default.discriminator.use_spectral_norm)
        ),
    )

    adversarial = replace(
        default.adversarial,
        enabled=bool(adversarial_payload.get("enabled", default.adversarial.enabled)),
        loss_mode=str(adversarial_payload.get("loss_mode", default.adversarial.loss_mode)),
        weight=float(adversarial_payload.get("weight", default.adversarial.weight)),
        ramp_epochs=int(adversarial_payload.get("ramp_epochs", default.adversarial.ramp_epochs)),
        discriminator_lr_scale=float(
            adversarial_payload.get(
                "discriminator_lr_scale",
                default.adversarial.discriminator_lr_scale,
            )
        ),
    )

    evaluation = replace(
        default.evaluation,
        snr=int(payload.get("eval_snr", model.multiple_snr[0])),
        rate=int(payload.get("eval_rate", model.channel_numbers[0])),
        save_images=bool(payload.get("save_images", default.evaluation.save_images)),
        max_saved_images=int(payload.get("max_saved_images", default.evaluation.max_saved_images)),
        lpips_network=str(payload.get("lpips_network", default.evaluation.lpips_network)),
    )

    checkpoint_raw = payload.get("checkpoint")
    checkpoint = None if checkpoint_raw in (None, "") else _resolve_path(str(checkpoint_raw), base_dir=base_dir)
    if require_checkpoint and checkpoint is None:
        raise ValueError(f"{path} must define checkpoint for evaluation.")
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    return LoadedRoute2Experiment(
        path=path.resolve(),
        name=name,
        dataset_manifest=dataset_manifest,
        config=Route2ExperimentConfig(
            data=data,
            model=model,
            optimizer=optimizer,
            discriminator=discriminator,
            adversarial=adversarial,
            training=training,
            evaluation=evaluation,
        ),
        checkpoint=checkpoint,
    )

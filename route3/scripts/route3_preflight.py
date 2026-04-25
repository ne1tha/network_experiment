from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Route3ImageManifestDataset, load_dataset_manifest, validate_dataset_manifest
from evaluators import budget_metrics_to_dict, summarize_single_user_budget
from models.ultimate import (
    ConditionalSSDDDecoder,
    LayerWiseAdaptiveDistillation,
    PatchGANDiscriminator,
    PerceptualAdversarialEnhancer,
    SemanticDistillationModule,
    SemanticTeacherEncoder,
    UltimateDualPathEncoder,
    UltimateMultiUserTransmission,
    UltimateSingleUserTransmission,
    VGGFeatureExtractor,
    VGGPerceptualLoss,
)
from losses import ConditionalSSDDDecoderLoss
from optim import move_to_device, select_torch_device


@dataclass(frozen=True)
class DatasetConfig:
    manifest_path: str
    train_split: str
    val_split: str
    image_size: tuple[int, int]
    train_image_size: tuple[int, int] | None
    val_image_size: tuple[int, int] | None
    train_transform_mode: str
    val_transform_mode: str
    dry_run_samples: int


@dataclass(frozen=True)
class WeightConfig:
    teacher_checkpoint: str | None
    vgg_checkpoint: str | None
    allow_untrained_vgg: bool


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    device: str
    semantic_channel_type: str
    detail_channel_type: str
    operating_mode: str
    target_effective_cbr: float | None
    target_effective_cbr_tolerance: float
    enable_distillation: bool
    enable_perceptual: bool
    enable_adversarial: bool
    sem_rate_ratio: float
    det_rate_ratio: float
    semantic_bandwidth_budget: float
    snr_db: float
    train_snr_db_choices: tuple[float, ...] | None
    train_sem_rate_ratio_range: tuple[float, float] | None
    train_det_rate_ratio_range: tuple[float, float] | None


@dataclass(frozen=True)
class ArtifactConfig:
    output_dir: str
    checkpoint_dir: str
    report_dir: str


@dataclass(frozen=True)
class Route3PreflightConfig:
    dataset: DatasetConfig
    weights: WeightConfig
    runtime: RuntimeConfig
    artifacts: ArtifactConfig


def _parse_optional_float_sequence(
    value: Any,
    *,
    field_name: str,
    expected_len: int | None = None,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple when provided.")
    parsed = tuple(float(item) for item in value)
    if expected_len is not None and len(parsed) != expected_len:
        raise ValueError(f"{field_name} must contain exactly {expected_len} values, got {len(parsed)}.")
    if len(parsed) == 0:
        raise ValueError(f"{field_name} cannot be empty when provided.")
    return parsed


def _require_file(path: str | None, label: str) -> None:
    if path is None:
        raise RuntimeError(f"{label} is required but missing from config.")
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_preflight_config(path: str | Path) -> Route3PreflightConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset = DatasetConfig(
        manifest_path=raw["dataset"]["manifest_path"],
        train_split=raw["dataset"].get("train_split", "train"),
        val_split=raw["dataset"].get("val_split", "val"),
        image_size=tuple(raw["dataset"]["image_size"]),
        train_image_size=(
            tuple(raw["dataset"]["train_image_size"])
            if raw["dataset"].get("train_image_size") is not None
            else None
        ),
        val_image_size=(
            tuple(raw["dataset"]["val_image_size"])
            if raw["dataset"].get("val_image_size") is not None
            else None
        ),
        train_transform_mode=str(raw["dataset"].get("train_transform_mode", "resize")),
        val_transform_mode=str(raw["dataset"].get("val_transform_mode", "resize")),
        dry_run_samples=int(raw["dataset"].get("dry_run_samples", 2)),
    )
    weights = WeightConfig(
        teacher_checkpoint=raw["weights"].get("teacher_checkpoint"),
        vgg_checkpoint=raw["weights"].get("vgg_checkpoint"),
        allow_untrained_vgg=bool(raw["weights"].get("allow_untrained_vgg", False)),
    )
    runtime_raw = raw["runtime"]
    if "enable_enhancement" in runtime_raw:
        raise ValueError(
            "runtime.enable_enhancement has been retired. "
            "Use runtime.enable_perceptual and runtime.enable_adversarial explicitly."
        )
    if "enable_perceptual" not in runtime_raw or "enable_adversarial" not in runtime_raw:
        raise ValueError(
            "runtime.enable_perceptual and runtime.enable_adversarial must both be set explicitly."
        )
    runtime = RuntimeConfig(
        mode=runtime_raw["mode"],
        device=runtime_raw.get("device", "auto"),
        semantic_channel_type=runtime_raw.get("semantic_channel_type", "awgn"),
        detail_channel_type=runtime_raw.get("detail_channel_type", "awgn"),
        operating_mode=str(runtime_raw.get("operating_mode", "open_quality")),
        target_effective_cbr=(
            float(runtime_raw["target_effective_cbr"])
            if runtime_raw.get("target_effective_cbr") is not None
            else None
        ),
        target_effective_cbr_tolerance=float(runtime_raw.get("target_effective_cbr_tolerance", 0.05)),
        enable_distillation=bool(runtime_raw.get("enable_distillation", True)),
        enable_perceptual=bool(runtime_raw["enable_perceptual"]),
        enable_adversarial=bool(runtime_raw["enable_adversarial"]),
        sem_rate_ratio=float(runtime_raw.get("sem_rate_ratio", 0.5)),
        det_rate_ratio=float(runtime_raw.get("det_rate_ratio", 0.75)),
        semantic_bandwidth_budget=float(runtime_raw.get("semantic_bandwidth_budget", 10.0)),
        snr_db=float(runtime_raw.get("snr_db", 10.0)),
        train_snr_db_choices=_parse_optional_float_sequence(
            runtime_raw.get("train_snr_db_choices"),
            field_name="runtime.train_snr_db_choices",
        ),
        train_sem_rate_ratio_range=_parse_optional_float_sequence(
            runtime_raw.get("train_sem_rate_ratio_range"),
            field_name="runtime.train_sem_rate_ratio_range",
            expected_len=2,
        ),
        train_det_rate_ratio_range=_parse_optional_float_sequence(
            runtime_raw.get("train_det_rate_ratio_range"),
            field_name="runtime.train_det_rate_ratio_range",
            expected_len=2,
        ),
    )
    artifacts = ArtifactConfig(
        output_dir=raw["artifacts"]["output_dir"],
        checkpoint_dir=raw["artifacts"]["checkpoint_dir"],
        report_dir=raw["artifacts"]["report_dir"],
    )
    return Route3PreflightConfig(dataset=dataset, weights=weights, runtime=runtime, artifacts=artifacts)


def validate_preflight_config(config: Route3PreflightConfig) -> None:
    if config.runtime.mode not in {"single_user", "multi_user"}:
        raise ValueError(f"Unsupported runtime mode: {config.runtime.mode}")
    if config.runtime.operating_mode not in {"open_quality", "matched_budget"}:
        raise ValueError(f"Unsupported runtime operating_mode: {config.runtime.operating_mode}")
    if config.runtime.device not in {"auto", "cpu"} and not config.runtime.device.startswith("cuda"):
        raise ValueError(f"Unsupported runtime device: {config.runtime.device}")
    for label, image_size in (
        ("dataset.image_size", config.dataset.image_size),
        ("dataset.train_image_size", config.dataset.train_image_size),
        ("dataset.val_image_size", config.dataset.val_image_size),
    ):
        if image_size is None:
            continue
        if image_size[0] % 16 != 0 or image_size[1] % 16 != 0:
            raise ValueError(f"{label} must be divisible by 16, got {image_size}")
    for field_name, transform_mode in (
        ("dataset.train_transform_mode", config.dataset.train_transform_mode),
        ("dataset.val_transform_mode", config.dataset.val_transform_mode),
    ):
        if transform_mode not in {"resize", "center_crop", "random_crop"}:
            raise ValueError(
                f"{field_name} must be one of resize, center_crop, random_crop, got {transform_mode}"
            )
    if not math.isfinite(config.runtime.snr_db):
        raise ValueError(f"runtime.snr_db must be finite, got {config.runtime.snr_db}")
    if config.runtime.semantic_bandwidth_budget <= 0:
        raise ValueError(
            "runtime.semantic_bandwidth_budget must be positive, "
            f"got {config.runtime.semantic_bandwidth_budget}"
        )
    if config.runtime.target_effective_cbr is not None:
        if not math.isfinite(config.runtime.target_effective_cbr) or config.runtime.target_effective_cbr <= 0.0:
            raise ValueError(
                "runtime.target_effective_cbr must be a finite positive value when provided, "
                f"got {config.runtime.target_effective_cbr}"
            )
    if (
        not math.isfinite(config.runtime.target_effective_cbr_tolerance)
        or config.runtime.target_effective_cbr_tolerance < 0.0
        or config.runtime.target_effective_cbr_tolerance >= 1.0
    ):
        raise ValueError(
            "runtime.target_effective_cbr_tolerance must be in [0, 1), "
            f"got {config.runtime.target_effective_cbr_tolerance}"
        )
    if config.runtime.mode == "single_user" and config.runtime.operating_mode == "matched_budget":
        if config.runtime.target_effective_cbr is None:
            raise RuntimeError(
                "single_user matched_budget mode requires runtime.target_effective_cbr to be set explicitly."
            )
    for field_name, value in (
        ("runtime.sem_rate_ratio", config.runtime.sem_rate_ratio),
        ("runtime.det_rate_ratio", config.runtime.det_rate_ratio),
    ):
        if value <= 0 or value > 1:
            raise ValueError(f"{field_name} must be in (0, 1], got {value}")
    if config.runtime.train_snr_db_choices is not None:
        for value in config.runtime.train_snr_db_choices:
            if not math.isfinite(value):
                raise ValueError(
                    "runtime.train_snr_db_choices must contain only finite values, "
                    f"got {config.runtime.train_snr_db_choices}"
                )
    for field_name, value_range in (
        ("runtime.train_sem_rate_ratio_range", config.runtime.train_sem_rate_ratio_range),
        ("runtime.train_det_rate_ratio_range", config.runtime.train_det_rate_ratio_range),
    ):
        if value_range is None:
            continue
        lower, upper = value_range
        if lower <= 0 or upper > 1 or lower > upper:
            raise ValueError(f"{field_name} must satisfy 0 < lower <= upper <= 1, got {value_range}")
    if config.runtime.enable_distillation:
        _require_file(config.weights.teacher_checkpoint, "teacher_checkpoint")
    if config.runtime.enable_perceptual and not (config.weights.vgg_checkpoint or config.weights.allow_untrained_vgg):
        raise RuntimeError(
            "enable_perceptual=True requires either vgg_checkpoint or allow_untrained_vgg=True."
        )

    manifest = load_dataset_manifest(config.dataset.manifest_path)
    validate_dataset_manifest(manifest, config.dataset.manifest_path, required_splits=(config.dataset.train_split, config.dataset.val_split))


def prepare_artifact_dirs(config: Route3PreflightConfig) -> None:
    for path in (config.artifacts.output_dir, config.artifacts.checkpoint_dir, config.artifacts.report_dir):
        Path(path).mkdir(parents=True, exist_ok=True)


def _resolve_dataset_resize_hw(config: Route3PreflightConfig, split: str) -> tuple[int, int]:
    if split == config.dataset.train_split and config.dataset.train_image_size is not None:
        return config.dataset.train_image_size
    if split == config.dataset.val_split and config.dataset.val_image_size is not None:
        return config.dataset.val_image_size
    return config.dataset.image_size


def _resolve_dataset_transform_mode(config: Route3PreflightConfig, split: str) -> str:
    if split == config.dataset.train_split:
        return config.dataset.train_transform_mode
    if split == config.dataset.val_split:
        return config.dataset.val_transform_mode
    return "resize"


def _build_single_user_model(config: Route3PreflightConfig) -> UltimateSingleUserTransmission:
    semantic_distillation = None
    if config.runtime.enable_distillation:
        teacher = SemanticTeacherEncoder(
            backbone=UltimateDualPathEncoder().semantic_branch,
            checkpoint_path=config.weights.teacher_checkpoint,
        )
        semantic_distillation = SemanticDistillationModule(
            teacher=teacher,
            distiller=LayerWiseAdaptiveDistillation(num_stages=3),
        )

    enhancer = None
    if config.runtime.enable_perceptual or config.runtime.enable_adversarial:
        feature_extractor = None
        if config.runtime.enable_perceptual:
            feature_extractor = VGGFeatureExtractor(
                checkpoint_path=config.weights.vgg_checkpoint,
                allow_untrained=config.weights.allow_untrained_vgg,
            )
        enhancer = PerceptualAdversarialEnhancer(
            perceptual_loss=VGGPerceptualLoss(feature_extractor=feature_extractor) if feature_extractor is not None else None,
            discriminator=PatchGANDiscriminator() if config.runtime.enable_adversarial else None,
        )

    return UltimateSingleUserTransmission(
        semantic_channel_type=config.runtime.semantic_channel_type,
        detail_channel_type=config.runtime.detail_channel_type,
        semantic_distillation=semantic_distillation,
        decoder=ConditionalSSDDDecoder(),
        decoder_loss=ConditionalSSDDDecoderLoss(),
        enhancer=enhancer,
    )


def build_runtime_model(config: Route3PreflightConfig):
    single_user = _build_single_user_model(config)
    if config.runtime.mode == "single_user":
        return single_user
    return UltimateMultiUserTransmission(single_user_model=single_user)


def load_dry_run_batch(config: Route3PreflightConfig, split: str) -> torch.Tensor:
    dataset = Route3ImageManifestDataset(
        manifest_path=config.dataset.manifest_path,
        split=split,
        resize_hw=_resolve_dataset_resize_hw(config, split),
        transform_mode=_resolve_dataset_transform_mode(config, split),
    )
    num_samples = min(config.dataset.dry_run_samples, len(dataset))
    if config.runtime.mode == "multi_user" and num_samples % 2 != 0:
        if len(dataset) < 2:
            raise RuntimeError("Multi-user dry-run requires at least two samples.")
        num_samples -= 1
    if num_samples <= 0:
        raise RuntimeError("Dry-run did not collect any samples.")

    samples = [dataset[idx][0] for idx in range(num_samples)]
    return torch.stack(samples, dim=0)


@torch.no_grad()
def run_forward_dry_run(config: Route3PreflightConfig, split: str = "train") -> dict[str, Any]:
    validate_preflight_config(config)
    prepare_artifact_dirs(config)
    device = select_torch_device(config.runtime.device)
    model = build_runtime_model(config)
    model.to(device)
    model.eval()
    batch = move_to_device(load_dry_run_batch(config, split), device)

    snr = config.runtime.snr_db
    if config.runtime.mode == "single_user":
        output = model(
            batch,
            snr_db=snr,
            sem_rate_ratio=config.runtime.sem_rate_ratio,
            det_rate_ratio=config.runtime.det_rate_ratio,
            decode_stochastic=False,
        )
        summary = {
            "mode": config.runtime.mode,
            "operating_mode": config.runtime.operating_mode,
            "device": str(device),
            "batch_shape": list(batch.shape),
            "z_sem_shape": list(output.encoder.z_sem.shape),
            "z_det_shape": list(output.encoder.z_det.shape),
            "base_reconstruction_shape": list(output.base_reconstruction.x_hat.shape) if output.base_reconstruction is not None else None,
            "final_reconstruction_shape": list(output.final_reconstruction.x_hat.shape) if output.final_reconstruction is not None else None,
            "reconstruction_shape": list(output.reconstruction.x_hat.shape) if output.reconstruction is not None else None,
            "reconstruction_output_kind": output.reconstruction.output_kind if output.reconstruction is not None else None,
            "reconstruction_stochastic": bool(output.reconstruction.stochastic) if output.reconstruction is not None else None,
            "budget": budget_metrics_to_dict(
                summarize_single_user_budget(
                    input_image=batch,
                    output=output,
                    operating_mode=config.runtime.operating_mode,
                    target_effective_cbr=config.runtime.target_effective_cbr,
                    target_effective_cbr_tolerance=config.runtime.target_effective_cbr_tolerance,
                )
            )
            if output.reconstruction is not None
            else None,
            "enhancement_status": asdict(output.enhancement_status) if output.enhancement_status is not None else None,
        }
    else:
        output = model(
            batch,
            snr_db=snr,
            sem_rate_ratio=config.runtime.sem_rate_ratio,
            det_rate_ratio=config.runtime.det_rate_ratio,
            semantic_bandwidth_budget=config.runtime.semantic_bandwidth_budget,
            decode_stochastic=False,
        )
        summary = {
            "mode": config.runtime.mode,
            "device": str(device),
            "batch_shape": list(batch.shape),
            "pair_indices_shape": list(output.pairing.pair_indices.shape),
            "pair_bandwidth_shape": list(output.bandwidth.pair_bandwidth.shape),
            "shared_rx_sem_shape": list(output.shared_rx_sem.shape),
            "private_rx_det_shape": list(output.private_rx_det.shape),
            "base_shared_reconstruction_shape": list(output.base_shared_reconstruction.x_hat.shape) if output.base_shared_reconstruction is not None else None,
            "final_shared_reconstruction_shape": list(output.final_shared_reconstruction.x_hat.shape) if output.final_shared_reconstruction is not None else None,
            "shared_reconstruction_shape": list(output.shared_reconstruction.x_hat.shape) if output.shared_reconstruction is not None else None,
            "shared_reconstruction_output_kind": output.shared_reconstruction.output_kind if output.shared_reconstruction is not None else None,
            "shared_reconstruction_stochastic": bool(output.shared_reconstruction.stochastic) if output.shared_reconstruction is not None else None,
            "enhancement_status": asdict(output.shared_enhancement_status) if output.shared_enhancement_status is not None else None,
        }
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description="Route 3 preflight and dry-run validation. This script does not train.")
    parser.add_argument("--config", required=True, help="Path to the route-3 JSON config.")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split for forward dry-run.")
    args = parser.parse_args()

    config = load_preflight_config(args.config)
    summary = run_forward_dry_run(config, split=args.split)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Route3ImageManifestDataset
from evaluators import (
    MultiUserComparison,
    Phase8ReportWriter,
    ReconstructionMetrics,
    average_single_user_budget_metrics,
)
from scripts.route3_preflight import (
    _resolve_dataset_resize_hw,
    _resolve_dataset_transform_mode,
    Route3PreflightConfig,
    build_runtime_model,
    load_preflight_config,
    prepare_artifact_dirs,
    validate_preflight_config,
)
from trainers import MultiUserTrainConfig, MultiUserTrainer, SingleUserTrainConfig, SingleUserTrainer


@dataclass(frozen=True)
class TrainerRuntimeConfig:
    epochs: int
    batch_size: int
    num_workers: int
    seed: int
    learning_rate: float
    discriminator_learning_rate: float | None
    weight_decay: float
    distillation_weight: float
    enhancement_weight: float
    phase1_epochs: int
    phase2_epochs: int
    perceptual_weight_max: float
    perceptual_loss_scale: float
    adversarial_weight: float
    adversarial_ramp_epochs: int
    decoder_ms_ssim_weight: float
    decoder_l1_weight: float
    decoder_mse_weight: float
    decoder_residual_weight: float
    base_decoder_aux_weight: float
    final_decoder_weight: float
    rate_regularization_weight: float
    refinement_consistency_weight: float
    refinement_delta_weight: float
    max_grad_norm: float | None
    train_decode_stochastic: bool
    val_decode_stochastic: bool
    log_every_steps: int
    validate_every_epochs: int
    checkpoint_every_epochs: int
    max_train_batches: int | None
    max_val_batches: int | None
    trainable_prefixes: tuple[str, ...] | None
    init_model_checkpoint: str | None
    resume_checkpoint: str | None


@dataclass(frozen=True)
class Route3TrainConfig:
    base: Route3PreflightConfig
    trainer: TrainerRuntimeConfig
    config_path: str


@dataclass(frozen=True)
class SampledTrainRuntime:
    snr_db: torch.Tensor | float
    sem_rate_ratio: torch.Tensor | float
    det_rate_ratio: torch.Tensor | float


@dataclass(frozen=True)
class SelectionMetric:
    name: str
    value: float
    sort_key: tuple[float, ...]


_TRAINER_ALLOWED_KEYS = frozenset(
    {
        "epochs",
        "batch_size",
        "num_workers",
        "seed",
        "learning_rate",
        "discriminator_learning_rate",
        "weight_decay",
        "distillation_weight",
        "enhancement_weight",
        "phase1_epochs",
        "phase2_epochs",
        "perceptual_weight_max",
        "perceptual_loss_scale",
        "adversarial_weight",
        "adversarial_ramp_epochs",
        "decoder_ms_ssim_weight",
        "decoder_l1_weight",
        "decoder_mse_weight",
        "decoder_residual_weight",
        "base_decoder_aux_weight",
        "final_decoder_weight",
        "rate_regularization_weight",
        "refinement_consistency_weight",
        "refinement_delta_weight",
        "max_grad_norm",
        "train_decode_stochastic",
        "val_decode_stochastic",
        "log_every_steps",
        "validate_every_epochs",
        "checkpoint_every_epochs",
        "max_train_batches",
        "max_val_batches",
        "trainable_prefixes",
        "init_model_checkpoint",
        "resume_checkpoint",
    }
)

_TRAINER_RETIRED_FIELD_MESSAGES = {
    "adversarial_warmup_epochs": (
        "trainer.adversarial_warmup_epochs has been retired. "
        "Use trainer.phase1_epochs, trainer.phase2_epochs, and trainer.adversarial_ramp_epochs explicitly."
    ),
    "init_learning_rate_scale": (
        "trainer.init_learning_rate_scale is not supported by the current Route 3 training entry. "
        "Set trainer.learning_rate explicitly or add a real LR scheduler."
    ),
}


def load_training_config(path: str | Path) -> Route3TrainConfig:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    trainer_raw = raw["trainer"]
    retired_keys = sorted(key for key in trainer_raw if key in _TRAINER_RETIRED_FIELD_MESSAGES)
    if retired_keys:
        if len(retired_keys) == 1:
            raise ValueError(_TRAINER_RETIRED_FIELD_MESSAGES[retired_keys[0]])
        raise ValueError(
            "Retired trainer config fields detected: "
            + ", ".join(retired_keys)
            + ". Update the config to the current Route 3 trainer schema."
        )
    unexpected_keys = sorted(key for key in trainer_raw if key not in _TRAINER_ALLOWED_KEYS)
    if unexpected_keys:
        raise ValueError(
            "Unsupported trainer config field(s): "
            + ", ".join(unexpected_keys)
        )
    trainer = TrainerRuntimeConfig(
        epochs=int(trainer_raw["epochs"]),
        batch_size=int(trainer_raw["batch_size"]),
        num_workers=int(trainer_raw.get("num_workers", 0)),
        seed=int(trainer_raw.get("seed", 2026)),
        learning_rate=float(trainer_raw.get("learning_rate", 1e-4)),
        discriminator_learning_rate=float(trainer_raw["discriminator_learning_rate"])
        if trainer_raw.get("discriminator_learning_rate") is not None
        else None,
        weight_decay=float(trainer_raw.get("weight_decay", 0.0)),
        distillation_weight=float(trainer_raw.get("distillation_weight", 1.0)),
        enhancement_weight=float(trainer_raw.get("enhancement_weight", 1.0)),
        phase1_epochs=int(trainer_raw.get("phase1_epochs", 0)),
        phase2_epochs=int(trainer_raw.get("phase2_epochs", 0)),
        perceptual_weight_max=float(trainer_raw.get("perceptual_weight_max", 1.0)),
        perceptual_loss_scale=float(trainer_raw.get("perceptual_loss_scale", 1.0)),
        adversarial_weight=float(trainer_raw.get("adversarial_weight", 0.1)),
        adversarial_ramp_epochs=int(trainer_raw.get("adversarial_ramp_epochs", 0)),
        decoder_ms_ssim_weight=float(trainer_raw.get("decoder_ms_ssim_weight", 0.0)),
        decoder_l1_weight=float(trainer_raw.get("decoder_l1_weight", 1.0)),
        decoder_mse_weight=float(trainer_raw.get("decoder_mse_weight", 1.0)),
        decoder_residual_weight=float(trainer_raw.get("decoder_residual_weight", 0.25)),
        base_decoder_aux_weight=float(trainer_raw.get("base_decoder_aux_weight", 0.5)),
        final_decoder_weight=float(trainer_raw.get("final_decoder_weight", 1.0)),
        rate_regularization_weight=float(trainer_raw.get("rate_regularization_weight", 0.0)),
        refinement_consistency_weight=float(trainer_raw.get("refinement_consistency_weight", 1.0)),
        refinement_delta_weight=float(trainer_raw.get("refinement_delta_weight", 0.1)),
        max_grad_norm=float(trainer_raw["max_grad_norm"]) if trainer_raw.get("max_grad_norm") is not None else None,
        train_decode_stochastic=bool(trainer_raw.get("train_decode_stochastic", False)),
        val_decode_stochastic=bool(trainer_raw.get("val_decode_stochastic", False)),
        log_every_steps=int(trainer_raw.get("log_every_steps", 10)),
        validate_every_epochs=int(trainer_raw.get("validate_every_epochs", 1)),
        checkpoint_every_epochs=int(trainer_raw.get("checkpoint_every_epochs", 1)),
        max_train_batches=int(trainer_raw["max_train_batches"]) if trainer_raw.get("max_train_batches") is not None else None,
        max_val_batches=int(trainer_raw["max_val_batches"]) if trainer_raw.get("max_val_batches") is not None else None,
        trainable_prefixes=tuple(str(item) for item in trainer_raw["trainable_prefixes"])
        if trainer_raw.get("trainable_prefixes")
        else None,
        init_model_checkpoint=trainer_raw.get("init_model_checkpoint"),
        resume_checkpoint=trainer_raw.get("resume_checkpoint"),
    )
    return Route3TrainConfig(
        base=load_preflight_config(path),
        trainer=trainer,
        config_path=str(path.resolve()),
    )


def validate_training_config(config: Route3TrainConfig) -> None:
    validate_preflight_config(config.base)

    trainer = config.trainer
    if trainer.epochs <= 0:
        raise ValueError(f"trainer.epochs must be positive, got {trainer.epochs}")
    if trainer.batch_size <= 0:
        raise ValueError(f"trainer.batch_size must be positive, got {trainer.batch_size}")
    if trainer.num_workers < 0:
        raise ValueError(f"trainer.num_workers must be non-negative, got {trainer.num_workers}")
    if trainer.phase1_epochs < 0:
        raise ValueError(f"trainer.phase1_epochs must be non-negative, got {trainer.phase1_epochs}")
    if trainer.phase2_epochs < 0:
        raise ValueError(f"trainer.phase2_epochs must be non-negative, got {trainer.phase2_epochs}")
    if trainer.perceptual_weight_max < 0.0:
        raise ValueError(f"trainer.perceptual_weight_max must be non-negative, got {trainer.perceptual_weight_max}")
    if trainer.perceptual_loss_scale < 0.0:
        raise ValueError(f"trainer.perceptual_loss_scale must be non-negative, got {trainer.perceptual_loss_scale}")
    if trainer.discriminator_learning_rate is not None and trainer.discriminator_learning_rate < 0.0:
        raise ValueError(
            "trainer.discriminator_learning_rate must be non-negative when provided, "
            f"got {trainer.discriminator_learning_rate}"
        )
    if trainer.adversarial_weight < 0.0:
        raise ValueError(f"trainer.adversarial_weight must be non-negative, got {trainer.adversarial_weight}")
    if trainer.adversarial_ramp_epochs < 0:
        raise ValueError(
            f"trainer.adversarial_ramp_epochs must be non-negative, got {trainer.adversarial_ramp_epochs}"
        )
    for field_name, value in (
        ("trainer.decoder_ms_ssim_weight", trainer.decoder_ms_ssim_weight),
        ("trainer.decoder_l1_weight", trainer.decoder_l1_weight),
        ("trainer.decoder_mse_weight", trainer.decoder_mse_weight),
        ("trainer.decoder_residual_weight", trainer.decoder_residual_weight),
        ("trainer.base_decoder_aux_weight", trainer.base_decoder_aux_weight),
        ("trainer.final_decoder_weight", trainer.final_decoder_weight),
        ("trainer.rate_regularization_weight", trainer.rate_regularization_weight),
        ("trainer.refinement_consistency_weight", trainer.refinement_consistency_weight),
        ("trainer.refinement_delta_weight", trainer.refinement_delta_weight),
    ):
        if value < 0.0:
            raise ValueError(f"{field_name} must be non-negative, got {value}")
    if trainer.log_every_steps <= 0:
        raise ValueError(f"trainer.log_every_steps must be positive, got {trainer.log_every_steps}")
    if trainer.validate_every_epochs <= 0:
        raise ValueError(f"trainer.validate_every_epochs must be positive, got {trainer.validate_every_epochs}")
    if trainer.checkpoint_every_epochs <= 0:
        raise ValueError(f"trainer.checkpoint_every_epochs must be positive, got {trainer.checkpoint_every_epochs}")
    if trainer.max_train_batches is not None and trainer.max_train_batches <= 0:
        raise ValueError(f"trainer.max_train_batches must be positive when provided, got {trainer.max_train_batches}")
    if trainer.max_val_batches is not None and trainer.max_val_batches <= 0:
        raise ValueError(f"trainer.max_val_batches must be positive when provided, got {trainer.max_val_batches}")
    if trainer.trainable_prefixes is not None and len(trainer.trainable_prefixes) == 0:
        raise ValueError("trainer.trainable_prefixes must be non-empty when provided.")
    if trainer.phase1_epochs + trainer.phase2_epochs > trainer.epochs:
        raise ValueError(
            "trainer.phase1_epochs + trainer.phase2_epochs must not exceed trainer.epochs, got "
            f"{trainer.phase1_epochs} + {trainer.phase2_epochs} > {trainer.epochs}"
        )
    if config.base.runtime.enable_adversarial and trainer.adversarial_weight > 0.0:
        if trainer.phase1_epochs + trainer.phase2_epochs >= trainer.epochs:
            raise ValueError("trainer progressive schedule leaves no epoch for the adversarial stage.")
    if config.base.runtime.mode == "multi_user" and trainer.batch_size % 2 != 0:
        raise ValueError(f"Multi-user training requires an even batch_size, got {trainer.batch_size}")
    if trainer.init_model_checkpoint is not None and not Path(trainer.init_model_checkpoint).exists():
        raise FileNotFoundError(f"Init model checkpoint not found: {trainer.init_model_checkpoint}")
    if trainer.resume_checkpoint is not None and not Path(trainer.resume_checkpoint).exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {trainer.resume_checkpoint}")
    if trainer.init_model_checkpoint is not None and trainer.resume_checkpoint is not None:
        raise ValueError("init_model_checkpoint and resume_checkpoint are mutually exclusive.")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_accelerator_runtime(device: str) -> dict[str, Any]:
    settings = {
        "device": device,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "cudnn_benchmark": False,
        "float32_matmul_precision": None,
    }
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return settings

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        settings["float32_matmul_precision"] = "high"
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
        settings["tf32_matmul"] = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        settings["tf32_cudnn"] = True
        settings["cudnn_benchmark"] = True
    return settings


def _compose_seed(base_seed: int, *components: int) -> int:
    seed = int(base_seed) & 0xFFFFFFFFFFFFFFFF
    for component in components:
        seed = (seed * 6364136223846793005 + 1442695040888963407 + int(component)) & 0xFFFFFFFFFFFFFFFF
    return int(seed % (2**63 - 1))


@contextlib.contextmanager
def _seed_scope(seed: int):
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    _set_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _build_epoch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _sample_choice_batch(choices: tuple[float, ...], batch_size: int) -> torch.Tensor:
    return torch.tensor(
        [random.choice(choices) for _ in range(batch_size)],
        dtype=torch.float32,
    )


def _sample_uniform_batch(value_range: tuple[float, float], batch_size: int) -> torch.Tensor:
    lower, upper = value_range
    if lower == upper:
        return torch.full((batch_size,), fill_value=lower, dtype=torch.float32)
    return torch.tensor(
        [random.uniform(lower, upper) for _ in range(batch_size)],
        dtype=torch.float32,
    )


def _resolve_train_runtime(config: Route3TrainConfig, batch_size: int) -> SampledTrainRuntime:
    runtime = config.base.runtime
    snr_db: torch.Tensor | float = runtime.snr_db
    sem_rate_ratio: torch.Tensor | float = runtime.sem_rate_ratio
    det_rate_ratio: torch.Tensor | float = runtime.det_rate_ratio

    if runtime.train_snr_db_choices is not None:
        snr_db = _sample_choice_batch(runtime.train_snr_db_choices, batch_size)
    if runtime.train_sem_rate_ratio_range is not None:
        sem_rate_ratio = _sample_uniform_batch(runtime.train_sem_rate_ratio_range, batch_size)
    if runtime.train_det_rate_ratio_range is not None:
        det_rate_ratio = _sample_uniform_batch(runtime.train_det_rate_ratio_range, batch_size)

    return SampledTrainRuntime(
        snr_db=snr_db,
        sem_rate_ratio=sem_rate_ratio,
        det_rate_ratio=det_rate_ratio,
    )


def _build_loader(
    config: Route3TrainConfig,
    split: str,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = Route3ImageManifestDataset(
        manifest_path=config.base.dataset.manifest_path,
        split=split,
        resize_hw=_resolve_dataset_resize_hw(config.base, split),
        transform_mode=_resolve_dataset_transform_mode(config.base, split),
    )
    drop_last = config.base.runtime.mode == "multi_user"
    if drop_last and len(dataset) < config.trainer.batch_size:
        raise RuntimeError(
            f"Dataset split '{split}' has only {len(dataset)} samples, smaller than batch_size={config.trainer.batch_size}."
        )
    loader_kwargs: dict[str, Any] = {}
    if config.trainer.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    pin_memory = torch.cuda.is_available() and str(config.base.runtime.device).startswith("cuda")
    return DataLoader(
        dataset,
        batch_size=config.trainer.batch_size,
        shuffle=shuffle,
        num_workers=config.trainer.num_workers,
        drop_last=drop_last,
        generator=generator,
        pin_memory=pin_memory,
        **loader_kwargs,
    )


def _merge_numeric_dict(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + float(value)


def _mean_numeric_dict(source: dict[str, float], count: int) -> dict[str, float]:
    if count <= 0:
        return {}
    return {key: value / count for key, value in source.items()}


def _metrics_to_dict(metrics: ReconstructionMetrics) -> dict[str, float]:
    return {
        "mse": float(metrics.mse),
        "psnr": float(metrics.psnr),
        "mean_abs_error": float(metrics.mean_abs_error),
    }


def _average_metrics(metrics_list: list[ReconstructionMetrics]) -> ReconstructionMetrics:
    if not metrics_list:
        raise RuntimeError("Cannot average an empty metrics list.")
    count = len(metrics_list)
    return ReconstructionMetrics(
        mse=sum(metric.mse for metric in metrics_list) / count,
        psnr=sum(metric.psnr for metric in metrics_list) / count,
        mean_abs_error=sum(metric.mean_abs_error for metric in metrics_list) / count,
    )


def _average_optional_metrics(metrics_list: list[ReconstructionMetrics | None]) -> ReconstructionMetrics | None:
    available = [metric for metric in metrics_list if metric is not None]
    if not available:
        return None
    return _average_metrics(available)


def _average_multi_user_comparison(comparisons: list[MultiUserComparison]) -> MultiUserComparison:
    if not comparisons:
        raise RuntimeError("Cannot average an empty comparison list.")
    count = len(comparisons)
    return MultiUserComparison(
        single_user_metrics=_average_metrics([comparison.single_user_metrics for comparison in comparisons]),
        shared_base_metrics=_average_optional_metrics([comparison.shared_base_metrics for comparison in comparisons]),
        shared_final_metrics=_average_optional_metrics([comparison.shared_final_metrics for comparison in comparisons]),
        multi_user_metrics=_average_metrics([comparison.multi_user_metrics for comparison in comparisons]),
        no_sharing_metrics=_average_metrics([comparison.no_sharing_metrics for comparison in comparisons]),
        semantic_sharing_gain_psnr=sum(comparison.semantic_sharing_gain_psnr for comparison in comparisons) / count,
        semantic_sharing_gain_mae=sum(comparison.semantic_sharing_gain_mae for comparison in comparisons) / count,
        shared_refinement_gain_psnr=(
            sum(comparison.shared_refinement_gain_psnr for comparison in comparisons if comparison.shared_refinement_gain_psnr is not None)
            / len([comparison for comparison in comparisons if comparison.shared_refinement_gain_psnr is not None])
            if any(comparison.shared_refinement_gain_psnr is not None for comparison in comparisons)
            else None
        ),
        pair_cost_mean=sum(comparison.pair_cost_mean for comparison in comparisons) / count,
        total_bandwidth_budget=sum(comparison.total_bandwidth_budget for comparison in comparisons) / count,
        notes=comparisons[-1].notes,
    )


def _print_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _run_single_user_train_epoch(
    trainer: SingleUserTrainer,
    loader: DataLoader,
    config: Route3TrainConfig,
    epoch: int,
) -> dict[str, Any]:
    trainer.set_epoch(epoch)
    term_totals: dict[str, float] = {}
    metrics_list: list[ReconstructionMetrics] = []
    budget_metrics_list = []
    total_loss = 0.0
    step_count = 0

    for step_index, (images, _) in enumerate(loader, start=1):
        step_seed = _compose_seed(config.trainer.seed, 101, epoch, step_index)
        with _seed_scope(step_seed):
            sampled_runtime = _resolve_train_runtime(config, batch_size=images.shape[0])
            output = trainer.train_step(
                images,
                snr_db=sampled_runtime.snr_db,
                sem_rate_ratio=sampled_runtime.sem_rate_ratio,
                det_rate_ratio=sampled_runtime.det_rate_ratio,
            )
        step_count += 1
        total_loss += output.total_loss
        _merge_numeric_dict(term_totals, output.terms)
        metrics_list.append(output.metrics)
        budget_metrics_list.append(output.budget_metrics)

        if step_index % config.trainer.log_every_steps == 0:
            _print_event(
                {
                    "event": "train_step",
                    "mode": config.base.runtime.mode,
                    "epoch": epoch,
                    "step": step_index,
                    "global_step": output.global_step,
                    "total_loss": output.total_loss,
                    "psnr": output.metrics.psnr,
                    "effective_cbr": output.budget_metrics.effective_cbr if output.budget_metrics is not None else None,
                }
            )
        if config.trainer.max_train_batches is not None and step_index >= config.trainer.max_train_batches:
            break

    if step_count == 0:
        raise RuntimeError("Training loader produced zero steps.")

    return {
        "steps": step_count,
        "total_loss": total_loss / step_count,
        "terms": _mean_numeric_dict(term_totals, step_count),
        "metrics": _metrics_to_dict(_average_metrics(metrics_list)),
        "budget": average_single_user_budget_metrics(budget_metrics_list),
        "stage": trainer.current_stage_summary(),
        "adversarial_active": trainer.adversarial_active(),
    }


def _run_multi_user_train_epoch(
    trainer: MultiUserTrainer,
    loader: DataLoader,
    config: Route3TrainConfig,
    epoch: int,
) -> dict[str, Any]:
    trainer.set_epoch(epoch)
    term_totals: dict[str, float] = {}
    metrics_list: list[ReconstructionMetrics] = []
    total_loss = 0.0
    step_count = 0

    for step_index, (images, _) in enumerate(loader, start=1):
        step_seed = _compose_seed(config.trainer.seed, 201, epoch, step_index)
        with _seed_scope(step_seed):
            sampled_runtime = _resolve_train_runtime(config, batch_size=images.shape[0])
            output = trainer.train_step(
                images,
                snr_db=sampled_runtime.snr_db,
                sem_rate_ratio=sampled_runtime.sem_rate_ratio,
                det_rate_ratio=sampled_runtime.det_rate_ratio,
                semantic_bandwidth_budget=config.base.runtime.semantic_bandwidth_budget,
            )
        step_count += 1
        total_loss += output.total_loss
        _merge_numeric_dict(term_totals, output.terms)
        metrics_list.append(output.metrics)

        if step_index % config.trainer.log_every_steps == 0:
            _print_event(
                {
                    "event": "train_step",
                    "mode": config.base.runtime.mode,
                    "epoch": epoch,
                    "step": step_index,
                    "global_step": output.global_step,
                    "total_loss": output.total_loss,
                    "psnr": output.metrics.psnr,
                }
            )
        if config.trainer.max_train_batches is not None and step_index >= config.trainer.max_train_batches:
            break

    if step_count == 0:
        raise RuntimeError("Training loader produced zero steps.")

    return {
        "steps": step_count,
        "total_loss": total_loss / step_count,
        "terms": _mean_numeric_dict(term_totals, step_count),
        "metrics": _metrics_to_dict(_average_metrics(metrics_list)),
        "stage": trainer.current_stage_summary(),
        "adversarial_active": trainer.adversarial_active(),
    }


def _run_single_user_validation_epoch(
    trainer: SingleUserTrainer,
    loader: DataLoader,
    config: Route3TrainConfig,
    epoch: int,
) -> dict[str, Any]:
    trainer.set_epoch(epoch)
    term_totals: dict[str, float] = {}
    metrics_list: list[ReconstructionMetrics] = []
    base_metrics_list: list[ReconstructionMetrics | None] = []
    final_metrics_list: list[ReconstructionMetrics | None] = []
    budget_metrics_list = []
    enhancement_status: dict[str, Any] | None = None
    step_count = 0

    for step_index, (images, _) in enumerate(loader, start=1):
        step_seed = _compose_seed(config.trainer.seed, 301, step_index)
        with _seed_scope(step_seed):
            output = trainer.validate_step(
                images,
                snr_db=config.base.runtime.snr_db,
                sem_rate_ratio=config.base.runtime.sem_rate_ratio,
                det_rate_ratio=config.base.runtime.det_rate_ratio,
            )
        step_count += 1
        _merge_numeric_dict(term_totals, output.terms)
        metrics_list.append(output.metrics)
        base_metrics_list.append(output.base_metrics)
        final_metrics_list.append(output.final_metrics)
        budget_metrics_list.append(output.budget_metrics)
        enhancement_status = output.enhancement_status
        if config.trainer.max_val_batches is not None and step_index >= config.trainer.max_val_batches:
            break

    if step_count == 0:
        raise RuntimeError("Validation loader produced zero steps.")

    return {
        "steps": step_count,
        "terms": _mean_numeric_dict(term_totals, step_count),
        "metrics": _metrics_to_dict(_average_metrics(metrics_list)),
        "base_metrics": (
            _metrics_to_dict(_average_optional_metrics(base_metrics_list))
            if _average_optional_metrics(base_metrics_list) is not None
            else None
        ),
        "final_metrics": (
            _metrics_to_dict(_average_optional_metrics(final_metrics_list))
            if _average_optional_metrics(final_metrics_list) is not None
            else None
        ),
        "budget": average_single_user_budget_metrics(budget_metrics_list),
        "enhancement_status": enhancement_status,
        "stage": trainer.current_stage_summary(),
        "adversarial_active": trainer.adversarial_active(),
    }


def _run_multi_user_validation_epoch(
    trainer: MultiUserTrainer,
    loader: DataLoader,
    config: Route3TrainConfig,
    epoch: int,
) -> dict[str, Any]:
    trainer.set_epoch(epoch)
    term_totals: dict[str, float] = {}
    metrics_list: list[ReconstructionMetrics] = []
    base_metrics_list: list[ReconstructionMetrics | None] = []
    final_metrics_list: list[ReconstructionMetrics | None] = []
    comparisons: list[MultiUserComparison] = []
    enhancement_status: dict[str, Any] | None = None
    step_count = 0

    for step_index, (images, _) in enumerate(loader, start=1):
        step_seed = _compose_seed(config.trainer.seed, 401, step_index)
        with _seed_scope(step_seed):
            output = trainer.validate_step(
                images,
                snr_db=config.base.runtime.snr_db,
                sem_rate_ratio=config.base.runtime.sem_rate_ratio,
                det_rate_ratio=config.base.runtime.det_rate_ratio,
                semantic_bandwidth_budget=config.base.runtime.semantic_bandwidth_budget,
            )
        step_count += 1
        _merge_numeric_dict(term_totals, output.terms)
        metrics_list.append(output.metrics)
        base_metrics_list.append(output.base_metrics)
        final_metrics_list.append(output.final_metrics)
        comparisons.append(output.comparison)
        enhancement_status = output.enhancement_status
        if config.trainer.max_val_batches is not None and step_index >= config.trainer.max_val_batches:
            break

    if step_count == 0:
        raise RuntimeError("Validation loader produced zero steps.")

    mean_comparison = _average_multi_user_comparison(comparisons)
    return {
        "steps": step_count,
        "terms": _mean_numeric_dict(term_totals, step_count),
        "metrics": _metrics_to_dict(_average_metrics(metrics_list)),
        "base_metrics": (
            _metrics_to_dict(_average_optional_metrics(base_metrics_list))
            if _average_optional_metrics(base_metrics_list) is not None
            else None
        ),
        "final_metrics": (
            _metrics_to_dict(_average_optional_metrics(final_metrics_list))
            if _average_optional_metrics(final_metrics_list) is not None
            else None
        ),
        "enhancement_status": enhancement_status,
        "comparison": mean_comparison,
        "stage": trainer.current_stage_summary(),
        "adversarial_active": trainer.adversarial_active(),
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _serialize_training_config(config: Route3TrainConfig) -> dict[str, Any]:
    return asdict(config)


def _resume_compatibility_payload(config: Route3TrainConfig) -> dict[str, Any]:
    payload = _serialize_training_config(config)
    payload.pop("config_path", None)
    trainer_payload = dict(payload["trainer"])
    trainer_payload.pop("epochs", None)
    trainer_payload.pop("resume_checkpoint", None)
    trainer_payload.pop("init_model_checkpoint", None)
    payload["trainer"] = trainer_payload
    return payload


def _config_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_snapshot_path(report_dir: Path) -> Path:
    return report_dir / "train_config.snapshot.json"


def _write_config_snapshot(report_dir: Path, config: Route3TrainConfig) -> tuple[Path, str]:
    full_payload = _serialize_training_config(config)
    compatibility_payload = _resume_compatibility_payload(config)
    fingerprint = _config_fingerprint(compatibility_payload)
    snapshot_path = _config_snapshot_path(report_dir)
    snapshot_payload = {
        "config_fingerprint": fingerprint,
        "resume_compatibility_payload": compatibility_payload,
        "full_config": full_payload,
    }
    _save_json(snapshot_path, snapshot_payload)
    return snapshot_path, fingerprint


def _ensure_fresh_run_target(report_dir: Path, checkpoint_dir: Path) -> None:
    blocking_paths = [
        report_dir / "train_history.json",
        report_dir / "train_summary.json",
        _config_snapshot_path(report_dir),
        checkpoint_dir / "latest.pt",
        checkpoint_dir / "best.pt",
        checkpoint_dir / "best_psnr.pt",
    ]
    existing = [path for path in blocking_paths if path.exists()]
    if existing:
        raise RuntimeError(
            "Refusing to start a fresh run in a non-empty artifact directory. "
            "Use a new output dir or resume_checkpoint. Existing paths: "
            + ", ".join(str(path) for path in existing[:8])
        )


def _validate_resume_compatibility(
    *,
    config: Route3TrainConfig,
    report_dir: Path,
    checkpoint_extra_state: dict[str, Any],
) -> tuple[Path | None, str | None]:
    history_path = report_dir / "train_history.json"
    if not history_path.exists():
        raise RuntimeError(
            "resume_checkpoint was provided, but report_dir has no train_history.json. "
            "Use init_model_checkpoint to branch from an old checkpoint into a new artifact dir."
        )

    history_payload = _load_json(history_path)
    if history_payload is None:
        raise RuntimeError(f"Failed to load existing history: {history_path}")

    current_payload = _resume_compatibility_payload(config)
    current_fingerprint = _config_fingerprint(current_payload)
    snapshot_path = _config_snapshot_path(report_dir)
    snapshot_payload = _load_json(snapshot_path)
    if snapshot_payload is not None:
        previous_fingerprint = snapshot_payload.get("config_fingerprint")
        if previous_fingerprint != current_fingerprint:
            raise RuntimeError(
                "Resume configuration mismatch: current config does not match the frozen config snapshot "
                f"at {snapshot_path}."
            )
        return snapshot_path, str(previous_fingerprint)

    previous_mode = history_payload.get("mode")
    current_mode = config.base.runtime.mode
    if previous_mode is not None and previous_mode != current_mode:
        raise RuntimeError(
            f"Resume mode mismatch: history mode is {previous_mode}, current config mode is {current_mode}."
        )

    checkpoint_mode = checkpoint_extra_state.get("mode")
    if checkpoint_mode is not None and checkpoint_mode != current_mode:
        raise RuntimeError(
            f"Resume checkpoint mode mismatch: checkpoint mode is {checkpoint_mode}, current config mode is {current_mode}."
        )

    previous_summary = _load_json(report_dir / "train_summary.json") or {}
    mismatches: list[str] = []
    if "trainer_seed" in previous_summary and int(previous_summary["trainer_seed"]) != config.trainer.seed:
        mismatches.append("trainer.seed")
    if (
        "train_decode_stochastic" in previous_summary
        and bool(previous_summary["train_decode_stochastic"]) != config.trainer.train_decode_stochastic
    ):
        mismatches.append("trainer.train_decode_stochastic")
    if (
        "val_decode_stochastic" in previous_summary
        and bool(previous_summary["val_decode_stochastic"]) != config.trainer.val_decode_stochastic
    ):
        mismatches.append("trainer.val_decode_stochastic")
    if mismatches:
        raise RuntimeError(
            "Resume configuration mismatch against existing training summary: "
            + ", ".join(mismatches)
        )
    return None, None


def _load_model_state_from_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise RuntimeError(f"Unsupported checkpoint format for model initialization: {path}")


_INIT_ALLOWED_MISSING_PREFIXES = (
    "semantic_distillation.teacher.",
    "single_user_model.semantic_distillation.teacher.",
    "enhancer.discriminator.",
    "enhancer.refiner.",
    "enhancer.perceptual_loss.feature_extractor.",
    "single_user_model.enhancer.discriminator.",
    "single_user_model.enhancer.refiner.",
    "single_user_model.enhancer.perceptual_loss.feature_extractor.",
)

_INIT_ALLOWED_UNEXPECTED_PREFIXES = (
    "enhancer.discriminator.",
    "enhancer.refiner.",
    "single_user_model.enhancer.discriminator.",
    "single_user_model.enhancer.refiner.",
)


def _initialize_model_from_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    init_state = _load_model_state_from_checkpoint(path)
    incompatible = model.load_state_dict(init_state, strict=False)
    disallowed_missing = [
        key for key in incompatible.missing_keys if not key.startswith(_INIT_ALLOWED_MISSING_PREFIXES)
    ]
    if disallowed_missing:
        raise RuntimeError(
            "Init model checkpoint is missing non-teacher weights: "
            + ", ".join(disallowed_missing[:16])
        )
    disallowed_unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith(_INIT_ALLOWED_UNEXPECTED_PREFIXES)
    ]
    if disallowed_unexpected:
        raise RuntimeError(
            "Init model checkpoint contains unexpected weights: "
            + ", ".join(disallowed_unexpected[:16])
        )


def _apply_trainable_prefixes(model: torch.nn.Module, prefixes: tuple[str, ...] | None) -> None:
    if prefixes is None:
        return

    matched_prefixes = {prefix: False for prefix in prefixes}
    for _, parameter in model.named_parameters():
        parameter.requires_grad_(False)

    for name, parameter in model.named_parameters():
        for prefix in prefixes:
            if name.startswith(prefix):
                parameter.requires_grad_(True)
                matched_prefixes[prefix] = True
                break

    unmatched = [prefix for prefix, matched in matched_prefixes.items() if not matched]
    if unmatched:
        raise ValueError(
            "trainer.trainable_prefixes did not match any model parameters: "
            + ", ".join(unmatched)
        )


def _extract_selection_metric(record: dict[str, Any], mode: str) -> SelectionMetric | None:
    validation = record.get("validation")
    if validation is None:
        return None
    metrics = validation.get("final_metrics") or validation.get("metrics", {})
    psnr = metrics.get("psnr")
    if psnr is None:
        return None
    psnr = float(psnr)
    if mode == "single_user":
        budget = validation.get("budget")
        if isinstance(budget, dict) and budget.get("operating_mode") == "matched_budget":
            relative_gap = budget.get("cbr_relative_gap")
            if relative_gap is None:
                relative_gap = float("inf")
            relative_gap = float(relative_gap)
            tolerance = float(budget.get("target_effective_cbr_tolerance", 0.0) or 0.0)
            within_target = bool(budget.get("within_target_tolerance"))
            overflow = max(relative_gap - tolerance, 0.0)
            if within_target:
                return SelectionMetric(
                    name="matched_budget_psnr",
                    value=psnr,
                    sort_key=(1.0, psnr, -relative_gap),
                )
            return SelectionMetric(
                name="matched_budget_psnr",
                value=psnr,
                sort_key=(0.0, -overflow, psnr),
            )
    return SelectionMetric(name="psnr", value=psnr, sort_key=(psnr,))


def _is_better_selection_metric(candidate: SelectionMetric, incumbent: SelectionMetric | None) -> bool:
    if incumbent is None:
        return True
    return candidate.sort_key > incumbent.sort_key


def _find_best_selection_metric_in_history(
    history: list[dict[str, Any]],
    *,
    mode: str,
) -> SelectionMetric | None:
    best_selection: SelectionMetric | None = None
    for record in history:
        if not isinstance(record, dict):
            continue
        selection_metric = _extract_selection_metric(record, mode)
        if selection_metric is None:
            continue
        if _is_better_selection_metric(selection_metric, best_selection):
            best_selection = selection_metric
    return best_selection


def _load_resume_history(
    report_dir: Path,
    start_epoch: int,
    mode: str,
) -> tuple[list[dict[str, Any]], float | None, str | None, str | None, int | None, str | None]:
    history_payload = _load_json(report_dir / "train_history.json")
    if history_payload is None:
        return [], None, None, None, None, None

    loaded_history = history_payload.get("history", [])
    if not isinstance(loaded_history, list):
        raise RuntimeError("train_history.json is malformed: history must be a list.")

    history: list[dict[str, Any]] = []
    best_metric: float | None = None
    best_checkpoint_path: str | None = None
    best_psnr_checkpoint_path: str | None = None
    best_epoch: int | None = None
    best_metric_name: str | None = None
    best_selection: SelectionMetric | None = None
    best_selection: SelectionMetric | None = None
    for record in loaded_history:
        if not isinstance(record, dict):
            raise RuntimeError("train_history.json is malformed: history entries must be objects.")
        epoch = int(record.get("epoch", 0))
        if epoch >= start_epoch:
            continue
        history.append(record)
        selection_metric = _extract_selection_metric(record, mode)
        if selection_metric is None:
            continue
        if _is_better_selection_metric(selection_metric, best_selection):
            best_selection = selection_metric
            best_metric = selection_metric.value
            candidate_path = record.get("best_checkpoint")
            candidate_psnr_path = record.get("best_psnr_checkpoint", candidate_path)
            best_checkpoint_path = str(candidate_path) if candidate_path is not None else None
            best_psnr_checkpoint_path = str(candidate_psnr_path) if candidate_psnr_path is not None else None
            best_epoch = int(record.get("best_epoch", epoch))
            best_metric_name = selection_metric.name

    return history, best_metric, best_checkpoint_path, best_psnr_checkpoint_path, best_epoch, best_metric_name


def run_training(config: Route3TrainConfig) -> dict[str, Any]:
    validate_training_config(config)
    prepare_artifact_dirs(config.base)
    _set_seed(config.trainer.seed)
    accelerator_runtime = _configure_accelerator_runtime(config.base.runtime.device)
    checkpoint_dir = Path(config.base.artifacts.checkpoint_dir)
    report_dir = Path(config.base.artifacts.report_dir)

    if config.trainer.resume_checkpoint is None:
        _ensure_fresh_run_target(report_dir, checkpoint_dir)

    model = build_runtime_model(config.base)
    _apply_trainable_prefixes(model, config.trainer.trainable_prefixes)
    if config.trainer.init_model_checkpoint is not None:
        _initialize_model_from_checkpoint(model, config.trainer.init_model_checkpoint)
    if config.base.runtime.mode == "single_user":
        trainer: SingleUserTrainer | MultiUserTrainer = SingleUserTrainer(
            model=model,
            config=SingleUserTrainConfig(
                learning_rate=config.trainer.learning_rate,
                discriminator_learning_rate=config.trainer.discriminator_learning_rate,
                weight_decay=config.trainer.weight_decay,
                distillation_weight=config.trainer.distillation_weight,
                enhancement_weight=config.trainer.enhancement_weight,
                max_grad_norm=config.trainer.max_grad_norm,
                train_decode_stochastic=config.trainer.train_decode_stochastic,
                val_decode_stochastic=config.trainer.val_decode_stochastic,
                device=config.base.runtime.device,
                operating_mode=config.base.runtime.operating_mode,
                target_effective_cbr=config.base.runtime.target_effective_cbr,
                target_effective_cbr_tolerance=config.base.runtime.target_effective_cbr_tolerance,
                total_epochs=config.trainer.epochs,
                phase1_epochs=config.trainer.phase1_epochs,
                phase2_epochs=config.trainer.phase2_epochs,
                perceptual_weight_max=config.trainer.perceptual_weight_max,
                perceptual_loss_scale=config.trainer.perceptual_loss_scale,
                adversarial_weight=config.trainer.adversarial_weight,
                adversarial_ramp_epochs=config.trainer.adversarial_ramp_epochs,
                decoder_ms_ssim_weight=config.trainer.decoder_ms_ssim_weight,
                decoder_l1_weight=config.trainer.decoder_l1_weight,
                decoder_mse_weight=config.trainer.decoder_mse_weight,
                decoder_residual_weight=config.trainer.decoder_residual_weight,
                base_decoder_aux_weight=config.trainer.base_decoder_aux_weight,
                final_decoder_weight=config.trainer.final_decoder_weight,
                rate_regularization_weight=config.trainer.rate_regularization_weight,
                refinement_consistency_weight=config.trainer.refinement_consistency_weight,
                refinement_delta_weight=config.trainer.refinement_delta_weight,
            ),
        )
    else:
        trainer = MultiUserTrainer(
            model=model,
            config=MultiUserTrainConfig(
                learning_rate=config.trainer.learning_rate,
                discriminator_learning_rate=config.trainer.discriminator_learning_rate,
                weight_decay=config.trainer.weight_decay,
                distillation_weight=config.trainer.distillation_weight,
                enhancement_weight=config.trainer.enhancement_weight,
                max_grad_norm=config.trainer.max_grad_norm,
                train_decode_stochastic=config.trainer.train_decode_stochastic,
                val_decode_stochastic=config.trainer.val_decode_stochastic,
                device=config.base.runtime.device,
                total_epochs=config.trainer.epochs,
                phase1_epochs=config.trainer.phase1_epochs,
                phase2_epochs=config.trainer.phase2_epochs,
                perceptual_weight_max=config.trainer.perceptual_weight_max,
                perceptual_loss_scale=config.trainer.perceptual_loss_scale,
                adversarial_weight=config.trainer.adversarial_weight,
                adversarial_ramp_epochs=config.trainer.adversarial_ramp_epochs,
                decoder_ms_ssim_weight=config.trainer.decoder_ms_ssim_weight,
                decoder_l1_weight=config.trainer.decoder_l1_weight,
                decoder_mse_weight=config.trainer.decoder_mse_weight,
                decoder_residual_weight=config.trainer.decoder_residual_weight,
            ),
        )

    val_loader = _build_loader(config, split=config.base.dataset.val_split, shuffle=False)

    history: list[dict[str, Any]] = []
    best_metric: float | None = None
    best_checkpoint_path: str | None = None
    best_psnr_checkpoint_path: str | None = None
    best_epoch: int | None = None
    best_metric_name: str | None = None
    best_selection: SelectionMetric | None = None
    start_epoch = 1
    config_snapshot_path: Path | None = None
    config_fingerprint: str | None = None

    if config.trainer.resume_checkpoint is not None:
        extra_state = trainer.load_checkpoint(config.trainer.resume_checkpoint)
        config_snapshot_path, config_fingerprint = _validate_resume_compatibility(
            config=config,
            report_dir=report_dir,
            checkpoint_extra_state=extra_state,
        )
        start_epoch = int(extra_state.get("epoch", 0)) + 1
        history, best_metric, best_checkpoint_path, best_psnr_checkpoint_path, best_epoch, best_metric_name = _load_resume_history(
            report_dir=report_dir,
            start_epoch=start_epoch,
            mode=config.base.runtime.mode,
        )
        best_selection = _find_best_selection_metric_in_history(history, mode=config.base.runtime.mode)

    if config_snapshot_path is None or config_fingerprint is None:
        config_snapshot_path, config_fingerprint = _write_config_snapshot(report_dir, config)

    report_writer = Phase8ReportWriter() if config.base.runtime.mode == "multi_user" else None

    for epoch in range(start_epoch, config.trainer.epochs + 1):
        train_loader = _build_loader(
            config,
            split=config.base.dataset.train_split,
            shuffle=True,
            generator=_build_epoch_generator(_compose_seed(config.trainer.seed, 1, epoch)),
        )
        if config.base.runtime.mode == "single_user":
            train_summary = _run_single_user_train_epoch(trainer, train_loader, config, epoch)
        else:
            train_summary = _run_multi_user_train_epoch(trainer, train_loader, config, epoch)

        validation_summary = None
        if epoch % config.trainer.validate_every_epochs == 0:
            if config.base.runtime.mode == "single_user":
                validation_summary = _run_single_user_validation_epoch(trainer, val_loader, config, epoch)
            else:
                validation_summary = _run_multi_user_validation_epoch(trainer, val_loader, config, epoch)
            selection_metric = _extract_selection_metric(
                {
                    "validation": validation_summary,
                },
                mode=config.base.runtime.mode,
            )
            if selection_metric is None:
                raise RuntimeError("Validation completed but no selection metric could be extracted.")

            if _is_better_selection_metric(selection_metric, best_selection):
                best_selection = selection_metric
                best_metric = selection_metric.value
                best_metric_name = selection_metric.name
                best_epoch = epoch
                best_checkpoint_path = str(
                    trainer.save_checkpoint(
                        checkpoint_dir / "best.pt",
                        extra_state={
                            "epoch": epoch,
                            "config_path": config.config_path,
                            "config_fingerprint": config_fingerprint,
                            "mode": config.base.runtime.mode,
                            "trainer_seed": config.trainer.seed,
                            "train_decode_stochastic": config.trainer.train_decode_stochastic,
                            "val_decode_stochastic": config.trainer.val_decode_stochastic,
                        },
                    )
                )
                best_psnr_checkpoint_path = str(
                    trainer.save_checkpoint(
                        checkpoint_dir / "best_psnr.pt",
                        extra_state={
                            "epoch": epoch,
                            "config_path": config.config_path,
                            "config_fingerprint": config_fingerprint,
                            "mode": config.base.runtime.mode,
                            "selection_metric_name": selection_metric.name,
                            "selection_metric_value": selection_metric.value,
                            "selection_metric_sort_key": list(selection_metric.sort_key),
                            "trainer_seed": config.trainer.seed,
                            "train_decode_stochastic": config.trainer.train_decode_stochastic,
                            "val_decode_stochastic": config.trainer.val_decode_stochastic,
                        },
                    )
                )

        if epoch % config.trainer.checkpoint_every_epochs == 0 or epoch == config.trainer.epochs:
            trainer.save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                extra_state={
                    "epoch": epoch,
                    "config_path": config.config_path,
                    "config_fingerprint": config_fingerprint,
                    "mode": config.base.runtime.mode,
                    "trainer_seed": config.trainer.seed,
                    "train_decode_stochastic": config.trainer.train_decode_stochastic,
                    "val_decode_stochastic": config.trainer.val_decode_stochastic,
                },
            )

        latest_checkpoint_path = str(
            trainer.save_checkpoint(
                checkpoint_dir / "latest.pt",
                extra_state={
                    "epoch": epoch,
                    "config_path": config.config_path,
                    "config_fingerprint": config_fingerprint,
                    "mode": config.base.runtime.mode,
                    "trainer_seed": config.trainer.seed,
                    "train_decode_stochastic": config.trainer.train_decode_stochastic,
                    "val_decode_stochastic": config.trainer.val_decode_stochastic,
                },
            )
        )

        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "train": train_summary,
            "validation": None,
            "trainer_seed": config.trainer.seed,
            "adversarial_active": train_summary["adversarial_active"],
            "train_decode_stochastic": config.trainer.train_decode_stochastic,
            "val_decode_stochastic": config.trainer.val_decode_stochastic,
            "config_fingerprint": config_fingerprint,
            "config_snapshot_path": str(config_snapshot_path),
            "best_epoch": best_epoch,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric,
            "best_psnr_checkpoint": best_psnr_checkpoint_path,
            "latest_checkpoint": latest_checkpoint_path,
            "best_checkpoint": best_checkpoint_path,
        }
        if validation_summary is not None:
            serializable_validation = dict(validation_summary)
            if "comparison" in serializable_validation:
                comparison = serializable_validation.pop("comparison")
                serializable_validation["comparison"] = asdict(comparison)
                if report_writer is not None:
                    report_writer.write_report(
                        report_dir / f"phase8_epoch_{epoch:03d}.md",
                        comparison,
                        extra={
                            "epoch": epoch,
                            "checkpoint": latest_checkpoint_path,
                            "config_path": config.config_path,
                        },
                    )
            epoch_record["validation"] = serializable_validation

        history.append(epoch_record)
        _save_json(
            report_dir / "train_history.json",
            {
                "config_path": config.config_path,
                "config_fingerprint": config_fingerprint,
                "config_snapshot_path": str(config_snapshot_path),
                "mode": config.base.runtime.mode,
                "history": history,
            },
        )
        val_total_loss = None
        if validation_summary is not None:
            if config.base.runtime.mode == "single_user":
                val_total_loss = validation_summary["terms"]["validation_total"]
            else:
                val_total_loss = validation_summary["terms"]["multi_user_validation_total"]

        _print_event(
            {
                "event": "epoch_end",
                "mode": config.base.runtime.mode,
                "operating_mode": config.base.runtime.operating_mode,
                "epoch": epoch,
                "stage": train_summary["stage"],
                "train_total_loss": train_summary["total_loss"],
                "val_total_loss": val_total_loss,
                "adversarial_active": train_summary["adversarial_active"],
                "selection_metric_name": best_metric_name,
                "selection_metric_value": best_metric,
                "latest_checkpoint": latest_checkpoint_path,
                "best_checkpoint": best_checkpoint_path,
            }
        )

    summary = {
        "mode": config.base.runtime.mode,
        "operating_mode": config.base.runtime.operating_mode,
        "device": config.base.runtime.device,
        "accelerator_runtime": accelerator_runtime,
        "epochs_completed": len(history),
        "config_path": config.config_path,
        "history_path": str(report_dir / "train_history.json"),
        "trainer_seed": config.trainer.seed,
        "adversarial_active": history[-1]["adversarial_active"],
        "final_stage": history[-1]["train"]["stage"],
        "train_decode_stochastic": config.trainer.train_decode_stochastic,
        "val_decode_stochastic": config.trainer.val_decode_stochastic,
        "config_fingerprint": config_fingerprint,
        "config_snapshot_path": str(config_snapshot_path),
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric,
        "best_psnr_checkpoint": best_psnr_checkpoint_path,
        "latest_checkpoint": history[-1]["latest_checkpoint"],
        "best_checkpoint": best_checkpoint_path,
    }
    _save_json(report_dir / "train_summary.json", summary)
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description="Route-3 formal training launcher.")
    parser.add_argument("--config", required=True, help="Path to the route-3 training JSON config.")
    args = parser.parse_args()

    config = load_training_config(args.config)
    summary = run_training(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()

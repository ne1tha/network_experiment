from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from route2_swinjscc_gan.common.checks import require_positive_int
from route2_swinjscc_gan.common.io import ensure_dir, save_json
from route2_swinjscc_gan.configs.manifest_loader import LoadedRoute2Experiment, load_experiment_config


CHECKPOINT_PATTERN = re.compile(r"epoch_(\d+)\.pt$")
VALIDATION_PATTERN = re.compile(r"epoch_(\d+)$")


@dataclass(slots=True)
class GPUSnapshot:
    index: int
    utilization_gpu: int
    memory_used: int
    memory_total: int


@dataclass(slots=True)
class TrainingRunSnapshot:
    observed_at: str
    run_dir: str
    total_epochs: int
    latest_checkpoint_epoch: int | None
    latest_validation_epoch: int | None
    next_expected_checkpoint_epoch: int | None
    next_expected_validation_epoch: int | None
    has_last_checkpoint: bool
    has_training_log: bool
    has_test_summary: bool
    completed: bool
    stage: str
    latest_metrics: dict[str, float | int] | None
    gpu: list[GPUSnapshot] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor a Route 2 training run until it completes.")
    parser.add_argument("--config-json", type=Path, required=True, help="Training experiment JSON config path.")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval in seconds.")
    parser.add_argument(
        "--stale-timeout-minutes",
        type=int,
        default=90,
        help="Abort if there is no observed progress for this many minutes and no active GPU load.",
    )
    parser.add_argument("--history-log", type=Path, default=None, help="Optional JSONL history output path.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional final summary JSON output path.")
    parser.add_argument("--skip-gpu-query", action="store_true", help="Disable nvidia-smi polling.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit immediately.")
    return parser.parse_args()


def scheduled_epochs(total_epochs: int, every_epochs: int) -> tuple[int, ...]:
    require_positive_int("total_epochs", total_epochs)
    require_positive_int("every_epochs", every_epochs)
    return tuple(range(every_epochs, total_epochs + 1, every_epochs))


def latest_epoch(paths: list[Path], pattern: re.Pattern[str]) -> int | None:
    latest: int | None = None
    for path in paths:
        match = pattern.match(path.name)
        if match is None:
            continue
        value = int(match.group(1))
        if latest is None or value > latest:
            latest = value
    return latest


def next_expected_epoch(current_epoch: int | None, scheduled: tuple[int, ...]) -> int | None:
    for epoch in scheduled:
        if current_epoch is None or epoch > current_epoch:
            return epoch
    return None


def current_stage_label(epoch: int | None, *, phase1_epochs: int, phase2_epochs: int, completed: bool) -> str:
    if completed:
        return "completed"
    if epoch is None or epoch <= 0:
        return "starting"
    if epoch <= phase1_epochs:
        return "stage1_reconstruction"
    if epoch <= phase1_epochs + phase2_epochs:
        return "stage2_perceptual"
    return "stage3_adversarial"


def is_run_completed(
    *,
    has_last_checkpoint: bool,
    has_training_log: bool,
    has_test_summary: bool,
    require_test_summary: bool,
) -> bool:
    if not has_last_checkpoint or not has_training_log:
        return False
    if require_test_summary and not has_test_summary:
        return False
    return True


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)!r}.")
    return payload


def query_gpu_snapshots() -> list[GPUSnapshot] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return None

    snapshots: list[GPUSnapshot] = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 4:
            continue
        snapshots.append(
            GPUSnapshot(
                index=int(parts[0]),
                utilization_gpu=int(parts[1]),
                memory_used=int(parts[2]),
                memory_total=int(parts[3]),
            )
        )
    return snapshots


def any_gpu_active(gpu_snapshots: list[GPUSnapshot] | None) -> bool:
    if gpu_snapshots is None:
        return False
    return any(snapshot.utilization_gpu >= 10 or snapshot.memory_used >= 1024 for snapshot in gpu_snapshots)


def relevant_artifact_mtime(run_dir: Path, snapshot: TrainingRunSnapshot) -> float | None:
    candidates = [
        run_dir / "experiment_config.json",
        run_dir / "training_log.json",
        run_dir / "test_summary.json",
        run_dir / "checkpoints" / "last.pt",
    ]
    if snapshot.latest_checkpoint_epoch is not None:
        candidates.append(run_dir / "checkpoints" / f"epoch_{snapshot.latest_checkpoint_epoch:04d}.pt")
    if snapshot.latest_validation_epoch is not None:
        candidates.append(run_dir / "validation" / f"epoch_{snapshot.latest_validation_epoch:04d}" / "metrics.json")

    mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
    return max(mtimes) if mtimes else None


def build_snapshot(loaded: LoadedRoute2Experiment, *, include_gpu: bool) -> TrainingRunSnapshot:
    config = loaded.config
    run_dir = config.training.output_path
    checkpoints_dir = run_dir / "checkpoints"
    validation_dir = run_dir / "validation"

    checkpoint_paths = list(checkpoints_dir.glob("epoch_*.pt")) if checkpoints_dir.exists() else []
    validation_paths = [path for path in validation_dir.iterdir() if path.is_dir()] if validation_dir.exists() else []
    latest_checkpoint_epoch = latest_epoch(checkpoint_paths, CHECKPOINT_PATTERN)
    latest_validation_epoch = latest_epoch(validation_paths, VALIDATION_PATTERN)

    expected_checkpoints = scheduled_epochs(config.training.total_epochs, config.training.save_every_epochs)
    expected_validations = scheduled_epochs(config.training.total_epochs, config.training.eval_every_epochs)

    has_last_checkpoint = (checkpoints_dir / "last.pt").exists()
    has_training_log = (run_dir / "training_log.json").exists()
    has_test_summary = (run_dir / "test_summary.json").exists()
    completed = is_run_completed(
        has_last_checkpoint=has_last_checkpoint,
        has_training_log=has_training_log,
        has_test_summary=has_test_summary,
        require_test_summary=bool(config.data.test_roots),
    )
    reference_epoch = latest_validation_epoch if latest_validation_epoch is not None else latest_checkpoint_epoch
    stage = current_stage_label(
        reference_epoch,
        phase1_epochs=config.training.phase1_epochs,
        phase2_epochs=config.training.phase2_epochs,
        completed=completed,
    )

    latest_metrics = None
    if latest_validation_epoch is not None:
        latest_metrics = load_json_if_exists(validation_dir / f"epoch_{latest_validation_epoch:04d}" / "metrics.json")

    return TrainingRunSnapshot(
        observed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        run_dir=str(run_dir),
        total_epochs=config.training.total_epochs,
        latest_checkpoint_epoch=latest_checkpoint_epoch,
        latest_validation_epoch=latest_validation_epoch,
        next_expected_checkpoint_epoch=next_expected_epoch(latest_checkpoint_epoch, expected_checkpoints),
        next_expected_validation_epoch=next_expected_epoch(latest_validation_epoch, expected_validations),
        has_last_checkpoint=has_last_checkpoint,
        has_training_log=has_training_log,
        has_test_summary=has_test_summary,
        completed=completed,
        stage=stage,
        latest_metrics=latest_metrics,
        gpu=None if not include_gpu else query_gpu_snapshots(),
    )


def format_snapshot(snapshot: TrainingRunSnapshot) -> str:
    checkpoint_text = (
        f"{snapshot.latest_checkpoint_epoch}/{snapshot.total_epochs}"
        if snapshot.latest_checkpoint_epoch is not None
        else f"none/{snapshot.total_epochs}"
    )
    validation_text = (
        f"{snapshot.latest_validation_epoch}/{snapshot.total_epochs}"
        if snapshot.latest_validation_epoch is not None
        else f"none/{snapshot.total_epochs}"
    )
    parts = [
        f"[{snapshot.observed_at}]",
        f"stage={snapshot.stage}",
        f"checkpoint={checkpoint_text}",
        f"validation={validation_text}",
        f"done={'yes' if snapshot.completed else 'no'}",
    ]
    if snapshot.next_expected_checkpoint_epoch is not None:
        parts.append(f"next_ckpt={snapshot.next_expected_checkpoint_epoch}")
    if snapshot.next_expected_validation_epoch is not None:
        parts.append(f"next_val={snapshot.next_expected_validation_epoch}")
    if snapshot.latest_metrics is not None:
        parts.append(
            "metrics="
            + ",".join(
                [
                    f"psnr={float(snapshot.latest_metrics['psnr']):.4f}",
                    f"ms_ssim={float(snapshot.latest_metrics['ms_ssim']):.4f}",
                    f"lpips={float(snapshot.latest_metrics['lpips']):.4f}",
                ]
            )
        )
    if snapshot.gpu is not None:
        active = [
            f"gpu{gpu.index}:{gpu.utilization_gpu}% {gpu.memory_used}/{gpu.memory_total}MiB"
            for gpu in snapshot.gpu
            if gpu.utilization_gpu > 0 or gpu.memory_used > 256
        ]
        if active:
            parts.append("gpu=" + "; ".join(active))
    return " ".join(parts)


def append_history(path: Path, snapshot: TrainingRunSnapshot) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(snapshot), ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    require_positive_int("poll_seconds", args.poll_seconds)
    require_positive_int("stale_timeout_minutes", args.stale_timeout_minutes)

    loaded = load_experiment_config(args.config_json)
    run_dir = loaded.config.training.output_path
    history_log = args.history_log if args.history_log is not None else run_dir / "watch_history.jsonl"
    summary_json = args.summary_json if args.summary_json is not None else run_dir / "watch_summary.json"

    last_progress_at = time.time()
    last_progress_key: tuple[int | None, int | None, bool, bool, bool] | None = None
    last_artifact_mtime: float | None = None

    while True:
        snapshot = build_snapshot(loaded, include_gpu=not args.skip_gpu_query)
        print(format_snapshot(snapshot), flush=True)
        append_history(history_log, snapshot)

        current_progress_key = (
            snapshot.latest_checkpoint_epoch,
            snapshot.latest_validation_epoch,
            snapshot.has_last_checkpoint,
            snapshot.has_training_log,
            snapshot.has_test_summary,
        )
        current_artifact_mtime = relevant_artifact_mtime(run_dir, snapshot)
        if current_progress_key != last_progress_key or (
            current_artifact_mtime is not None
            and (last_artifact_mtime is None or current_artifact_mtime > last_artifact_mtime)
        ):
            last_progress_at = time.time()
            last_progress_key = current_progress_key
            last_artifact_mtime = current_artifact_mtime

        if snapshot.completed:
            save_json(summary_json, asdict(snapshot))
            print(f"Training completed. Summary saved to {summary_json}", flush=True)
            return 0

        if args.once:
            return 0

        if time.time() - last_progress_at > args.stale_timeout_minutes * 60 and not any_gpu_active(snapshot.gpu):
            save_json(summary_json, asdict(snapshot))
            print(
                f"No progress observed for {args.stale_timeout_minutes} minutes and no active GPU load detected. "
                f"Summary saved to {summary_json}",
                flush=True,
            )
            return 2

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

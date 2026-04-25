from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.route3_train import _extract_selection_metric, _is_better_selection_metric


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _checkpoint_epoch(path: Path) -> int | None:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return None
    extra_state = checkpoint.get("extra_state", {})
    if not isinstance(extra_state, dict) or "epoch" not in extra_state:
        return None
    return int(extra_state["epoch"])


def _copy_checkpoint_atomically(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.tmp.",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        shutil.copy2(source, tmp_path)
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _resolve_best_source_checkpoint(checkpoint_dir: Path, *, epoch: int, last_epoch: int) -> Path:
    epoch_checkpoint = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    if epoch_checkpoint.exists():
        return epoch_checkpoint

    latest_checkpoint = checkpoint_dir / "latest.pt"
    if epoch == last_epoch and latest_checkpoint.exists() and _checkpoint_epoch(latest_checkpoint) == epoch:
        return latest_checkpoint

    raise FileNotFoundError(
        "Could not locate a materialized checkpoint for the selected best epoch "
        f"{epoch}. Expected {epoch_checkpoint} or a matching latest.pt."
    )


def reselect_best_checkpoint(report_dir: str | Path, checkpoint_dir: str | Path | None = None) -> dict[str, Any]:
    report_dir = Path(report_dir).resolve()
    checkpoint_dir = (Path(checkpoint_dir).resolve() if checkpoint_dir is not None else report_dir.parent / "checkpoints")
    history_path = report_dir / "train_history.json"
    history_payload = _load_json(history_path)
    history = history_payload.get("history", [])
    if not isinstance(history, list) or not history:
        raise RuntimeError(f"History is empty or malformed: {history_path}")

    scored_records: list[tuple[Any, dict[str, Any]]] = []
    for record in history:
        if not isinstance(record, dict):
            raise RuntimeError("History entries must be objects.")
        selection_metric = _extract_selection_metric(record, mode=history_payload.get("mode", "single_user"))
        if selection_metric is None:
            continue
        scored_records.append((selection_metric, record))

    if not scored_records:
        raise RuntimeError(f"No validation selection metrics found in {history_path}")

    last_epoch = max(int(record["epoch"]) for record in history if isinstance(record, dict) and "epoch" in record)
    overall_best_metric, overall_best_record = scored_records[0]
    for candidate_metric, candidate_record in scored_records[1:]:
        if _is_better_selection_metric(candidate_metric, overall_best_metric):
            overall_best_metric = candidate_metric
            overall_best_record = candidate_record
    overall_best_epoch = int(overall_best_record["epoch"])

    materialized_records: list[tuple[Any, dict[str, Any], Path]] = []
    skipped_unmaterialized_epochs: list[int] = []
    for metric_value, record in scored_records:
        epoch = int(record["epoch"])
        try:
            checkpoint_path = _resolve_best_source_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                last_epoch=last_epoch,
            )
        except FileNotFoundError:
            skipped_unmaterialized_epochs.append(epoch)
            continue
        materialized_records.append((metric_value, record, checkpoint_path))

    if not materialized_records:
        raise RuntimeError(
            f"No materialized checkpoints were found in {checkpoint_dir} for the validation history at {history_path}."
        )

    best_metric, best_record, source_checkpoint = materialized_records[0]
    for candidate_metric, candidate_record, candidate_checkpoint in materialized_records[1:]:
        if _is_better_selection_metric(candidate_metric, best_metric):
            best_metric = candidate_metric
            best_record = candidate_record
            source_checkpoint = candidate_checkpoint
    best_epoch = int(best_record["epoch"])
    best_checkpoint_path = checkpoint_dir / "best.pt"
    best_psnr_checkpoint_path = checkpoint_dir / "best_psnr.pt"
    if source_checkpoint.resolve() != best_checkpoint_path.resolve():
        _copy_checkpoint_atomically(source_checkpoint, best_checkpoint_path)
    if source_checkpoint.resolve() != best_psnr_checkpoint_path.resolve():
        _copy_checkpoint_atomically(source_checkpoint, best_psnr_checkpoint_path)

    selection_report = {
        "selection_metric_name": best_metric.name,
        "selection_metric_value": best_metric.value,
        "selection_metric_sort_key": list(best_metric.sort_key),
        "best_epoch": best_epoch,
        "best_overall_epoch": overall_best_epoch,
        "best_overall_metric_value": overall_best_metric.value,
        "used_materialized_checkpoint_only": best_epoch != overall_best_epoch,
        "skipped_unmaterialized_epochs": skipped_unmaterialized_epochs,
        "source_checkpoint": str(source_checkpoint),
        "best_checkpoint": str(best_checkpoint_path),
        "best_psnr_checkpoint": str(best_psnr_checkpoint_path),
        "history_path": str(history_path),
    }
    _save_json(report_dir / "best_reselection.json", selection_report)

    summary_path = report_dir / "train_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        summary["best_epoch"] = best_epoch
        summary["best_metric_name"] = best_metric.name
        summary["best_metric_value"] = best_metric.value
        summary["best_checkpoint"] = str(best_checkpoint_path)
        summary["best_psnr_checkpoint"] = str(best_psnr_checkpoint_path)
        _save_json(summary_path, summary)

    return selection_report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Reselect Route-3 best checkpoint by validation PSNR.")
    parser.add_argument("--report-dir", required=True, help="Path to the Route-3 reports directory.")
    parser.add_argument("--checkpoint-dir", default=None, help="Optional checkpoint directory override.")
    args = parser.parse_args()

    summary = reselect_best_checkpoint(args.report_dir, checkpoint_dir=args.checkpoint_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()

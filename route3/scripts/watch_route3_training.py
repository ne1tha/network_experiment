from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.route3_train import load_training_config


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_training_pids(config_path: Path) -> list[int]:
    matched: list[int] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return matched

    target = str(config_path.resolve())
    for proc_dir in proc_root.iterdir():
        if not proc_dir.name.isdigit():
            continue
        cmdline_path = proc_dir / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        args = [part for part in raw.decode(errors="ignore").split("\x00") if part]
        joined = " ".join(args)
        if "route3_train.py" in joined and target in joined:
            matched.append(int(proc_dir.name))
    return sorted(matched)


def _pid_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _epoch_payload(record: dict[str, Any]) -> dict[str, Any]:
    validation = record.get("validation")
    comparison = validation.get("comparison") if validation else None
    payload: dict[str, Any] = {
        "event": "epoch_observed",
        "epoch": record["epoch"],
        "train_total_loss": record["train"]["total_loss"],
        "train_psnr": record["train"]["metrics"]["psnr"],
        "latest_checkpoint": record.get("latest_checkpoint"),
        "best_checkpoint": record.get("best_checkpoint"),
    }
    if validation is not None:
        val_terms = validation.get("terms", {})
        if "multi_user_validation_total" in val_terms:
            payload["val_total_loss"] = val_terms["multi_user_validation_total"]
        elif "validation_total" in val_terms:
            payload["val_total_loss"] = val_terms["validation_total"]
        payload["val_psnr"] = validation["metrics"]["psnr"]
    if comparison is not None:
        payload["semantic_sharing_gain_psnr"] = comparison["semantic_sharing_gain_psnr"]
    return payload


def watch_training(
    config_path: str | Path,
    poll_seconds: float = 30.0,
    once: bool = False,
    pid: int | None = None,
    stall_seconds: float | None = None,
) -> int:
    config = load_training_config(config_path)
    resolved_config_path = Path(config.config_path)
    history_path = Path(config.base.artifacts.report_dir) / "train_history.json"
    summary_path = Path(config.base.artifacts.report_dir) / "train_summary.json"
    expected_epochs = config.trainer.epochs

    matched_pids = _find_training_pids(resolved_config_path) if pid is None else [pid]
    tracked_pid = matched_pids[0] if len(matched_pids) == 1 else None

    print(
        json.dumps(
            {
                "event": "monitor_start",
                "config_path": str(resolved_config_path),
                "expected_epochs": expected_epochs,
                "history_path": str(history_path),
                "summary_path": str(summary_path),
                "tracked_pid": tracked_pid,
                "matched_pids": matched_pids,
            },
            ensure_ascii=False,
        )
    )

    last_epoch_seen = 0
    last_progress_time = time.time()

    while True:
        summary = _read_json(summary_path)
        if summary is not None:
            print(json.dumps({"event": "training_complete", **summary}, ensure_ascii=False))
            return 0

        history = _read_json(history_path)
        records = history.get("history", []) if history is not None else []
        if len(records) > last_epoch_seen:
            for record in records[last_epoch_seen:]:
                print(json.dumps(_epoch_payload(record), ensure_ascii=False))
            last_epoch_seen = len(records)
            last_progress_time = time.time()

        if once:
            print(
                json.dumps(
                    {
                        "event": "monitor_snapshot",
                        "epochs_recorded": last_epoch_seen,
                        "expected_epochs": expected_epochs,
                        "tracked_pid": tracked_pid,
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        if tracked_pid is not None and not _pid_exists(tracked_pid):
            print(
                json.dumps(
                    {
                        "event": "training_process_missing",
                        "tracked_pid": tracked_pid,
                        "epochs_recorded": last_epoch_seen,
                        "summary_found": False,
                    },
                    ensure_ascii=False,
                )
            )
            return 1

        if stall_seconds is not None and time.time() - last_progress_time > stall_seconds:
            print(
                json.dumps(
                    {
                        "event": "monitor_stalled",
                        "stall_seconds": stall_seconds,
                        "epochs_recorded": last_epoch_seen,
                        "tracked_pid": tracked_pid,
                    },
                    ensure_ascii=False,
                )
            )
            return 2

        time.sleep(poll_seconds)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Watch a route-3 training run until completion.")
    parser.add_argument("--config", required=True, help="Path to the route-3 training JSON config.")
    parser.add_argument("--poll-seconds", type=float, default=30.0, help="Polling interval in seconds.")
    parser.add_argument("--stall-seconds", type=float, default=None, help="Optional stall timeout in seconds.")
    parser.add_argument("--pid", type=int, default=None, help="Optional explicit training PID to watch.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    args = parser.parse_args()
    raise SystemExit(
        watch_training(
            config_path=args.config,
            poll_seconds=args.poll_seconds,
            once=args.once,
            pid=args.pid,
            stall_seconds=args.stall_seconds,
        )
    )


if __name__ == "__main__":
    _main()

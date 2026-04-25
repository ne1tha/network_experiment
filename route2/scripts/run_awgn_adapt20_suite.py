from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from route2_swinjscc_gan.configs.manifest_loader import load_experiment_config


DEFAULT_CONFIGS = (
    PROJECT_PARENT / "route2_swinjscc_gan" / "configs" / "experiments" / "route2_awgn_adapt20_adv_off.local.json",
    PROJECT_PARENT / "route2_swinjscc_gan" / "configs" / "experiments" / "route2_awgn_adapt20_adv_ramp.local.json",
    PROJECT_PARENT / "route2_swinjscc_gan" / "configs" / "experiments" / "route2_awgn_adapt20_cond_disc_v1.local.json",
    PROJECT_PARENT / "route2_swinjscc_gan" / "configs" / "experiments" / "route2_awgn_adapt20_best375.local.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the route2 AWGN 20-epoch adaptation suite sequentially.")
    parser.add_argument("--device", default=None, help="Optional device override such as cuda:0 or cuda:1.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override for shared GPUs.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional dataloader worker override.")
    parser.add_argument(
        "--config-json",
        type=Path,
        action="append",
        default=None,
        help="Optional config override. When omitted, uses the default four-route suite.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip runs that already have last.pt and training_log.json artifacts.",
    )
    return parser.parse_args()


def _is_completed(config_json: Path) -> bool:
    loaded = load_experiment_config(config_json)
    run_dir = loaded.config.training.output_path
    if not (run_dir / "checkpoints" / "last.pt").exists():
        return False
    if not (run_dir / "training_log.json").exists():
        return False
    if not (run_dir / "test_summary.json").exists():
        return False

    total_epochs = loaded.config.training.total_epochs
    if total_epochs % loaded.config.training.save_every_epochs == 0:
        final_checkpoint = run_dir / "checkpoints" / f"epoch_{total_epochs:04d}.pt"
        if not final_checkpoint.exists():
            return False
    if total_epochs % loaded.config.training.eval_every_epochs == 0:
        final_validation = run_dir / "validation" / f"epoch_{total_epochs:04d}" / "metrics.json"
        if not final_validation.exists():
            return False
    return True


def _launch_train(
    config_json: Path,
    *,
    device: str | None,
    batch_size: int | None,
    num_workers: int | None,
) -> int:
    command = [
        sys.executable,
        str(PROJECT_PARENT / "route2_swinjscc_gan" / "scripts" / "train_from_config.py"),
        "--config-json",
        str(config_json),
    ]
    if device is not None:
        command.extend(["--device", device])
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    if num_workers is not None:
        command.extend(["--num-workers", str(num_workers)])
    process = subprocess.run(command, cwd=str(PROJECT_PARENT))
    return int(process.returncode)


def main() -> int:
    args = parse_args()
    configs = tuple(args.config_json) if args.config_json else DEFAULT_CONFIGS

    for config_json in configs:
        config_json = config_json.resolve()
        if args.skip_completed and _is_completed(config_json):
            print(f"[adapt20] skip completed: {config_json}", flush=True)
            continue
        print(f"[adapt20] start: {config_json}", flush=True)
        returncode = _launch_train(
            config_json,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if returncode != 0:
            print(f"[adapt20] failed ({returncode}): {config_json}", flush=True)
            return returncode
        print(f"[adapt20] done: {config_json}", flush=True)

    print("[adapt20] suite finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

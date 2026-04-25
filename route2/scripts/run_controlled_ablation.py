from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from route2_swinjscc_gan.configs.manifest_loader import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled Route 2 protocol: shared phase1+2, then two branched phase3 ablations."
    )
    parser.add_argument("--common-config", type=Path, required=True)
    parser.add_argument("--branch-config", type=Path, action="append", required=True)
    parser.add_argument("--common-device", default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def _launch_train(config_json: Path, *, device: str | None) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        str(PROJECT_PARENT / "route2_swinjscc_gan" / "scripts" / "train_from_config.py"),
        "--config-json",
        str(config_json),
    ]
    if device is not None:
        command.extend(["--device", device])
    return subprocess.Popen(command, cwd=str(PROJECT_PARENT))


def _branch_device(config_json: Path) -> str | None:
    loaded = load_experiment_config(config_json)
    return loaded.config.model.device


def main() -> int:
    args = parse_args()
    common_loaded = load_experiment_config(args.common_config)
    common_last = common_loaded.config.training.output_path / "checkpoints" / "last.pt"

    if not common_last.exists():
        print(f"[controlled] launching shared phase1+2 run: {args.common_config}", flush=True)
        common_process = _launch_train(args.common_config, device=args.common_device)
        common_returncode = common_process.wait()
        if common_returncode != 0:
            print(f"[controlled] shared phase1+2 run failed with code {common_returncode}", flush=True)
            return common_returncode
    else:
        print(f"[controlled] shared phase1+2 already finished: {common_last}", flush=True)

    while not common_last.exists():
        print(f"[controlled] waiting for shared phase1+2 completion: {common_last}", flush=True)
        time.sleep(args.poll_seconds)

    branch_processes: list[subprocess.Popen[bytes]] = []
    for branch_config in args.branch_config:
        device = _branch_device(branch_config)
        print(f"[controlled] launching branch {branch_config} on {device}", flush=True)
        branch_processes.append(_launch_train(branch_config, device=device))

    exit_code = 0
    for process in branch_processes:
        returncode = process.wait()
        if returncode != 0 and exit_code == 0:
            exit_code = returncode
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

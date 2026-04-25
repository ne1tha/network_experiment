from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from route2_swinjscc_gan.configs.manifest_loader import load_experiment_config
from route2_swinjscc_gan.scripts.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Route 2 from a JSON experiment config.")
    parser.add_argument("--config-json", type=Path, required=True, help="Experiment JSON config path.")
    parser.add_argument("--device", default=None, help="Optional device override, for example cpu, cuda:0, or cuda:1.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional dataloader worker override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loaded = load_experiment_config(args.config_json)
    config = loaded.config
    if args.device is not None:
        config = replace(config, model=replace(config.model, device=args.device))
    if args.batch_size is not None or args.num_workers is not None:
        config = replace(
            config,
            data=replace(
                config.data,
                batch_size=args.batch_size if args.batch_size is not None else config.data.batch_size,
                num_workers=args.num_workers if args.num_workers is not None else config.data.num_workers,
            ),
        )
    if args.max_steps is not None:
        config = replace(config, training=replace(config.training, max_steps=args.max_steps))
    return run_training(config)


if __name__ == "__main__":
    raise SystemExit(main())

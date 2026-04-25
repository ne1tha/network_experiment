from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from route2_swinjscc_gan.configs.manifest_loader import load_experiment_config
from route2_swinjscc_gan.scripts.eval import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Route 2 from a JSON experiment config.")
    parser.add_argument("--config-json", type=Path, required=True, help="Experiment JSON config path.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override.")
    parser.add_argument("--output-dir", default=None, help="Optional evaluation output directory override.")
    parser.add_argument("--device", default=None, help="Optional device override, for example cpu, cuda:0, or cuda:1.")
    parser.add_argument("--channel-type", default=None, choices=["awgn", "rayleigh"], help="Optional channel type override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional DataLoader worker override.")
    parser.add_argument("--save-images", action="store_true", help="Force saving visual pairs.")
    parser.add_argument("--snr", type=int, default=None, help="Optional SNR override.")
    parser.add_argument("--rate", type=int, default=None, help="Optional rate override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loaded = load_experiment_config(args.config_json, require_checkpoint=args.checkpoint is None)
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint is not None else loaded.checkpoint
    if checkpoint is None:
        raise ValueError("A checkpoint must be provided either in the config JSON or via --checkpoint.")

    config = loaded.config
    device = args.device if args.device is not None else config.model.device
    output_dir = args.output_dir if args.output_dir is not None else config.training.output_dir
    return run_evaluation(
        checkpoint_path=checkpoint,
        test_roots=config.data.test_roots,
        output_dir=output_dir,
        device=device,
        channel_type=args.channel_type if args.channel_type is not None else config.model.channel_type,
        num_workers=args.num_workers if args.num_workers is not None else config.data.num_workers,
        allow_eval_size_adjustment=config.data.allow_eval_size_adjustment,
        snr=args.snr if args.snr is not None else config.evaluation.snr,
        rate=args.rate if args.rate is not None else config.evaluation.rate,
        save_images=args.save_images or config.evaluation.save_images,
    )


if __name__ == "__main__":
    raise SystemExit(main())

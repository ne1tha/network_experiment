from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.reference_loader import load_reference_experiment
from datasets.integrity import summarize_image_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a route 1 JSON config and summarize datasets.")
    parser.add_argument("--config-json", type=Path, required=True, help="Experiment JSON config path.")
    parser.add_argument("--sample-limit", type=int, default=8, help="How many sample images to record per split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment, manifest = load_reference_experiment(args.config_json)
    train_summary = summarize_image_dirs(
        manifest.train_dirs,
        max_samples=args.sample_limit,
    )
    eval_summary = summarize_image_dirs(
        manifest.test_dirs,
        require_multiple_of=128 if not experiment.allow_eval_size_adjustment else None,
        max_samples=args.sample_limit,
    )

    payload = {
        "config_json": str(args.config_json.resolve()),
        "dataset_manifest": str(manifest.path.resolve()),
        "run_name": experiment.run_name,
        "model": experiment.model,
        "channel_type": experiment.channel_type,
        "channels_csv": experiment.channels_csv,
        "snrs_csv": experiment.snrs_csv,
        "device": experiment.device,
        "allow_eval_size_adjustment": experiment.allow_eval_size_adjustment,
        "train_summary": {
            "root_paths": train_summary.root_paths,
            "num_images": train_summary.num_images,
            "min_width": train_summary.min_width,
            "max_width": train_summary.max_width,
            "min_height": train_summary.min_height,
            "max_height": train_summary.max_height,
            "sample_images": [asdict(sample) for sample in train_summary.sample_images],
        },
        "eval_summary": {
            "root_paths": eval_summary.root_paths,
            "num_images": eval_summary.num_images,
            "min_width": eval_summary.min_width,
            "max_width": eval_summary.max_width,
            "min_height": eval_summary.min_height,
            "max_height": eval_summary.max_height,
            "sample_images": [asdict(sample) for sample in eval_summary.sample_images],
            "bad_eval_images": eval_summary.bad_eval_images,
        },
    }

    if not experiment.allow_eval_size_adjustment and eval_summary.bad_eval_images:
        raise ValueError(
            "Some evaluation images are not divisible by 128. "
            "Either preprocess them or set allow_eval_size_adjustment explicitly."
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

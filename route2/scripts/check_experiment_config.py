from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from route2_swinjscc_gan.configs.manifest_loader import load_experiment_config
from route2_swinjscc_gan.datasets import DatasetSummary, summarize_image_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Route 2 JSON experiment config and summarize datasets.")
    parser.add_argument("--config-json", type=Path, required=True, help="Experiment JSON config path.")
    parser.add_argument("--sample-limit", type=int, default=8, help="How many sample images to record per split.")
    return parser.parse_args()


def _summary_dict(summary: DatasetSummary) -> dict[str, object]:
    return {
        "root_paths": summary.root_paths,
        "num_images": summary.num_images,
        "min_width": summary.min_width,
        "max_width": summary.max_width,
        "min_height": summary.min_height,
        "max_height": summary.max_height,
        "sample_images": [asdict(sample) for sample in summary.sample_images],
        "bad_eval_images": summary.bad_eval_images,
        "too_small_images": summary.too_small_images,
    }


def _raise_if_bad_images(summary: DatasetSummary, *, label: str, min_size: int) -> None:
    if summary.too_small_images:
        raise ValueError(
            f"{label} contains images smaller than required size {min_size}: "
            + ", ".join(summary.too_small_images[:8])
        )


def main() -> int:
    args = parse_args()
    loaded = load_experiment_config(args.config_json)
    config = loaded.config

    train_summary = summarize_image_dirs(
        loaded.dataset_manifest.train_roots,
        min_size=config.data.crop_size,
        max_samples=args.sample_limit,
    )
    _raise_if_bad_images(train_summary, label="Training dataset", min_size=config.data.crop_size)

    if config.data.val_roots:
        val_summary = summarize_image_dirs(
            loaded.dataset_manifest.val_roots,
            min_size=config.data.crop_size,
            max_samples=args.sample_limit,
        )
        _raise_if_bad_images(val_summary, label="Validation dataset", min_size=config.data.crop_size)
    else:
        val_summary = None

    eval_summary = summarize_image_dirs(
        loaded.dataset_manifest.test_roots,
        require_multiple_of=None if config.data.allow_eval_size_adjustment else config.data.eval_divisible_by,
        min_size=config.data.eval_divisible_by,
        max_samples=args.sample_limit,
    )
    _raise_if_bad_images(eval_summary, label="Evaluation dataset", min_size=config.data.eval_divisible_by)
    if not config.data.allow_eval_size_adjustment and eval_summary.bad_eval_images:
        raise ValueError(
            "Some evaluation images are not divisible by "
            f"{config.data.eval_divisible_by}: " + ", ".join(eval_summary.bad_eval_images[:8])
        )

    effective_train_images = train_summary.num_images
    if not config.data.val_roots:
        effective_train_images = int(train_summary.num_images * (1.0 - config.data.val_split_ratio))
    if effective_train_images < config.data.batch_size:
        raise ValueError(
            f"Effective training images {effective_train_images} are smaller than batch_size={config.data.batch_size}."
        )

    payload: dict[str, object] = {
        "config_json": str(loaded.path),
        "dataset_manifest": str(loaded.dataset_manifest.path),
        "checkpoint": str(loaded.checkpoint) if loaded.checkpoint is not None else None,
        "experiment_name": loaded.name,
        "output_dir": str(config.training.output_path.resolve()),
        "model_variant": config.model.model_variant,
        "model_size": config.model.model_size,
        "channel_type": config.model.channel_type,
        "multiple_snr": list(config.model.multiple_snr),
        "channel_numbers": list(config.model.channel_numbers),
        "device": config.model.device,
        "allow_eval_size_adjustment": config.data.allow_eval_size_adjustment,
        "train_summary": _summary_dict(train_summary),
        "val_summary": None if val_summary is None else _summary_dict(val_summary),
        "eval_summary": _summary_dict(eval_summary),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

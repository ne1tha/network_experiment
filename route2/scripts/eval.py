from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import torch

from route2_swinjscc_gan.common.device import prepare_runtime_device, resolve_runtime_device
from route2_swinjscc_gan.common.io import ensure_dir, save_json
from route2_swinjscc_gan.configs.defaults import (
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    Route2ExperimentConfig,
    build_default_experiment_config,
)
from route2_swinjscc_gan.datasets import build_test_loader
from route2_swinjscc_gan.evaluators import ImageQualityMetricSuite, SwinJSCCGANEvaluator
from route2_swinjscc_gan.models.swinjscc_gan.generator import SwinJSCCGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Route 2 SwinJSCC-GAN reproduction.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint created by scripts/train.py")
    parser.add_argument("--test-roots", nargs="+", required=True, help="Evaluation image directories.")
    parser.add_argument("--output-dir", default="route2_swinjscc_gan/artifacts/eval_run")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--channel-type", default=None, choices=["awgn", "rayleigh"])
    parser.add_argument("--num-workers", type=int, default=None, help="Optional DataLoader worker override.")
    parser.add_argument("--allow-eval-size-adjustment", action="store_true")
    parser.add_argument("--snr", type=int, default=None)
    parser.add_argument("--rate", type=int, default=None)
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def build_eval_config(
    *,
    checkpoint_payload: dict[str, object],
    device: str,
    test_roots: tuple[str, ...],
    num_workers: int | None,
    allow_eval_size_adjustment: bool,
    channel_type: str | None,
) -> Route2ExperimentConfig:
    default = build_default_experiment_config()
    checkpoint_data_config = checkpoint_payload.get("data_config")
    if checkpoint_data_config is None:
        base_data_config = default.data
    else:
        base_data_config = DataConfig(**checkpoint_data_config)
    checkpoint_model_config = checkpoint_payload.get("model_config")
    if checkpoint_model_config is None:
        model_config = replace(
            default.model,
            device=device,
            channel_type=default.model.channel_type if channel_type is None else channel_type,
        )
    else:
        model_config = ModelConfig(**checkpoint_model_config)
        model_config = replace(
            model_config,
            device=device,
            channel_type=model_config.channel_type if channel_type is None else channel_type,
        )
    checkpoint_evaluation_config = checkpoint_payload.get("evaluation_config")
    if checkpoint_evaluation_config is None:
        evaluation_config = default.evaluation
    else:
        evaluation_config = EvaluationConfig(**checkpoint_evaluation_config)
    if num_workers is None:
        eval_num_workers = 0 if device == "cpu" else default.data.num_workers
    else:
        eval_num_workers = num_workers
    data_config = replace(
        base_data_config,
        test_roots=test_roots,
        num_workers=eval_num_workers,
        allow_eval_size_adjustment=(
            allow_eval_size_adjustment or base_data_config.allow_eval_size_adjustment
        ),
    )
    return replace(default, data=data_config, model=model_config, evaluation=evaluation_config)


def run_evaluation(
    *,
    checkpoint_path: str | Path,
    test_roots: tuple[str, ...],
    output_dir: str | Path,
    device: str,
    channel_type: str | None = None,
    num_workers: int | None = None,
    allow_eval_size_adjustment: bool = False,
    snr: int | None = None,
    rate: int | None = None,
    save_images: bool = False,
) -> int:
    resolved_device = resolve_runtime_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    experiment = build_eval_config(
        checkpoint_payload=checkpoint,
        device=resolved_device,
        test_roots=test_roots,
        num_workers=num_workers,
        allow_eval_size_adjustment=allow_eval_size_adjustment,
        channel_type=channel_type,
    )

    runtime_device = prepare_runtime_device(experiment.model.device)
    loader = build_test_loader(experiment.data)

    generator = SwinJSCCGenerator(experiment.model.build_generator_config()).to(runtime_device)
    generator.load_state_dict(checkpoint["generator"], strict=True)
    snr = snr if snr is not None else int(experiment.evaluation.snr)
    rate = rate if rate is not None else int(experiment.evaluation.rate)

    metric_suite = ImageQualityMetricSuite(lpips_network=experiment.evaluation.lpips_network).to(runtime_device)
    evaluator = SwinJSCCGANEvaluator(generator, metric_suite)
    output_dir = ensure_dir(output_dir)
    summary = evaluator.evaluate(
        loader,
        device=runtime_device,
        snr=snr,
        rate=rate,
        output_dir=output_dir,
        max_saved_images=experiment.evaluation.max_saved_images if save_images else 0,
    )

    save_json(
        output_dir / "summary.json",
        {
            "num_samples": summary.num_samples,
            "psnr": summary.psnr,
            "ms_ssim": summary.ms_ssim,
            "lpips": summary.lpips,
            "snr": snr,
            "rate": rate,
            "channel_type": experiment.model.channel_type,
        },
    )
    return 0


def main() -> int:
    args = parse_args()
    return run_evaluation(
        checkpoint_path=args.checkpoint,
        test_roots=tuple(args.test_roots),
        output_dir=args.output_dir,
        device=args.device,
        channel_type=args.channel_type,
        num_workers=args.num_workers,
        allow_eval_size_adjustment=args.allow_eval_size_adjustment,
        snr=args.snr,
        rate=args.rate,
        save_images=args.save_images,
    )


if __name__ == "__main__":
    raise SystemExit(main())

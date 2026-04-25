from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.route1_reference import build_div2k_reference_config
from datasets.hr_datasets import build_hr_eval_loader
from evaluators.reference_evaluator import evaluate_reference_model
from models.swinjscc.upstream_reference import build_upstream_model
from support.runtime import build_logger, ensure_run_dirs, load_checkpoint_strict, save_config_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the route 1 SwinJSCC reference pipeline.")
    parser.add_argument("--train-dir", action="append", required=True, help="Training image directory metadata. Repeat as needed.")
    parser.add_argument("--test-dir", action="append", required=True, help="Evaluation image directory. Repeat as needed.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--testset", default="kodak", choices=["kodak", "CLIC21", "ffhq"])
    parser.add_argument("--distortion-metric", default="MSE", choices=["MSE", "MS-SSIM"])
    parser.add_argument(
        "--model",
        default="SwinJSCC_w/_SAandRA",
        choices=[
            "SwinJSCC_w/o_SAandRA",
            "SwinJSCC_w/_SA",
            "SwinJSCC_w/_RA",
            "SwinJSCC_w/_SAandRA",
        ],
    )
    parser.add_argument("--channel-type", default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--channels", default="32,64,96,128,192")
    parser.add_argument("--snrs", default="1,4,7,10,13")
    parser.add_argument("--model-size", default="base", choices=["small", "base", "large"])
    parser.add_argument("--run-name", default="route1_reference_eval")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--allow-eval-size-adjustment", action="store_true")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--no-save-logs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_div2k_reference_config(
        workspace_root=REPO_ROOT,
        train_dirs=[Path(path) for path in args.train_dir],
        test_dirs=[Path(path) for path in args.test_dir],
        testset=args.testset,
        model=args.model,
        channel_type=args.channel_type,
        channels_csv=args.channels,
        snrs_csv=args.snrs,
        model_size=args.model_size,
        run_name=args.run_name,
        training=False,
        distortion_metric=args.distortion_metric,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        save_logs=not args.no_save_logs,
        max_eval_samples=args.max_eval_samples,
        allow_eval_size_adjustment=args.allow_eval_size_adjustment,
    )

    model, bundle, runtime_config = build_upstream_model(config)
    device = runtime_config.device
    model = model.to(device)

    ensure_run_dirs(Path(config.run_paths.workdir), Path(config.run_paths.models_dir))
    logger = build_logger(
        f"route1.{config.run_name}.eval",
        log_path=None if args.no_save_logs else Path(config.run_paths.log_path),
    )
    save_config_snapshot(config, Path(config.run_paths.workdir) / "config_snapshot.json")

    metadata = load_checkpoint_strict(
        model=model,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )
    logger.info("Loaded evaluation checkpoint from %s", args.checkpoint_path)
    logger.info("Checkpoint metadata: %s", metadata)

    eval_loader = build_hr_eval_loader(config)
    output_path = args.output_path or (Path(config.run_paths.workdir) / "eval_results.json")
    results = evaluate_reference_model(
        model=model,
        eval_loader=eval_loader,
        experiment=config,
        bundle=bundle,
        device=device,
        logger=logger,
        output_path=output_path,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

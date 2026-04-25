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
from trainers.reference_trainer import ReferenceTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the route 1 SwinJSCC reference pipeline.")
    parser.add_argument("--train-dir", action="append", required=True, help="Training image directory. Repeat as needed.")
    parser.add_argument("--test-dir", action="append", required=True, help="Evaluation image directory. Repeat as needed.")
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
    parser.add_argument("--channels", default="32,64,96,128,192", help="CSV bottleneck dimensions.")
    parser.add_argument("--snrs", default="1,4,7,10,13", help="CSV SNR values.")
    parser.add_argument("--model-size", default="base", choices=["small", "base", "large"])
    parser.add_argument("--run-name", default="route1_reference_train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-model-freq", type=int, default=1)
    parser.add_argument("--print-step", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Optional model-only checkpoint to initialize from.")
    parser.add_argument("--resume-path", type=Path, default=None, help="Optional full training checkpoint to resume from.")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-save-logs", action="store_true")
    parser.add_argument("--allow-eval-size-adjustment", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
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
        training=True,
        distortion_metric=args.distortion_metric,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        total_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_model_freq=args.save_model_freq,
        print_step=args.print_step,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        resume_path=args.resume_path,
        save_logs=not args.no_save_logs,
        max_train_steps=args.max_train_steps,
        max_eval_samples=args.max_eval_samples,
        allow_eval_size_adjustment=args.allow_eval_size_adjustment,
    )

    trainer = ReferenceTrainer(config)
    summary = trainer.train()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

DEFAULT_ROUTE1_ROOT = PROJECT_PARENT / "codex_route1_neu"
if str(DEFAULT_ROUTE1_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROUTE1_ROOT))

import torch
from PIL import Image

from route2_swinjscc_gan.common.device import prepare_runtime_device, resolve_runtime_device
from route2_swinjscc_gan.common.io import ensure_dir, save_json
from route2_swinjscc_gan.evaluators.metrics import ImageQualityMetricSuite

from configs.route1_reference import build_div2k_reference_config
from datasets.hr_datasets import build_hr_eval_loader
from models.swinjscc.upstream_reference import build_upstream_model
from support.runtime import load_checkpoint_compatible, load_checkpoint_strict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a route1 checkpoint with route2 metric definitions."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Route1 checkpoint path.")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_ROUTE1_ROOT / "configs" / "datasets" / "route1_div2k_hr.local.json",
        help="Route1 dataset manifest JSON.",
    )
    parser.add_argument(
        "--route1-root",
        type=Path,
        default=DEFAULT_ROUTE1_ROOT,
        help="Route1 workspace root used to build the reference model.",
    )
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
    parser.add_argument(
        "--channel-type",
        default="rayleigh",
        choices=["awgn", "rayleigh"],
        help="Channel simulation branch used during inference.",
    )
    parser.add_argument(
        "--channels-csv",
        default="32,64,96,128,192",
        help="Channel list used to instantiate the route1 model.",
    )
    parser.add_argument("--snr", type=int, required=True, help="Evaluation SNR.")
    parser.add_argument("--rate", type=int, required=True, help="Evaluation rate.")
    parser.add_argument(
        "--model-size",
        default="base",
        choices=["small", "base", "large"],
        help="Route1 model size.",
    )
    parser.add_argument(
        "--distortion-metric",
        default="MSE",
        choices=["MSE", "MS-SSIM"],
        help="Reference config distortion metric.",
    )
    parser.add_argument(
        "--checkpoint-load-mode",
        default="strict",
        choices=["strict", "compatible"],
        help="How to load the route1 checkpoint.",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:N, or auto.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--allow-eval-size-adjustment", action="store_true")
    parser.add_argument(
        "--lpips-network",
        default="vgg",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone used by route2 metrics.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for outputs.")
    parser.add_argument(
        "--max-saved-images",
        type=int,
        default=0,
        help="How many visual pairs to save. Use 0 to disable.",
    )
    parser.add_argument(
        "--run-name",
        default="route1_eval_with_route2_metrics",
        help="Synthetic run name used to build the route1 config.",
    )
    return parser.parse_args()


def load_dataset_manifest(path: Path) -> tuple[list[Path], list[Path], bool]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    train_dirs = [Path(item) for item in payload["train_dirs"]]
    test_dirs = [Path(item) for item in payload["test_dirs"]]
    allow_eval_size_adjustment = bool(payload.get("allow_eval_size_adjustment", False))
    return train_dirs, test_dirs, allow_eval_size_adjustment


def maybe_save_visual_pair(
    output_dir: Path,
    image_name: str,
    source: torch.Tensor,
    reconstruction: torch.Tensor,
) -> None:
    source_image = tensor_to_image(source)
    reconstruction_image = tensor_to_image(reconstruction)
    source_image.save(output_dir / f"{image_name}_source.png")
    reconstruction_image.save(output_dir / f"{image_name}_reconstruction.png")


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def main() -> int:
    args = parse_args()
    train_dirs, test_dirs, manifest_allow_adjustment = load_dataset_manifest(args.dataset_config)
    resolved_device = resolve_runtime_device(args.device)

    experiment = build_div2k_reference_config(
        workspace_root=args.route1_root,
        train_dirs=train_dirs,
        test_dirs=test_dirs,
        testset="kodak",
        model=args.model,
        channel_type=args.channel_type,
        channels_csv=args.channels_csv,
        snrs_csv=str(args.snr),
        model_size=args.model_size,
        run_name=args.run_name,
        training=False,
        distortion_metric=args.distortion_metric,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        device=resolved_device,
        checkpoint_path=args.checkpoint,
        save_logs=False,
        allow_eval_size_adjustment=(
            args.allow_eval_size_adjustment or manifest_allow_adjustment
        ),
    )

    model, _bundle, runtime_config = build_upstream_model(experiment)
    device = prepare_runtime_device(runtime_config.device.type if runtime_config.device.type == "cpu" else str(runtime_config.device))
    model = model.to(device)

    if args.checkpoint_load_mode == "strict":
        metadata = load_checkpoint_strict(
            model=model,
            checkpoint_path=args.checkpoint,
            device=device,
        )
    else:
        metadata = load_checkpoint_compatible(
            model=model,
            checkpoint_path=args.checkpoint,
            device=device,
        )

    eval_loader = build_hr_eval_loader(experiment)
    metric_suite = ImageQualityMetricSuite(lpips_network=args.lpips_network).to(device).eval()
    output_dir = ensure_dir(args.output_dir)

    model.eval()
    total_psnr = 0.0
    total_ms_ssim = 0.0
    total_lpips = 0.0
    total_samples = 0

    with torch.no_grad():
        for sample_index, (inputs, names) in enumerate(eval_loader):
            inputs = inputs.to(device, non_blocking=True)
            reconstruction, _cbr, _used_snr, _mse, _ = model(
                inputs,
                given_SNR=args.snr,
                given_rate=args.rate,
            )
            metrics = metric_suite(reconstruction, inputs)

            batch_size = inputs.shape[0]
            total_psnr += metrics.psnr * batch_size
            total_ms_ssim += metrics.ms_ssim * batch_size
            total_lpips += metrics.lpips * batch_size
            total_samples += batch_size

            if args.max_saved_images > 0 and sample_index < args.max_saved_images:
                maybe_save_visual_pair(output_dir, names[0], inputs[0], reconstruction[0])

    summary = {
        "num_samples": total_samples,
        "psnr": total_psnr / total_samples,
        "ms_ssim": total_ms_ssim / total_samples,
        "lpips": total_lpips / total_samples,
        "snr": args.snr,
        "rate": args.rate,
        "channel_type": args.channel_type,
        "model": args.model,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_load_mode": args.checkpoint_load_mode,
        "checkpoint_metadata": metadata,
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

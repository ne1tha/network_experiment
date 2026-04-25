from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

DEFAULT_ROUTE3_ROOT = PROJECT_PARENT / "route3"
LEGACY_ROUTE3_ROOT = PROJECT_PARENT / "codex_route3_neu"
if not DEFAULT_ROUTE3_ROOT.exists():
    DEFAULT_ROUTE3_ROOT = LEGACY_ROUTE3_ROOT
if str(DEFAULT_ROUTE3_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROUTE3_ROOT))

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from route2_swinjscc_gan.common.device import prepare_runtime_device, resolve_runtime_device
from route2_swinjscc_gan.common.io import ensure_dir, save_json
from route2_swinjscc_gan.evaluators.metrics import ImageQualityMetricSuite

from scripts.route3_preflight import build_runtime_model
from scripts.route3_train import load_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a route3 single-user checkpoint with route2 metric definitions."
    )
    parser.add_argument("--config", type=Path, required=True, help="Route3 training config JSON.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Route3 checkpoint path.")
    parser.add_argument(
        "--kodak-dir",
        type=Path,
        default=PROJECT_PARENT / "route3" / "data" / "Kodak",
        help="Kodak image directory.",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:N, or auto.")
    parser.add_argument(
        "--channel-type",
        default=None,
        choices=["awgn", "rayleigh"],
        help="Optional shared override for both route3 branches.",
    )
    parser.add_argument(
        "--semantic-channel-type",
        default=None,
        choices=["awgn", "rayleigh"],
        help="Optional override for the semantic branch channel.",
    )
    parser.add_argument(
        "--detail-channel-type",
        default=None,
        choices=["awgn", "rayleigh"],
        help="Optional override for the detail branch channel.",
    )
    parser.add_argument("--snr-db", type=float, default=None, help="Optional evaluation SNR override.")
    parser.add_argument("--sem-rate-ratio", type=float, default=None, help="Optional semantic rate ratio override.")
    parser.add_argument("--det-rate-ratio", type=float, default=None, help="Optional detail rate ratio override.")
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
        help="How many source/reconstruction pairs to save.",
    )
    return parser.parse_args()


def _load_model_state_from_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"Unsupported route3 checkpoint format: {path}")
    extra_state = checkpoint.get("extra_state", {})
    if not isinstance(extra_state, dict):
        extra_state = {}
    return checkpoint["model"], extra_state


def _load_single_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return TF.to_tensor(image).unsqueeze(0)


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0)
    return TF.to_pil_image(image)


def _save_visual_pair(output_dir: Path, image_name: str, source: torch.Tensor, reconstruction: torch.Tensor) -> None:
    _tensor_to_image(source).save(output_dir / f"{image_name}_source.png")
    _tensor_to_image(reconstruction).save(output_dir / f"{image_name}_reconstruction.png")


def _effective_cbr(
    *,
    input_image: torch.Tensor,
    semantic_active_channels: torch.Tensor,
    detail_active_channels: torch.Tensor,
    semantic_shape: torch.Size,
    detail_shape: torch.Size,
) -> torch.Tensor:
    _, _, height, width = input_image.shape
    semantic_spatial = int(semantic_shape[-2]) * int(semantic_shape[-1])
    detail_spatial = int(detail_shape[-2]) * int(detail_shape[-1])
    input_real_values = 3 * int(height) * int(width)

    semantic_real_values = semantic_active_channels.to(dtype=input_image.dtype) * float(semantic_spatial)
    detail_real_values = detail_active_channels.to(dtype=input_image.dtype) * float(detail_spatial)
    return (semantic_real_values + detail_real_values) / float(2 * input_real_values)


def main() -> int:
    args = parse_args()
    config = load_training_config(args.config)
    if config.base.runtime.mode != "single_user":
        raise ValueError(
            f"Route3 unified evaluation expects a single_user config, got {config.base.runtime.mode!r}."
        )

    resolved_device = resolve_runtime_device(args.device)
    shared_channel_override = args.channel_type
    runtime = replace(
        config.base.runtime,
        device=resolved_device,
        semantic_channel_type=(
            args.semantic_channel_type
            or shared_channel_override
            or config.base.runtime.semantic_channel_type
        ),
        detail_channel_type=(
            args.detail_channel_type
            or shared_channel_override
            or config.base.runtime.detail_channel_type
        ),
        snr_db=float(args.snr_db) if args.snr_db is not None else config.base.runtime.snr_db,
        sem_rate_ratio=(
            float(args.sem_rate_ratio)
            if args.sem_rate_ratio is not None
            else config.base.runtime.sem_rate_ratio
        ),
        det_rate_ratio=(
            float(args.det_rate_ratio)
            if args.det_rate_ratio is not None
            else config.base.runtime.det_rate_ratio
        ),
    )
    base_config = replace(config.base, runtime=runtime)

    device = prepare_runtime_device(resolved_device)
    model = build_runtime_model(base_config).to(device)
    model_state, checkpoint_extra_state = _load_model_state_from_checkpoint(args.checkpoint)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    metric_suite = ImageQualityMetricSuite(lpips_network=args.lpips_network).to(device).eval()
    output_dir = ensure_dir(args.output_dir)

    image_paths = sorted(Path(args.kodak_dir).glob("kodim*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No Kodak images found under: {args.kodak_dir}")

    total_psnr = 0.0
    total_ms_ssim = 0.0
    total_lpips = 0.0
    total_effective_cbr = 0.0
    total_semantic_active_channels = 0.0
    total_detail_active_channels = 0.0
    total_samples = 0
    per_image_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for sample_index, image_path in enumerate(image_paths):
            source = _load_single_image(image_path).to(device)
            output = model(
                source,
                snr_db=runtime.snr_db,
                sem_rate_ratio=runtime.sem_rate_ratio,
                det_rate_ratio=runtime.det_rate_ratio,
                decode_stochastic=False,
                run_enhancement=False,
                compute_enhancement_discriminator_loss=False,
            )
            if output.reconstruction is None:
                raise RuntimeError("Route3 evaluation requires reconstruction output, but got None.")

            reconstruction = output.reconstruction.x_hat.detach()
            metrics = metric_suite(reconstruction, source)
            batch_cbr = _effective_cbr(
                input_image=source,
                semantic_active_channels=output.semantic.mask.active_channels,
                detail_active_channels=output.detail.mask.active_channels,
                semantic_shape=output.semantic.tx.shape,
                detail_shape=output.detail.tx.shape,
            )

            semantic_active_channels = float(
                output.semantic.mask.active_channels.to(dtype=torch.float32).mean().item()
            )
            detail_active_channels = float(
                output.detail.mask.active_channels.to(dtype=torch.float32).mean().item()
            )
            effective_cbr = float(batch_cbr.to(dtype=torch.float32).mean().item())

            total_psnr += metrics.psnr
            total_ms_ssim += metrics.ms_ssim
            total_lpips += metrics.lpips
            total_effective_cbr += effective_cbr
            total_semantic_active_channels += semantic_active_channels
            total_detail_active_channels += detail_active_channels
            total_samples += 1

            per_image_rows.append(
                {
                    "image": image_path.name,
                    "psnr": float(metrics.psnr),
                    "ms_ssim": float(metrics.ms_ssim),
                    "lpips": float(metrics.lpips),
                    "effective_cbr": effective_cbr,
                    "semantic_active_channels": semantic_active_channels,
                    "detail_active_channels": detail_active_channels,
                }
            )

            if args.max_saved_images > 0 and sample_index < args.max_saved_images:
                _save_visual_pair(output_dir, image_path.name, source[0], reconstruction[0])

    if total_samples == 0:
        raise RuntimeError("Route3 evaluation produced zero samples.")

    summary = {
        "num_samples": total_samples,
        "psnr": total_psnr / total_samples,
        "ms_ssim": total_ms_ssim / total_samples,
        "lpips": total_lpips / total_samples,
        "effective_cbr": total_effective_cbr / total_samples,
        "semantic_active_channels_mean": total_semantic_active_channels / total_samples,
        "detail_active_channels_mean": total_detail_active_channels / total_samples,
        "snr_db": float(runtime.snr_db),
        "semantic_channel_type": runtime.semantic_channel_type,
        "detail_channel_type": runtime.detail_channel_type,
        "sem_rate_ratio": float(runtime.sem_rate_ratio),
        "det_rate_ratio": float(runtime.det_rate_ratio),
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": (
            int(checkpoint_extra_state["epoch"])
            if "epoch" in checkpoint_extra_state
            else None
        ),
    }
    save_json(output_dir / "summary.json", summary)
    save_json(
        output_dir / "per_image.json",
        {
            "summary": summary,
            "per_image": per_image_rows,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

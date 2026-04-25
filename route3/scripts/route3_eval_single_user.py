from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluators import compute_reconstruction_metrics
from evaluators import budget_metrics_to_dict, summarize_single_user_budget
from optim import move_to_device, select_torch_device
from scripts.route3_preflight import build_runtime_model
from scripts.route3_train import load_training_config

_EVAL_ALLOWED_MISSING_PREFIXES = (
    "enhancer.discriminator.",
    "enhancer.perceptual_loss.feature_extractor.",
    "enhancer.refiner.",
)

_EVAL_ALLOWED_UNEXPECTED_PREFIXES = (
    "enhancer.discriminator.",
    "enhancer.refiner.",
)

try:
    from PIL import Image, ImageDraw
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required for route3 evaluation.") from exc

try:
    from torchvision.transforms import functional as TF
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torchvision is required for route3 evaluation.") from exc


def _load_model_state_from_checkpoint(path: Path) -> tuple[dict[str, Any], int | None]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    extra_state = checkpoint.get("extra_state", {})
    checkpoint_epoch = int(extra_state["epoch"]) if "epoch" in extra_state else None
    return checkpoint["model"], checkpoint_epoch


def _load_eval_state_dict(model: torch.nn.Module, model_state: dict[str, Any]) -> None:
    incompatible = model.load_state_dict(model_state, strict=False)
    disallowed_missing = [
        key for key in incompatible.missing_keys if not key.startswith(_EVAL_ALLOWED_MISSING_PREFIXES)
    ]
    if disallowed_missing:
        raise RuntimeError(
            "Route3 eval checkpoint is missing required weights: "
            + ", ".join(disallowed_missing[:16])
        )
    disallowed_unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith(_EVAL_ALLOWED_UNEXPECTED_PREFIXES)
    ]
    if disallowed_unexpected:
        raise RuntimeError(
            "Route3 eval checkpoint contains unexpected non-discriminator weights: "
            + ", ".join(disallowed_unexpected[:16])
        )


def _load_single_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = TF.to_tensor(image).unsqueeze(0)
    return tensor


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0.0, 1.0).squeeze(0)
    return TF.to_pil_image(image)


def _build_absdiff_image(x_gt: torch.Tensor, x_hat: torch.Tensor, amplify: float = 4.0) -> Image.Image:
    diff = (x_gt.detach().cpu() - x_hat.detach().cpu()).abs().mul(amplify).clamp(0.0, 1.0)
    return _tensor_to_pil(diff)


def _build_comparison_image(
    source: Image.Image,
    reconstruction: Image.Image,
    absdiff_x4: Image.Image,
) -> Image.Image:
    gap = 24
    label_height = 34
    width = source.width + reconstruction.width + absdiff_x4.width + gap * 4
    height = max(source.height, reconstruction.height, absdiff_x4.height) + label_height + gap * 2
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    panels = [
        ("source", source),
        ("recon", reconstruction),
        ("absdiff x4", absdiff_x4),
    ]
    cursor_x = gap
    for label, image in panels:
        draw.text((cursor_x, 10), label, fill=(0, 0, 0))
        canvas.paste(image, (cursor_x, label_height))
        cursor_x += image.width + gap
    return canvas


def _metrics_to_payload(metrics) -> dict[str, float]:
    return {
        "psnr": float(metrics.psnr),
        "mse": float(metrics.mse),
        "mae": float(metrics.mean_abs_error),
    }


@torch.no_grad()
def _run_single_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    *,
    snr_db: float,
    sem_rate_ratio: float,
    det_rate_ratio: float,
    decode_stochastic: bool,
    run_enhancement: bool,
    operating_mode: str = "open_quality",
    target_effective_cbr: float | None = None,
    target_effective_cbr_tolerance: float = 0.05,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source = move_to_device(_load_single_image(image_path), device)
    output = model(
        source,
        snr_db=snr_db,
        sem_rate_ratio=sem_rate_ratio,
        det_rate_ratio=det_rate_ratio,
        decode_stochastic=decode_stochastic,
        run_enhancement=run_enhancement,
        compute_enhancement_discriminator_loss=False,
    )
    if output.reconstruction is None:
        raise RuntimeError("Single-user evaluation requires reconstruction output.")
    base_reconstruction = getattr(output, "base_reconstruction", None)
    final_reconstruction = getattr(output, "final_reconstruction", None)
    x_hat = output.reconstruction.x_hat.detach()
    metrics = compute_reconstruction_metrics(source, x_hat)
    payload: dict[str, Any] = {
        **_metrics_to_payload(metrics),
        "reconstruction_output_kind": getattr(output.reconstruction, "output_kind", None),
        "base_metrics": None,
        "final_metrics": None,
        "refinement_gain_psnr": None,
        "budget": None,
    }
    if hasattr(output, "semantic") and hasattr(output, "detail"):
        payload["budget"] = budget_metrics_to_dict(
            summarize_single_user_budget(
                input_image=source,
                output=output,
                operating_mode=operating_mode,
                target_effective_cbr=target_effective_cbr,
                target_effective_cbr_tolerance=target_effective_cbr_tolerance,
            )
        )
    if base_reconstruction is not None:
        payload["base_metrics"] = _metrics_to_payload(
            compute_reconstruction_metrics(source, base_reconstruction.x_hat.detach())
        )
        payload["base_output_kind"] = getattr(base_reconstruction, "output_kind", None)
    if final_reconstruction is not None:
        payload["final_metrics"] = _metrics_to_payload(
            compute_reconstruction_metrics(source, final_reconstruction.x_hat.detach())
        )
        payload["final_output_kind"] = getattr(final_reconstruction, "output_kind", None)
    if payload["base_metrics"] is not None and payload["final_metrics"] is not None:
        payload["refinement_gain_psnr"] = (
            payload["final_metrics"]["psnr"] - payload["base_metrics"]["psnr"]
        )
    return x_hat, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a route3 single-user checkpoint on Kodak.")
    parser.add_argument("--config", required=True, help="Training config used to build the model.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path, usually best.pt.")
    parser.add_argument("--device", default=None, help="Override runtime device, e.g. cuda:0.")
    parser.add_argument("--kodak-dir", default=str(ROOT / "data" / "Kodak"), help="Kodak image directory.")
    parser.add_argument("--output-dir", default=None, help="Directory for evaluation outputs.")
    parser.add_argument("--snr-db", type=float, default=None, help="Override SNR for evaluation.")
    parser.add_argument("--sem-rate-ratio", type=float, default=None, help="Override semantic rate ratio.")
    parser.add_argument("--det-rate-ratio", type=float, default=None, help="Override detail rate ratio.")
    parser.add_argument("--focus-image", default="kodim01.png", help="Kodak image name for visualization.")
    parser.add_argument("--seed", type=int, default=None, help="Optional torch seed for deterministic channel noise.")
    parser.add_argument(
        "--decode-stochastic",
        action="store_true",
        help="Enable stochastic decoder sampling during evaluation.",
    )
    args = parser.parse_args()

    config = load_training_config(args.config)
    if config.base.runtime.mode != "single_user":
        raise ValueError(f"Expected single_user config, got {config.base.runtime.mode}")

    if args.device is not None:
        runtime_config = config.base.runtime
        config = type(config)(
            base=type(config.base)(
                dataset=config.base.dataset,
                weights=config.base.weights,
                runtime=type(runtime_config)(
                    mode=runtime_config.mode,
                    device=args.device,
                    semantic_channel_type=runtime_config.semantic_channel_type,
                    detail_channel_type=runtime_config.detail_channel_type,
                    enable_distillation=runtime_config.enable_distillation,
                    enable_perceptual=runtime_config.enable_perceptual,
                    enable_adversarial=runtime_config.enable_adversarial,
                    sem_rate_ratio=runtime_config.sem_rate_ratio,
                    det_rate_ratio=runtime_config.det_rate_ratio,
                    semantic_bandwidth_budget=runtime_config.semantic_bandwidth_budget,
                    snr_db=runtime_config.snr_db,
                    train_snr_db_choices=runtime_config.train_snr_db_choices,
                    train_sem_rate_ratio_range=runtime_config.train_sem_rate_ratio_range,
                    train_det_rate_ratio_range=runtime_config.train_det_rate_ratio_range,
                ),
                artifacts=config.base.artifacts,
            ),
            trainer=config.trainer,
            config_path=config.config_path,
        )

    device = select_torch_device(config.base.runtime.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(config.base.artifacts.output_dir) / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    model = build_runtime_model(config.base)
    model_state, checkpoint_epoch = _load_model_state_from_checkpoint(checkpoint_path)
    _load_eval_state_dict(model, model_state)
    model.to(device)
    model.eval()

    snr_db = args.snr_db if args.snr_db is not None else config.base.runtime.snr_db
    sem_rate_ratio = args.sem_rate_ratio if args.sem_rate_ratio is not None else config.base.runtime.sem_rate_ratio
    det_rate_ratio = args.det_rate_ratio if args.det_rate_ratio is not None else config.base.runtime.det_rate_ratio
    run_enhancement = bool(config.base.runtime.enable_perceptual or config.base.runtime.enable_adversarial)

    kodak_dir = Path(args.kodak_dir).resolve()
    image_paths = sorted(kodak_dir.glob("kodim*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No Kodak images found under: {kodak_dir}")

    summary_rows: list[dict[str, Any]] = []
    mse_total = 0.0
    psnr_total = 0.0
    mae_total = 0.0
    base_mse_total = 0.0
    base_psnr_total = 0.0
    base_mae_total = 0.0
    base_metric_count = 0
    final_mse_total = 0.0
    final_psnr_total = 0.0
    final_mae_total = 0.0
    final_metric_count = 0

    focus_source_path = (kodak_dir / args.focus_image).resolve()
    if not focus_source_path.exists():
        raise FileNotFoundError(f"Focus image not found: {focus_source_path}")

    focus_x_hat, focus_metrics = _run_single_image(
        model,
        focus_source_path,
        device,
        snr_db=snr_db,
        sem_rate_ratio=sem_rate_ratio,
        det_rate_ratio=det_rate_ratio,
        decode_stochastic=args.decode_stochastic,
        run_enhancement=run_enhancement,
        operating_mode=config.base.runtime.operating_mode,
        target_effective_cbr=config.base.runtime.target_effective_cbr,
        target_effective_cbr_tolerance=config.base.runtime.target_effective_cbr_tolerance,
    )
    focus_source_tensor = _load_single_image(focus_source_path)
    focus_source_image = _tensor_to_pil(focus_source_tensor)
    focus_reconstruction_image = _tensor_to_pil(focus_x_hat)
    focus_absdiff_image = _build_absdiff_image(focus_source_tensor, focus_x_hat)
    focus_compare_image = _build_comparison_image(
        focus_source_image,
        focus_reconstruction_image,
        focus_absdiff_image,
    )

    focus_source_out = output_dir / f"{focus_source_path.stem}_source.png"
    focus_recon_out = output_dir / f"{focus_source_path.stem}_best_recon.png"
    focus_absdiff_out = output_dir / f"{focus_source_path.stem}_best_absdiff_x4.png"
    focus_compare_out = output_dir / f"{focus_source_path.stem}_best_compare.png"
    focus_source_image.save(focus_source_out)
    focus_reconstruction_image.save(focus_recon_out)
    focus_absdiff_image.save(focus_absdiff_out)
    focus_compare_image.save(focus_compare_out)

    for image_path in image_paths:
        x_hat, metrics = _run_single_image(
            model,
            image_path,
            device,
            snr_db=snr_db,
            sem_rate_ratio=sem_rate_ratio,
            det_rate_ratio=det_rate_ratio,
            decode_stochastic=args.decode_stochastic,
            run_enhancement=run_enhancement,
            operating_mode=config.base.runtime.operating_mode,
            target_effective_cbr=config.base.runtime.target_effective_cbr,
            target_effective_cbr_tolerance=config.base.runtime.target_effective_cbr_tolerance,
        )
        del x_hat
        summary_rows.append(
            {
                "image": image_path.name,
                "psnr": metrics["psnr"],
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "reconstruction_output_kind": metrics.get("reconstruction_output_kind"),
                "base_metrics": metrics.get("base_metrics"),
                "final_metrics": metrics.get("final_metrics"),
                "refinement_gain_psnr": metrics.get("refinement_gain_psnr"),
                "budget": metrics.get("budget"),
            }
        )
        mse_total += metrics["mse"]
        psnr_total += metrics["psnr"]
        mae_total += metrics["mae"]
        if metrics.get("base_metrics") is not None:
            base_metric_count += 1
            base_mse_total += metrics["base_metrics"]["mse"]
            base_psnr_total += metrics["base_metrics"]["psnr"]
            base_mae_total += metrics["base_metrics"]["mae"]
        if metrics.get("final_metrics") is not None:
            final_metric_count += 1
            final_mse_total += metrics["final_metrics"]["mse"]
            final_psnr_total += metrics["final_metrics"]["psnr"]
            final_mae_total += metrics["final_metrics"]["mae"]

    num_samples = len(summary_rows)
    kodak_summary = {
        "config_path": config.config_path,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "device": str(device),
        "seed": args.seed,
        "decode_stochastic": args.decode_stochastic,
        "run_enhancement": run_enhancement,
        "num_samples": num_samples,
        "mean_psnr": psnr_total / num_samples,
        "mean_mse": mse_total / num_samples,
        "mean_mae": mae_total / num_samples,
        "mean_base_metrics": (
            {
                "psnr": base_psnr_total / base_metric_count,
                "mse": base_mse_total / base_metric_count,
                "mae": base_mae_total / base_metric_count,
            }
            if base_metric_count > 0
            else None
        ),
        "mean_final_metrics": (
            {
                "psnr": final_psnr_total / final_metric_count,
                "mse": final_mse_total / final_metric_count,
                "mae": final_mae_total / final_metric_count,
            }
            if final_metric_count > 0
            else None
        ),
        "operating_mode": config.base.runtime.operating_mode,
        "target_effective_cbr": config.base.runtime.target_effective_cbr,
        "target_effective_cbr_tolerance": config.base.runtime.target_effective_cbr_tolerance,
        "per_image": summary_rows,
    }
    kodak_summary_path = output_dir / "kodak24_eval.json"
    kodak_summary_path.write_text(json.dumps(kodak_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    focus_payload = {
        "config_path": config.config_path,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "source_image": str(focus_source_path),
        "device": str(device),
        "seed": args.seed,
        "decode_stochastic": args.decode_stochastic,
        "run_enhancement": run_enhancement,
        "snr_db": snr_db,
        "sem_rate_ratio": sem_rate_ratio,
        "det_rate_ratio": det_rate_ratio,
        "metrics": {
            "psnr": focus_metrics["psnr"],
            "mse": focus_metrics["mse"],
            "mean_abs_error": focus_metrics["mae"],
            "reconstruction_output_kind": focus_metrics.get("reconstruction_output_kind"),
            "base_metrics": focus_metrics.get("base_metrics"),
            "final_metrics": focus_metrics.get("final_metrics"),
            "refinement_gain_psnr": focus_metrics.get("refinement_gain_psnr"),
            "budget": focus_metrics.get("budget"),
        },
        "outputs": {
            "source": str(focus_source_out),
            "reconstruction": str(focus_recon_out),
            "absdiff_x4": str(focus_absdiff_out),
            "comparison": str(focus_compare_out),
        },
    }
    focus_metrics_path = output_dir / f"{focus_source_path.stem}_best_metrics.json"
    focus_metrics_path.write_text(json.dumps(focus_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "kodak24_eval": str(kodak_summary_path),
                "focus_metrics": str(focus_metrics_path),
                "mean_psnr": kodak_summary["mean_psnr"],
                "mean_base_metrics": kodak_summary["mean_base_metrics"],
                "mean_final_metrics": kodak_summary["mean_final_metrics"],
                "checkpoint_epoch": checkpoint_epoch,
                "device": str(device),
                "decode_stochastic": args.decode_stochastic,
                "run_enhancement": run_enhancement,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

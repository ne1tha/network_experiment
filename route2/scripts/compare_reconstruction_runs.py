from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from route2_swinjscc_gan.common.io import ensure_dir, save_json
from route2_swinjscc_gan.evaluators.metrics import ImageQualityMetricSuite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create side-by-side visual and metric comparisons for two reconstruction runs."
    )
    parser.add_argument("--baseline-dir", required=True, help="Baseline run test_eval directory.")
    parser.add_argument("--candidate-dir", required=True, help="Candidate run test_eval directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison artifacts.")
    parser.add_argument("--baseline-label", default="baseline", help="Label shown for baseline column.")
    parser.add_argument("--candidate-label", default="candidate", help="Label shown for candidate column.")
    parser.add_argument("--tile-width", type=int, default=256, help="Output tile width in pixels.")
    parser.add_argument(
        "--lpips-network",
        default="vgg",
        choices=("alex", "vgg"),
        help="LPIPS backbone used for per-image metrics.",
    )
    return parser.parse_args()


def collect_sample_paths(directory: Path) -> dict[str, dict[str, Path]]:
    samples: dict[str, dict[str, Path]] = {}
    for image_path in sorted(directory.glob("*.png")):
        if image_path.name.endswith("_source.png"):
            sample_id = image_path.name[: -len("_source.png")]
            entry = samples.setdefault(sample_id, {})
            entry["source"] = image_path
        elif image_path.name.endswith("_reconstruction.png"):
            sample_id = image_path.name[: -len("_reconstruction.png")]
            entry = samples.setdefault(sample_id, {})
            entry["reconstruction"] = image_path
    return samples


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def compute_metrics(
    metric_suite: ImageQualityMetricSuite,
    *,
    source_image: Image.Image,
    reconstructed_image: Image.Image,
) -> dict[str, float]:
    with torch.no_grad():
        metrics = metric_suite(to_tensor(reconstructed_image), to_tensor(source_image))
    return {
        "psnr": float(metrics.psnr),
        "ms_ssim": float(metrics.ms_ssim),
        "lpips": float(metrics.lpips),
    }


def resize_for_grid(image: Image.Image, *, tile_width: int) -> Image.Image:
    scale = tile_width / image.width
    tile_height = max(1, round(image.height * scale))
    return image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)


def delta_summary(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> dict[str, float | str]:
    delta_psnr = candidate_metrics["psnr"] - baseline_metrics["psnr"]
    delta_ms_ssim = candidate_metrics["ms_ssim"] - baseline_metrics["ms_ssim"]
    delta_lpips = candidate_metrics["lpips"] - baseline_metrics["lpips"]
    wins = 0
    wins += 1 if delta_psnr > 0 else 0
    wins += 1 if delta_ms_ssim > 0 else 0
    wins += 1 if delta_lpips < 0 else 0
    return {
        "delta_psnr": delta_psnr,
        "delta_ms_ssim": delta_ms_ssim,
        "delta_lpips": delta_lpips,
        "candidate_wins": wins,
        "result": "candidate_better" if wins >= 2 else "baseline_better_or_mixed",
    }


def build_summary(
    rows: list[dict[str, Any]],
    *,
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    sample_count = len(rows)
    aggregate = {
        "baseline": {"psnr": 0.0, "ms_ssim": 0.0, "lpips": 0.0},
        "candidate": {"psnr": 0.0, "ms_ssim": 0.0, "lpips": 0.0},
        "candidate_better_count": {"psnr": 0, "ms_ssim": 0, "lpips": 0},
    }
    for row in rows:
        for metric in ("psnr", "ms_ssim", "lpips"):
            aggregate["baseline"][metric] += row["baseline_metrics"][metric]
            aggregate["candidate"][metric] += row["candidate_metrics"][metric]
        aggregate["candidate_better_count"]["psnr"] += int(row["delta"]["delta_psnr"] > 0)
        aggregate["candidate_better_count"]["ms_ssim"] += int(row["delta"]["delta_ms_ssim"] > 0)
        aggregate["candidate_better_count"]["lpips"] += int(row["delta"]["delta_lpips"] < 0)

    for bucket_name in ("baseline", "candidate"):
        for metric in ("psnr", "ms_ssim", "lpips"):
            aggregate[bucket_name][metric] /= max(sample_count, 1)

    return {
        "labels": {"baseline": baseline_label, "candidate": candidate_label},
        "num_samples": sample_count,
        "aggregate": aggregate,
        "rows": rows,
    }


def render_grid(
    rows: list[dict[str, Any]],
    *,
    baseline_label: str,
    candidate_label: str,
    tile_width: int,
    output_path: Path,
) -> None:
    font = ImageFont.load_default()
    sample_image = resize_for_grid(rows[0]["source_image"], tile_width=tile_width)
    tile_height = sample_image.height
    margin = 16
    gap = 12
    header_height = 50
    row_title_height = 44
    label_height = 18
    footer_height = 12
    columns = ["source", "baseline", "candidate"]
    column_labels = {
        "source": "source",
        "baseline": baseline_label,
        "candidate": candidate_label,
    }

    grid_width = margin * 2 + tile_width * len(columns) + gap * (len(columns) - 1)
    row_height = row_title_height + label_height + tile_height + gap
    grid_height = header_height + margin + row_height * len(rows) + footer_height
    canvas = Image.new("RGB", (grid_width, grid_height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (margin, 14),
        f"source vs {baseline_label} vs {candidate_label}",
        fill=(20, 20, 20),
        font=font,
    )

    for row_index, row in enumerate(rows):
        top = header_height + margin + row_index * row_height
        title = (
            f"{row['sample_id']} | "
            f"dPSNR {row['delta']['delta_psnr']:+.3f} | "
            f"dMS-SSIM {row['delta']['delta_ms_ssim']:+.4f} | "
            f"dLPIPS {row['delta']['delta_lpips']:+.4f}"
        )
        draw.text((margin, top), title, fill=(30, 30, 30), font=font)

        images = {
            "source": resize_for_grid(row["source_image"], tile_width=tile_width),
            "baseline": resize_for_grid(row["baseline_image"], tile_width=tile_width),
            "candidate": resize_for_grid(row["candidate_image"], tile_width=tile_width),
        }
        metrics_text = {
            "source": "",
            "baseline": (
                f"{baseline_label} | P {row['baseline_metrics']['psnr']:.2f} "
                f"M {row['baseline_metrics']['ms_ssim']:.4f} "
                f"L {row['baseline_metrics']['lpips']:.4f}"
            ),
            "candidate": (
                f"{candidate_label} | P {row['candidate_metrics']['psnr']:.2f} "
                f"M {row['candidate_metrics']['ms_ssim']:.4f} "
                f"L {row['candidate_metrics']['lpips']:.4f}"
            ),
        }

        label_y = top + row_title_height
        image_y = label_y + label_height
        for column_index, column in enumerate(columns):
            left = margin + column_index * (tile_width + gap)
            draw.text((left, label_y), column_labels[column], fill=(60, 60, 60), font=font)
            canvas.paste(images[column], (left, image_y))
            if metrics_text[column]:
                draw.text(
                    (left, image_y + tile_height - 14),
                    metrics_text[column],
                    fill=(255, 255, 255),
                    font=font,
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )

    ensure_dir(output_path.parent)
    canvas.save(output_path)


def main() -> int:
    args = parse_args()

    baseline_dir = Path(args.baseline_dir).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    output_dir = ensure_dir(args.output_dir)

    baseline_samples = collect_sample_paths(baseline_dir)
    candidate_samples = collect_sample_paths(candidate_dir)
    shared_ids = sorted(
        sample_id
        for sample_id in baseline_samples
        if sample_id in candidate_samples
        and "source" in baseline_samples[sample_id]
        and "reconstruction" in baseline_samples[sample_id]
        and "reconstruction" in candidate_samples[sample_id]
    )
    if not shared_ids:
        raise SystemExit("No overlapping samples with source/reconstruction pairs were found.")

    metric_suite = ImageQualityMetricSuite(lpips_network=args.lpips_network).cpu().eval()
    rows: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []

    for sample_id in shared_ids:
        source_path = baseline_samples[sample_id]["source"]
        baseline_path = baseline_samples[sample_id]["reconstruction"]
        candidate_path = candidate_samples[sample_id]["reconstruction"]

        source_image = load_rgb(source_path)
        baseline_image = load_rgb(baseline_path)
        candidate_image = load_rgb(candidate_path)

        baseline_metrics = compute_metrics(
            metric_suite, source_image=source_image, reconstructed_image=baseline_image
        )
        candidate_metrics = compute_metrics(
            metric_suite, source_image=source_image, reconstructed_image=candidate_image
        )
        deltas = delta_summary(baseline_metrics, candidate_metrics)

        rows.append(
            {
                "sample_id": sample_id,
                "source_path": str(source_path),
                "baseline_path": str(baseline_path),
                "candidate_path": str(candidate_path),
                "baseline_metrics": baseline_metrics,
                "candidate_metrics": candidate_metrics,
                "delta": deltas,
            }
        )
        render_rows.append(
            {
                "sample_id": sample_id,
                "source_image": source_image,
                "baseline_image": baseline_image,
                "candidate_image": candidate_image,
                "baseline_metrics": baseline_metrics,
                "candidate_metrics": candidate_metrics,
                "delta": deltas,
            }
        )

    summary = build_summary(rows, baseline_label=args.baseline_label, candidate_label=args.candidate_label)
    grid_path = Path(output_dir) / "comparison_grid.png"
    summary_path = Path(output_dir) / "comparison_metrics.json"
    render_grid(
        render_rows,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        tile_width=args.tile_width,
        output_path=grid_path,
    )
    save_json(summary_path, summary)

    print(f"comparison_grid: {grid_path}")
    print(f"comparison_metrics: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

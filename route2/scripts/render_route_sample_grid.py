from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from route2_swinjscc_gan.common.io import ensure_dir, save_json
from route2_swinjscc_gan.evaluators.metrics import ImageQualityMetricSuite


@dataclass(frozen=True)
class TileSpec:
    label: str
    image_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a single-sample multi-route comparison grid.")
    parser.add_argument("--title", required=True, help="Canvas title.")
    parser.add_argument("--source", required=True, help="Source image path.")
    parser.add_argument("--output-image", required=True, help="Output PNG path.")
    parser.add_argument("--output-json", required=True, help="Output JSON summary path.")
    parser.add_argument("--tile-width", type=int, default=320, help="Tile width in pixels.")
    parser.add_argument("--columns", type=int, default=4, help="Number of columns in the grid.")
    parser.add_argument(
        "--lpips-network",
        default="vgg",
        choices=("alex", "vgg"),
        help="LPIPS backbone used for per-tile metrics.",
    )
    parser.add_argument(
        "--tile",
        action="append",
        default=[],
        help="Tile in the form label=/abs/path/to/image.png. Repeat for multiple routes.",
    )
    return parser.parse_args()


def parse_tile(raw: str) -> TileSpec:
    if "=" not in raw:
        raise ValueError(f"Tile must be in label=path form, got: {raw}")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Tile must define both label and path, got: {raw}")
    return TileSpec(label=label, image_path=path)


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


def main() -> int:
    args = parse_args()
    tiles = [parse_tile(raw) for raw in args.tile]
    if not tiles:
        raise SystemExit("At least one --tile must be provided.")

    source_path = Path(args.source).resolve()
    source_image = load_rgb(source_path)
    metric_suite = ImageQualityMetricSuite(lpips_network=args.lpips_network).cpu().eval()
    font = ImageFont.load_default()

    resized_source = resize_for_grid(source_image, tile_width=args.tile_width)
    tile_height = resized_source.height
    margin = 16
    gap_x = 12
    gap_y = 18
    title_height = 42
    label_height = 18
    metrics_height = 18
    tile_block_height = label_height + tile_height + metrics_height
    columns = max(1, args.columns)
    rows = (len(tiles) + columns - 1) // columns
    canvas_width = margin * 2 + columns * args.tile_width + (columns - 1) * gap_x
    canvas_height = margin * 2 + title_height + rows * tile_block_height + (rows - 1) * gap_y
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, margin), args.title, fill=(18, 18, 18), font=font)

    summary_rows: list[dict[str, object]] = []
    for index, tile in enumerate(tiles):
        row_index = index // columns
        column_index = index % columns
        left = margin + column_index * (args.tile_width + gap_x)
        top = margin + title_height + row_index * (tile_block_height + gap_y)

        tile_path = Path(tile.image_path).resolve()
        tile_image = load_rgb(tile_path)
        resized_tile = resize_for_grid(tile_image, tile_width=args.tile_width)
        metrics = compute_metrics(metric_suite, source_image=source_image, reconstructed_image=tile_image)

        draw.text((left, top), tile.label, fill=(50, 50, 50), font=font)
        image_top = top + label_height
        canvas.paste(resized_tile, (left, image_top))
        metrics_text = (
            f"P {metrics['psnr']:.2f}  "
            f"M {metrics['ms_ssim']:.4f}  "
            f"L {metrics['lpips']:.4f}"
        )
        draw.text(
            (left, image_top + tile_height + 2),
            metrics_text,
            fill=(40, 40, 40),
            font=font,
        )
        summary_rows.append(
            {
                "label": tile.label,
                "image_path": str(tile_path),
                "metrics": metrics,
            }
        )

    output_image = Path(args.output_image).resolve()
    output_json = Path(args.output_json).resolve()
    ensure_dir(output_image.parent)
    ensure_dir(output_json.parent)
    canvas.save(output_image)
    save_json(
        output_json,
        {
            "title": args.title,
            "source_path": str(source_path),
            "tile_width": args.tile_width,
            "columns": columns,
            "tiles": [asdict(tile) for tile in tiles],
            "rows": summary_rows,
        },
    )
    print(f"output_image: {output_image}")
    print(f"output_json: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

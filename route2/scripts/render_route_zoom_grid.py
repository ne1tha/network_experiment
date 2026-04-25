from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from route2_swinjscc_gan.common.io import ensure_dir, save_json


@dataclass(frozen=True)
class TileSpec:
    label: str
    image_path: str


@dataclass(frozen=True)
class CropSpec:
    label: str
    x: int
    y: int
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render multi-route zoomed crop comparisons for a sample.")
    parser.add_argument("--title", required=True, help="Canvas title.")
    parser.add_argument("--source", required=True, help="Source image path.")
    parser.add_argument("--output-image", required=True, help="Output PNG path.")
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    parser.add_argument("--overview-width", type=int, default=520, help="Source overview width in pixels.")
    parser.add_argument("--zoom-scale", type=int, default=3, help="Integer zoom scale for crops.")
    parser.add_argument(
        "--tile",
        action="append",
        default=[],
        help="Tile in the form label=/abs/path/to/image.png. Repeat for multiple routes.",
    )
    parser.add_argument(
        "--crop",
        action="append",
        default=[],
        help="Crop in the form label=x,y,w,h. Repeat for multiple crop regions.",
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


def parse_crop(raw: str) -> CropSpec:
    if "=" not in raw:
        raise ValueError(f"Crop must be in label=x,y,w,h form, got: {raw}")
    label, payload = raw.split("=", 1)
    label = label.strip()
    parts = [item.strip() for item in payload.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Crop must define x,y,w,h, got: {raw}")
    x, y, width, height = (int(item) for item in parts)
    return CropSpec(label=label, x=x, y=y, width=width, height=height)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_overview(image: Image.Image, *, target_width: int) -> Image.Image:
    scale = target_width / image.width
    target_height = max(1, round(image.height * scale))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def crop_and_zoom(image: Image.Image, crop: CropSpec, *, zoom_scale: int) -> Image.Image:
    patch = image.crop((crop.x, crop.y, crop.x + crop.width, crop.y + crop.height))
    return patch.resize((crop.width * zoom_scale, crop.height * zoom_scale), Image.Resampling.NEAREST)


def draw_crop_boxes(
    image: Image.Image,
    crops: list[CropSpec],
    *,
    target_width: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    overview = resize_overview(image, target_width=target_width)
    draw = ImageDraw.Draw(overview)
    scale = overview.width / image.width
    colors = [(216, 78, 61), (44, 126, 200), (76, 166, 76), (173, 95, 189)]
    for index, crop in enumerate(crops):
        color = colors[index % len(colors)]
        left = round(crop.x * scale)
        top = round(crop.y * scale)
        right = round((crop.x + crop.width) * scale)
        bottom = round((crop.y + crop.height) * scale)
        draw.rectangle((left, top, right, bottom), outline=color, width=3)
        draw.rectangle((left, top - 16, left + 24, top), fill=color)
        draw.text((left + 6, top - 15), crop.label, fill=(255, 255, 255), font=font)
    return overview


def main() -> int:
    args = parse_args()
    tiles = [parse_tile(raw) for raw in args.tile]
    crops = [parse_crop(raw) for raw in args.crop]
    if not tiles:
        raise SystemExit("At least one --tile must be provided.")
    if not crops:
        raise SystemExit("At least one --crop must be provided.")

    source_path = Path(args.source).resolve()
    source_image = load_rgb(source_path)
    tile_images = [(tile.label, load_rgb(Path(tile.image_path).resolve()), str(Path(tile.image_path).resolve())) for tile in tiles]

    font = ImageFont.load_default()
    overview = draw_crop_boxes(source_image, crops, target_width=args.overview_width, font=font)

    zoom_tile_width = crops[0].width * args.zoom_scale
    zoom_tile_height = crops[0].height * args.zoom_scale
    for crop in crops[1:]:
        if crop.width != crops[0].width or crop.height != crops[0].height:
            raise ValueError("All crops must currently share the same width and height.")

    margin = 18
    gap_x = 12
    gap_y = 18
    title_height = 28
    header_height = 18
    route_label_height = 18
    columns = len(tile_images)
    rows = len(crops)
    grid_width = columns * zoom_tile_width + (columns - 1) * gap_x
    grid_height = rows * (header_height + route_label_height + zoom_tile_height) + (rows - 1) * gap_y
    canvas_width = margin * 2 + max(overview.width, grid_width)
    canvas_height = margin * 2 + title_height + overview.height + 24 + grid_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, margin), args.title, fill=(20, 20, 20), font=font)
    overview_top = margin + title_height
    canvas.paste(overview, (margin, overview_top))

    grid_top = overview_top + overview.height + 24
    rows_payload: list[dict[str, object]] = []
    for row_index, crop in enumerate(crops):
        row_top = grid_top + row_index * (header_height + route_label_height + zoom_tile_height + gap_y)
        draw.text((margin, row_top), f"{crop.label}: ({crop.x}, {crop.y}, {crop.width}, {crop.height})", fill=(45, 45, 45), font=font)
        label_y = row_top + header_height
        image_y = label_y + route_label_height

        crop_payload: dict[str, object] = {
            "crop": asdict(crop),
            "tiles": [],
        }
        for column_index, (label, image, image_path) in enumerate(tile_images):
            left = margin + column_index * (zoom_tile_width + gap_x)
            draw.text((left, label_y), label, fill=(60, 60, 60), font=font)
            zoomed = crop_and_zoom(image, crop, zoom_scale=args.zoom_scale)
            canvas.paste(zoomed, (left, image_y))
            crop_payload["tiles"].append(
                {
                    "label": label,
                    "image_path": image_path,
                }
            )
        rows_payload.append(crop_payload)

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
            "overview_width": args.overview_width,
            "zoom_scale": args.zoom_scale,
            "tiles": [asdict(tile) for tile in tiles],
            "crops": [asdict(crop) for crop in crops],
            "rows": rows_payload,
        },
    )
    print(f"output_image: {output_image}")
    print(f"output_json: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

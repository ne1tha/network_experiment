from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from PIL import Image, ImageDraw

from route2_swinjscc_gan.common.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic dataset for Route 2 smoke tests.")
    parser.add_argument("--output-root", required=True, help="Dataset root to create.")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_image(size: int, rng: random.Random, index: int) -> Image.Image:
    image = Image.new("RGB", (size, size), color=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    draw = ImageDraw.Draw(image)

    for _ in range(12):
        x0 = rng.randint(0, size - 32)
        y0 = rng.randint(0, size - 32)
        x1 = rng.randint(x0 + 8, min(size, x0 + size // 2))
        y1 = rng.randint(y0 + 8, min(size, y0 + size // 2))
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        if rng.random() > 0.5:
            draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        else:
            draw.ellipse((x0, y0, x1, y1), outline=color, width=3)

    for step in range(0, size, max(8, size // 16)):
        draw.line((0, step, size, (step + index * 7) % size), fill=(255, 255, 255), width=1)
    return image


def populate(split_dir: Path, count: int, size: int, rng: random.Random) -> None:
    ensure_dir(split_dir)
    for index in range(count):
        image = make_image(size, rng, index=index)
        image.save(split_dir / f"{index:04d}.png")


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    root = Path(args.output_root)
    train_dir = ensure_dir(root / "train")
    test_dir = ensure_dir(root / "test")
    populate(train_dir, count=args.train_count, size=args.size, rng=rng)
    populate(test_dir, count=args.test_count, size=args.size, rng=rng)
    print(f"created synthetic dataset under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

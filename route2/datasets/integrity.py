from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from route2_swinjscc_gan.common.checks import require_positive_int
from route2_swinjscc_gan.datasets.image_folder import _collect_images


@dataclass(slots=True)
class ImageSample:
    path: str
    width: int
    height: int


@dataclass(slots=True)
class DatasetSummary:
    root_paths: list[str]
    num_images: int
    sample_images: list[ImageSample]
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    bad_eval_images: list[str]
    too_small_images: list[str]


def summarize_image_dirs(
    image_dirs: Sequence[str | Path],
    *,
    require_multiple_of: int | None = None,
    min_size: int | None = None,
    max_samples: int = 16,
) -> DatasetSummary:
    require_positive_int("max_samples", max_samples)
    roots = [Path(path).expanduser().resolve() for path in image_dirs]
    for path in roots:
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {path}")

    image_files = _collect_images(roots)
    widths: list[int] = []
    heights: list[int] = []
    samples: list[ImageSample] = []
    bad_eval_images: list[str] = []
    too_small_images: list[str] = []

    for index, image_path in enumerate(image_files):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size

        widths.append(width)
        heights.append(height)
        if index < max_samples:
            samples.append(ImageSample(path=str(image_path), width=width, height=height))
        if require_multiple_of is not None and (width % require_multiple_of != 0 or height % require_multiple_of != 0):
            bad_eval_images.append(str(image_path))
        if min_size is not None and (width < min_size or height < min_size):
            too_small_images.append(str(image_path))

    return DatasetSummary(
        root_paths=[str(path) for path in roots],
        num_images=len(image_files),
        sample_images=samples,
        min_width=min(widths),
        max_width=max(widths),
        min_height=min(heights),
        max_height=max(heights),
        bad_eval_images=bad_eval_images,
        too_small_images=too_small_images,
    )

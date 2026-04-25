from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from datasets.hr_datasets import _collect_image_files


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


def summarize_image_dirs(
    image_dirs: list[Path],
    *,
    require_multiple_of: int | None = None,
    max_samples: int = 16,
) -> DatasetSummary:
    image_files = _collect_image_files(image_dirs)
    widths: list[int] = []
    heights: list[int] = []
    samples: list[ImageSample] = []
    bad_eval_images: list[str] = []

    for index, image_path in enumerate(image_files):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size

        widths.append(width)
        heights.append(height)
        if index < max_samples:
            samples.append(ImageSample(path=str(image_path), width=width, height=height))
        if require_multiple_of is not None:
            if width % require_multiple_of != 0 or height % require_multiple_of != 0:
                bad_eval_images.append(str(image_path))

    return DatasetSummary(
        root_paths=[str(path) for path in image_dirs],
        num_images=len(image_files),
        sample_images=samples,
        min_width=min(widths),
        max_width=max(widths),
        min_height=min(heights),
        max_height=max(heights),
        bad_eval_images=bad_eval_images,
    )

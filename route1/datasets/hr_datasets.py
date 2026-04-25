from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from configs.route1_reference import Route1ExperimentConfig
from datasets.path_checks import IMAGE_SUFFIXES, require_existing_paths


def _collect_image_files(paths: list[Path]) -> list[Path]:
    image_files: list[Path] = []
    for root in require_existing_paths(paths, label="dataset"):
        if root.is_file():
            if root.suffix.lower() in IMAGE_SUFFIXES:
                image_files.append(root)
            continue

        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                image_files.append(candidate)

    image_files = sorted(set(image_files))
    if not image_files:
        joined = ", ".join(str(path) for path in paths)
        raise FileNotFoundError(f"No image files were found in dataset paths: {joined}")
    return image_files


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an RGB image array, got shape {array.shape!r}.")
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous() / 255.0
    return tensor


def _random_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    if width < crop_size or height < crop_size:
        raise ValueError(
            f"Image size {(width, height)} is smaller than required crop size {crop_size}."
        )
    if width == crop_size and height == crop_size:
        return image

    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    return image.crop((left, top, left + crop_size, top + crop_size))


def _center_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid crop target {(width, height)}.")
    image_width, image_height = image.size
    if width > image_width or height > image_height:
        raise ValueError(
            f"Crop target {(width, height)} exceeds image size {(image_width, image_height)}."
        )

    left = (image_width - width) // 2
    top = (image_height - height) // 2
    return image.crop((left, top, left + width, top + height))


class Div2KTrainDataset(Dataset[torch.Tensor]):
    def __init__(self, image_dirs: list[Path], crop_size: int = 256) -> None:
        self.image_files = _collect_image_files(image_dirs)
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_files[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            cropped = _random_crop(image, self.crop_size)
        return _to_tensor(cropped)


class HREvalDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(
        self,
        image_dirs: list[Path],
        *,
        require_multiple_of: int = 128,
        allow_size_adjustment: bool = False,
    ) -> None:
        self.image_files = _collect_image_files(image_dirs)
        self.require_multiple_of = require_multiple_of
        self.allow_size_adjustment = allow_size_adjustment

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_files[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            cropped_width = width - (width % self.require_multiple_of)
            cropped_height = height - (height % self.require_multiple_of)

            if cropped_width != width or cropped_height != height:
                if not self.allow_size_adjustment:
                    raise ValueError(
                        "Evaluation image size must be divisible by "
                        f"{self.require_multiple_of}, got {(width, height)} for {image_path}."
                    )
                image = _center_crop(image, cropped_width, cropped_height)

            tensor = _to_tensor(image)
        return tensor, image_path.name


def build_hr_train_loader(experiment: Route1ExperimentConfig) -> DataLoader[torch.Tensor]:
    if experiment.trainset != "DIV2K":
        raise NotImplementedError(
            f"HR dataloaders currently support only DIV2K trainset, got {experiment.trainset!r}."
        )

    train_dataset = Div2KTrainDataset(
        image_dirs=experiment.dataset_paths.train_dirs,
        crop_size=256,
    )
    if len(train_dataset) < experiment.batch_size:
        raise ValueError(
            f"Training dataset has only {len(train_dataset)} images, which is smaller than batch_size={experiment.batch_size}."
        )
    return DataLoader(
        dataset=train_dataset,
        batch_size=experiment.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=experiment.num_workers,
        pin_memory=experiment.pin_memory,
    )


def build_hr_eval_loader(experiment: Route1ExperimentConfig) -> DataLoader[tuple[torch.Tensor, str]]:
    eval_dataset = HREvalDataset(
        image_dirs=experiment.dataset_paths.test_dirs,
        allow_size_adjustment=experiment.allow_eval_size_adjustment,
    )
    return DataLoader(
        dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=experiment.num_workers,
        pin_memory=experiment.pin_memory,
    )


def build_hr_dataloaders(
    experiment: Route1ExperimentConfig,
) -> tuple[DataLoader[torch.Tensor], DataLoader[tuple[torch.Tensor, str]]]:
    return build_hr_train_loader(experiment), build_hr_eval_loader(experiment)
    return train_loader, eval_loader

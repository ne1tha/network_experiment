from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from route2_swinjscc_gan.common.checks import require, require_positive_int
from route2_swinjscc_gan.configs.defaults import DataConfig


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _validate_roots(name: str, roots: Sequence[str]) -> list[Path]:
    require(len(roots) > 0, f"`{name}` must contain at least one directory.")
    paths = [Path(root).expanduser().resolve() for root in roots]
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {path}")
    return paths


def _collect_images(roots: Sequence[Path]) -> list[Path]:
    images: list[Path] = []
    for root in roots:
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(candidate)
    images.sort()
    if not images:
        raise FileNotFoundError(f"No supported image files found under: {[str(root) for root in roots]}")
    return images


class TrainingImageDataset(Dataset[torch.Tensor]):
    def __init__(self, image_paths: Sequence[Path], crop_size: int) -> None:
        require_positive_int("crop_size", crop_size)
        self.image_paths = list(image_paths)
        self.transform = transforms.Compose(
            [
                transforms.RandomCrop((crop_size, crop_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if min(width, height) < self.transform.transforms[0].size[0]:
                raise RuntimeError(f"Training image is smaller than crop size: {image_path}")
            return self.transform(image)


class ValidationImageDataset(Dataset[torch.Tensor]):
    def __init__(self, image_paths: Sequence[Path], crop_size: int) -> None:
        require_positive_int("crop_size", crop_size)
        self.image_paths = list(image_paths)
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if min(width, height) < self.transform.transforms[0].size[0]:
                raise RuntimeError(f"Validation image is smaller than crop size: {image_path}")
            return self.transform(image)


@dataclass(frozen=True)
class EvaluationSample:
    image: torch.Tensor
    name: str


class EvaluationImageDataset(Dataset[EvaluationSample]):
    def __init__(self, image_paths: Sequence[Path], divisible_by: int, allow_size_adjustment: bool = False) -> None:
        require_positive_int("divisible_by", divisible_by)
        self.image_paths = list(image_paths)
        self.divisible_by = divisible_by
        self.allow_size_adjustment = allow_size_adjustment
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> EvaluationSample:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            cropped_width = width - (width % self.divisible_by)
            cropped_height = height - (height % self.divisible_by)
            if cropped_width == 0 or cropped_height == 0:
                raise RuntimeError(
                    f"Evaluation image {image_path} is smaller than divisibility constraint {self.divisible_by}."
                )
            if cropped_width != width or cropped_height != height:
                if not self.allow_size_adjustment:
                    raise RuntimeError(
                        f"Evaluation image {image_path} with size {(width, height)} is not divisible "
                        f"by {self.divisible_by}. Enable size adjustment explicitly to allow center cropping."
                    )
            left = (width - cropped_width) // 2
            top = (height - cropped_height) // 2
            image = image.crop((left, top, left + cropped_width, top + cropped_height))
            tensor = self.to_tensor(image)
            return EvaluationSample(image=tensor, name=image_path.name)


def _collate_eval(samples: Sequence[EvaluationSample]) -> tuple[torch.Tensor, list[str]]:
    images = torch.stack([sample.image for sample in samples], dim=0)
    names = [sample.name for sample in samples]
    return images, names


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_train_val_loaders(config: DataConfig) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
    train_roots = _validate_roots("train_roots", config.train_roots)
    image_paths = _collect_images(train_roots)

    if config.val_roots:
        val_roots = _validate_roots("val_roots", config.val_roots)
        train_dataset: Dataset[torch.Tensor] = TrainingImageDataset(image_paths, crop_size=config.crop_size)
        val_dataset: Dataset[torch.Tensor] = ValidationImageDataset(
            _collect_images(val_roots),
            crop_size=config.crop_size,
        )
    else:
        generator = torch.Generator().manual_seed(config.seed)
        permutation = torch.randperm(len(image_paths), generator=generator).tolist()
        train_length = int(len(image_paths) * (1.0 - config.val_split_ratio))
        val_length = len(image_paths) - train_length
        require(train_length > 0 and val_length > 0, "Train/val split produced an empty subset.")
        train_paths = [image_paths[index] for index in permutation[:train_length]]
        val_paths = [image_paths[index] for index in permutation[train_length:]]
        train_dataset = TrainingImageDataset(train_paths, crop_size=config.crop_size)
        val_dataset = ValidationImageDataset(val_paths, crop_size=config.crop_size)

    require(
        len(train_dataset) >= config.batch_size,
        f"Training dataset has {len(train_dataset)} images after split, smaller than batch_size={config.batch_size}.",
    )

    pin_memory = torch.cuda.is_available()
    loader_generator = torch.Generator().manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        generator=loader_generator,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
    )
    return train_loader, val_loader


def build_test_loader(config: DataConfig) -> DataLoader[tuple[torch.Tensor, list[str]]]:
    test_roots = _validate_roots("test_roots", config.test_roots)
    dataset = EvaluationImageDataset(
        _collect_images(test_roots),
        divisible_by=config.eval_divisible_by,
        allow_size_adjustment=config.allow_eval_size_adjustment,
    )
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_eval,
        worker_init_fn=_seed_worker,
    )

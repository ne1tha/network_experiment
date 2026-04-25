from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from models.ultimate.fusion_interface import assert_valid_image_size

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required for route-3 dataset loading.") from exc

try:
    from torchvision.transforms import functional as TF
    from torchvision.transforms import RandomCrop
    from torchvision.transforms.functional import InterpolationMode
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torchvision is required for route-3 dataset loading.") from exc


SUPPORTED_TRANSFORM_MODES = {"resize", "center_crop", "random_crop"}


@dataclass(frozen=True)
class ManifestSample:
    image: str
    user_id: str | None = None


@dataclass(frozen=True)
class Route3DatasetManifest:
    root: str | None
    splits: dict[str, tuple[ManifestSample, ...]]


def load_dataset_manifest(path: str | Path) -> Route3DatasetManifest:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    splits = {}
    for split_name, samples in raw.get("splits", {}).items():
        splits[split_name] = tuple(ManifestSample(**sample) for sample in samples)
    manifest = Route3DatasetManifest(root=raw.get("root"), splits=splits)
    return manifest


def validate_dataset_manifest(
    manifest: Route3DatasetManifest,
    manifest_path: str | Path,
    required_splits: tuple[str, ...] = ("train", "val"),
) -> None:
    manifest_path = Path(manifest_path)
    root = Path(manifest.root) if manifest.root is not None else manifest_path.parent
    for split_name in required_splits:
        if split_name not in manifest.splits:
            raise ValueError(f"Dataset manifest is missing required split '{split_name}'.")
        if not manifest.splits[split_name]:
            raise ValueError(f"Dataset split '{split_name}' is empty.")
        for sample in manifest.splits[split_name]:
            image_path = root / sample.image
            if not image_path.exists():
                raise FileNotFoundError(f"Dataset sample image not found: {image_path}")


class Route3ImageManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        resize_hw: tuple[int, int] | None = None,
        transform_mode: str = "resize",
    ):
        self.manifest_path = Path(manifest_path)
        self.manifest = load_dataset_manifest(self.manifest_path)
        validate_dataset_manifest(self.manifest, self.manifest_path, required_splits=(split,))
        self.root = Path(self.manifest.root) if self.manifest.root is not None else self.manifest_path.parent
        self.split = split
        self.samples = self.manifest.splits[split]
        self.resize_hw = resize_hw
        if transform_mode not in SUPPORTED_TRANSFORM_MODES:
            raise ValueError(f"Unsupported transform_mode {transform_mode!r}. Expected one of {sorted(SUPPORTED_TRANSFORM_MODES)}.")
        self.transform_mode = transform_mode
        if resize_hw is not None:
            assert_valid_image_size(resize_hw[0], resize_hw[1])

    @staticmethod
    def _resize_to_cover(image: Image.Image, target_hw: tuple[int, int]) -> Image.Image:
        target_h, target_w = target_hw
        source_w, source_h = image.size
        scale = max(target_h / float(source_h), target_w / float(source_w))
        resized_h = max(target_h, int(round(source_h * scale)))
        resized_w = max(target_w, int(round(source_w * scale)))
        return TF.resize(image, [resized_h, resized_w], interpolation=InterpolationMode.BICUBIC, antialias=True)

    def _apply_transform(self, image: Image.Image) -> Image.Image:
        if self.resize_hw is None:
            return image
        if self.transform_mode == "resize":
            return TF.resize(image, list(self.resize_hw), interpolation=InterpolationMode.BICUBIC, antialias=True)

        if image.height < self.resize_hw[0] or image.width < self.resize_hw[1]:
            image = self._resize_to_cover(image, self.resize_hw)

        if self.transform_mode == "center_crop":
            return TF.center_crop(image, list(self.resize_hw))

        top, left, height, width = RandomCrop.get_params(image, output_size=self.resize_hw)
        return TF.crop(image, top, left, height, width)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, str | None]]:
        sample = self.samples[index]
        image_path = self.root / sample.image
        image = Image.open(image_path).convert("RGB")
        image = self._apply_transform(image)
        tensor = TF.to_tensor(image)
        height, width = tensor.shape[-2:]
        assert_valid_image_size(height, width)
        meta = {
            "image_path": str(image_path),
            "user_id": sample.user_id,
            "split": self.split,
        }
        return tensor, meta

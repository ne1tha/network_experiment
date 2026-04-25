from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from route2_swinjscc_gan.configs.defaults import DataConfig
from route2_swinjscc_gan.datasets.image_folder import EvaluationImageDataset, build_train_val_loaders


def _write_rgb(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size=size, color=(255, 128, 0)).save(path)


def test_eval_dataset_rejects_implicit_size_adjustment(tmp_path: Path) -> None:
    image_path = tmp_path / "eval" / "sample.png"
    _write_rgb(image_path, size=(260, 260))
    dataset = EvaluationImageDataset([image_path], divisible_by=128, allow_size_adjustment=False)

    with pytest.raises(RuntimeError, match="not divisible"):
        dataset[0]


def test_eval_dataset_allows_explicit_size_adjustment(tmp_path: Path) -> None:
    image_path = tmp_path / "eval" / "sample.png"
    _write_rgb(image_path, size=(260, 260))
    dataset = EvaluationImageDataset([image_path], divisible_by=128, allow_size_adjustment=True)

    sample = dataset[0]
    assert tuple(sample.image.shape) == (3, 256, 256)


def test_train_loader_rejects_dataset_smaller_than_batch_size(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _write_rgb(train_dir / "0000.png", size=(256, 256))
    _write_rgb(train_dir / "0001.png", size=(256, 256))
    _write_rgb(val_dir / "0000.png", size=(256, 256))

    config = DataConfig(
        train_roots=(str(train_dir),),
        val_roots=(str(val_dir),),
        crop_size=256,
        batch_size=3,
        num_workers=0,
    )

    with pytest.raises(RuntimeError, match="smaller than batch_size"):
        build_train_val_loaders(config)

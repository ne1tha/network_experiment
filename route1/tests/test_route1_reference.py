from __future__ import annotations

from pathlib import Path
import sys
import json

import pytest
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from channels.specs import validate_channel_type
from configs.route1_reference import (
    Route1ExperimentConfig,
    build_div2k_reference_config,
    parse_csv_ints,
)
from configs.reference_loader import load_dataset_manifest, load_reference_experiment
from datasets.hr_datasets import HREvalDataset
from datasets.path_checks import require_existing_paths, require_images_present
from support.runtime import load_checkpoint_compatible, save_checkpoint


def test_parse_csv_ints_keeps_order() -> None:
    assert parse_csv_ints("32, 64,96") == [32, 64, 96]


def test_parse_csv_ints_rejects_empty_item() -> None:
    with pytest.raises(ValueError):
        parse_csv_ints("32,,96")


def test_validate_channel_type_rejects_none() -> None:
    with pytest.raises(ValueError):
        validate_channel_type("none")


def test_require_existing_paths_raises_for_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        require_existing_paths([missing], label="training dataset")


def test_require_images_present_fails_when_directory_has_no_images(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        require_images_present([empty_dir], label="evaluation dataset")


def test_build_div2k_reference_config_requires_existing_dirs(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    (train_dir / "a.png").write_bytes(b"not-a-real-png")
    (test_dir / "b.jpg").write_bytes(b"not-a-real-jpg")

    config = build_div2k_reference_config(
        workspace_root=tmp_path,
        train_dirs=[train_dir],
        test_dirs=[test_dir],
        channels_csv="96",
        snrs_csv="10",
    )
    assert config.channel_values == [96]


def test_experiment_config_rejects_checkpoint_and_resume_together(tmp_path: Path) -> None:
    ckpt = tmp_path / "a.pt"
    resume = tmp_path / "b.pt"
    ckpt.write_bytes(b"x")
    resume.write_bytes(b"y")
    config = Route1ExperimentConfig(
        workspace_root=tmp_path,
        dataset_paths=type("DatasetPathsLike", (), {"train_dirs": [tmp_path], "test_dirs": [tmp_path]})(),
        checkpoint_path=ckpt,
        resume_path=resume,
    )
    with pytest.raises(ValueError):
        config.validate()


def test_experiment_config_rejects_unknown_checkpoint_load_mode(tmp_path: Path) -> None:
    config = Route1ExperimentConfig(
        workspace_root=tmp_path,
        dataset_paths=type("DatasetPathsLike", (), {"train_dirs": [tmp_path], "test_dirs": [tmp_path]})(),
        checkpoint_load_mode="partial",
    )
    with pytest.raises(ValueError):
        config.validate()


def test_eval_dataset_rejects_non_multiple_size_by_default(tmp_path: Path) -> None:
    image_path = tmp_path / "bad.png"
    Image.new("RGB", (257, 257), color=(255, 0, 0)).save(image_path)
    dataset = HREvalDataset([tmp_path], allow_size_adjustment=False)
    with pytest.raises(ValueError):
        dataset[0]


def test_eval_dataset_adjusts_size_only_when_enabled(tmp_path: Path) -> None:
    image_path = tmp_path / "crop.png"
    Image.new("RGB", (257, 257), color=(0, 255, 0)).save(image_path)
    dataset = HREvalDataset([tmp_path], allow_size_adjustment=True)
    tensor, name = dataset[0]
    assert tuple(tensor.shape) == (3, 256, 256)
    assert name == "crop.png"


def test_load_dataset_manifest_rejects_placeholder_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "placeholder_manifest",
                "train_dirs": ["<fill-me>/train"],
                "test_dirs": ["<fill-me>/test"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_dataset_manifest(manifest_path)


def test_load_reference_experiment_resolves_relative_paths(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    test_dir = tmp_path / "data" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    Image.new("RGB", (320, 320), color=(1, 2, 3)).save(train_dir / "train.png")
    Image.new("RGB", (256, 256), color=(4, 5, 6)).save(test_dir / "test.png")

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    manifest_path = configs_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "local_manifest",
                "train_dirs": ["../data/train"],
                "test_dirs": ["../data/test"],
            }
        ),
        encoding="utf-8",
    )
    experiment_path = configs_dir / "experiment.json"
    experiment_path.write_text(
        json.dumps(
            {
                "dataset_manifest": "manifest.json",
                "run_name": "relative_path_case",
                "device": "cpu",
                "channels_csv": "96",
                "snrs_csv": "10",
                "batch_size": 1,
                "total_epochs": 1,
                "save_model_freq": 1,
                "print_step": 1,
                "checkpoint_load_mode": "compatible",
            }
        ),
        encoding="utf-8",
    )

    experiment, manifest = load_reference_experiment(experiment_path)
    assert manifest.name == "local_manifest"
    assert experiment.run_name == "relative_path_case"
    assert experiment.checkpoint_load_mode == "compatible"
    assert experiment.dataset_paths.train_dirs == [train_dir.resolve()]
    assert experiment.dataset_paths.test_dirs == [test_dir.resolve()]


def test_load_checkpoint_compatible_only_loads_matching_shapes(tmp_path: Path) -> None:
    class SourceNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 3)
            self.head = nn.Linear(3, 2)

    class TargetNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 3)
            self.head = nn.Linear(3, 4)
            self.extra = nn.Linear(3, 1)

    source = SourceNet()
    target = TargetNet()

    with torch.no_grad():
        source.shared.weight.fill_(1.25)
        source.shared.bias.fill_(0.75)
        source.head.weight.fill_(2.5)
        source.head.bias.fill_(1.5)

    original_target_head = target.head.weight.detach().clone()
    original_target_extra = target.extra.weight.detach().clone()

    checkpoint_path = tmp_path / "source.pt"
    save_checkpoint(
        model=source,
        optimizer=None,
        epoch=3,
        global_step=7,
        experiment={"run_name": "source"},
        output_path=checkpoint_path,
    )

    metadata = load_checkpoint_compatible(
        model=target,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    assert metadata["load_mode"] == "compatible"
    assert metadata["loaded_key_count"] == 2
    assert metadata["shape_mismatch_count"] == 2
    assert "head.weight" in metadata["shape_mismatch_examples"]
    assert torch.allclose(target.shared.weight, source.shared.weight)
    assert torch.allclose(target.shared.bias, source.shared.bias)
    assert torch.allclose(target.head.weight, original_target_head)
    assert torch.allclose(target.extra.weight, original_target_extra)

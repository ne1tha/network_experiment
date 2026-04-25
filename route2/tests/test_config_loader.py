from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from route2_swinjscc_gan.configs.manifest_loader import load_dataset_manifest, load_experiment_config


def _write_rgb(path: Path, size: tuple[int, int] = (256, 256)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size=size, color=(64, 128, 192)).save(path)


def test_load_dataset_manifest_rejects_placeholder_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "placeholder_manifest",
                "train_roots": ["<fill-me>/DIV2K_train_HR"],
                "test_roots": ["<fill-me>/Kodak"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="placeholder path"):
        load_dataset_manifest(manifest_path)


def test_load_experiment_config_resolves_relative_paths(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    test_dir = tmp_path / "data" / "test"
    _write_rgb(train_dir / "train.png")
    _write_rgb(test_dir / "test.png")

    manifest_dir = tmp_path / "configs" / "datasets"
    experiments_dir = tmp_path / "configs" / "experiments"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / "route2_test_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "route2_test_manifest",
                "train_roots": ["../../data/train"],
                "test_roots": ["../../data/test"],
                "allow_eval_size_adjustment": False,
            }
        ),
        encoding="utf-8",
    )
    experiment_path = experiments_dir / "route2_test_experiment.json"
    experiment_path.write_text(
        json.dumps(
            {
                "name": "route2_test_experiment",
                "dataset_manifest": "../datasets/route2_test_manifest.json",
                "output_dir": "../../runs/route2_test_experiment",
                "device": "cpu",
                "image_size": 128,
                "batch_size": 1,
                "num_workers": 0,
                "multiple_snr": [10],
                "channel_numbers": [24],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_experiment_config(experiment_path)
    assert loaded.name == "route2_test_experiment"
    assert loaded.config.data.train_roots == (str(train_dir.resolve()),)
    assert loaded.config.data.test_roots == (str(test_dir.resolve()),)
    assert loaded.config.data.crop_size == 128
    assert loaded.config.training.output_dir == str((tmp_path / "runs" / "route2_test_experiment").resolve())
    assert loaded.config.discriminator.kind == "legacy_patchgan"


def test_load_experiment_config_parses_adversarial_ablation_and_route2_checkpoint(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    test_dir = tmp_path / "data" / "test"
    checkpoint_path = tmp_path / "artifacts" / "route2_resume.pt"
    _write_rgb(train_dir / "train.png")
    _write_rgb(test_dir / "test.png")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"route2")

    manifest_dir = tmp_path / "configs" / "datasets"
    experiments_dir = tmp_path / "configs" / "experiments"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / "route2_test_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "route2_test_manifest",
                "train_roots": ["../../data/train"],
                "test_roots": ["../../data/test"],
            }
        ),
        encoding="utf-8",
    )

    experiment_path = experiments_dir / "route2_ablation_experiment.json"
    experiment_path.write_text(
        json.dumps(
            {
                "name": "route2_ablation_experiment",
                "dataset_manifest": "../datasets/route2_test_manifest.json",
                "device": "cpu",
                "checkpoint_path": "../../artifacts/route2_resume.pt",
                "optimizer": {
                    "generator_lr": 2e-4,
                    "discriminator_lr": 5e-5,
                    "warmup_epochs": 2,
                },
                "discriminator": {
                    "kind": "legacy_patchgan",
                    "norm_type": "batch",
                    "use_spectral_norm": False,
                },
                "adversarial": {
                    "enabled": False,
                    "loss_mode": "bce",
                    "weight": 0.02,
                    "ramp_epochs": 5,
                    "discriminator_lr_scale": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_experiment_config(experiment_path)

    assert loaded.config.training.checkpoint_path == str(checkpoint_path.resolve())
    assert loaded.config.optimizer.generator_lr == 2e-4
    assert loaded.config.optimizer.discriminator_lr == 5e-5
    assert loaded.config.optimizer.warmup_epochs == 2
    assert loaded.config.discriminator.kind == "legacy_patchgan"
    assert loaded.config.discriminator.norm_type == "batch"
    assert loaded.config.discriminator.use_spectral_norm is False
    assert loaded.config.adversarial.enabled is False
    assert loaded.config.adversarial.loss_mode == "bce"
    assert loaded.config.adversarial.weight == 0.02
    assert loaded.config.adversarial.ramp_epochs == 5
    assert loaded.config.adversarial.discriminator_lr_scale == 0.5

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Route3ImageManifestDataset
from models.ultimate import UltimateDualPathEncoder
from scripts.route3_preflight import load_preflight_config, run_forward_dry_run, validate_preflight_config


def _write_image(path: Path, size: tuple[int, int] = (128, 128)) -> None:
    tensor = (torch.rand(size[0], size[1], 3) * 255).to(torch.uint8).numpy()
    Image.fromarray(tensor).save(path)


class PreflightInfraTests(unittest.TestCase):
    def test_manifest_dataset_loads_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train").mkdir()
            image_path = root / "train" / "sample.png"
            _write_image(image_path)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [{"image": "train/sample.png", "user_id": "u1"}],
                            "val": [{"image": "train/sample.png", "user_id": "u2"}],
                        },
                    }
                ),
                encoding="utf-8",
            )

            dataset = Route3ImageManifestDataset(manifest_path=manifest_path, split="train", resize_hw=(128, 128))
            image, meta = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 128, 128))
            self.assertEqual(meta["user_id"], "u1")

    def test_preflight_validation_rejects_missing_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train").mkdir()
            image_path = root / "train" / "sample.png"
            _write_image(image_path)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [{"image": "train/sample.png"}],
                            "val": [{"image": "train/sample.png"}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "manifest_path": str(manifest_path),
                            "train_split": "train",
                            "val_split": "val",
                            "image_size": [128, 128],
                            "dry_run_samples": 2,
                        },
                        "weights": {
                            "teacher_checkpoint": str(root / "missing_teacher.pt"),
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": True,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "enable_distillation": True,
                            "enable_perceptual": False,
                            "enable_adversarial": False,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "semantic_bandwidth_budget": 10.0,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "ckpt"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_preflight_config(config_path)
            with self.assertRaisesRegex(FileNotFoundError, "teacher_checkpoint"):
                validate_preflight_config(config)

    def test_single_and_multi_user_dry_run_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train").mkdir()
            for idx in range(4):
                _write_image(root / "train" / f"sample_{idx}.png")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [
                                {"image": f"train/sample_{idx}.png", "user_id": f"u{idx}"}
                                for idx in range(4)
                            ],
                            "val": [
                                {"image": f"train/sample_{idx}.png", "user_id": f"v{idx}"}
                                for idx in range(4)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            teacher_path = root / "teacher.pt"
            torch.save(UltimateDualPathEncoder().semantic_branch.state_dict(), teacher_path)

            single_config_path = root / "single.json"
            single_config_path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "manifest_path": str(manifest_path),
                            "train_split": "train",
                            "val_split": "val",
                            "image_size": [128, 128],
                            "dry_run_samples": 2,
                        },
                        "weights": {
                            "teacher_checkpoint": str(teacher_path),
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": True,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "enable_distillation": True,
                            "enable_perceptual": True,
                            "enable_adversarial": True,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "semantic_bandwidth_budget": 10.0,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs_single"),
                            "checkpoint_dir": str(root / "outputs_single" / "ckpt"),
                            "report_dir": str(root / "outputs_single" / "reports"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            multi_config_path = root / "multi.json"
            multi_config_path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "manifest_path": str(manifest_path),
                            "train_split": "train",
                            "val_split": "val",
                            "image_size": [128, 128],
                            "dry_run_samples": 4,
                        },
                        "weights": {
                            "teacher_checkpoint": str(teacher_path),
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": True,
                        },
                        "runtime": {
                            "mode": "multi_user",
                            "enable_distillation": True,
                            "enable_perceptual": True,
                            "enable_adversarial": True,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "semantic_bandwidth_budget": 10.0,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs_multi"),
                            "checkpoint_dir": str(root / "outputs_multi" / "ckpt"),
                            "report_dir": str(root / "outputs_multi" / "reports"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            single_summary = run_forward_dry_run(load_preflight_config(single_config_path), split="train")
            multi_summary = run_forward_dry_run(load_preflight_config(multi_config_path), split="train")

            self.assertEqual(single_summary["mode"], "single_user")
            self.assertEqual(single_summary["operating_mode"], "open_quality")
            self.assertIn(single_summary["device"], {"cpu", "cuda", "cuda:0"})
            self.assertEqual(single_summary["batch_shape"], [2, 3, 128, 128])
            self.assertFalse(single_summary["reconstruction_stochastic"])
            self.assertIsNotNone(single_summary["budget"])
            self.assertGreater(single_summary["budget"]["effective_cbr"], 0.0)
            self.assertEqual(multi_summary["mode"], "multi_user")
            self.assertIn(multi_summary["device"], {"cpu", "cuda", "cuda:0"})
            self.assertEqual(multi_summary["pair_indices_shape"], [2, 2])
            self.assertEqual(multi_summary["shared_reconstruction_shape"], [4, 3, 128, 128])
            self.assertFalse(multi_summary["shared_reconstruction_stochastic"])

    def test_preflight_supports_perceptual_without_adversarial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train").mkdir()
            for idx in range(2):
                _write_image(root / "train" / f"sample_{idx}.png")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [
                                {"image": f"train/sample_{idx}.png", "user_id": f"u{idx}"}
                                for idx in range(2)
                            ],
                            "val": [
                                {"image": f"train/sample_{idx}.png", "user_id": f"v{idx}"}
                                for idx in range(2)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            config_path = root / "perceptual_only.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "manifest_path": str(manifest_path),
                            "train_split": "train",
                            "val_split": "val",
                            "image_size": [128, 128],
                            "dry_run_samples": 2,
                        },
                        "weights": {
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": True,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "device": "cpu",
                            "enable_distillation": False,
                            "enable_perceptual": True,
                            "enable_adversarial": False,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "semantic_bandwidth_budget": 10.0,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "ckpt"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = run_forward_dry_run(load_preflight_config(config_path), split="train")

            self.assertTrue(summary["enhancement_status"]["perceptual_enabled"])
            self.assertFalse(summary["enhancement_status"]["adversarial_enabled"])

    def test_preflight_supports_split_specific_image_sizes_and_transforms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train").mkdir()
            for idx in range(2):
                _write_image(root / "train" / f"sample_{idx}.png", size=(192, 192))

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [
                                {"image": f"train/sample_{idx}.png", "user_id": f"u{idx}"}
                                for idx in range(2)
                            ],
                            "val": [
                                {"image": f"train/sample_{idx}.png", "user_id": f"v{idx}"}
                                for idx in range(2)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "split_sizes.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "manifest_path": str(manifest_path),
                            "train_split": "train",
                            "val_split": "val",
                            "image_size": [128, 128],
                            "train_image_size": [160, 160],
                            "val_image_size": [96, 96],
                            "train_transform_mode": "random_crop",
                            "val_transform_mode": "center_crop",
                            "dry_run_samples": 2,
                        },
                        "weights": {
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": False,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "device": "cpu",
                            "enable_distillation": False,
                            "enable_perceptual": False,
                            "enable_adversarial": False,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "semantic_bandwidth_budget": 10.0,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "ckpt"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_preflight_config(config_path)
            train_summary = run_forward_dry_run(config, split="train")
            val_summary = run_forward_dry_run(config, split="val")

            self.assertEqual(train_summary["batch_shape"], [2, 3, 160, 160])
            self.assertEqual(val_summary["batch_shape"], [2, 3, 96, 96])

    def test_preflight_accepts_train_time_runtime_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train").mkdir()
            for idx in range(2):
                _write_image(root / "train" / f"sample_{idx}.png")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [
                                {"image": f"train/sample_{idx}.png", "user_id": f"u{idx}"}
                                for idx in range(2)
                            ],
                            "val": [
                                {"image": f"train/sample_{idx}.png", "user_id": f"v{idx}"}
                                for idx in range(2)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            config_path = root / "sampled_runtime.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "manifest_path": str(manifest_path),
                            "train_split": "train",
                            "val_split": "val",
                            "image_size": [128, 128],
                            "dry_run_samples": 2,
                        },
                        "weights": {
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": False,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "device": "cpu",
                            "enable_distillation": False,
                            "enable_perceptual": False,
                            "enable_adversarial": False,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "semantic_bandwidth_budget": 10.0,
                            "snr_db": 10.0,
                            "train_snr_db_choices": [7.0, 10.0, 13.0],
                            "train_sem_rate_ratio_range": [0.45, 0.55],
                            "train_det_rate_ratio_range": [0.65, 0.85],
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "ckpt"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_preflight_config(config_path)
            validate_preflight_config(config)

            self.assertEqual(config.runtime.train_snr_db_choices, (7.0, 10.0, 13.0))
            self.assertEqual(config.runtime.train_sem_rate_ratio_range, (0.45, 0.55))
            self.assertEqual(config.runtime.train_det_rate_ratio_range, (0.65, 0.85))

    def test_preflight_rejects_invalid_train_det_rate_ratio_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train").mkdir()
            _write_image(root / "train" / "sample.png")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [{"image": "train/sample.png", "user_id": "u1"}],
                            "val": [{"image": "train/sample.png", "user_id": "v1"}],
                        },
                    }
                ),
                encoding="utf-8",
            )

            config_path = root / "invalid_runtime.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "manifest_path": str(manifest_path),
                            "train_split": "train",
                            "val_split": "val",
                            "image_size": [128, 128],
                            "dry_run_samples": 2,
                        },
                        "weights": {
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": False,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "device": "cpu",
                            "enable_distillation": False,
                            "enable_perceptual": False,
                            "enable_adversarial": False,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "semantic_bandwidth_budget": 10.0,
                            "snr_db": 10.0,
                            "train_det_rate_ratio_range": [0.0, 1.1],
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "ckpt"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_preflight_config(config_path)
            with self.assertRaisesRegex(ValueError, "train_det_rate_ratio_range"):
                validate_preflight_config(config)

    def test_preflight_rejects_retired_enable_enhancement_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train").mkdir()
            _write_image(root / "train" / "sample.png")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [{"image": "train/sample.png", "user_id": "u1"}],
                            "val": [{"image": "train/sample.png", "user_id": "u2"}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "legacy.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "manifest_path": str(manifest_path),
                            "train_split": "train",
                            "val_split": "val",
                            "image_size": [128, 128],
                            "dry_run_samples": 2,
                        },
                        "weights": {
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": True,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "device": "cpu",
                            "enable_distillation": False,
                            "enable_enhancement": True,
                            "enable_perceptual": True,
                            "enable_adversarial": True,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "semantic_bandwidth_budget": 10.0,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "ckpt"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "enable_enhancement has been retired"):
                load_preflight_config(config_path)

    def test_preflight_rejects_matched_budget_without_target_cbr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train").mkdir()
            _write_image(root / "train" / "sample.png")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [{"image": "train/sample.png", "user_id": "u1"}],
                            "val": [{"image": "train/sample.png", "user_id": "u2"}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "matched_budget_missing_target.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "manifest_path": str(manifest_path),
                            "train_split": "train",
                            "val_split": "val",
                            "image_size": [128, 128],
                            "dry_run_samples": 2,
                        },
                        "weights": {
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": False,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "device": "cpu",
                            "operating_mode": "matched_budget",
                            "enable_distillation": False,
                            "enable_perceptual": False,
                            "enable_adversarial": False,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "semantic_bandwidth_budget": 10.0,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "ckpt"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_preflight_config(config_path)
            with self.assertRaisesRegex(RuntimeError, "target_effective_cbr"):
                validate_preflight_config(config)


if __name__ == "__main__":
    unittest.main()

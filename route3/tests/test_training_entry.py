from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reselect_route3_best import reselect_best_checkpoint
from scripts.route3_train import (
    _apply_trainable_prefixes,
    _extract_selection_metric,
    _is_better_selection_metric,
    _initialize_model_from_checkpoint,
    _load_resume_history,
    load_training_config,
    run_training,
    validate_training_config,
)
from scripts.route3_preflight import build_runtime_model


def _write_image(path: Path, size: tuple[int, int] = (128, 128)) -> None:
    tensor = (torch.rand(size[0], size[1], 3) * 255).to(torch.uint8).numpy()
    Image.fromarray(tensor).save(path)


def _write_minimal_single_user_config(
    root: Path,
    *,
    enable_perceptual: bool = False,
    enable_adversarial: bool = False,
) -> Path:
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({"root": str(root), "splits": {"train": [], "val": []}}), encoding="utf-8")

    config_path = root / "single_user_model.json"
    config_path.write_text(
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
                    "teacher_checkpoint": None,
                    "vgg_checkpoint": None,
                    "allow_untrained_vgg": True,
                },
                "runtime": {
                    "mode": "single_user",
                    "device": "cpu",
                    "semantic_channel_type": "awgn",
                    "detail_channel_type": "awgn",
                    "enable_distillation": False,
                    "enable_perceptual": enable_perceptual,
                    "enable_adversarial": enable_adversarial,
                    "sem_rate_ratio": 0.5,
                    "det_rate_ratio": 0.75,
                    "semantic_bandwidth_budget": 10.0,
                    "snr_db": 10.0,
                },
                "artifacts": {
                    "output_dir": str(root / "outputs"),
                    "checkpoint_dir": str(root / "outputs" / "checkpoints"),
                    "report_dir": str(root / "outputs" / "reports"),
                },
                "trainer": {
                    "epochs": 1,
                    "batch_size": 2,
                    "num_workers": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


class TrainingEntryTests(unittest.TestCase):
    def test_load_training_config_preserves_legacy_decoder_loss_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"root": str(root), "splits": {"train": [], "val": []}}), encoding="utf-8")
            config_path = root / "legacy_train.json"
            config_path.write_text(
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
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": True,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "device": "cpu",
                            "semantic_channel_type": "awgn",
                            "detail_channel_type": "awgn",
                            "enable_distillation": False,
                            "enable_perceptual": True,
                            "enable_adversarial": True,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "checkpoints"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_training_config(config_path)

            self.assertAlmostEqual(config.trainer.decoder_ms_ssim_weight, 0.0)
            self.assertAlmostEqual(config.trainer.decoder_l1_weight, 1.0)
            self.assertAlmostEqual(config.trainer.decoder_mse_weight, 1.0)
            self.assertAlmostEqual(config.trainer.decoder_residual_weight, 0.25)

    def test_load_training_config_preserves_legacy_decode_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"root": str(root), "splits": {"train": [], "val": []}}), encoding="utf-8")
            config_path = root / "legacy_decode.json"
            config_path.write_text(
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
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": True,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "device": "cpu",
                            "semantic_channel_type": "awgn",
                            "detail_channel_type": "awgn",
                            "enable_distillation": False,
                            "enable_perceptual": True,
                            "enable_adversarial": True,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "checkpoints"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_training_config(config_path)

            self.assertFalse(config.trainer.train_decode_stochastic)
            self.assertFalse(config.trainer.val_decode_stochastic)

    def test_load_training_config_supports_decode_stochastic_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "train.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "manifest_path": str(root / "manifest.json"),
                            "train_split": "train",
                            "val_split": "val",
                            "image_size": [128, 128],
                            "dry_run_samples": 4,
                        },
                        "weights": {
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": True,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "device": "cpu",
                            "semantic_channel_type": "awgn",
                            "detail_channel_type": "awgn",
                            "enable_distillation": False,
                            "enable_perceptual": False,
                            "enable_adversarial": False,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "checkpoints"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0,
                            "phase1_epochs": 2,
                            "phase2_epochs": 3,
                            "perceptual_weight_max": 0.7,
                            "perceptual_loss_scale": 0.04,
                            "discriminator_learning_rate": 0.00005,
                            "adversarial_weight": 0.15,
                            "adversarial_ramp_epochs": 4,
                            "decoder_ms_ssim_weight": 1.25,
                            "decoder_l1_weight": 0.2,
                            "decoder_mse_weight": 0.3,
                            "decoder_residual_weight": 0.4,
                            "train_decode_stochastic": False,
                            "val_decode_stochastic": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"root": str(root), "splits": {"train": [], "val": []}}), encoding="utf-8")

            config = load_training_config(config_path)

            self.assertEqual(config.trainer.phase1_epochs, 2)
            self.assertEqual(config.trainer.phase2_epochs, 3)
            self.assertAlmostEqual(config.trainer.perceptual_weight_max, 0.7)
            self.assertAlmostEqual(config.trainer.perceptual_loss_scale, 0.04)
            self.assertAlmostEqual(config.trainer.discriminator_learning_rate, 0.00005)
            self.assertAlmostEqual(config.trainer.adversarial_weight, 0.15)
            self.assertEqual(config.trainer.adversarial_ramp_epochs, 4)
            self.assertAlmostEqual(config.trainer.decoder_ms_ssim_weight, 1.25)
            self.assertAlmostEqual(config.trainer.decoder_l1_weight, 0.2)
            self.assertAlmostEqual(config.trainer.decoder_mse_weight, 0.3)
            self.assertAlmostEqual(config.trainer.decoder_residual_weight, 0.4)
            self.assertFalse(config.trainer.train_decode_stochastic)
            self.assertTrue(config.trainer.val_decode_stochastic)

    def test_load_training_config_rejects_retired_trainer_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"root": str(root), "splits": {"train": [], "val": []}}), encoding="utf-8")
            config_path = root / "retired_trainer_key.json"
            config_path.write_text(
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
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": True,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "device": "cpu",
                            "semantic_channel_type": "awgn",
                            "detail_channel_type": "awgn",
                            "enable_distillation": False,
                            "enable_perceptual": True,
                            "enable_adversarial": True,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "checkpoints"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0,
                            "adversarial_warmup_epochs": 20,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "adversarial_warmup_epochs has been retired"):
                load_training_config(config_path)

    def test_load_training_config_rejects_unknown_trainer_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"root": str(root), "splits": {"train": [], "val": []}}), encoding="utf-8")
            config_path = root / "unknown_trainer_key.json"
            config_path.write_text(
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
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": True,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "device": "cpu",
                            "semantic_channel_type": "awgn",
                            "detail_channel_type": "awgn",
                            "enable_distillation": False,
                            "enable_perceptual": False,
                            "enable_adversarial": False,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "checkpoints"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0,
                            "mystery_flag": 123,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unsupported trainer config field\\(s\\): mystery_flag"):
                load_training_config(config_path)

    def test_validate_training_config_rejects_schedule_without_adversarial_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            _write_image(root / "images" / "sample_0.png")
            _write_image(root / "images" / "sample_1.png")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [{"image": "images/sample_0.png", "user_id": "train_0"}],
                            "val": [{"image": "images/sample_1.png", "user_id": "val_0"}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "schedule_invalid.json"
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
                            "semantic_channel_type": "awgn",
                            "detail_channel_type": "awgn",
                            "enable_distillation": False,
                            "enable_perceptual": True,
                            "enable_adversarial": True,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "semantic_bandwidth_budget": 10.0,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "checkpoints"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                        "trainer": {
                            "epochs": 4,
                            "batch_size": 1,
                            "num_workers": 0,
                            "phase1_epochs": 2,
                            "phase2_epochs": 2,
                            "adversarial_weight": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = load_training_config(config_path)
            bad_config = type(config)(
                base=config.base,
                trainer=type(config.trainer)(
                    **{
                        **config.trainer.__dict__,
                        "epochs": 4,
                    }
                ),
                config_path=config.config_path,
            )

            with self.assertRaisesRegex(ValueError, "leaves no epoch for the adversarial stage"):
                validate_training_config(bad_config)

    def test_apply_trainable_prefixes_restricts_stage_b_student_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "train.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "manifest_path": str(root / "manifest.json"),
                            "train_split": "train",
                            "val_split": "val",
                            "image_size": [128, 128],
                            "dry_run_samples": 4,
                        },
                        "weights": {
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": True,
                        },
                        "runtime": {
                            "mode": "multi_user",
                            "device": "cpu",
                            "semantic_channel_type": "awgn",
                            "detail_channel_type": "awgn",
                            "enable_distillation": False,
                            "enable_perceptual": True,
                            "enable_adversarial": True,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "semantic_bandwidth_budget": 10.0,
                            "snr_db": 10.0,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "checkpoints"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 4,
                            "num_workers": 0,
                            "seed": 123,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 1.0,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "trainable_prefixes": ["single_user_model.encoder.semantic_branch"],
                            "init_model_checkpoint": None,
                            "resume_checkpoint": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"root": str(root), "splits": {"train": [], "val": []}}), encoding="utf-8")

            config = load_training_config(config_path)
            model = build_runtime_model(config.base)
            _apply_trainable_prefixes(model, config.trainer.trainable_prefixes)

            semantic_trainable = sum(
                p.numel()
                for name, p in model.named_parameters()
                if p.requires_grad and name.startswith("single_user_model.encoder.semantic_branch")
            )
            detail_trainable = sum(
                p.numel()
                for name, p in model.named_parameters()
                if p.requires_grad and name.startswith("single_user_model.encoder.detail_branch")
            )
            decoder_trainable = sum(
                p.numel()
                for name, p in model.named_parameters()
                if p.requires_grad and name.startswith("single_user_model.decoder")
            )
            discriminator_trainable = sum(
                p.numel()
                for name, p in model.named_parameters()
                if p.requires_grad and name.startswith("single_user_model.enhancer.discriminator")
            )

            self.assertGreater(semantic_trainable, 0)
            self.assertEqual(detail_trainable, 0)
            self.assertEqual(decoder_trainable, 0)
            self.assertEqual(discriminator_trainable, 0)

    def test_multi_user_training_entry_runs_single_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            for idx in range(8):
                _write_image(root / "images" / f"sample_{idx}.png")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"train_{idx}"}
                                for idx in range(4)
                            ],
                            "val": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"val_{idx}"}
                                for idx in range(4, 8)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            config_path = root / "train.json"
            config_path.write_text(
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
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": False,
                        },
                        "runtime": {
                            "mode": "multi_user",
                            "device": "cpu",
                            "semantic_channel_type": "awgn",
                            "detail_channel_type": "awgn",
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
                            "checkpoint_dir": str(root / "outputs" / "checkpoints"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 4,
                            "num_workers": 0,
                            "seed": 123,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 1.0,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "resume_checkpoint": None,
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = run_training(load_training_config(config_path))
            history_payload = json.loads((root / "outputs" / "reports" / "train_history.json").read_text(encoding="utf-8"))
            epoch_record = history_payload["history"][0]

            self.assertEqual(summary["mode"], "multi_user")
            self.assertEqual(summary["epochs_completed"], 1)
            self.assertIn("final_stage", summary)
            self.assertEqual(summary["final_stage"]["name"], "adversarial_refinement")
            self.assertEqual(epoch_record["train"]["stage"]["name"], "adversarial_refinement")
            self.assertEqual(epoch_record["validation"]["stage"]["name"], "adversarial_refinement")
            self.assertTrue(Path(summary["latest_checkpoint"]).exists())
            self.assertTrue(Path(summary["best_checkpoint"]).exists())
            self.assertTrue(Path(summary["best_psnr_checkpoint"]).exists())
            self.assertTrue(Path(summary["config_snapshot_path"]).exists())
            self.assertTrue(summary["config_fingerprint"])
            self.assertTrue((root / "outputs" / "reports" / "train_history.json").exists())
            self.assertTrue((root / "outputs" / "reports" / "phase8_epoch_001.md").exists())

    def test_multi_user_training_entry_supports_init_model_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            for idx in range(8):
                _write_image(root / "images" / f"sample_{idx}.png")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"train_{idx}"}
                                for idx in range(4)
                            ],
                            "val": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"val_{idx}"}
                                for idx in range(4, 8)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            base_config = {
                "dataset": {
                    "manifest_path": str(manifest_path),
                    "train_split": "train",
                    "val_split": "val",
                    "image_size": [128, 128],
                    "dry_run_samples": 4,
                },
                "weights": {
                    "teacher_checkpoint": None,
                    "vgg_checkpoint": None,
                    "allow_untrained_vgg": False,
                },
                "runtime": {
                    "mode": "multi_user",
                    "device": "cpu",
                    "semantic_channel_type": "awgn",
                    "detail_channel_type": "awgn",
                    "enable_distillation": False,
                    "enable_perceptual": False,
                    "enable_adversarial": False,
                    "sem_rate_ratio": 0.5,
                    "det_rate_ratio": 0.75,
                    "semantic_bandwidth_budget": 10.0,
                    "snr_db": 10.0,
                },
            }

            source_config_path = root / "train_source.json"
            source_config_path.write_text(
                json.dumps(
                    {
                        **base_config,
                        "artifacts": {
                            "output_dir": str(root / "outputs_source"),
                            "checkpoint_dir": str(root / "outputs_source" / "checkpoints"),
                            "report_dir": str(root / "outputs_source" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 4,
                            "num_workers": 0,
                            "seed": 123,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 1.0,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "init_model_checkpoint": None,
                            "resume_checkpoint": None,
                        },
                    }
                ),
                encoding="utf-8",
            )

            source_summary = run_training(load_training_config(source_config_path))

            init_config_path = root / "train_init.json"
            init_config_path.write_text(
                json.dumps(
                    {
                        **base_config,
                        "artifacts": {
                            "output_dir": str(root / "outputs_init"),
                            "checkpoint_dir": str(root / "outputs_init" / "checkpoints"),
                            "report_dir": str(root / "outputs_init" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 4,
                            "num_workers": 0,
                            "seed": 456,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 1.0,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "init_model_checkpoint": str(source_summary["best_checkpoint"]),
                            "resume_checkpoint": None,
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = run_training(load_training_config(init_config_path))

            self.assertEqual(summary["mode"], "multi_user")
            self.assertEqual(summary["epochs_completed"], 1)
            self.assertTrue(Path(summary["latest_checkpoint"]).exists())
            self.assertTrue(Path(summary["best_checkpoint"]).exists())
            self.assertTrue(Path(summary["best_psnr_checkpoint"]).exists())

    def test_single_user_training_entry_supports_sampled_runtime_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            for idx in range(8):
                _write_image(root / "images" / f"sample_{idx}.png")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"train_{idx}"}
                                for idx in range(4)
                            ],
                            "val": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"val_{idx}"}
                                for idx in range(4, 8)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            config_path = root / "train_sampled.json"
            config_path.write_text(
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
                            "teacher_checkpoint": None,
                            "vgg_checkpoint": None,
                            "allow_untrained_vgg": False,
                        },
                        "runtime": {
                            "mode": "single_user",
                            "device": "cpu",
                            "semantic_channel_type": "awgn",
                            "detail_channel_type": "awgn",
                            "enable_distillation": False,
                            "enable_perceptual": False,
                            "enable_adversarial": False,
                            "sem_rate_ratio": 0.5,
                            "det_rate_ratio": 0.75,
                            "semantic_bandwidth_budget": 10.0,
                            "snr_db": 10.0,
                            "train_snr_db_choices": [7.0, 10.0, 13.0],
                            "train_det_rate_ratio_range": [0.65, 0.85],
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "outputs" / "checkpoints"),
                            "report_dir": str(root / "outputs" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0,
                            "seed": 123,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 0.5,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "resume_checkpoint": None,
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = run_training(load_training_config(config_path))
            history_payload = json.loads((root / "outputs" / "reports" / "train_history.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["mode"], "single_user")
            self.assertEqual(summary["epochs_completed"], 1)
            self.assertEqual(len(history_payload["history"]), 1)
            self.assertEqual(summary["final_stage"]["name"], "adversarial_refinement")
            self.assertEqual(history_payload["history"][0]["train"]["stage"]["name"], "adversarial_refinement")
            self.assertEqual(history_payload["history"][0]["validation"]["stage"]["name"], "adversarial_refinement")
            self.assertTrue(Path(summary["latest_checkpoint"]).exists())
            self.assertTrue(Path(summary["best_checkpoint"]).exists())
            self.assertTrue(Path(summary["best_psnr_checkpoint"]).exists())

    def test_single_user_init_checkpoint_allows_adding_discriminator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            for idx in range(8):
                _write_image(root / "images" / f"sample_{idx}.png")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"train_{idx}"}
                                for idx in range(4)
                            ],
                            "val": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"val_{idx}"}
                                for idx in range(4, 8)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            base_config = {
                "dataset": {
                    "manifest_path": str(manifest_path),
                    "train_split": "train",
                    "val_split": "val",
                    "image_size": [128, 128],
                    "dry_run_samples": 4,
                },
                "weights": {
                    "teacher_checkpoint": None,
                    "vgg_checkpoint": None,
                    "allow_untrained_vgg": True,
                },
                "runtime": {
                    "mode": "single_user",
                    "device": "cpu",
                    "semantic_channel_type": "awgn",
                    "detail_channel_type": "awgn",
                    "enable_distillation": False,
                    "enable_perceptual": False,
                    "enable_adversarial": False,
                    "sem_rate_ratio": 0.5,
                    "det_rate_ratio": 0.75,
                    "semantic_bandwidth_budget": 10.0,
                    "snr_db": 10.0,
                },
            }

            source_config_path = root / "single_source.json"
            source_config_path.write_text(
                json.dumps(
                    {
                        **base_config,
                        "artifacts": {
                            "output_dir": str(root / "outputs_source"),
                            "checkpoint_dir": str(root / "outputs_source" / "checkpoints"),
                            "report_dir": str(root / "outputs_source" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0,
                            "seed": 123,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 0.5,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "init_model_checkpoint": None,
                            "resume_checkpoint": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
            source_summary = run_training(load_training_config(source_config_path))

            init_config_path = root / "single_init_adv.json"
            init_config_path.write_text(
                json.dumps(
                    {
                        **base_config,
                        "runtime": {
                            **base_config["runtime"],
                            "enable_perceptual": True,
                            "enable_adversarial": True,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs_init"),
                            "checkpoint_dir": str(root / "outputs_init" / "checkpoints"),
                            "report_dir": str(root / "outputs_init" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0,
                            "seed": 456,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 0.5,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "init_model_checkpoint": str(source_summary["best_checkpoint"]),
                            "resume_checkpoint": None,
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = run_training(load_training_config(init_config_path))

            self.assertEqual(summary["mode"], "single_user")
            self.assertEqual(summary["epochs_completed"], 1)
            self.assertTrue(Path(summary["latest_checkpoint"]).exists())
            self.assertTrue(Path(summary["best_checkpoint"]).exists())
            self.assertTrue(Path(summary["best_psnr_checkpoint"]).exists())

    def test_init_model_checkpoint_allows_ignoring_old_discriminator_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = load_training_config(
                _write_minimal_single_user_config(
                    root,
                    enable_perceptual=True,
                    enable_adversarial=True,
                )
            )
            model = build_runtime_model(config.base)
            state = {key: value.clone() for key, value in model.state_dict().items()}
            state["enhancer.discriminator.downsample_blocks.0.0.weight"] = torch.randn(16, 6, 4, 4)

            checkpoint_path = root / "old_disc_init.pt"
            torch.save({"model": state}, checkpoint_path)

            _initialize_model_from_checkpoint(model, checkpoint_path)

    def test_single_user_init_checkpoint_rejects_unrelated_missing_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = load_training_config(
                _write_minimal_single_user_config(
                    root,
                    enable_perceptual=True,
                    enable_adversarial=True,
                )
            )
            model = build_runtime_model(config.base)
            state = model.state_dict()
            non_migration_key = next(
                key
                for key in state
                if key.startswith("encoder.") or key.startswith("decoder.initial.")
            )
            broken_state = {key: value.clone() for key, value in state.items() if key != non_migration_key}

            checkpoint_path = root / "broken_init.pt"
            torch.save({"model": broken_state}, checkpoint_path)

            with self.assertRaisesRegex(RuntimeError, "missing non-teacher weights"):
                _initialize_model_from_checkpoint(model, checkpoint_path)

    def test_resume_allows_clearing_init_model_checkpoint_after_branching(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            for idx in range(8):
                _write_image(root / "images" / f"sample_{idx}.png")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"train_{idx}"}
                                for idx in range(4)
                            ],
                            "val": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"val_{idx}"}
                                for idx in range(4, 8)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            shared_config = {
                "dataset": {
                    "manifest_path": str(manifest_path),
                    "train_split": "train",
                    "val_split": "val",
                    "image_size": [128, 128],
                    "dry_run_samples": 4,
                },
                "weights": {
                    "teacher_checkpoint": None,
                    "vgg_checkpoint": None,
                    "allow_untrained_vgg": True,
                },
                "runtime": {
                    "mode": "single_user",
                    "device": "cpu",
                    "semantic_channel_type": "awgn",
                    "detail_channel_type": "awgn",
                    "enable_distillation": False,
                    "enable_perceptual": False,
                    "enable_adversarial": False,
                    "sem_rate_ratio": 0.5,
                    "det_rate_ratio": 0.75,
                    "semantic_bandwidth_budget": 10.0,
                    "snr_db": 10.0,
                },
            }

            source_config_path = root / "source_train.json"
            source_config_path.write_text(
                json.dumps(
                    {
                        **shared_config,
                        "artifacts": {
                            "output_dir": str(root / "source_outputs"),
                            "checkpoint_dir": str(root / "source_outputs" / "checkpoints"),
                            "report_dir": str(root / "source_outputs" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0,
                            "seed": 123,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 1.0,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "resume_checkpoint": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
            source_summary = run_training(load_training_config(source_config_path))

            first_run_config_path = root / "branch_first.json"
            first_run_config_path.write_text(
                json.dumps(
                    {
                        **shared_config,
                        "artifacts": {
                            "output_dir": str(root / "branch_outputs"),
                            "checkpoint_dir": str(root / "branch_outputs" / "checkpoints"),
                            "report_dir": str(root / "branch_outputs" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0,
                            "seed": 123,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 1.0,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "init_model_checkpoint": str(source_summary["best_checkpoint"]),
                            "resume_checkpoint": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
            first_summary = run_training(load_training_config(first_run_config_path))

            resumed_config_path = root / "branch_resume.json"
            resumed_config_path.write_text(
                json.dumps(
                    {
                        **shared_config,
                        "artifacts": {
                            "output_dir": str(root / "branch_outputs"),
                            "checkpoint_dir": str(root / "branch_outputs" / "checkpoints"),
                            "report_dir": str(root / "branch_outputs" / "reports"),
                        },
                        "trainer": {
                            "epochs": 2,
                            "batch_size": 2,
                            "num_workers": 0,
                            "seed": 123,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 1.0,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "init_model_checkpoint": None,
                            "resume_checkpoint": str(first_summary["latest_checkpoint"]),
                        },
                    }
                ),
                encoding="utf-8",
            )

            resumed_summary = run_training(load_training_config(resumed_config_path))

            self.assertEqual(resumed_summary["epochs_completed"], 2)
            self.assertTrue(Path(resumed_summary["latest_checkpoint"]).exists())

    def test_single_user_init_checkpoint_allows_adding_distillation_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            for idx in range(8):
                _write_image(root / "images" / f"sample_{idx}.png")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"train_{idx}"}
                                for idx in range(4)
                            ],
                            "val": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"val_{idx}"}
                                for idx in range(4, 8)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            base_config = {
                "dataset": {
                    "manifest_path": str(manifest_path),
                    "train_split": "train",
                    "val_split": "val",
                    "image_size": [128, 128],
                    "dry_run_samples": 4,
                },
                "weights": {
                    "teacher_checkpoint": None,
                    "vgg_checkpoint": None,
                    "allow_untrained_vgg": True,
                },
                "runtime": {
                    "mode": "single_user",
                    "device": "cpu",
                    "semantic_channel_type": "awgn",
                    "detail_channel_type": "awgn",
                    "enable_distillation": False,
                    "enable_perceptual": False,
                    "enable_adversarial": False,
                    "sem_rate_ratio": 0.5,
                    "det_rate_ratio": 0.75,
                    "semantic_bandwidth_budget": 10.0,
                    "snr_db": 10.0,
                },
            }

            source_config_path = root / "single_source_distill.json"
            source_config_path.write_text(
                json.dumps(
                    {
                        **base_config,
                        "artifacts": {
                            "output_dir": str(root / "outputs_source"),
                            "checkpoint_dir": str(root / "outputs_source" / "checkpoints"),
                            "report_dir": str(root / "outputs_source" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0,
                            "seed": 123,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 0.5,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "init_model_checkpoint": None,
                            "resume_checkpoint": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
            source_summary = run_training(load_training_config(source_config_path))

            source_checkpoint = torch.load(source_summary["best_checkpoint"], map_location="cpu")
            teacher_state = {
                key[len("encoder.semantic_branch."):]: value
                for key, value in source_checkpoint["model"].items()
                if key.startswith("encoder.semantic_branch.")
            }
            teacher_path = root / "teacher_semantic.pt"
            torch.save(teacher_state, teacher_path)

            init_config_path = root / "single_init_distill.json"
            init_config_path.write_text(
                json.dumps(
                    {
                        **base_config,
                        "weights": {
                            **base_config["weights"],
                            "teacher_checkpoint": str(teacher_path),
                        },
                        "runtime": {
                            **base_config["runtime"],
                            "enable_distillation": True,
                        },
                        "artifacts": {
                            "output_dir": str(root / "outputs_init"),
                            "checkpoint_dir": str(root / "outputs_init" / "checkpoints"),
                            "report_dir": str(root / "outputs_init" / "reports"),
                        },
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0,
                            "seed": 456,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 0.5,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "trainable_prefixes": [
                                "encoder.semantic_branch",
                                "semantic_tx",
                            ],
                            "init_model_checkpoint": str(source_summary["best_checkpoint"]),
                            "resume_checkpoint": None,
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = run_training(load_training_config(init_config_path))

            self.assertEqual(summary["mode"], "single_user")
            self.assertEqual(summary["epochs_completed"], 1)
            self.assertTrue(Path(summary["latest_checkpoint"]).exists())
            self.assertTrue(Path(summary["best_checkpoint"]).exists())
            self.assertTrue(Path(summary["best_psnr_checkpoint"]).exists())

    def test_single_user_resume_preserves_existing_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            for idx in range(8):
                _write_image(root / "images" / f"sample_{idx}.png")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"train_{idx}"}
                                for idx in range(4)
                            ],
                            "val": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"val_{idx}"}
                                for idx in range(4, 8)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            shared_config = {
                "dataset": {
                    "manifest_path": str(manifest_path),
                    "train_split": "train",
                    "val_split": "val",
                    "image_size": [128, 128],
                    "dry_run_samples": 4,
                },
                "weights": {
                    "teacher_checkpoint": None,
                    "vgg_checkpoint": None,
                    "allow_untrained_vgg": False,
                },
                "runtime": {
                    "mode": "single_user",
                    "device": "cpu",
                    "semantic_channel_type": "awgn",
                    "detail_channel_type": "awgn",
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
                    "checkpoint_dir": str(root / "outputs" / "checkpoints"),
                    "report_dir": str(root / "outputs" / "reports"),
                },
            }

            first_config_path = root / "train_first.json"
            first_config_path.write_text(
                json.dumps(
                    {
                        **shared_config,
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0,
                            "seed": 123,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 1.0,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "resume_checkpoint": None,
                        },
                    }
                ),
                encoding="utf-8",
            )

            first_summary = run_training(load_training_config(first_config_path))

            second_config_path = root / "train_resume.json"
            second_config_path.write_text(
                json.dumps(
                    {
                        **shared_config,
                        "trainer": {
                            "epochs": 2,
                            "batch_size": 2,
                            "num_workers": 0,
                            "seed": 123,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 1.0,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "resume_checkpoint": str(first_summary["latest_checkpoint"]),
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = run_training(load_training_config(second_config_path))
            history_payload = json.loads((root / "outputs" / "reports" / "train_history.json").read_text(encoding="utf-8"))
            history = history_payload["history"]

            self.assertEqual(summary["mode"], "single_user")
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertEqual([record["epoch"] for record in history], [1, 2])
            self.assertTrue(Path(summary["latest_checkpoint"]).exists())
            self.assertTrue(Path(summary["best_checkpoint"]).exists())
            self.assertTrue(Path(summary["best_psnr_checkpoint"]).exists())

    def test_single_user_resume_matches_uninterrupted_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            for idx in range(8):
                _write_image(root / "images" / f"sample_{idx}.png")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"train_{idx}"}
                                for idx in range(4)
                            ],
                            "val": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"val_{idx}"}
                                for idx in range(4, 8)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            def _payload(epochs: int, output_root: Path, resume_checkpoint: str | None = None) -> dict[str, Any]:
                return {
                    "dataset": {
                        "manifest_path": str(manifest_path),
                        "train_split": "train",
                        "val_split": "val",
                        "image_size": [128, 128],
                        "dry_run_samples": 4,
                    },
                    "weights": {
                        "teacher_checkpoint": None,
                        "vgg_checkpoint": None,
                        "allow_untrained_vgg": False,
                    },
                    "runtime": {
                        "mode": "single_user",
                        "device": "cpu",
                        "semantic_channel_type": "awgn",
                        "detail_channel_type": "awgn",
                        "enable_distillation": False,
                        "enable_perceptual": False,
                        "enable_adversarial": False,
                        "sem_rate_ratio": 0.5,
                        "det_rate_ratio": 0.75,
                        "semantic_bandwidth_budget": 10.0,
                        "snr_db": 10.0,
                    },
                    "artifacts": {
                        "output_dir": str(output_root),
                        "checkpoint_dir": str(output_root / "checkpoints"),
                        "report_dir": str(output_root / "reports"),
                    },
                    "trainer": {
                        "epochs": epochs,
                        "batch_size": 2,
                        "num_workers": 0,
                        "seed": 123,
                        "learning_rate": 0.0001,
                        "weight_decay": 0.0,
                        "distillation_weight": 1.0,
                        "enhancement_weight": 1.0,
                        "max_grad_norm": 1.0,
                        "log_every_steps": 1,
                        "validate_every_epochs": 1,
                        "checkpoint_every_epochs": 1,
                        "max_train_batches": 2,
                        "max_val_batches": 2,
                        "resume_checkpoint": resume_checkpoint,
                    },
                }

            uninterrupted_config_path = root / "continuous.json"
            uninterrupted_config_path.write_text(
                json.dumps(_payload(epochs=2, output_root=root / "continuous")),
                encoding="utf-8",
            )
            uninterrupted_summary = run_training(load_training_config(uninterrupted_config_path))
            uninterrupted_history = json.loads(
                (root / "continuous" / "reports" / "train_history.json").read_text(encoding="utf-8")
            )["history"]

            first_config_path = root / "resume_first.json"
            first_config_path.write_text(
                json.dumps(_payload(epochs=1, output_root=root / "resume")),
                encoding="utf-8",
            )
            first_summary = run_training(load_training_config(first_config_path))

            resumed_config_path = root / "resume_second.json"
            resumed_config_path.write_text(
                json.dumps(
                    _payload(
                        epochs=2,
                        output_root=root / "resume",
                        resume_checkpoint=str(first_summary["latest_checkpoint"]),
                    )
                ),
                encoding="utf-8",
            )
            resumed_summary = run_training(load_training_config(resumed_config_path))
            resumed_history = json.loads((root / "resume" / "reports" / "train_history.json").read_text(encoding="utf-8"))[
                "history"
            ]

            self.assertEqual(uninterrupted_summary["epochs_completed"], 2)
            self.assertEqual(resumed_summary["epochs_completed"], 2)
            self.assertEqual([record["epoch"] for record in uninterrupted_history], [1, 2])
            self.assertEqual([record["epoch"] for record in resumed_history], [1, 2])

            uninterrupted_epoch2 = uninterrupted_history[1]
            resumed_epoch2 = resumed_history[1]
            self.assertAlmostEqual(uninterrupted_epoch2["train"]["total_loss"], resumed_epoch2["train"]["total_loss"], places=6)
            self.assertAlmostEqual(
                uninterrupted_epoch2["validation"]["terms"]["validation_total"],
                resumed_epoch2["validation"]["terms"]["validation_total"],
                places=6,
            )
            self.assertAlmostEqual(
                uninterrupted_epoch2["validation"]["metrics"]["psnr"],
                resumed_epoch2["validation"]["metrics"]["psnr"],
                places=6,
            )

    def test_resume_rejects_changed_runtime_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            for idx in range(8):
                _write_image(root / "images" / f"sample_{idx}.png")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"train_{idx}"}
                                for idx in range(4)
                            ],
                            "val": [
                                {"image": f"images/sample_{idx}.png", "user_id": f"val_{idx}"}
                                for idx in range(4, 8)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            shared_config = {
                "dataset": {
                    "manifest_path": str(manifest_path),
                    "train_split": "train",
                    "val_split": "val",
                    "image_size": [128, 128],
                    "dry_run_samples": 4,
                },
                "weights": {
                    "teacher_checkpoint": None,
                    "vgg_checkpoint": None,
                    "allow_untrained_vgg": False,
                },
                "runtime": {
                    "mode": "single_user",
                    "device": "cpu",
                    "semantic_channel_type": "awgn",
                    "detail_channel_type": "awgn",
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
                    "checkpoint_dir": str(root / "outputs" / "checkpoints"),
                    "report_dir": str(root / "outputs" / "reports"),
                },
            }

            first_config_path = root / "train_first.json"
            first_config_path.write_text(
                json.dumps(
                    {
                        **shared_config,
                        "trainer": {
                            "epochs": 1,
                            "batch_size": 2,
                            "num_workers": 0,
                            "seed": 123,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 1.0,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "resume_checkpoint": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
            first_summary = run_training(load_training_config(first_config_path))

            resume_config_path = root / "train_resume_mismatch.json"
            mismatched_runtime = dict(shared_config["runtime"])
            mismatched_runtime["det_rate_ratio"] = 0.5
            resume_config_path.write_text(
                json.dumps(
                    {
                        **shared_config,
                        "runtime": mismatched_runtime,
                        "trainer": {
                            "epochs": 2,
                            "batch_size": 2,
                            "num_workers": 0,
                            "seed": 123,
                            "learning_rate": 0.0001,
                            "weight_decay": 0.0,
                            "distillation_weight": 1.0,
                            "enhancement_weight": 1.0,
                            "max_grad_norm": 1.0,
                            "log_every_steps": 1,
                            "validate_every_epochs": 1,
                            "checkpoint_every_epochs": 1,
                            "max_train_batches": 1,
                            "max_val_batches": 1,
                            "resume_checkpoint": str(first_summary["latest_checkpoint"]),
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Resume configuration mismatch"):
                run_training(load_training_config(resume_config_path))

    def test_resume_history_prefers_higher_validation_psnr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            (report_dir / "train_history.json").write_text(
                json.dumps(
                    {
                        "config_path": "dummy.json",
                        "mode": "single_user",
                        "history": [
                            {
                                "epoch": 1,
                                "validation": {
                                    "terms": {"validation_total": 1.0},
                                    "metrics": {"psnr": 20.0},
                                },
                                "best_checkpoint": "epoch1.pt",
                            },
                            {
                                "epoch": 2,
                                "validation": {
                                    "terms": {"validation_total": 2.0},
                                    "metrics": {"psnr": 25.0},
                                },
                                "best_checkpoint": "epoch2.pt",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            history, best_metric, best_checkpoint, best_psnr_checkpoint, best_epoch, best_metric_name = _load_resume_history(
                report_dir=report_dir,
                start_epoch=3,
                mode="single_user",
            )

            self.assertEqual(len(history), 2)
            self.assertEqual(best_metric, 25.0)
            self.assertEqual(best_checkpoint, "epoch2.pt")
            self.assertEqual(best_psnr_checkpoint, "epoch2.pt")
            self.assertEqual(best_epoch, 2)
            self.assertEqual(best_metric_name, "psnr")

    def test_reselect_best_checkpoint_uses_highest_psnr_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_dir = root / "reports"
            checkpoint_dir = root / "checkpoints"
            report_dir.mkdir()
            checkpoint_dir.mkdir()

            torch.save({"model": {}, "extra_state": {"epoch": 1}}, checkpoint_dir / "epoch_001.pt")
            torch.save({"model": {}, "extra_state": {"epoch": 2}}, checkpoint_dir / "epoch_002.pt")
            (report_dir / "train_history.json").write_text(
                json.dumps(
                    {
                        "config_path": "dummy.json",
                        "mode": "single_user",
                        "history": [
                            {
                                "epoch": 1,
                                "validation": {"metrics": {"psnr": 20.0}},
                            },
                            {
                                "epoch": 2,
                                "validation": {"metrics": {"psnr": 24.0}},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = reselect_best_checkpoint(report_dir=report_dir, checkpoint_dir=checkpoint_dir)
            selected_epoch = torch.load(checkpoint_dir / "best.pt", map_location="cpu")["extra_state"]["epoch"]
            selected_psnr_epoch = torch.load(checkpoint_dir / "best_psnr.pt", map_location="cpu")["extra_state"]["epoch"]

            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(result["selection_metric_name"], "psnr")
            self.assertEqual(selected_epoch, 2)
            self.assertEqual(selected_psnr_epoch, 2)
            self.assertTrue((report_dir / "best_reselection.json").exists())

    def test_selection_metric_prefers_budget_feasible_single_user_checkpoint(self) -> None:
        over_budget = _extract_selection_metric(
            {
                "validation": {
                    "metrics": {"psnr": 28.0},
                    "budget": {
                        "operating_mode": "matched_budget",
                        "cbr_relative_gap": 0.20,
                        "target_effective_cbr_tolerance": 0.05,
                        "within_target_tolerance": False,
                    },
                }
            },
            mode="single_user",
        )
        in_budget = _extract_selection_metric(
            {
                "validation": {
                    "metrics": {"psnr": 24.0},
                    "budget": {
                        "operating_mode": "matched_budget",
                        "cbr_relative_gap": 0.01,
                        "target_effective_cbr_tolerance": 0.05,
                        "within_target_tolerance": True,
                    },
                }
            },
            mode="single_user",
        )

        self.assertIsNotNone(over_budget)
        self.assertIsNotNone(in_budget)
        self.assertEqual(in_budget.name, "matched_budget_psnr")
        self.assertTrue(_is_better_selection_metric(in_budget, over_budget))

    def test_reselect_best_checkpoint_prefers_budget_feasible_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_dir = root / "reports"
            checkpoint_dir = root / "checkpoints"
            report_dir.mkdir()
            checkpoint_dir.mkdir()

            torch.save({"model": {}, "extra_state": {"epoch": 1}}, checkpoint_dir / "epoch_001.pt")
            torch.save({"model": {}, "extra_state": {"epoch": 2}}, checkpoint_dir / "epoch_002.pt")
            (report_dir / "train_history.json").write_text(
                json.dumps(
                    {
                        "config_path": "dummy.json",
                        "mode": "single_user",
                        "history": [
                            {
                                "epoch": 1,
                                "validation": {
                                    "metrics": {"psnr": 29.0},
                                    "budget": {
                                        "operating_mode": "matched_budget",
                                        "cbr_relative_gap": 0.20,
                                        "target_effective_cbr_tolerance": 0.05,
                                        "within_target_tolerance": False,
                                    },
                                },
                            },
                            {
                                "epoch": 2,
                                "validation": {
                                    "metrics": {"psnr": 25.0},
                                    "budget": {
                                        "operating_mode": "matched_budget",
                                        "cbr_relative_gap": 0.01,
                                        "target_effective_cbr_tolerance": 0.05,
                                        "within_target_tolerance": True,
                                    },
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = reselect_best_checkpoint(report_dir=report_dir, checkpoint_dir=checkpoint_dir)
            selected_epoch = torch.load(checkpoint_dir / "best.pt", map_location="cpu")["extra_state"]["epoch"]

            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(result["selection_metric_name"], "matched_budget_psnr")
            self.assertEqual(selected_epoch, 2)


if __name__ == "__main__":
    unittest.main()

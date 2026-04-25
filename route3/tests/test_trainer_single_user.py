from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses import ConditionalSSDDDecoderLoss
from models.ultimate import (
    ConditionalSSDDDecoder,
    LayerWiseAdaptiveDistillation,
    PatchGANDiscriminator,
    PerceptualAdversarialEnhancer,
    SemanticDistillationModule,
    SemanticTeacherEncoder,
    UltimateDualPathEncoder,
    UltimateSingleUserTransmission,
    VGGFeatureExtractor,
    VGGPerceptualLoss,
)
from trainers import SingleUserTrainConfig, SingleUserTrainer


class TinyVGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
            ]
        )

    def __iter__(self):
        return iter(self.layers)


class RecordingSingleUserTransmission(UltimateSingleUserTransmission):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decode_stochastic_calls: list[bool] = []
        self.noise_image_calls: list[torch.Tensor | None] = []

    def forward(self, *args, **kwargs):
        self.decode_stochastic_calls.append(bool(kwargs.get("decode_stochastic", True)))
        noise_image = kwargs.get("noise_image")
        self.noise_image_calls.append(None if noise_image is None else noise_image.detach().clone())
        return super().forward(*args, **kwargs)


class RecordingEnhancer(PerceptualAdversarialEnhancer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator_step_calls: list[dict[str, bool]] = []

    def generator_step(
        self,
        x_gt: torch.Tensor,
        x_pred: torch.Tensor | None = None,
        *,
        base_reconstruction=None,
        compute_discriminator_loss: bool = False,
        enable_perceptual: bool | None = None,
        enable_adversarial: bool | None = None,
    ):
        self.generator_step_calls.append(
            {
                "compute_discriminator_loss": bool(compute_discriminator_loss),
                "enable_perceptual": bool(enable_perceptual),
                "enable_adversarial": bool(enable_adversarial),
            }
        )
        return super().generator_step(
            x_gt=x_gt,
            x_pred=x_pred,
            base_reconstruction=base_reconstruction,
            compute_discriminator_loss=compute_discriminator_loss,
            enable_perceptual=enable_perceptual,
            enable_adversarial=enable_adversarial,
        )


def build_training_model(
    record_decode_calls: bool = False,
    record_enhancement_calls: bool = False,
) -> UltimateSingleUserTransmission:
    student_encoder = UltimateDualPathEncoder()
    teacher_encoder = UltimateDualPathEncoder().semantic_branch

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "teacher_semantic.pt"
        torch.save(student_encoder.semantic_branch.state_dict(), checkpoint_path)
        teacher = SemanticTeacherEncoder(backbone=teacher_encoder, checkpoint_path=checkpoint_path)
        distillation = SemanticDistillationModule(
            teacher=teacher,
            distiller=LayerWiseAdaptiveDistillation(num_stages=3),
        )
        enhancer_cls = RecordingEnhancer if record_enhancement_calls else PerceptualAdversarialEnhancer
        enhancer = enhancer_cls(
            perceptual_loss=VGGPerceptualLoss(
                feature_extractor=VGGFeatureExtractor(layers=(1, 3, 5), feature_extractor=TinyVGGFeatures()),
            ),
            discriminator=PatchGANDiscriminator(base_channels=16, n_layers=2),
        )
        model_cls = RecordingSingleUserTransmission if record_decode_calls else UltimateSingleUserTransmission
        model = model_cls(
            encoder=student_encoder,
            semantic_distillation=distillation,
            decoder=ConditionalSSDDDecoder(),
            decoder_loss=ConditionalSSDDDecoderLoss(),
            enhancer=enhancer,
        )
    return model


class TrainerSingleUserTests(unittest.TestCase):
    def test_train_step_updates_global_step(self) -> None:
        torch.manual_seed(5)
        trainer = SingleUserTrainer(
            model=build_training_model(),
            config=SingleUserTrainConfig(max_grad_norm=1.0),
        )
        batch = torch.rand(2, 3, 128, 128)

        result = trainer.train_step(batch, snr_db=torch.tensor([8.0, 12.0]), sem_rate_ratio=0.5, det_rate_ratio=0.75)

        self.assertEqual(result.global_step, 1)
        self.assertIn("train_total", result.terms)
        self.assertIn("distillation_total", result.terms)
        self.assertIn("enhancement_g_total", result.terms)
        self.assertIn("enhancement_d_total", result.terms)
        self.assertIn("enhancement_weighted_total", result.terms)
        self.assertIn("base_decoder_aux_weighted_total", result.terms)
        self.assertIn("final_decoder_weighted_total", result.terms)
        self.assertIn("enhancement_refinement_consistency_weighted", result.terms)
        self.assertIn("enhancement_refinement_delta_weighted", result.terms)
        self.assertGreaterEqual(result.total_loss, 0.0)

    def test_train_uses_stochastic_decoder_and_validation_stays_deterministic(self) -> None:
        torch.manual_seed(17)
        model = build_training_model(record_decode_calls=True)
        trainer = SingleUserTrainer(
            model=model,
            config=SingleUserTrainConfig(train_decode_stochastic=True),
        )
        batch = torch.rand(1, 3, 128, 128)

        trainer.train_step(batch, snr_db=10.0, sem_rate_ratio=0.5, det_rate_ratio=0.5)
        trainer.validate_step(batch, snr_db=10.0, sem_rate_ratio=0.5, det_rate_ratio=0.5)

        self.assertEqual(model.decode_stochastic_calls, [True, True, False])
        self.assertIsNotNone(model.noise_image_calls[0])
        self.assertIsNotNone(model.noise_image_calls[1])
        self.assertTrue(torch.equal(model.noise_image_calls[0], model.noise_image_calls[1]))
        self.assertIsNone(model.noise_image_calls[2])

    def test_validation_and_checkpoint_roundtrip(self) -> None:
        torch.manual_seed(7)
        trainer = SingleUserTrainer(model=build_training_model())
        batch = torch.rand(1, 3, 128, 128)

        trainer.train_step(batch, snr_db=10.0, sem_rate_ratio=0.5, det_rate_ratio=0.5)
        validation = trainer.validate_step(batch, snr_db=10.0, sem_rate_ratio=0.5, det_rate_ratio=0.5)

        self.assertIn("validation_total", validation.terms)
        self.assertIn("enhancement_weighted_total", validation.terms)
        self.assertIn("base_decoder_aux_weighted_total", validation.terms)
        self.assertIn("final_decoder_weighted_total", validation.terms)
        self.assertIn("enhancement_refinement_consistency_weighted", validation.terms)
        self.assertIn("enhancement_refinement_delta_weighted", validation.terms)
        self.assertIsNotNone(validation.base_metrics)
        self.assertIsNotNone(validation.final_metrics)
        self.assertAlmostEqual(validation.metrics.psnr, validation.final_metrics.psnr, places=6)
        self.assertIsNotNone(validation.enhancement_status)
        self.assertTrue(validation.enhancement_status["perceptual_enabled"])

    def test_trainer_applies_perceptual_loss_scale_to_enhancer(self) -> None:
        torch.manual_seed(31)
        model = build_training_model()
        trainer = SingleUserTrainer(
            model=model,
            config=SingleUserTrainConfig(perceptual_loss_scale=0.05),
        )

        self.assertAlmostEqual(model.enhancer.perceptual_loss.loss_scale, 0.05)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "single_user.pt"
            trainer.save_checkpoint(ckpt_path, extra_state={"phase": "phase6"})

            reloaded = SingleUserTrainer(model=build_training_model())
            extra_state = reloaded.load_checkpoint(ckpt_path)

            self.assertEqual(extra_state["phase"], "phase6")
            self.assertEqual(reloaded.global_step, trainer.global_step)
            self.assertEqual(reloaded.validation_step_count, trainer.validation_step_count)
            self.assertIsNotNone(reloaded.discriminator_optimizer)

    def test_generator_step_keeps_spectral_norm_buffers_stable(self) -> None:
        torch.manual_seed(11)
        model = build_training_model()
        model.train()
        batch = torch.rand(1, 3, 128, 128)
        fake = torch.rand(1, 3, 128, 128)
        discriminator = model.enhancer.discriminator
        self.assertIsNotNone(discriminator)

        tracked_buffers = {
            name: buffer.detach().clone()
            for name, buffer in discriminator.named_buffers()
            if "weight_u" in name or "weight_v" in name
        }

        model.enhancer.generator_step(batch, fake, compute_discriminator_loss=False)

        for name, before in tracked_buffers.items():
            after = dict(discriminator.named_buffers())[name]
            self.assertTrue(torch.equal(before, after), msg=f"spectral norm buffer changed: {name}")

    def test_ablation_runner_executes(self) -> None:
        torch.manual_seed(9)
        trainer = SingleUserTrainer(model=build_training_model())
        batch = torch.rand(1, 3, 128, 128)

        report = trainer.run_ablations(batch, snr_db=10.0, sem_rate_ratio=0.5, det_rate_ratio=0.5)

        self.assertTrue(report.requires_retraining)
        self.assertGreaterEqual(report.full_model.psnr, 0.0)
        self.assertIsNotNone(report.base_model)
        self.assertGreaterEqual(report.no_detail_branch.psnr, 0.0)
        self.assertGreaterEqual(report.deterministic_baseline.psnr, 0.0)
        self.assertIsNotNone(report.refinement_gain_psnr)
        self.assertIsNotNone(report.no_distillation_objective)

    def test_progressive_stage_schedule_gates_enhancement_and_discriminator(self) -> None:
        torch.manual_seed(23)
        model = build_training_model(record_enhancement_calls=True)
        trainer = SingleUserTrainer(
            model=model,
            config=SingleUserTrainConfig(
                total_epochs=6,
                phase1_epochs=2,
                phase2_epochs=2,
                perceptual_weight_max=0.8,
                adversarial_weight=0.2,
            ),
        )
        batch = torch.rand(2, 3, 128, 128)

        trainer.set_epoch(1)
        stage1 = trainer.train_step(batch, snr_db=10.0, sem_rate_ratio=0.5, det_rate_ratio=0.75)
        self.assertEqual(trainer.current_stage_summary()["name"], "structural_reconstruction")
        self.assertNotIn("enhancement_g_total", stage1.terms)
        self.assertNotIn("enhancement_d_total", stage1.terms)
        self.assertEqual(model.enhancer.generator_step_calls, [])

        trainer.set_epoch(4)
        stage2 = trainer.train_step(batch, snr_db=10.0, sem_rate_ratio=0.5, det_rate_ratio=0.75)
        self.assertEqual(trainer.current_stage_summary()["name"], "perceptual_enhancement")
        self.assertIn("enhancement_g_total", stage2.terms)
        self.assertNotIn("enhancement_d_total", stage2.terms)
        self.assertEqual(
            model.enhancer.generator_step_calls[-1],
            {
                "compute_discriminator_loss": False,
                "enable_perceptual": True,
                "enable_adversarial": False,
            },
        )

        trainer.set_epoch(5)
        stage3 = trainer.train_step(batch, snr_db=10.0, sem_rate_ratio=0.5, det_rate_ratio=0.75)
        self.assertEqual(trainer.current_stage_summary()["name"], "adversarial_refinement")
        self.assertIn("enhancement_g_total", stage3.terms)
        self.assertIn("enhancement_d_total", stage3.terms)
        self.assertEqual(
            model.enhancer.generator_step_calls[-1],
            {
                "compute_discriminator_loss": False,
                "enable_perceptual": True,
                "enable_adversarial": True,
            },
        )

    def test_progressive_perceptual_stage_starts_with_nonzero_weight(self) -> None:
        trainer = SingleUserTrainer(
            model=build_training_model(),
            config=SingleUserTrainConfig(
                total_epochs=6,
                phase1_epochs=2,
                phase2_epochs=2,
                perceptual_weight_max=0.8,
                adversarial_weight=0.2,
            ),
        )

        trainer.set_epoch(3)

        stage = trainer.current_stage_summary()
        self.assertEqual(stage["name"], "perceptual_enhancement")
        self.assertGreater(stage["perceptual_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()

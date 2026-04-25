from __future__ import annotations

import sys
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
    PatchGANDiscriminator,
    PerceptualAdversarialEnhancer,
    UltimateDualPathEncoder,
    UltimateMultiUserTransmission,
    UltimateSingleUserTransmission,
    VGGFeatureExtractor,
    VGGPerceptualLoss,
)
from trainers import MultiUserTrainConfig, MultiUserTrainer


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


def build_multi_user_training_model(record_decode_calls: bool = False) -> UltimateMultiUserTransmission:
    enhancer = RecordingEnhancer(
        perceptual_loss=VGGPerceptualLoss(
            feature_extractor=VGGFeatureExtractor(layers=(1, 3, 5), feature_extractor=TinyVGGFeatures()),
        ),
        discriminator=PatchGANDiscriminator(base_channels=16, n_layers=2),
    )
    model_cls = RecordingSingleUserTransmission if record_decode_calls else UltimateSingleUserTransmission
    single_user_model = model_cls(
        encoder=UltimateDualPathEncoder(),
        decoder=ConditionalSSDDDecoder(),
        decoder_loss=ConditionalSSDDDecoderLoss(),
        enhancer=enhancer,
    )
    return UltimateMultiUserTransmission(single_user_model=single_user_model)


class TrainerMultiUserTests(unittest.TestCase):
    def test_multi_user_forward_only_enhances_shared_branch(self) -> None:
        torch.manual_seed(13)
        model = build_multi_user_training_model()
        batch = torch.rand(2, 3, 128, 128)

        output = model(
            batch,
            snr_db=10.0,
            sem_rate_ratio=0.5,
            det_rate_ratio=0.75,
            semantic_bandwidth_budget=10.0,
            run_enhancement=True,
            compute_enhancement_discriminator_loss=False,
        )

        enhancer = model.single_user_model.enhancer
        self.assertIsNotNone(enhancer)
        self.assertIsNone(output.single_user.enhancement_losses)
        self.assertIsNotNone(output.shared_enhancement_losses)
        self.assertIsNotNone(output.single_user.reconstruction)
        self.assertIsNotNone(output.shared_reconstruction)
        self.assertTrue(
            torch.equal(output.single_user.reconstruction.noisy_input, output.shared_reconstruction.noisy_input)
        )
        self.assertEqual(
            enhancer.generator_step_calls,
            [
                {
                    "compute_discriminator_loss": False,
                    "enable_perceptual": False,
                    "enable_adversarial": False,
                }
            ],
        )

    def test_train_step_uses_single_generator_enhancement_pass(self) -> None:
        torch.manual_seed(19)
        model = build_multi_user_training_model(record_decode_calls=True)
        trainer = MultiUserTrainer(
            model=model,
            config=MultiUserTrainConfig(max_grad_norm=1.0, train_decode_stochastic=True),
        )
        batch = torch.rand(2, 3, 128, 128)

        train_output = trainer.train_step(
            batch,
            snr_db=10.0,
            sem_rate_ratio=0.5,
            det_rate_ratio=0.75,
            semantic_bandwidth_budget=10.0,
        )
        validation_output = trainer.validate_step(
            batch,
            snr_db=10.0,
            sem_rate_ratio=0.5,
            det_rate_ratio=0.75,
            semantic_bandwidth_budget=10.0,
        )

        enhancer = model.single_user_model.enhancer
        self.assertIsNotNone(enhancer)
        self.assertEqual(
            enhancer.generator_step_calls,
            [
                {
                    "compute_discriminator_loss": False,
                    "enable_perceptual": True,
                    "enable_adversarial": True,
                },
                {
                    "compute_discriminator_loss": True,
                    "enable_perceptual": True,
                    "enable_adversarial": True,
                },
            ],
        )
        self.assertEqual(model.single_user_model.decode_stochastic_calls, [True, True, False])
        self.assertIsNotNone(model.single_user_model.noise_image_calls[0])
        self.assertIsNotNone(model.single_user_model.noise_image_calls[1])
        self.assertTrue(
            torch.equal(
                model.single_user_model.noise_image_calls[0],
                model.single_user_model.noise_image_calls[1],
            )
        )
        self.assertIsNone(model.single_user_model.noise_image_calls[2])
        self.assertIn("enhancement_g_total", train_output.terms)
        self.assertIn("enhancement_d_total", train_output.terms)
        self.assertIn("enhancement_weighted_total", train_output.terms)
        self.assertIn("enhancement_d_total", validation_output.terms)
        self.assertIsNotNone(validation_output.base_metrics)
        self.assertIsNotNone(validation_output.final_metrics)
        self.assertIsNotNone(validation_output.comparison.shared_base_metrics)
        self.assertIsNotNone(validation_output.comparison.shared_final_metrics)

    def test_progressive_stage_schedule_gates_multi_user_enhancement(self) -> None:
        torch.manual_seed(29)
        model = build_multi_user_training_model(record_decode_calls=True)
        trainer = MultiUserTrainer(
            model=model,
            config=MultiUserTrainConfig(
                total_epochs=6,
                phase1_epochs=2,
                phase2_epochs=2,
                perceptual_weight_max=0.8,
                adversarial_weight=0.2,
            ),
        )
        batch = torch.rand(2, 3, 128, 128)
        enhancer = model.single_user_model.enhancer
        self.assertIsNotNone(enhancer)

        trainer.set_epoch(1)
        stage1 = trainer.train_step(
            batch,
            snr_db=10.0,
            sem_rate_ratio=0.5,
            det_rate_ratio=0.75,
            semantic_bandwidth_budget=10.0,
        )
        self.assertEqual(trainer.current_stage_summary()["name"], "structural_reconstruction")
        self.assertNotIn("enhancement_g_total", stage1.terms)
        self.assertNotIn("enhancement_d_total", stage1.terms)
        self.assertEqual(enhancer.generator_step_calls, [])

        trainer.set_epoch(4)
        stage2 = trainer.train_step(
            batch,
            snr_db=10.0,
            sem_rate_ratio=0.5,
            det_rate_ratio=0.75,
            semantic_bandwidth_budget=10.0,
        )
        self.assertEqual(trainer.current_stage_summary()["name"], "perceptual_enhancement")
        self.assertIn("enhancement_g_total", stage2.terms)
        self.assertNotIn("enhancement_d_total", stage2.terms)
        self.assertEqual(
            enhancer.generator_step_calls[-1],
            {
                "compute_discriminator_loss": False,
                "enable_perceptual": True,
                "enable_adversarial": False,
            },
        )

        trainer.set_epoch(5)
        stage3 = trainer.train_step(
            batch,
            snr_db=10.0,
            sem_rate_ratio=0.5,
            det_rate_ratio=0.75,
            semantic_bandwidth_budget=10.0,
        )
        self.assertEqual(trainer.current_stage_summary()["name"], "adversarial_refinement")
        self.assertIn("enhancement_g_total", stage3.terms)
        self.assertIn("enhancement_d_total", stage3.terms)
        self.assertEqual(
            enhancer.generator_step_calls[-1],
            {
                "compute_discriminator_loss": False,
                "enable_perceptual": True,
                "enable_adversarial": True,
            },
        )

    def test_trainer_applies_perceptual_loss_scale_to_shared_enhancer(self) -> None:
        model = build_multi_user_training_model()
        trainer = MultiUserTrainer(
            model=model,
            config=MultiUserTrainConfig(perceptual_loss_scale=0.05),
        )

        self.assertAlmostEqual(model.single_user_model.enhancer.perceptual_loss.loss_scale, 0.05)


if __name__ == "__main__":
    unittest.main()

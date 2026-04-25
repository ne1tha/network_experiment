from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ultimate import (
    ConditionalSSDDDecoder,
    DecoderOutput,
    PatchGANDiscriminator,
    PerceptualAdversarialEnhancer,
    UltimateSingleUserTransmission,
    VGGFeatureExtractor,
    VGGPerceptualLoss,
)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class RecordingConditionalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, source: torch.Tensor, candidate: torch.Tensor | None = None) -> torch.Tensor:
        paired_candidate = source if candidate is None else candidate
        self.calls.append((source.detach().clone(), paired_candidate.detach().clone()))
        merged = torch.cat([source, paired_candidate], dim=1).mean(dim=1, keepdim=True)
        return F.avg_pool2d(merged, kernel_size=4, stride=4)


class PerceptualGANTests(unittest.TestCase):
    def test_vgg_feature_extractor_requires_explicit_backbone(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires an explicit checkpoint"):
            VGGFeatureExtractor(allow_untrained=False)

    def test_dual_branch_patchgan_emits_two_patch_maps(self) -> None:
        discriminator = PatchGANDiscriminator(base_channels=16, n_layers=2)
        source = torch.rand(2, 3, 128, 128)
        candidate = torch.rand(2, 3, 128, 128)

        logits = discriminator(source, candidate)

        self.assertIsInstance(logits, tuple)
        self.assertEqual(len(logits), 2)
        for logit in logits:
            self.assertEqual(logit.shape[0], 2)
            self.assertEqual(logit.shape[1], 1)
            self.assertGreater(logit.shape[-2], 1)
            self.assertGreater(logit.shape[-1], 1)
            self.assertTrue(torch.isfinite(logit).all().item())

    def test_legacy_and_concat_conditional_patchgan_still_work(self) -> None:
        legacy = PatchGANDiscriminator(base_channels=16, n_layers=2, kind="legacy_patchgan")
        conditional = PatchGANDiscriminator(base_channels=16, n_layers=2, kind="conditional_multiscale_v1")
        source = torch.rand(2, 3, 128, 128)
        candidate = torch.rand(2, 3, 128, 128)

        legacy_logits = legacy(source, candidate)
        conditional_logits = conditional(source, candidate)

        self.assertIsInstance(legacy_logits, torch.Tensor)
        self.assertIsInstance(conditional_logits, tuple)
        self.assertEqual(len(conditional_logits), 2)

    def test_perceptual_and_adversarial_losses_are_emitted(self) -> None:
        feature_extractor = VGGFeatureExtractor(
            layers=(1, 3, 5),
            feature_extractor=TinyVGGFeatures(),
        )
        enhancer = PerceptualAdversarialEnhancer(
            perceptual_loss=VGGPerceptualLoss(feature_extractor=feature_extractor, layer_weights=(1.0, 0.5, 0.25)),
            discriminator=PatchGANDiscriminator(base_channels=16, n_layers=2),
        )
        model = UltimateSingleUserTransmission(
            decoder=ConditionalSSDDDecoder(),
            enhancer=enhancer,
        )
        image = torch.rand(2, 3, 128, 128)

        output = model(
            image,
            snr_db=torch.tensor([8.0, 12.0]),
            sem_rate_ratio=0.5,
            det_rate_ratio=0.75,
            decode_stochastic=False,
            compute_enhancement_discriminator_loss=True,
        )

        self.assertIsNotNone(output.enhancement_losses)
        self.assertIsNotNone(output.enhancement_status)
        self.assertIsNotNone(output.base_reconstruction)
        self.assertIsNotNone(output.final_reconstruction)
        self.assertEqual(output.base_reconstruction.output_kind, "base")
        self.assertEqual(output.final_reconstruction.output_kind, "final")
        self.assertIs(output.reconstruction, output.final_reconstruction)
        self.assertTrue(output.enhancement_status.perceptual_enabled)
        self.assertTrue(output.enhancement_status.adversarial_enabled)
        self.assertIn("perceptual_vgg", output.enhancement_losses.generator_terms)
        self.assertIn("perceptual_vgg_raw", output.enhancement_losses.generator_terms)
        self.assertIn("perceptual_vgg_effective", output.enhancement_losses.generator_terms)
        self.assertIn("adversarial_g", output.enhancement_losses.generator_terms)
        self.assertIn("adversarial_g_effective", output.enhancement_losses.generator_terms)
        self.assertTrue(torch.isfinite(output.enhancement_losses.generator_total_loss).item())
        self.assertIsNotNone(output.enhancement_losses.discriminator_loss)
        self.assertGreaterEqual(float(output.enhancement_losses.discriminator_loss.item()), 0.0)

    def test_refine_reconstruction_preserves_base_metadata_and_marks_final_output(self) -> None:
        enhancer = PerceptualAdversarialEnhancer()
        base = DecoderOutput(
            x_hat=torch.rand(1, 3, 32, 32),
            predicted_residual=torch.rand(1, 3, 32, 32),
            noisy_input=torch.rand(1, 3, 32, 32),
            conditioning_map=torch.rand(1, 128, 8, 8),
            semantic_condition_map=torch.rand(1, 128, 8, 8),
            detail_condition_map=torch.rand(1, 128, 8, 8),
            bottleneck_condition_map=torch.rand(1, 128, 4, 4),
            context_vector=torch.rand(1, 512),
            decode_steps=1,
            stochastic=False,
            output_kind="base",
        )

        refined = enhancer.refine_reconstruction(base_reconstruction=base)

        self.assertEqual(refined.output_kind, "final")
        self.assertIsNotNone(refined.base_x_hat)
        self.assertIsNotNone(refined.refinement_delta)
        self.assertTrue(torch.equal(refined.base_x_hat, base.x_hat))
        self.assertTrue(torch.equal(refined.predicted_residual, base.predicted_residual))
        self.assertTrue(torch.equal(refined.conditioning_map, base.conditioning_map))
        self.assertTrue(torch.equal(refined.context_vector, base.context_vector))

    def test_adversarial_step_conditions_on_base_reconstruction(self) -> None:
        discriminator = RecordingConditionalDiscriminator()
        enhancer = PerceptualAdversarialEnhancer(discriminator=discriminator)
        base = DecoderOutput(
            x_hat=torch.rand(1, 3, 32, 32),
            predicted_residual=torch.rand(1, 3, 32, 32),
            noisy_input=torch.rand(1, 3, 32, 32),
            conditioning_map=torch.rand(1, 128, 8, 8),
            semantic_condition_map=torch.rand(1, 128, 8, 8),
            detail_condition_map=torch.rand(1, 128, 8, 8),
            bottleneck_condition_map=torch.rand(1, 128, 4, 4),
            context_vector=torch.rand(1, 512),
            output_kind="base",
        )
        x_gt = torch.rand(1, 3, 32, 32)

        output = enhancer.generator_step(
            x_gt=x_gt,
            base_reconstruction=base,
            compute_discriminator_loss=True,
            enable_perceptual=False,
            enable_adversarial=True,
        )

        self.assertEqual(len(discriminator.calls), 3)
        for source, _candidate in discriminator.calls:
            self.assertTrue(torch.allclose(source, base.x_hat))
        self.assertTrue(torch.allclose(discriminator.calls[1][1], x_gt))
        self.assertTrue(torch.allclose(discriminator.calls[2][1], output.refined_reconstruction.x_hat))

    def test_vgg_perceptual_loss_scale_is_applied_after_raw_feature_loss(self) -> None:
        feature_extractor = VGGFeatureExtractor(
            layers=(1, 3, 5),
            feature_extractor=TinyVGGFeatures(),
        )
        perceptual = VGGPerceptualLoss(
            feature_extractor=feature_extractor,
            layer_weights=(1.0, 0.5, 0.25),
            loss_scale=0.125,
        )
        x_gt = torch.rand(2, 3, 32, 32)
        x_pred = torch.rand(2, 3, 32, 32)

        raw = perceptual.raw_loss(x_gt, x_pred)
        scaled = perceptual(x_gt, x_pred)

        self.assertTrue(torch.allclose(scaled, raw * 0.125, atol=1e-6, rtol=1e-6))

    def test_enhancer_can_temporarily_disable_adversarial_loss(self) -> None:
        feature_extractor = VGGFeatureExtractor(
            layers=(1, 3, 5),
            feature_extractor=TinyVGGFeatures(),
        )
        enhancer = PerceptualAdversarialEnhancer(
            perceptual_loss=VGGPerceptualLoss(feature_extractor=feature_extractor, layer_weights=(1.0, 0.5, 0.25)),
            discriminator=PatchGANDiscriminator(base_channels=16, n_layers=2),
        )
        x_gt = torch.rand(2, 3, 128, 128)
        x_pred = torch.rand(2, 3, 128, 128)

        output = enhancer.generator_step(
            x_gt=x_gt,
            x_pred=x_pred,
            compute_discriminator_loss=True,
            enable_perceptual=True,
            enable_adversarial=False,
        )

        self.assertTrue(output.status.perceptual_enabled)
        self.assertFalse(output.status.adversarial_enabled)
        self.assertTrue(output.status.discriminator_ready)
        self.assertIsNone(output.discriminator_loss)
        self.assertEqual(float(output.generator_terms["adversarial_g"].item()), 0.0)

    def test_enhancer_requires_main_decoder(self) -> None:
        feature_extractor = VGGFeatureExtractor(
            layers=(1, 3, 5),
            feature_extractor=TinyVGGFeatures(),
        )
        enhancer = PerceptualAdversarialEnhancer(
            perceptual_loss=VGGPerceptualLoss(feature_extractor=feature_extractor),
            discriminator=PatchGANDiscriminator(base_channels=16, n_layers=2),
        )
        model = UltimateSingleUserTransmission(enhancer=enhancer)
        image = torch.rand(1, 3, 128, 128)

        with self.assertRaisesRegex(RuntimeError, "requires the main decoder"):
            model(image, snr_db=10.0, sem_rate_ratio=0.5, det_rate_ratio=0.5)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses import ConditionalSSDDDecoderLoss
from models.ultimate import ConditionalSSDDDecoder, UltimateSingleUserTransmission


class DecoderSSDDTests(unittest.TestCase):
    def test_decoder_emits_image_with_expected_contract(self) -> None:
        torch.manual_seed(11)
        model = UltimateSingleUserTransmission(
            decoder=ConditionalSSDDDecoder(),
            decoder_loss=ConditionalSSDDDecoderLoss(),
        )
        image = torch.rand(2, 3, 128, 128)

        output = model(
            image,
            snr_db=torch.tensor([8.0, 12.0]),
            sem_rate_ratio=0.5,
            det_rate_ratio=0.75,
            decode_stochastic=False,
        )

        self.assertIsNotNone(output.reconstruction)
        self.assertEqual(output.reconstruction.x_hat.shape, image.shape)
        self.assertEqual(output.reconstruction.predicted_residual.shape, image.shape)
        self.assertEqual(output.reconstruction.noisy_input.shape, image.shape)
        self.assertEqual(tuple(output.reconstruction.context_vector.shape), (2, 512))
        self.assertEqual(output.reconstruction.semantic_condition_map.shape, output.reconstruction.detail_condition_map.shape)
        self.assertEqual(output.reconstruction.conditioning_map.shape, output.reconstruction.semantic_condition_map.shape)
        self.assertEqual(
            output.reconstruction.bottleneck_condition_map.shape[-2:],
            tuple(size // 2 for size in output.reconstruction.conditioning_map.shape[-2:]),
        )
        self.assertEqual(output.reconstruction.decode_steps, 1)
        self.assertFalse(output.reconstruction.stochastic)
        self.assertIsNotNone(output.base_decoder_losses)
        self.assertIsNotNone(output.final_decoder_losses)
        self.assertIsNotNone(output.decoder_losses)
        self.assertIn("decoder_ms_ssim", output.decoder_losses.terms)
        self.assertIn("decoder_l1", output.decoder_losses.terms)
        self.assertIn("decoder_mse", output.decoder_losses.terms)
        self.assertIn("decoder_residual", output.decoder_losses.terms)
        self.assertIs(output.base_decoder_losses, output.final_decoder_losses)
        self.assertIs(output.final_decoder_losses, output.decoder_losses)

    def test_stochastic_and_deterministic_decode_differ(self) -> None:
        torch.manual_seed(19)
        model = UltimateSingleUserTransmission(decoder=ConditionalSSDDDecoder())
        image = torch.rand(1, 3, 128, 128)

        deterministic = model(
            image,
            snr_db=10.0,
            sem_rate_ratio=0.5,
            det_rate_ratio=0.5,
            decode_stochastic=False,
        ).reconstruction.x_hat
        stochastic = model(
            image,
            snr_db=10.0,
            sem_rate_ratio=0.5,
            det_rate_ratio=0.5,
            decode_stochastic=True,
        ).reconstruction.x_hat

        self.assertFalse(torch.allclose(deterministic, stochastic))

    def test_decoder_loss_rejects_shape_mismatch(self) -> None:
        loss_block = ConditionalSSDDDecoderLoss()
        with self.assertRaisesRegex(ValueError, "matching image shapes"):
            loss_block(
                x_gt=torch.rand(1, 3, 128, 128),
                x_hat=torch.rand(1, 3, 64, 64),
            )


if __name__ == "__main__":
    unittest.main()

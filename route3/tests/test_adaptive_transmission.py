from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ultimate.code_mask import CodeMaskModule
from models.ultimate.full_model_single_user import UltimateSingleUserTransmission


class AdaptiveTransmissionTests(unittest.TestCase):
    def test_code_mask_prunes_requested_channels(self) -> None:
        module = CodeMaskModule()
        features = torch.ones(1, 8, 4, 4)
        scores = torch.tensor([[0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6]], dtype=features.dtype)

        output = module(features, channel_scores=scores, rate_ratio=0.5)

        self.assertEqual(int(output.active_channels[0].item()), 4)
        self.assertEqual(int(output.channel_mask.sum().item()), 4)
        self.assertFalse(torch.allclose(output.masked, features))

    def test_single_user_transmission_emits_tx_rx_and_metadata(self) -> None:
        torch.manual_seed(7)
        model = UltimateSingleUserTransmission()
        image = torch.randn(2, 3, 128, 128)

        output = model(image, snr_db=torch.tensor([6.0, 10.0]), sem_rate_ratio=0.5, det_rate_ratio=0.75)

        self.assertEqual(output.semantic.tx.shape, output.encoder.z_sem.shape)
        self.assertEqual(output.detail.tx.shape, output.encoder.z_det.shape)
        self.assertEqual(output.semantic.rx.shape, output.encoder.z_sem.shape)
        self.assertEqual(output.detail.rx.shape, output.encoder.z_det.shape)
        self.assertTrue(torch.all(output.semantic.mask.active_channels < output.encoder.z_sem.shape[1]))
        self.assertTrue(torch.all(output.detail.mask.active_channels < output.encoder.z_det.shape[1]))
        self.assertEqual(tuple(output.semantic.channel_state.snr_db.shape), (2,))

    def test_awgn_channel_changes_received_symbols(self) -> None:
        torch.manual_seed(3)
        model = UltimateSingleUserTransmission()
        image = torch.randn(1, 3, 128, 128)

        output = model(image, snr_db=4.0, sem_rate_ratio=0.5, det_rate_ratio=0.5)

        self.assertFalse(torch.allclose(output.semantic.tx, output.semantic.rx))
        self.assertFalse(torch.allclose(output.detail.tx, output.detail.rx))

    def test_invalid_rate_ratio_fails_fast(self) -> None:
        model = UltimateSingleUserTransmission()
        image = torch.randn(1, 3, 128, 128)

        with self.assertRaisesRegex(ValueError, r"rate_ratio must be in \(0, 1\]"):
            model(image, snr_db=10.0, sem_rate_ratio=0.0, det_rate_ratio=0.5)


if __name__ == "__main__":
    unittest.main()

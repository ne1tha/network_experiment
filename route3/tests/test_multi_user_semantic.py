from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses import ConditionalSSDDDecoderLoss
from models.ultimate import ConditionalSSDDDecoder, UltimateMultiUserTransmission, UltimateSingleUserTransmission


def build_multi_user_model() -> UltimateMultiUserTransmission:
    single_user = UltimateSingleUserTransmission(
        decoder=ConditionalSSDDDecoder(),
        decoder_loss=ConditionalSSDDDecoderLoss(),
    )
    return UltimateMultiUserTransmission(single_user_model=single_user)


class MultiUserSemanticTests(unittest.TestCase):
    def test_multi_user_shares_semantic_and_keeps_detail_private(self) -> None:
        torch.manual_seed(13)
        model = build_multi_user_model()
        batch = torch.rand(4, 3, 128, 128)

        output = model(
            batch,
            snr_db=torch.tensor([8.0, 10.0, 12.0, 14.0]),
            sem_rate_ratio=0.5,
            det_rate_ratio=0.75,
            semantic_bandwidth_budget=10.0,
        )

        self.assertEqual(output.shared_rx_sem.shape, output.single_user.semantic.rx.shape)
        self.assertEqual(output.private_rx_det.shape, output.single_user.detail.rx.shape)
        self.assertFalse(torch.allclose(output.shared_rx_sem, output.single_user.semantic.rx))
        self.assertTrue(torch.allclose(output.private_rx_det, output.single_user.detail.rx))
        self.assertEqual(tuple(output.pairing.pair_indices.shape), (2, 2))
        self.assertEqual(tuple(output.bandwidth.pair_bandwidth.shape), (2,))
        self.assertEqual(tuple(output.bandwidth.per_user_bandwidth.shape), (4,))

    def test_multi_user_requires_even_number_of_users(self) -> None:
        model = build_multi_user_model()
        batch = torch.rand(3, 3, 128, 128)

        with self.assertRaisesRegex(ValueError, "even number of users"):
            model(batch, snr_db=10.0, sem_rate_ratio=0.5, det_rate_ratio=0.5, semantic_bandwidth_budget=8.0)

    def test_bandwidth_budget_must_be_positive(self) -> None:
        model = build_multi_user_model()
        batch = torch.rand(4, 3, 128, 128)

        with self.assertRaisesRegex(ValueError, "total_budget must be positive"):
            model(batch, snr_db=10.0, sem_rate_ratio=0.5, det_rate_ratio=0.5, semantic_bandwidth_budget=0.0)

    def test_bandwidth_allocation_changes_shared_semantic_when_budget_changes(self) -> None:
        torch.manual_seed(17)
        model = build_multi_user_model()
        batch = torch.rand(4, 3, 128, 128)

        output_low = model(
            batch,
            snr_db=torch.tensor([8.0, 10.0, 12.0, 14.0]),
            sem_rate_ratio=0.5,
            det_rate_ratio=0.75,
            semantic_bandwidth_budget=4.0,
        )
        output_high = model(
            batch,
            snr_db=torch.tensor([8.0, 10.0, 12.0, 14.0]),
            sem_rate_ratio=0.5,
            det_rate_ratio=0.75,
            semantic_bandwidth_budget=20.0,
        )

        self.assertFalse(torch.allclose(output_low.shared_rx_sem, output_high.shared_rx_sem))


if __name__ == "__main__":
    unittest.main()

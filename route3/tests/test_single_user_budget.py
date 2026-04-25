from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluators import compute_effective_cbr_tensor, summarize_single_user_budget
from losses import ConditionalSSDDDecoderLoss
from models.ultimate import ConditionalSSDDDecoder, UltimateSingleUserTransmission
from scripts.route3_calibrate_single_user_budget import enumerate_budget_candidates, extract_candidate_frontier


class SingleUserBudgetTests(unittest.TestCase):
    def test_compute_effective_cbr_matches_closed_form(self) -> None:
        input_image = torch.rand(1, 3, 128, 128)
        semantic_active = torch.tensor([96.0])
        detail_active = torch.tensor([64.0])
        semantic_shape = torch.Size([1, 192, 8, 8])
        detail_shape = torch.Size([1, 128, 8, 8])

        effective_cbr = compute_effective_cbr_tensor(
            input_image=input_image,
            semantic_active_channels=semantic_active,
            detail_active_channels=detail_active,
            semantic_shape=semantic_shape,
            detail_shape=detail_shape,
        )

        expected = ((96.0 * 64.0) + (64.0 * 64.0)) / float(2 * 3 * 128 * 128)
        self.assertAlmostEqual(float(effective_cbr.item()), expected, places=8)

    def test_summarize_single_user_budget_marks_matching_target_as_feasible(self) -> None:
        torch.manual_seed(7)
        model = UltimateSingleUserTransmission(
            decoder=ConditionalSSDDDecoder(),
            decoder_loss=ConditionalSSDDDecoderLoss(),
        )
        model.eval()
        image = torch.rand(1, 3, 128, 128)

        with torch.no_grad():
            output = model(
                image,
                snr_db=10.0,
                sem_rate_ratio=0.5,
                det_rate_ratio=0.75,
                decode_stochastic=False,
                run_enhancement=False,
            )

        open_summary = summarize_single_user_budget(
            input_image=image,
            output=output,
            operating_mode="open_quality",
        )
        matched_summary = summarize_single_user_budget(
            input_image=image,
            output=output,
            operating_mode="matched_budget",
            target_effective_cbr=open_summary.effective_cbr,
            target_effective_cbr_tolerance=0.0,
        )

        self.assertGreater(open_summary.effective_cbr, 0.0)
        self.assertIsNone(open_summary.within_target_tolerance)
        self.assertEqual(matched_summary.semantic_total_channels, 192)
        self.assertEqual(matched_summary.detail_total_channels, 128)
        self.assertTrue(matched_summary.within_target_tolerance)
        self.assertAlmostEqual(matched_summary.cbr_absolute_gap or 0.0, 0.0, places=8)

    def test_budget_candidate_frontier_surfaces_all_exact_points_for_mixed_spatial_costs(self) -> None:
        input_image = torch.rand(1, 3, 128, 128)
        semantic_shape = torch.Size([1, 192, 8, 8])
        detail_shape = torch.Size([1, 128, 32, 32])
        target_effective_cbr = float(
            compute_effective_cbr_tensor(
                input_image=input_image,
                semantic_active_channels=torch.tensor([32.0]),
                detail_active_channels=torch.tensor([4.0]),
                semantic_shape=semantic_shape,
                detail_shape=detail_shape,
            ).item()
        )

        candidates = enumerate_budget_candidates(
            input_image=input_image,
            semantic_channels=192,
            detail_channels=128,
            semantic_shape=semantic_shape,
            detail_shape=detail_shape,
            target_effective_cbr=target_effective_cbr,
            tolerance=0.0,
            max_sem_active=80,
            max_det_active=5,
        )
        frontier = extract_candidate_frontier(candidates)

        self.assertTrue(candidates)
        self.assertTrue(candidates[0].within_target_tolerance)
        self.assertAlmostEqual(candidates[0].effective_cbr, target_effective_cbr, places=8)
        frontier_pairs = {
            (candidate.semantic_active_channels, candidate.detail_active_channels)
            for candidate in frontier
        }
        self.assertEqual(
            frontier_pairs,
            {
                (16, 5),
                (32, 4),
                (48, 3),
                (64, 2),
                (80, 1),
            },
        )
        self.assertGreater(len(frontier), 1)


if __name__ == "__main__":
    unittest.main()

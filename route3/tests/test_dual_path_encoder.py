from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ultimate.dual_path_encoder import UltimateDualPathEncoder
from models.ultimate.fusion_interface import DualPathEncoderOutput


class DualPathEncoderTests(unittest.TestCase):
    def test_dual_path_encoder_emits_expected_shapes(self) -> None:
        model = UltimateDualPathEncoder()
        image = torch.randn(2, 3, 128, 160)

        output = model(image)

        self.assertEqual(output.z_sem.shape, (2, 192, 8, 10))
        self.assertEqual(output.z_det.shape, (2, 128, 32, 40))
        self.assertEqual(tuple(t.shape[-2:] for t in output.sem_pyramid), ((32, 40), (16, 20), (8, 10)))
        self.assertEqual(tuple(t.shape[-2:] for t in output.det_pyramid), ((64, 80), (32, 40), (32, 40)))

    def test_dual_path_encoder_rejects_non_divisible_sizes(self) -> None:
        model = UltimateDualPathEncoder()
        image = torch.randn(1, 3, 130, 128)

        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            model(image)

    def test_fusion_interface_rejects_bad_semantic_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "z_sem must have H/16 x W/16 spatial size"):
            DualPathEncoderOutput(
                z_sem=torch.randn(1, 192, 9, 8),
                z_det=torch.randn(1, 128, 32, 32),
                sem_pyramid=(torch.randn(1, 64, 32, 32), torch.randn(1, 128, 16, 16), torch.randn(1, 192, 8, 8)),
                det_pyramid=(torch.randn(1, 64, 64, 64), torch.randn(1, 128, 32, 32), torch.randn(1, 128, 32, 32)),
                input_size=(128, 128),
            ).validate()

    def test_fusion_interface_rejects_non_monotonic_pyramid(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            DualPathEncoderOutput(
                z_sem=torch.randn(1, 192, 8, 8),
                z_det=torch.randn(1, 128, 32, 32),
                sem_pyramid=(torch.randn(1, 64, 16, 16), torch.randn(1, 128, 32, 32), torch.randn(1, 192, 8, 8)),
                det_pyramid=(torch.randn(1, 64, 64, 64), torch.randn(1, 128, 32, 32), torch.randn(1, 128, 32, 32)),
                input_size=(128, 128),
            ).validate()


if __name__ == "__main__":
    unittest.main()

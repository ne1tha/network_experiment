from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ultimate.dual_path_encoder import UltimateDualPathEncoder
from models.ultimate.full_model_single_user import UltimateSingleUserTransmission
from models.ultimate.semantic_distill import (
    LayerWiseAdaptiveDistillation,
    SemanticDistillationModule,
    SemanticTeacherEncoder,
)


class SemanticDistillationTests(unittest.TestCase):
    def test_missing_teacher_checkpoint_fails_fast(self) -> None:
        teacher = SemanticTeacherEncoder(backbone=UltimateDualPathEncoder().semantic_branch)
        distillation = SemanticDistillationModule(
            teacher=teacher,
            distiller=LayerWiseAdaptiveDistillation(num_stages=3),
        )
        model = UltimateSingleUserTransmission(semantic_distillation=distillation)
        image = torch.randn(1, 3, 128, 128)

        with self.assertRaisesRegex(RuntimeError, "no teacher checkpoint has been loaded"):
            model(image, snr_db=10.0, sem_rate_ratio=0.5, det_rate_ratio=0.5)

    def test_teacher_checkpoint_load_and_per_stage_outputs(self) -> None:
        student_encoder = UltimateDualPathEncoder()
        teacher_backbone = UltimateDualPathEncoder().semantic_branch

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "teacher_semantic.pt"
            torch.save(student_encoder.semantic_branch.state_dict(), checkpoint_path)

            teacher = SemanticTeacherEncoder(backbone=teacher_backbone, checkpoint_path=checkpoint_path)
            distillation = SemanticDistillationModule(
                teacher=teacher,
                distiller=LayerWiseAdaptiveDistillation(num_stages=3, beta=2.0),
            )
            model = UltimateSingleUserTransmission(encoder=student_encoder, semantic_distillation=distillation)

            image = torch.randn(2, 3, 128, 128)
            output = model(image, snr_db=torch.tensor([8.0, 12.0]), sem_rate_ratio=0.5, det_rate_ratio=0.75)

            self.assertIsNotNone(output.distillation)
            self.assertEqual(tuple(output.distillation.final_weights.shape), (3,))
            self.assertEqual(set(output.distillation.per_stage_loss.keys()), {"stage_1", "stage_2", "stage_3"})
            self.assertAlmostEqual(float(output.distillation.final_weights.sum().item()), 1.0, places=5)
            self.assertGreaterEqual(float(output.distillation.total_loss.item()), 0.0)

    def test_mismatched_stage_count_fails_fast(self) -> None:
        distiller = LayerWiseAdaptiveDistillation(num_stages=3)
        with self.assertRaisesRegex(ValueError, "expects exactly 3 semantic stages"):
            distiller(
                student_pyramid=(torch.randn(1, 8, 8, 8), torch.randn(1, 8, 4, 4)),
                teacher_pyramid=(torch.randn(1, 8, 8, 8), torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2)),
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.route3_eval_single_user import _run_single_image


def _write_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    tensor = (torch.rand(size[0], size[1], 3) * 255).to(torch.uint8).numpy()
    Image.fromarray(tensor).save(path)


class RecordingEvalModel:
    def __init__(self):
        self.decode_stochastic_calls: list[bool] = []
        self.run_enhancement_calls: list[bool] = []

    def __call__(self, source, **kwargs):
        self.decode_stochastic_calls.append(bool(kwargs.get("decode_stochastic", False)))
        self.run_enhancement_calls.append(bool(kwargs.get("run_enhancement", False)))
        return SimpleNamespace(reconstruction=SimpleNamespace(x_hat=source.clone()))


class Route3EvalSingleUserTests(unittest.TestCase):
    def test_run_single_image_respects_decode_stochastic_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            _write_image(image_path)
            model = RecordingEvalModel()

            x_hat, metrics = _run_single_image(
                model,
                image_path,
                torch.device("cpu"),
                snr_db=10.0,
                sem_rate_ratio=0.5,
                det_rate_ratio=0.75,
                decode_stochastic=True,
                run_enhancement=True,
            )

            self.assertEqual(model.decode_stochastic_calls, [True])
            self.assertEqual(model.run_enhancement_calls, [True])
            self.assertEqual(tuple(x_hat.shape), (1, 3, 32, 32))
            self.assertGreaterEqual(metrics["psnr"], 0.0)


if __name__ == "__main__":
    unittest.main()

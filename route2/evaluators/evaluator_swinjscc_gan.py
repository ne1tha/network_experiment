from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader

from route2_swinjscc_gan.common.io import ensure_dir, save_json
from route2_swinjscc_gan.evaluators.metrics import ImageQualityMetricSuite
from route2_swinjscc_gan.models.swinjscc_gan.generator import SwinJSCCGenerator


@dataclass
class EvaluationSummary:
    num_samples: int
    psnr: float
    ms_ssim: float
    lpips: float


class SwinJSCCGANEvaluator:
    def __init__(self, generator: SwinJSCCGenerator, metric_suite: ImageQualityMetricSuite) -> None:
        self.generator = generator
        self.metric_suite = metric_suite

    def evaluate(
        self,
        dataloader: DataLoader[tuple[torch.Tensor, list[str]]],
        *,
        device: torch.device,
        snr: int,
        rate: int,
        output_dir: str | Path | None = None,
        max_saved_images: int = 0,
    ) -> EvaluationSummary:
        self.generator.eval()
        self.metric_suite.eval()
        save_dir = ensure_dir(output_dir) if output_dir is not None else None

        total_psnr = 0.0
        total_ms_ssim = 0.0
        total_lpips = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_index, (images, names) in enumerate(dataloader):
                images = images.to(device, non_blocking=True)
                output = self.generator(images, snr=snr, rate=rate)
                reconstruction = output.reconstruction

                metrics = self.metric_suite(reconstruction, images)
                batch_size = images.shape[0]
                total_psnr += metrics.psnr * batch_size
                total_ms_ssim += metrics.ms_ssim * batch_size
                total_lpips += metrics.lpips * batch_size
                total_samples += batch_size

                if save_dir is not None and batch_index < max_saved_images:
                    self._save_visual_pair(save_dir, names[0], images[0], reconstruction[0])

        summary = EvaluationSummary(
            num_samples=total_samples,
            psnr=total_psnr / total_samples,
            ms_ssim=total_ms_ssim / total_samples,
            lpips=total_lpips / total_samples,
        )

        if save_dir is not None:
            save_json(
                save_dir / "metrics.json",
                {
                    "num_samples": summary.num_samples,
                    "psnr": summary.psnr,
                    "ms_ssim": summary.ms_ssim,
                    "lpips": summary.lpips,
                    "snr": snr,
                    "rate": rate,
                },
            )

        return summary

    def _save_visual_pair(self, save_dir: Path, image_name: str, source: torch.Tensor, reconstruction: torch.Tensor) -> None:
        source_image = self._to_image(source)
        reconstruction_image = self._to_image(reconstruction)
        source_image.save(save_dir / f"{image_name}_source.png")
        reconstruction_image.save(save_dir / f"{image_name}_reconstruction.png")

    @staticmethod
    def _to_image(tensor: torch.Tensor) -> Image.Image:
        array = tensor.clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(array)


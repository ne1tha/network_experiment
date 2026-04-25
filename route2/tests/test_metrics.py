import torch

from route2_swinjscc_gan.evaluators.metrics import ImageQualityMetricSuite, compute_psnr


def test_psnr_is_infinite_for_identical_images() -> None:
    image = torch.ones(1, 3, 64, 64)
    psnr = compute_psnr(image, image)
    assert torch.isinf(psnr).all()


def test_metric_suite_runs_on_small_batch() -> None:
    image = torch.rand(1, 3, 128, 128)
    metrics = ImageQualityMetricSuite(lpips_network="vgg")
    result = metrics(image, image)
    assert result.ms_ssim <= 1.0
    assert result.lpips >= 0.0

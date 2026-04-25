from route2_swinjscc_gan.scripts.compare_with_route1 import compare


def test_compare_marks_improvement_by_metric_direction() -> None:
    route1 = {"psnr": 20.0, "ms_ssim": 0.90, "lpips": 0.20}
    route2 = {"psnr": 21.5, "ms_ssim": 0.92, "lpips": 0.15}
    payload = compare(route1, route2)

    assert payload["psnr"]["route2_better"] == "yes"
    assert payload["ms_ssim"]["route2_better"] == "yes"
    assert payload["lpips"]["route2_better"] == "yes"

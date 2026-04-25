from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SingleUserBudgetMetrics:
    operating_mode: str
    effective_cbr: float
    target_effective_cbr: float | None
    target_effective_cbr_tolerance: float | None
    cbr_absolute_gap: float | None
    cbr_relative_gap: float | None
    within_target_tolerance: bool | None
    semantic_active_channels_mean: float
    detail_active_channels_mean: float
    semantic_rate_ratio_mean: float
    detail_rate_ratio_mean: float
    semantic_total_channels: int
    detail_total_channels: int
    semantic_latent_hw: tuple[int, int]
    detail_latent_hw: tuple[int, int]


def compute_effective_cbr_tensor(
    *,
    input_image: torch.Tensor,
    semantic_active_channels: torch.Tensor,
    detail_active_channels: torch.Tensor,
    semantic_shape: torch.Size,
    detail_shape: torch.Size,
) -> torch.Tensor:
    if input_image.ndim != 4:
        raise ValueError(f"Expected BCHW input_image, got shape {tuple(input_image.shape)}.")
    _, _, height, width = input_image.shape
    semantic_spatial = int(semantic_shape[-2]) * int(semantic_shape[-1])
    detail_spatial = int(detail_shape[-2]) * int(detail_shape[-1])
    input_real_values = 3 * int(height) * int(width)

    semantic_real_values = semantic_active_channels.to(dtype=input_image.dtype) * float(semantic_spatial)
    detail_real_values = detail_active_channels.to(dtype=input_image.dtype) * float(detail_spatial)
    return (semantic_real_values + detail_real_values) / float(2 * input_real_values)


def summarize_single_user_budget(
    *,
    input_image: torch.Tensor,
    output,
    operating_mode: str = "open_quality",
    target_effective_cbr: float | None = None,
    target_effective_cbr_tolerance: float = 0.05,
) -> SingleUserBudgetMetrics:
    semantic_active_channels = output.semantic.mask.active_channels.to(dtype=torch.float32)
    detail_active_channels = output.detail.mask.active_channels.to(dtype=torch.float32)
    effective_cbr_tensor = compute_effective_cbr_tensor(
        input_image=input_image,
        semantic_active_channels=semantic_active_channels,
        detail_active_channels=detail_active_channels,
        semantic_shape=output.semantic.tx.shape,
        detail_shape=output.detail.tx.shape,
    )
    effective_cbr = float(effective_cbr_tensor.mean().item())
    semantic_rate_ratio_mean = float(output.semantic.mask.rate_ratio.to(dtype=torch.float32).mean().item())
    detail_rate_ratio_mean = float(output.detail.mask.rate_ratio.to(dtype=torch.float32).mean().item())

    cbr_absolute_gap = None
    cbr_relative_gap = None
    within_target_tolerance = None
    if target_effective_cbr is not None:
        cbr_absolute_gap = abs(effective_cbr - float(target_effective_cbr))
        denominator = max(abs(float(target_effective_cbr)), 1e-12)
        cbr_relative_gap = cbr_absolute_gap / denominator
        within_target_tolerance = cbr_relative_gap <= float(target_effective_cbr_tolerance) + 1e-12

    return SingleUserBudgetMetrics(
        operating_mode=operating_mode,
        effective_cbr=effective_cbr,
        target_effective_cbr=float(target_effective_cbr) if target_effective_cbr is not None else None,
        target_effective_cbr_tolerance=(
            float(target_effective_cbr_tolerance) if target_effective_cbr is not None else None
        ),
        cbr_absolute_gap=cbr_absolute_gap,
        cbr_relative_gap=cbr_relative_gap,
        within_target_tolerance=within_target_tolerance,
        semantic_active_channels_mean=float(semantic_active_channels.mean().item()),
        detail_active_channels_mean=float(detail_active_channels.mean().item()),
        semantic_rate_ratio_mean=semantic_rate_ratio_mean,
        detail_rate_ratio_mean=detail_rate_ratio_mean,
        semantic_total_channels=int(output.semantic.tx.shape[1]),
        detail_total_channels=int(output.detail.tx.shape[1]),
        semantic_latent_hw=(int(output.semantic.tx.shape[-2]), int(output.semantic.tx.shape[-1])),
        detail_latent_hw=(int(output.detail.tx.shape[-2]), int(output.detail.tx.shape[-1])),
    )


def budget_metrics_to_dict(metrics: SingleUserBudgetMetrics) -> dict[str, Any]:
    return asdict(metrics)


def average_single_user_budget_metrics(
    metrics_list: list[SingleUserBudgetMetrics | None],
) -> dict[str, Any] | None:
    available = [metrics for metrics in metrics_list if metrics is not None]
    if not available:
        return None

    def _mean(values: list[float]) -> float:
        return sum(values) / float(len(values))

    target_available = [metrics for metrics in available if metrics.target_effective_cbr is not None]
    within_values = [
        1.0 if metrics.within_target_tolerance else 0.0
        for metrics in target_available
        if metrics.within_target_tolerance is not None
    ]

    summary: dict[str, Any] = {
        "operating_mode": available[0].operating_mode,
        "effective_cbr": _mean([metrics.effective_cbr for metrics in available]),
        "semantic_active_channels_mean": _mean([metrics.semantic_active_channels_mean for metrics in available]),
        "detail_active_channels_mean": _mean([metrics.detail_active_channels_mean for metrics in available]),
        "semantic_rate_ratio_mean": _mean([metrics.semantic_rate_ratio_mean for metrics in available]),
        "detail_rate_ratio_mean": _mean([metrics.detail_rate_ratio_mean for metrics in available]),
        "semantic_total_channels": available[0].semantic_total_channels,
        "detail_total_channels": available[0].detail_total_channels,
        "semantic_latent_hw": list(available[0].semantic_latent_hw),
        "detail_latent_hw": list(available[0].detail_latent_hw),
        "target_effective_cbr": available[0].target_effective_cbr,
        "target_effective_cbr_tolerance": available[0].target_effective_cbr_tolerance,
        "cbr_absolute_gap": (
            _mean([metrics.cbr_absolute_gap for metrics in target_available if metrics.cbr_absolute_gap is not None])
            if target_available
            else None
        ),
        "cbr_relative_gap": (
            _mean([metrics.cbr_relative_gap for metrics in target_available if metrics.cbr_relative_gap is not None])
            if target_available
            else None
        ),
        "within_target_tolerance": all(within_values) if within_values else None,
        "within_target_tolerance_ratio": _mean(within_values) if within_values else None,
    }
    return summary

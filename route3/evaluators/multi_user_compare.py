from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .single_user_metrics import ReconstructionMetrics, compute_reconstruction_metrics


@dataclass(frozen=True)
class MultiUserComparison:
    single_user_metrics: ReconstructionMetrics
    shared_base_metrics: ReconstructionMetrics | None
    shared_final_metrics: ReconstructionMetrics | None
    multi_user_metrics: ReconstructionMetrics
    no_sharing_metrics: ReconstructionMetrics
    semantic_sharing_gain_psnr: float
    semantic_sharing_gain_mae: float
    shared_refinement_gain_psnr: float | None
    pair_cost_mean: float
    total_bandwidth_budget: float
    notes: str


class MultiUserComparisonEvaluator:
    """Compare single-user, no-sharing, and semantic-sharing multi-user outputs."""

    @torch.no_grad()
    def evaluate_batch(
        self,
        x: torch.Tensor,
        multi_user_output,
    ) -> MultiUserComparison:
        if multi_user_output.single_user.reconstruction is None:
            raise RuntimeError("Single-user baseline reconstruction is missing.")
        if multi_user_output.shared_reconstruction is None:
            raise RuntimeError("Multi-user shared reconstruction is missing.")

        single_user_metrics = compute_reconstruction_metrics(x, multi_user_output.single_user.reconstruction.x_hat)
        shared_base_metrics = None
        if multi_user_output.base_shared_reconstruction is not None:
            shared_base_metrics = compute_reconstruction_metrics(x, multi_user_output.base_shared_reconstruction.x_hat)
        shared_final_metrics = compute_reconstruction_metrics(x, multi_user_output.shared_reconstruction.x_hat)
        multi_user_metrics = shared_final_metrics
        no_sharing_metrics = single_user_metrics
        shared_refinement_gain_psnr = None
        if shared_base_metrics is not None:
            shared_refinement_gain_psnr = shared_final_metrics.psnr - shared_base_metrics.psnr

        return MultiUserComparison(
            single_user_metrics=single_user_metrics,
            shared_base_metrics=shared_base_metrics,
            shared_final_metrics=shared_final_metrics,
            multi_user_metrics=multi_user_metrics,
            no_sharing_metrics=no_sharing_metrics,
            semantic_sharing_gain_psnr=multi_user_metrics.psnr - no_sharing_metrics.psnr,
            semantic_sharing_gain_mae=no_sharing_metrics.mean_abs_error - multi_user_metrics.mean_abs_error,
            shared_refinement_gain_psnr=shared_refinement_gain_psnr,
            pair_cost_mean=float(multi_user_output.pairing.pair_costs.mean().item()),
            total_bandwidth_budget=float(multi_user_output.bandwidth.total_budget),
            notes=(
                "Single-user baseline inside multi-user compare is the non-sharing decode path. "
                "shared_base/shared_final expose the refinement gain on the shared branch."
            ),
        )


class Phase8ReportWriter:
    """Write a compact markdown report for phase-8 comparison outputs."""

    def write_report(
        self,
        output_path: str | Path,
        comparison: MultiUserComparison,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        extra = extra or {}

        lines = [
            "# Route 3 Phase 8 Multi-user Evaluation Report",
            "",
            "## Comparison Summary",
            "",
            f"- single_user_psnr: {comparison.single_user_metrics.psnr:.4f}",
            f"- shared_base_psnr: {comparison.shared_base_metrics.psnr:.4f}" if comparison.shared_base_metrics is not None else "- shared_base_psnr: null",
            f"- shared_final_psnr: {comparison.shared_final_metrics.psnr:.4f}" if comparison.shared_final_metrics is not None else "- shared_final_psnr: null",
            f"- multi_user_psnr: {comparison.multi_user_metrics.psnr:.4f}",
            f"- no_sharing_psnr: {comparison.no_sharing_metrics.psnr:.4f}",
            f"- semantic_sharing_gain_psnr: {comparison.semantic_sharing_gain_psnr:.4f}",
            f"- semantic_sharing_gain_mae: {comparison.semantic_sharing_gain_mae:.6f}",
            (
                f"- shared_refinement_gain_psnr: {comparison.shared_refinement_gain_psnr:.4f}"
                if comparison.shared_refinement_gain_psnr is not None
                else "- shared_refinement_gain_psnr: null"
            ),
            f"- pair_cost_mean: {comparison.pair_cost_mean:.6f}",
            f"- total_bandwidth_budget: {comparison.total_bandwidth_budget:.4f}",
            "",
            "## Notes",
            "",
            f"- {comparison.notes}",
        ]

        if extra:
            lines.extend(["", "## Extra State", ""])
            for key, value in extra.items():
                lines.append(f"- {key}: {value}")

        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def comparison_to_dict(comparison: MultiUserComparison) -> dict[str, Any]:
    result = asdict(comparison)
    return result

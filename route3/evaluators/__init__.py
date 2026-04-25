"""Evaluation utilities for route 3."""

from .ablation_single_user import AblationComparison, SingleUserAblationRunner
from .multi_user_compare import MultiUserComparison, MultiUserComparisonEvaluator, Phase8ReportWriter, comparison_to_dict
from .single_user_budget import (
    SingleUserBudgetMetrics,
    average_single_user_budget_metrics,
    budget_metrics_to_dict,
    compute_effective_cbr_tensor,
    summarize_single_user_budget,
)
from .single_user_metrics import ReconstructionMetrics, compute_reconstruction_metrics

__all__ = [
    "AblationComparison",
    "MultiUserComparison",
    "MultiUserComparisonEvaluator",
    "Phase8ReportWriter",
    "ReconstructionMetrics",
    "SingleUserBudgetMetrics",
    "SingleUserAblationRunner",
    "average_single_user_budget_metrics",
    "budget_metrics_to_dict",
    "comparison_to_dict",
    "compute_effective_cbr_tensor",
    "compute_reconstruction_metrics",
    "summarize_single_user_budget",
]

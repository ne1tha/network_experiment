from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BandwidthAllocationOutput:
    pair_bandwidth: torch.Tensor
    per_user_bandwidth: torch.Tensor
    required_bandwidth: torch.Tensor
    total_budget: float


class SemanticBandwidthAllocator:
    """Allocate semantic bandwidth based on pair difficulty under a fixed budget."""

    def __call__(self, pair_costs: torch.Tensor, total_budget: float) -> BandwidthAllocationOutput:
        if pair_costs.ndim != 1:
            raise ValueError(f"pair_costs must be 1D, got {pair_costs.ndim}D.")
        if total_budget <= 0:
            raise ValueError(f"total_budget must be positive, got {total_budget}.")

        required = pair_costs + 1e-3
        required = required / required.sum().clamp_min(1e-8) * total_budget
        pair_bandwidth = required
        per_user_bandwidth = torch.repeat_interleave(pair_bandwidth, repeats=2)
        return BandwidthAllocationOutput(
            pair_bandwidth=pair_bandwidth,
            per_user_bandwidth=per_user_bandwidth,
            required_bandwidth=required,
            total_budget=total_budget,
        )


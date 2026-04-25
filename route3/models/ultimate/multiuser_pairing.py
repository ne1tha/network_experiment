from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairingOutput:
    pair_indices: torch.Tensor
    pair_costs: torch.Tensor
    cost_matrix: torch.Tensor


class SemanticPairingAllocator:
    """Greedy minimum-cost pairing over semantic embeddings."""

    def __call__(self, semantic_vectors: torch.Tensor) -> PairingOutput:
        if semantic_vectors.ndim != 2:
            raise ValueError(f"SemanticPairingAllocator expects [users, dim], got {tuple(semantic_vectors.shape)}.")
        num_users = semantic_vectors.shape[0]
        if num_users % 2 != 0:
            raise ValueError(f"SemanticPairingAllocator expects an even number of users, got {num_users}.")

        normalized = F.normalize(semantic_vectors, dim=-1)
        similarity = normalized @ normalized.transpose(0, 1)
        cost_matrix = 1.0 - similarity

        unused = list(range(num_users))
        pairs = []
        costs = []
        while unused:
            i = unused.pop(0)
            best_j = None
            best_cost = None
            for j in unused:
                candidate_cost = float(cost_matrix[i, j].item())
                if best_cost is None or candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_j = j
            if best_j is None:
                raise RuntimeError("Failed to build a valid semantic pairing.")
            unused.remove(best_j)
            pairs.append((i, best_j))
            costs.append(best_cost)

        return PairingOutput(
            pair_indices=torch.tensor(pairs, dtype=torch.long, device=semantic_vectors.device),
            pair_costs=torch.tensor(costs, dtype=semantic_vectors.dtype, device=semantic_vectors.device),
            cost_matrix=cost_matrix,
        )


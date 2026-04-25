from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return (a * b).sum(dim=-1)


@dataclass(frozen=True)
class CUAOutput:
    shared_semantic: torch.Tensor
    pair_indices: torch.Tensor
    pair_gates: torch.Tensor
    pair_similarity: torch.Tensor


class CrossUserAttention(nn.Module):
    """Semantic-only cross-user attention with cosine gating."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels {channels} must be divisible by num_heads {num_heads}.")
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        self.gate_scale = nn.Parameter(torch.tensor(2.0))
        self.mlp = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, z_sem: torch.Tensor, pair_indices: torch.Tensor) -> CUAOutput:
        if z_sem.ndim != 4:
            raise ValueError(f"CrossUserAttention expects 4D semantic features, got {z_sem.ndim}D.")
        batch_size, channels, height, width = z_sem.shape
        if batch_size % 2 != 0:
            raise ValueError(f"CrossUserAttention expects an even number of users, got {batch_size}.")
        if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
            raise ValueError(f"pair_indices must have shape [num_pairs, 2], got {tuple(pair_indices.shape)}.")

        tokens = z_sem.flatten(2).transpose(1, 2)
        updated_list = list(tokens.unbind(0))
        similarities = []
        gates = []

        for pair in pair_indices:
            i = int(pair[0].item())
            j = int(pair[1].item())
            base_i = updated_list[i]
            base_j = updated_list[j]
            q_i = self.norm(base_i).unsqueeze(0)
            q_j = self.norm(base_j).unsqueeze(0)

            cross_i, _ = self.attn(q_i, q_j, q_j, need_weights=False)
            cross_j, _ = self.attn(q_j, q_i, q_i, need_weights=False)

            sim = _pairwise_cosine(q_i.mean(dim=1), q_j.mean(dim=1)).squeeze(0)
            gate = torch.sigmoid(self.gate_scale * sim)

            new_i = base_i + self.res_scale * gate * cross_i.squeeze(0)
            new_j = base_j + self.res_scale * gate * cross_j.squeeze(0)
            new_i = new_i + self.mlp(new_i)
            new_j = new_j + self.mlp(new_j)

            updated_list[i] = new_i
            updated_list[j] = new_j

            similarities.append(sim)
            gates.append(gate)

        updated = torch.stack(updated_list, dim=0)
        shared = updated.transpose(1, 2).reshape(batch_size, channels, height, width)
        return CUAOutput(
            shared_semantic=shared,
            pair_indices=pair_indices,
            pair_gates=torch.stack(gates) if gates else z_sem.new_zeros((0,)),
            pair_similarity=torch.stack(similarities) if similarities else z_sem.new_zeros((0,)),
        )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    module.requires_grad_(False)
    return module


def _validate_stage_weights(num_stages: int, stage_weights: Sequence[float] | None) -> torch.Tensor:
    if stage_weights is None:
        return torch.ones(num_stages, dtype=torch.float32)
    if len(stage_weights) != num_stages:
        raise ValueError(f"Expected {num_stages} stage weights, got {len(stage_weights)}.")
    weights = torch.tensor(stage_weights, dtype=torch.float32)
    if torch.any(weights <= 0):
        raise ValueError(f"All stage weights must be positive, got {weights.tolist()}.")
    return weights


@dataclass(frozen=True)
class DistillationOutput:
    total_loss: torch.Tensor
    final_weights: torch.Tensor
    alignment_penalty: torch.Tensor
    base_weights: torch.Tensor
    per_stage_loss: dict[str, torch.Tensor]


class SemanticTeacherEncoder(nn.Module):
    """Frozen semantic teacher wrapper with explicit checkpoint loading rules."""

    def __init__(self, backbone: nn.Module, checkpoint_path: str | Path | None = None, strict: bool = True):
        super().__init__()
        self.backbone = _freeze_module(backbone)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
        self.strict = strict

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, strict=strict)

    def load_checkpoint(self, checkpoint_path: str | Path, strict: bool = True) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Semantic teacher checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.backbone.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "Teacher checkpoint is incompatible with the semantic teacher backbone; "
                f"missing={missing}, unexpected={unexpected}."
            )
        self.backbone = _freeze_module(self.backbone)
        self.checkpoint_path = str(checkpoint_path)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        with torch.no_grad():
            return self.backbone(x)


class LayerWiseAdaptiveDistillation(nn.Module):
    """UniFlow-style adaptive self-distillation for the semantic branch."""

    def __init__(self, num_stages: int, beta: float = 2.0, stage_weights: Sequence[float] | None = None):
        super().__init__()
        if num_stages <= 0:
            raise ValueError(f"num_stages must be positive, got {num_stages}.")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}.")

        self.num_stages = num_stages
        self.beta = beta
        self.register_buffer("stage_weights", _validate_stage_weights(num_stages, stage_weights), persistent=False)

    @staticmethod
    def _flatten_normalized(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"Expected 4D semantic feature map, got {feature.ndim}D.")
        flattened = feature.flatten(2).transpose(1, 2)
        return F.normalize(flattened, dim=-1)

    def forward(
        self,
        student_pyramid: Iterable[torch.Tensor],
        teacher_pyramid: Iterable[torch.Tensor],
    ) -> DistillationOutput:
        student_pyramid = tuple(student_pyramid)
        teacher_pyramid = tuple(teacher_pyramid)
        if len(student_pyramid) != self.num_stages or len(teacher_pyramid) != self.num_stages:
            raise ValueError(
                f"Distillation expects exactly {self.num_stages} semantic stages, got "
                f"student={len(student_pyramid)}, teacher={len(teacher_pyramid)}."
            )

        penalties = []
        losses = []
        for stage_idx, (student_feature, teacher_feature) in enumerate(zip(student_pyramid, teacher_pyramid), start=1):
            if student_feature.shape != teacher_feature.shape:
                raise ValueError(
                    f"Semantic distillation stage {stage_idx} shape mismatch: "
                    f"student={tuple(student_feature.shape)}, teacher={tuple(teacher_feature.shape)}."
                )
            student_tokens = self._flatten_normalized(student_feature)
            teacher_tokens = self._flatten_normalized(teacher_feature)
            cosine_distance = 1.0 - (student_tokens * teacher_tokens).sum(dim=-1)
            stage_loss = cosine_distance.mean().clamp_min(0.0)
            losses.append(stage_loss)
            penalties.append(stage_loss.detach())

        losses_tensor = torch.stack(losses)
        penalties_tensor = torch.stack(penalties)

        base = torch.arange(1, self.num_stages + 1, device=losses_tensor.device, dtype=losses_tensor.dtype)
        base = base / float(self.num_stages)
        base = base * self.stage_weights.to(device=losses_tensor.device, dtype=losses_tensor.dtype)

        final_weights = base * torch.exp(self.beta * penalties_tensor)
        final_weights = final_weights / final_weights.sum().clamp_min(1e-8)
        total_loss = torch.sum(final_weights * losses_tensor)

        per_stage_loss = {f"stage_{idx}": loss for idx, loss in enumerate(losses, start=1)}
        return DistillationOutput(
            total_loss=total_loss,
            final_weights=final_weights,
            alignment_penalty=penalties_tensor,
            base_weights=base,
            per_stage_loss=per_stage_loss,
        )


class SemanticDistillationModule(nn.Module):
    """Teacher plus adaptive layer-wise distillation wrapper."""

    def __init__(self, teacher: SemanticTeacherEncoder, distiller: LayerWiseAdaptiveDistillation):
        super().__init__()
        self.teacher = teacher
        self.distiller = distiller

    def forward(self, x: torch.Tensor, student_pyramid: Iterable[torch.Tensor]) -> DistillationOutput:
        if not self.teacher.checkpoint_path:
            raise RuntimeError("Semantic distillation is enabled but no teacher checkpoint has been loaded.")
        _, teacher_pyramid = self.teacher(x)
        return self.distiller(student_pyramid=student_pyramid, teacher_pyramid=teacher_pyramid)

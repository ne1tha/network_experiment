from __future__ import annotations

import torch
import torch.nn as nn

from .encoder_detail_cnn import DetailCNNEncoder
from .encoder_semantic_swin import SemanticSwinEncoder
from .fusion_interface import DualPathEncoderOutput, assert_valid_image_size


class UltimateDualPathEncoder(nn.Module):
    """Phase-1 route-3 encoder wrapper."""

    def __init__(
        self,
        semantic_branch: SemanticSwinEncoder | None = None,
        detail_branch: DetailCNNEncoder | None = None,
    ):
        super().__init__()
        self.semantic_branch = semantic_branch or SemanticSwinEncoder()
        self.detail_branch = detail_branch or DetailCNNEncoder()

        if self.semantic_branch is None:
            raise ValueError("semantic_branch must be enabled for route 3.")
        if self.detail_branch is None:
            raise ValueError("detail_branch must be enabled for route 3.")

    def forward(self, x: torch.Tensor) -> DualPathEncoderOutput:
        if x.ndim != 4:
            raise ValueError(f"UltimateDualPathEncoder expects a 4D tensor, got {x.ndim}D.")

        _, _, height, width = x.shape
        assert_valid_image_size(height, width)

        z_sem, sem_pyramid = self.semantic_branch(x)
        z_det, det_pyramid = self.detail_branch(x)
        return DualPathEncoderOutput(
            z_sem=z_sem,
            z_det=z_det,
            sem_pyramid=sem_pyramid,
            det_pyramid=det_pyramid,
            input_size=(height, width),
        ).validate()


from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .fusion_interface import assert_valid_image_size


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return self.act(x)


class DetailCNNEncoder(nn.Module):
    """High-resolution CNN detail branch with an H/4 output contract."""

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 32,
        stage_channels: Tuple[int, int] = (64, 128),
        blocks_per_stage: Tuple[int, int] = (2, 2),
        out_channels: int = 128,
    ):
        super().__init__()
        if len(stage_channels) != 2 or len(blocks_per_stage) != 2:
            raise ValueError("DetailCNNEncoder expects exactly two stages for the H/4 contract.")

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_channels, stride=1),
            ConvNormAct(stem_channels, stage_channels[0], stride=2),
        )
        self.stage1 = nn.Sequential(*[ResidualConvBlock(stage_channels[0]) for _ in range(blocks_per_stage[0])])
        self.downsample = ConvNormAct(stage_channels[0], stage_channels[1], stride=2)
        self.stage2 = nn.Sequential(*[ResidualConvBlock(stage_channels[1]) for _ in range(blocks_per_stage[1])])
        self.head = nn.Sequential(
            ConvNormAct(stage_channels[1], stage_channels[1], stride=1),
            nn.Conv2d(stage_channels[1], out_channels, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if x.ndim != 4:
            raise ValueError(f"DetailCNNEncoder expects a 4D tensor, got {x.ndim}D.")

        _, _, height, width = x.shape
        assert_valid_image_size(height, width)

        x = self.stem(x)
        stage1 = self.stage1(x)
        x = self.downsample(stage1)
        stage2 = self.stage2(x)
        z_det = self.head(stage2)
        det_pyramid = (stage1, stage2, z_det)
        return z_det, det_pyramid


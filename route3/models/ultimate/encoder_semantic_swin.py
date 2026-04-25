from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion_interface import assert_valid_image_size


def _to_2tuple(value: int) -> Tuple[int, int]:
    return (value, value)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, channels)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    batch = windows.shape[0] // ((height // window_size) * (width // window_size))
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}.")

        self.dim = dim
        self.window_size = _to_2tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        num_relative_positions = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_positions, num_heads))

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, num_tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        query = query * self.scale
        attention = query @ key.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(batch_windows // num_windows, num_windows, self.num_heads, num_tokens, num_tokens)
            attention = attention + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, num_tokens, num_tokens)

        attention = self.softmax(attention)
        x = (attention @ value).transpose(1, 2).reshape(batch_windows, num_tokens, channels)
        return self.proj(x)


class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class PatchMerging2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_channels)
        self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                "PatchMerging2D requires even spatial sizes, "
                f"received {(height, width)}."
            )

        x = x.permute(0, 2, 3, 1).contiguous()
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SwinTransformerBlock2D(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 4)

    def _attention_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        marker = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = marker
                marker += 1

        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, float("-inf"))
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float(0.0))
        return attention_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        shortcut = x

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm1(x)

        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = x.shape[1], x.shape[2]

        attention_mask = self._attention_mask(padded_h, padded_w, x.device)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)
        attn_windows = self.attn(x_windows, mask=attention_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        x = window_reverse(attn_windows, self.window_size, padded_h, padded_w)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        if pad_h or pad_w:
            x = x[:, :height, :width, :]

        x = shortcut + x.permute(0, 3, 1, 2).contiguous()

        y = x.permute(0, 2, 3, 1).contiguous()
        y = self.norm2(y)
        y = self.mlp(y)
        y = y.permute(0, 3, 1, 2).contiguous()
        return x + y


class SwinStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        num_heads: int,
        window_size: int,
        downsample: bool,
    ):
        super().__init__()
        self.downsample = PatchMerging2D(in_channels, out_channels) if downsample else None
        dim = out_channels if downsample else in_channels
        if not downsample and in_channels != out_channels:
            raise ValueError("Non-downsampling Swin stages must preserve channel width.")

        blocks = []
        for idx in range(depth):
            shift_size = 0 if idx % 2 == 0 else window_size // 2
            blocks.append(
                SwinTransformerBlock2D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class SemanticSwinEncoder(nn.Module):
    """Low-resolution Swin branch with an H/16 semantic contract."""

    def __init__(
        self,
        in_channels: int = 3,
        image_downsample: int = 2,
        patch_size: int = 2,
        embed_dims: Tuple[int, int, int] = (64, 128, 192),
        depths: Tuple[int, int, int] = (2, 2, 4),
        num_heads: Tuple[int, int, int] = (4, 8, 12),
        window_size: int = 4,
        out_channels: int = 192,
    ):
        super().__init__()
        if not (len(embed_dims) == len(depths) == len(num_heads) == 3):
            raise ValueError("SemanticSwinEncoder expects exactly three hierarchical stages.")

        total_downsample = image_downsample * patch_size * 2 ** (len(embed_dims) - 1)
        if total_downsample != 16:
            raise ValueError(
                "SemanticSwinEncoder must preserve the route-3 H/16 contract; "
                f"configured total downsample is {total_downsample}."
            )

        self.image_downsample = image_downsample
        self.input_downsample = nn.AvgPool2d(kernel_size=image_downsample, stride=image_downsample)
        self.patch_embed = PatchEmbed2D(in_channels=in_channels, embed_dim=embed_dims[0], patch_size=patch_size)
        self.stages = nn.ModuleList(
            [
                SwinStage(
                    in_channels=embed_dims[0],
                    out_channels=embed_dims[0],
                    depth=depths[0],
                    num_heads=num_heads[0],
                    window_size=window_size,
                    downsample=False,
                ),
                SwinStage(
                    in_channels=embed_dims[0],
                    out_channels=embed_dims[1],
                    depth=depths[1],
                    num_heads=num_heads[1],
                    window_size=window_size,
                    downsample=True,
                ),
                SwinStage(
                    in_channels=embed_dims[1],
                    out_channels=embed_dims[2],
                    depth=depths[2],
                    num_heads=num_heads[2],
                    window_size=window_size,
                    downsample=True,
                ),
            ]
        )
        self.out_norm = nn.LayerNorm(embed_dims[-1])
        self.out_proj = nn.Conv2d(embed_dims[-1], out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if x.ndim != 4:
            raise ValueError(f"SemanticSwinEncoder expects a 4D tensor, got {x.ndim}D.")

        _, _, height, width = x.shape
        assert_valid_image_size(height, width)

        x = self.input_downsample(x)
        x = self.patch_embed(x)

        pyramid = []
        for stage in self.stages:
            x = stage(x)
            pyramid.append(x)

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.out_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        z_sem = self.out_proj(x)
        return z_sem, tuple(pyramid)


from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from channels import ChannelState


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"Expected 1D timesteps, got shape {t.shape}.")
        half_dim = self.dim // 2
        freq = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device, dtype=t.dtype))
            * torch.arange(0, half_dim, device=t.device, dtype=t.dtype)
            / max(half_dim - 1, 1)
        )
        angles = t.unsqueeze(1) * freq.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class FiLMResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, context_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.context_proj = nn.Linear(context_dim, out_channels * 2)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale, shift = self.context_proj(context).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = self.conv1(x)
        h = self.norm1(h)
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.skip(x)


class SpatialTransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels {channels} must be divisible by num_heads {num_heads}.")
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(channels * mlp_ratio, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attn_input = self.norm1(tokens)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + attn_output
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class SemanticConditionPyramid(nn.Module):
    """Build semantic conditioning maps at multiple decoder resolutions."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.low = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        self.high = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        low_size: tuple[int, int],
        mid_size: tuple[int, int],
        high_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low = self.low(x)
        if low.shape[-2:] != low_size:
            low = F.interpolate(low, size=low_size, mode="bilinear", align_corners=False)
        mid = self.mid(low)
        if mid.shape[-2:] != mid_size:
            mid = F.interpolate(mid, size=mid_size, mode="bilinear", align_corners=False)
        high = self.high(mid)
        if high.shape[-2:] != high_size:
            high = F.interpolate(high, size=high_size, mode="bilinear", align_corners=False)
        return low, mid, high


class DetailConditionPyramid(nn.Module):
    """Build detail conditioning maps at multiple decoder resolutions."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.high = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        self.low = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        low_size: tuple[int, int],
        mid_size: tuple[int, int],
        high_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        high = self.high(x)
        if high.shape[-2:] != high_size:
            high = F.interpolate(high, size=high_size, mode="bilinear", align_corners=False)
        mid = self.mid(high)
        if mid.shape[-2:] != mid_size:
            mid = F.interpolate(mid, size=mid_size, mode="bilinear", align_corners=False)
        low = self.low(mid)
        if low.shape[-2:] != low_size:
            low = F.interpolate(low, size=low_size, mode="bilinear", align_corners=False)
        return low, mid, high


@dataclass(frozen=True)
class DecoderOutput:
    x_hat: torch.Tensor
    predicted_residual: torch.Tensor | None = None
    noisy_input: torch.Tensor | None = None
    conditioning_map: torch.Tensor | None = None
    semantic_condition_map: torch.Tensor | None = None
    detail_condition_map: torch.Tensor | None = None
    bottleneck_condition_map: torch.Tensor | None = None
    context_vector: torch.Tensor | None = None
    decode_steps: int = 1
    stochastic: bool = True
    base_x_hat: torch.Tensor | None = None
    refinement_delta: torch.Tensor | None = None
    output_kind: str = "base"


class ConditionalSSDDDecoder(nn.Module):
    """SSDD-inspired conditional single-step generative decoder for route 3."""

    def __init__(
        self,
        semantic_channels: int = 192,
        detail_channels: int = 128,
        hidden_channels: int = 128,
        output_channels: int = 3,
        transformer_heads: int = 4,
    ):
        super().__init__()
        self.semantic_channels = semantic_channels
        self.detail_channels = detail_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        self.detail_proj = nn.Sequential(
            nn.Conv2d(detail_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        self.condition_fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )

        self.noise_stem = nn.Sequential(
            nn.Conv2d(output_channels, hidden_channels // 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(_group_count(hidden_channels // 2), hidden_channels // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )

        context_dim = hidden_channels * 4
        self.time_embed = SinusoidalTimeEmbedding(hidden_channels)
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_channels + 8, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
        )

        self.down_block = FiLMResBlock(hidden_channels * 2, hidden_channels * 2, context_dim)
        self.downsample = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=2, padding=1)
        self.mid_block = SpatialTransformerBlock(hidden_channels * 2, num_heads=transformer_heads)
        self.mid_res = FiLMResBlock(hidden_channels * 2, hidden_channels * 2, context_dim)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        self.up_block = FiLMResBlock(hidden_channels * 2, hidden_channels, context_dim)
        self.out_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels // 2), hidden_channels // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, output_channels, kernel_size=3, padding=1),
        )

    @staticmethod
    def _expand_scalar(value: torch.Tensor | float, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(value):
            value = torch.tensor([value], device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype).reshape(-1)
        if value.numel() == 1:
            return value.expand(batch_size)
        if value.numel() != batch_size:
            raise ValueError(f"Expected scalar or batch-sized condition for batch {batch_size}, got {value.numel()}.")
        return value

    def _context_vector(
        self,
        sem_state: ChannelState,
        det_state: ChannelState,
        sem_rate_ratio: torch.Tensor | float,
        det_rate_ratio: torch.Tensor | float,
        decode_timestep: torch.Tensor | float,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sem_rate_ratio = self._expand_scalar(sem_rate_ratio, batch_size, device, dtype)
        det_rate_ratio = self._expand_scalar(det_rate_ratio, batch_size, device, dtype)
        decode_timestep = self._expand_scalar(decode_timestep, batch_size, device, dtype)
        time_emb = self.time_embed(decode_timestep)

        stats = torch.stack(
            [
                sem_state.snr_db.to(device=device, dtype=dtype),
                det_state.snr_db.to(device=device, dtype=dtype),
                sem_state.noise_std.to(device=device, dtype=dtype),
                det_state.noise_std.to(device=device, dtype=dtype),
                sem_state.tx_power.to(device=device, dtype=dtype),
                det_state.tx_power.to(device=device, dtype=dtype),
                sem_rate_ratio,
                det_rate_ratio,
            ],
            dim=1,
        )
        return self.context_mlp(torch.cat([time_emb, stats], dim=1))

    def forward(
        self,
        rx_sem: torch.Tensor,
        rx_det: torch.Tensor,
        sem_state: ChannelState,
        det_state: ChannelState,
        sem_rate_ratio: torch.Tensor | float,
        det_rate_ratio: torch.Tensor | float,
        output_size: tuple[int, int],
        noise_image: torch.Tensor | None = None,
        decode_timestep: torch.Tensor | float = 1.0,
        stochastic: bool = True,
    ) -> DecoderOutput:
        if rx_sem.ndim != 4 or rx_det.ndim != 4:
            raise ValueError("ConditionalSSDDDecoder expects 4D semantic and detail tensors.")

        batch_size = rx_sem.shape[0]
        if rx_det.shape[0] != batch_size:
            raise ValueError("Semantic and detail receive tensors must have the same batch size.")

        if noise_image is None:
            if stochastic:
                noise_image = torch.randn(batch_size, self.output_channels, output_size[0], output_size[1], device=rx_sem.device, dtype=rx_sem.dtype)
            else:
                noise_image = torch.zeros(batch_size, self.output_channels, output_size[0], output_size[1], device=rx_sem.device, dtype=rx_sem.dtype)
        elif noise_image.shape != (batch_size, self.output_channels, output_size[0], output_size[1]):
            raise ValueError(
                f"noise_image must have shape {(batch_size, self.output_channels, output_size[0], output_size[1])}, "
                f"got {tuple(noise_image.shape)}."
            )

        sem = self.semantic_proj(rx_sem)
        det = self.detail_proj(rx_det)
        if sem.shape[-2:] != det.shape[-2:]:
            sem = F.interpolate(sem, size=det.shape[-2:], mode="bilinear", align_corners=False)
        conditioning_map = self.condition_fuse(torch.cat([sem, det], dim=1))

        context = self._context_vector(
            sem_state=sem_state,
            det_state=det_state,
            sem_rate_ratio=sem_rate_ratio,
            det_rate_ratio=det_rate_ratio,
            decode_timestep=decode_timestep,
            batch_size=batch_size,
            device=rx_sem.device,
            dtype=rx_sem.dtype,
        )

        noise_feat = self.noise_stem(noise_image)
        if noise_feat.shape[-2:] != conditioning_map.shape[-2:]:
            noise_feat = F.interpolate(noise_feat, size=conditioning_map.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([noise_feat, conditioning_map], dim=1)
        x = self.down_block(x, context)
        x = self.downsample(x)
        bottleneck_condition = F.interpolate(conditioning_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.mid_block(x)
        x = self.mid_res(x, context)
        x = self.upsample(x)
        if x.shape[-2:] != conditioning_map.shape[-2:]:
            x = F.interpolate(x, size=conditioning_map.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, conditioning_map], dim=1)
        x = self.up_block(x, context)
        predicted_residual = self.out_head(x)
        predicted_residual = F.interpolate(predicted_residual, size=output_size, mode="bilinear", align_corners=False)
        x_hat = (noise_image + predicted_residual).clamp(0.0, 1.0)

        return DecoderOutput(
            x_hat=x_hat,
            predicted_residual=predicted_residual,
            noisy_input=noise_image,
            conditioning_map=conditioning_map,
            semantic_condition_map=sem,
            detail_condition_map=det,
            bottleneck_condition_map=bottleneck_condition,
            context_vector=context,
            decode_steps=1,
            stochastic=stochastic,
            output_kind="base",
        )

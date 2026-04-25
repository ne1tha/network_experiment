from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from route2_swinjscc_gan.common.checks import require, require_finite, require_same_shape


def create_window(window_size: int, sigma: float, channel: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)


def gaussian_filter(x: torch.Tensor, window_1d: torch.Tensor, use_padding: bool) -> torch.Tensor:
    channels = x.shape[1]
    padding = window_1d.shape[3] // 2 if use_padding else 0
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=channels)
    out = F.conv2d(out, window_1d.transpose(2, 3), stride=1, padding=(padding, 0), groups=channels)
    return out


def ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    data_range: float,
    use_padding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_x = gaussian_filter(x, window, use_padding)
    mu_y = gaussian_filter(y, window, use_padding)
    sigma_x = gaussian_filter(x * x, window, use_padding) - mu_x.pow(2)
    sigma_y = gaussian_filter(y * y, window, use_padding) - mu_y.pow(2)
    sigma_xy = gaussian_filter(x * y, window, use_padding) - mu_x * mu_y

    cs_map = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
    cs_map = F.relu(cs_map)
    ssim = ((2 * mu_x * mu_y + c1) / (mu_x.pow(2) + mu_y.pow(2) + c1)) * cs_map

    return ssim.mean(dim=(1, 2, 3)), cs_map.mean(dim=(1, 2, 3))


def ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    data_range: float,
    weights: torch.Tensor,
    use_padding: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    weights = weights[:, None]
    levels = int(weights.shape[0])
    values = []

    for level in range(levels):
        ssim_value, cs_value = ssim_map(x, y, window, data_range=data_range, use_padding=use_padding)
        if level < levels - 1:
            values.append(cs_value)
            x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, stride=2, ceil_mode=True)
        else:
            values.append(ssim_value)

    stacked = torch.stack(values, dim=0).clamp_min(eps)
    result = weighted_geometric_product(stacked, weights)
    return result


def weighted_geometric_product(stacked: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    require(
        stacked.ndim == 2 and weights.ndim == 2,
        f"Expected stacked and weights to be 2D tensors, got {stacked.ndim}D and {weights.ndim}D.",
    )
    require(
        stacked.shape[0] == weights.shape[0],
        f"Expected stacked and weights to have the same leading dimension, got {stacked.shape} and {weights.shape}.",
    )
    powered = stacked.pow(weights)
    result = powered[0]
    for level in range(1, powered.shape[0]):
        result = result * powered[level]
    return result


class MSE255Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.squared_difference = nn.MSELoss(reduction="none")

    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        require_same_shape("reconstructed", reconstructed, "target", target)
        loss = torch.mean(self.squared_difference(reconstructed * 255.0, target * 255.0))
        require_finite(loss, "mse255_loss")
        return loss


class MSSSIMLoss(nn.Module):
    def __init__(
        self,
        *,
        window_size: int = 11,
        window_sigma: float = 1.5,
        data_range: float = 1.0,
        channel: int = 3,
        use_padding: bool = False,
        weights: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        levels: int | None = 4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        require(window_size % 2 == 1, "MS-SSIM window size must be odd.")
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        if levels is not None:
            weights_tensor = weights_tensor[:levels]
            weights_tensor = weights_tensor / weights_tensor.sum()
        self.register_buffer("window", create_window(window_size, window_sigma, channel))
        self.register_buffer("weights", weights_tensor)
        self.data_range = data_range
        self.use_padding = use_padding
        self.eps = eps

    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        require_same_shape("reconstructed", reconstructed, "target", target)
        loss = 1.0 - ms_ssim(
            reconstructed,
            target,
            window=self.window,
            data_range=self.data_range,
            weights=self.weights,
            use_padding=self.use_padding,
            eps=self.eps,
        )
        mean_loss = loss.mean()
        require_finite(mean_loss, "ms_ssim_loss")
        return mean_loss


@dataclass(frozen=True)
class ReconstructionLossConfig:
    metric: str = "ms-ssim"

    def __post_init__(self) -> None:
        require(self.metric in {"ms-ssim", "mse"}, "Reconstruction loss must be `ms-ssim` or `mse`.")


def build_reconstruction_loss(config: ReconstructionLossConfig | None = None) -> nn.Module:
    config = config or ReconstructionLossConfig()
    if config.metric == "ms-ssim":
        return MSSSIMLoss()
    return MSE255Loss()

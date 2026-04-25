from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _create_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.reshape(1, 1, 1, -1).repeat(channels, 1, 1, 1)


def _gaussian_filter(x: torch.Tensor, window_1d: torch.Tensor, use_padding: bool) -> torch.Tensor:
    channels = x.shape[1]
    padding = window_1d.shape[3] // 2 if use_padding else 0
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=channels)
    out = F.conv2d(out, window_1d.transpose(2, 3), stride=1, padding=(padding, 0), groups=channels)
    return out


def _ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    *,
    data_range: float,
    use_padding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_x = _gaussian_filter(x, window, use_padding)
    mu_y = _gaussian_filter(y, window, use_padding)
    sigma_x = _gaussian_filter(x * x, window, use_padding) - mu_x.pow(2)
    sigma_y = _gaussian_filter(y * y, window, use_padding) - mu_y.pow(2)
    sigma_xy = _gaussian_filter(x * y, window, use_padding) - mu_x * mu_y

    cs_map = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
    cs_map = F.relu(cs_map)
    ssim = ((2 * mu_x * mu_y + c1) / (mu_x.pow(2) + mu_y.pow(2) + c1)) * cs_map
    return ssim.mean(dim=(1, 2, 3)), cs_map.mean(dim=(1, 2, 3))


def _weighted_geometric_product(stacked: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if stacked.ndim != 2 or weights.ndim != 2:
        raise ValueError(f"Expected stacked and weights to be 2D tensors, got {stacked.ndim}D and {weights.ndim}D.")
    if stacked.shape[0] != weights.shape[0]:
        raise ValueError(
            "Expected stacked and weights to share the same leading dimension, "
            f"got {tuple(stacked.shape)} and {tuple(weights.shape)}."
        )

    powered = stacked.pow(weights)
    result = powered[0]
    for level in range(1, powered.shape[0]):
        result = result * powered[level]
    return result


def _ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    *,
    data_range: float,
    weights: torch.Tensor,
    use_padding: bool,
    eps: float,
) -> torch.Tensor:
    weights = weights[:, None]
    levels = int(weights.shape[0])
    values = []

    for level in range(levels):
        ssim_value, cs_value = _ssim_map(x, y, window, data_range=data_range, use_padding=use_padding)
        if level < levels - 1:
            values.append(cs_value)
            x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, stride=2, ceil_mode=True)
        else:
            values.append(ssim_value)

    stacked = torch.stack(values, dim=0).clamp_min(eps)
    return _weighted_geometric_product(stacked, weights)


class MSSSIMLoss(nn.Module):
    """Route-3 local MS-SSIM loss, adapted from the validated Route-2 implementation."""

    def __init__(
        self,
        *,
        window_size: int = 11,
        window_sigma: float = 1.5,
        data_range: float = 1.0,
        channels: int = 3,
        use_padding: bool = False,
        weights: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        levels: int | None = 4,
        eps: float = 1e-8,
    ):
        super().__init__()
        if window_size % 2 != 1:
            raise ValueError(f"MS-SSIM window_size must be odd, got {window_size}")

        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        if levels is not None:
            weights_tensor = weights_tensor[:levels]
            weights_tensor = weights_tensor / weights_tensor.sum()
        self.register_buffer("window", _create_window(window_size, window_sigma, channels), persistent=False)
        self.register_buffer("weights", weights_tensor, persistent=False)
        self.data_range = data_range
        self.use_padding = use_padding
        self.eps = eps

    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if reconstructed.shape != target.shape:
            raise ValueError(
                "MS-SSIM expects matching image shapes, "
                f"got {tuple(reconstructed.shape)} and {tuple(target.shape)}."
            )
        loss = 1.0 - _ms_ssim(
            reconstructed,
            target,
            window=self.window.to(device=reconstructed.device, dtype=reconstructed.dtype),
            data_range=self.data_range,
            weights=self.weights.to(device=reconstructed.device, dtype=reconstructed.dtype),
            use_padding=self.use_padding,
            eps=self.eps,
        )
        mean_loss = loss.mean()
        if not torch.isfinite(mean_loss):
            raise RuntimeError("MS-SSIM loss produced non-finite values.")
        return mean_loss

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from route2_swinjscc_gan.common.checks import require, require_finite, require_same_shape


VGG19_LAYER_INDEX = {
    "relu1_1": 1,
    "relu1_2": 3,
    "relu2_1": 6,
    "relu2_2": 8,
    "relu3_1": 11,
    "relu3_2": 13,
    "relu3_3": 15,
    "relu3_4": 17,
    "relu4_1": 20,
    "relu4_2": 22,
    "relu4_3": 24,
    "relu4_4": 26,
    "relu5_1": 29,
    "relu5_2": 31,
    "relu5_3": 33,
    "relu5_4": 35,
}


def _import_torchvision_models():
    try:
        from torchvision import models  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "VGG perceptual loss requires `torchvision`, but it is not installed. "
            "Route 2 must not silently fall back to a pixel-only loss."
        ) from exc
    return models


@dataclass(frozen=True)
class VGGPerceptualConfig:
    layer_names: tuple[str, ...] = ("relu1_2", "relu2_2", "relu3_4", "relu4_4")
    layer_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    similarity: str = "cosine"
    weights_path: str | None = None

    def __post_init__(self) -> None:
        require(len(self.layer_names) > 0, "VGG perceptual layers must not be empty.")
        require(
            len(self.layer_names) == len(self.layer_weights),
            "VGG perceptual layer names and weights must have the same length.",
        )
        for layer_name in self.layer_names:
            require(layer_name in VGG19_LAYER_INDEX, f"Unsupported VGG19 layer `{layer_name}`.")
        require(self.similarity in {"cosine", "l1"}, "VGG perceptual similarity must be `cosine` or `l1`.")


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss with explicit fail-fast dependency handling.

    The paper does not specify the exact VGG19 feature layers. We expose the chosen
    layers as configuration instead of hard-coding a hidden default.
    """

    def __init__(self, config: VGGPerceptualConfig | None = None) -> None:
        super().__init__()
        self.config = config or VGGPerceptualConfig()
        models = _import_torchvision_models()

        if self.config.weights_path is not None:
            weights_path = Path(self.config.weights_path)
            if not weights_path.is_file():
                raise FileNotFoundError(f"VGG weights file not found: {weights_path}")
            vgg = models.vgg19(weights=None)
            state_dict = torch.load(weights_path, map_location="cpu")
            vgg.load_state_dict(state_dict, strict=True)
        else:
            try:
                vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            except Exception as exc:  # pragma: no cover - depends on environment/model cache.
                raise RuntimeError(
                    "Failed to build pretrained VGG19. Provide a local `weights_path` or "
                    "make the torchvision pretrained weights available."
                ) from exc

        self.feature_extractor = vgg.features.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        self.max_layer_index = max(VGG19_LAYER_INDEX[name] for name in self.config.layer_names)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        require(image.ndim == 4 and image.shape[1] == 3, "VGG perceptual loss expects an NCHW RGB tensor.")
        return (image - self.mean.to(image.device)) / self.std.to(image.device)

    def _extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features: dict[str, torch.Tensor] = {}
        x = self._preprocess(image)
        for index, layer in enumerate(self.feature_extractor):
            x = layer(x)
            for name in self.config.layer_names:
                if VGG19_LAYER_INDEX[name] == index:
                    features[name] = x
            if index >= self.max_layer_index:
                break
        if set(features.keys()) != set(self.config.layer_names):
            missing = sorted(set(self.config.layer_names) - set(features.keys()))
            raise RuntimeError(f"Failed to extract configured VGG feature layers: {missing}")
        return features

    def _layer_loss(self, reconstructed_feature: torch.Tensor, target_feature: torch.Tensor) -> torch.Tensor:
        if self.config.similarity == "cosine":
            loss = 1.0 - F.cosine_similarity(
                reconstructed_feature.flatten(1),
                target_feature.flatten(1),
                dim=1,
            )
            return loss.mean()
        return F.l1_loss(reconstructed_feature, target_feature)

    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        require_same_shape("reconstructed", reconstructed, "target", target)
        reconstructed_features = self._extract(reconstructed)
        target_features = self._extract(target)

        loss = reconstructed.new_tensor(0.0)
        for weight, layer_name in zip(self.config.layer_weights, self.config.layer_names):
            layer_loss = self._layer_loss(reconstructed_features[layer_name], target_features[layer_name])
            loss = loss + weight * layer_loss

        require_finite(loss, "vgg_perceptual_loss")
        return loss


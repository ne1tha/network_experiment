from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from route2_swinjscc_gan.common.checks import require, require_finite, require_in


@dataclass(frozen=True)
class AdversarialLossConfig:
    mode: str = "hinge"

    def __post_init__(self) -> None:
        require_in("mode", self.mode, {"hinge", "bce"})


class PatchAdversarialLoss(nn.Module):
    """Adversarial objective for PatchGAN training.

    The paper explicitly defines the generator loss as `-mean(d_fake)`, but it does
    not specify the discriminator objective. We expose the discriminator loss mode
    explicitly and default to a hinge discriminator because it is stable for PatchGAN.
    """

    def __init__(self, config: AdversarialLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or AdversarialLossConfig()

    def _as_logit_maps(
        self,
        logits: torch.Tensor | Sequence[torch.Tensor],
        *,
        label: str,
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(logits, torch.Tensor):
            maps = (logits,)
        else:
            maps = tuple(logits)
            require(len(maps) > 0, f"{label} must not be empty.")

        for index, logit in enumerate(maps):
            require(isinstance(logit, torch.Tensor), f"{label}[{index}] must be a tensor.")
            require(logit.ndim >= 3, f"{label}[{index}] must be a patch-level tensor, got shape {tuple(logit.shape)}.")
            require_finite(logit, f"{label}[{index}]")
        return maps

    def generator_loss(self, fake_logits: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        fake_logit_maps = self._as_logit_maps(fake_logits, label="fake_logits")
        loss = sum(-logit.mean() for logit in fake_logit_maps) / float(len(fake_logit_maps))
        require_finite(loss, "generator_adversarial_loss")
        return loss

    def discriminator_loss(
        self,
        real_logits: torch.Tensor | Sequence[torch.Tensor],
        fake_logits: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        real_logit_maps = self._as_logit_maps(real_logits, label="real_logits")
        fake_logit_maps = self._as_logit_maps(fake_logits, label="fake_logits")
        require(
            len(real_logit_maps) == len(fake_logit_maps),
            "Real and fake PatchGAN outputs must expose the same number of scales.",
        )

        losses: list[torch.Tensor] = []
        for index, (real_logit, fake_logit) in enumerate(zip(real_logit_maps, fake_logit_maps)):
            require(
                real_logit.shape == fake_logit.shape,
                f"Real and fake PatchGAN logits must share the same shape at scale {index}.",
            )
            if self.config.mode == "hinge":
                loss_real = F.relu(1.0 - real_logit).mean()
                loss_fake = F.relu(1.0 + fake_logit).mean()
                losses.append(0.5 * (loss_real + loss_fake))
            else:
                loss_real = F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
                loss_fake = F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
                losses.append(0.5 * (loss_real + loss_fake))

        loss = sum(losses) / float(len(losses))

        require_finite(loss, "discriminator_adversarial_loss")
        return loss

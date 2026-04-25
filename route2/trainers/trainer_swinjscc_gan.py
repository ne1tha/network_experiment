from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from route2_swinjscc_gan.common.checks import require, require_finite, require_has_gradients
from route2_swinjscc_gan.losses.adversarial import PatchAdversarialLoss
from route2_swinjscc_gan.models.swinjscc_gan.generator import SwinJSCCGenerator
from route2_swinjscc_gan.models.swinjscc_gan.training_stages import ActiveStage, ProgressiveStageController


@dataclass(frozen=True)
class TrainingStepResult:
    stage: ActiveStage
    generator_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    perceptual_loss: torch.Tensor | None
    adversarial_loss: torch.Tensor | None
    discriminator_loss: torch.Tensor | None
    cbr: float
    snr: int
    rate: int


class SwinJSCCGANTrainer:
    def __init__(
        self,
        *,
        generator: SwinJSCCGenerator,
        reconstruction_loss: nn.Module,
        stage_controller: ProgressiveStageController,
        discriminator: nn.Module | None = None,
        perceptual_loss: nn.Module | None = None,
        adversarial_loss: PatchAdversarialLoss | None = None,
    ) -> None:
        self.generator = generator
        self.reconstruction_loss = reconstruction_loss
        self.stage_controller = stage_controller
        self.discriminator = discriminator
        self.perceptual_loss = perceptual_loss
        self.adversarial_loss = adversarial_loss

    def _set_requires_grad(self, module: nn.Module | None, requires_grad: bool) -> None:
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def train_step(
        self,
        batch: torch.Tensor,
        *,
        epoch: int,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer | None,
        snr: int | None = None,
        rate: int | None = None,
    ) -> TrainingStepResult:
        stage = self.stage_controller.stage_at_epoch(epoch)
        self.generator.train()
        selected_snr, selected_rate = self.generator.sample_training_conditions(snr=snr, rate=rate)

        discriminator_loss_value: torch.Tensor | None = None
        if stage.train_discriminator:
            require(self.discriminator is not None, "Adversarial stage requires a discriminator.")
            require(self.adversarial_loss is not None, "Adversarial stage requires an adversarial loss.")
            require(discriminator_optimizer is not None, "Adversarial stage requires a discriminator optimizer.")

            self._set_requires_grad(self.discriminator, True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                generator_output = self.generator(batch, snr=selected_snr, rate=selected_rate)
                fake_image = generator_output.reconstruction.detach()
            real_logits = self.discriminator(batch)
            fake_logits = self.discriminator(batch, fake_image)
            discriminator_loss_value = self.adversarial_loss.discriminator_loss(real_logits, fake_logits)
            require_finite(discriminator_loss_value, "discriminator_loss")
            discriminator_loss_value.backward()
            require_has_gradients(self.discriminator, "discriminator")
            discriminator_optimizer.step()

        if self.discriminator is not None:
            self._set_requires_grad(self.discriminator, False)

        generator_optimizer.zero_grad(set_to_none=True)
        generator_output = self.generator(batch, snr=selected_snr, rate=selected_rate)
        reconstruction = generator_output.reconstruction
        reconstruction_loss_value = self.reconstruction_loss(reconstruction, batch)
        generator_loss = stage.ms_ssim_weight * reconstruction_loss_value
        perceptual_loss_value: torch.Tensor | None = None
        adversarial_loss_value: torch.Tensor | None = None

        if stage.perceptual_weight > 0.0:
            require(self.perceptual_loss is not None, "Perceptual stage requires a VGG perceptual loss.")
            perceptual_loss_value = self.perceptual_loss(reconstruction, batch)
            generator_loss = generator_loss + stage.perceptual_weight * perceptual_loss_value

        if stage.train_discriminator:
            require(self.discriminator is not None, "Adversarial stage requires a discriminator.")
            require(self.adversarial_loss is not None, "Adversarial stage requires an adversarial loss.")
            fake_logits = self.discriminator(batch, reconstruction)
            adversarial_loss_value = self.adversarial_loss.generator_loss(fake_logits)
            generator_loss = generator_loss + stage.adversarial_weight * adversarial_loss_value

        require_finite(generator_loss, "generator_loss")
        generator_loss.backward()
        require_has_gradients(self.generator, "generator")
        generator_optimizer.step()

        return TrainingStepResult(
            stage=stage,
            generator_loss=generator_loss.detach(),
            reconstruction_loss=reconstruction_loss_value.detach(),
            perceptual_loss=perceptual_loss_value.detach() if perceptual_loss_value is not None else None,
            adversarial_loss=adversarial_loss_value.detach() if adversarial_loss_value is not None else None,
            discriminator_loss=discriminator_loss_value.detach() if discriminator_loss_value is not None else None,
            cbr=generator_output.cbr,
            snr=generator_output.snr,
            rate=generator_output.rate,
        )

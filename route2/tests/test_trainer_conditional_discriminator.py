from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from route2_swinjscc_gan.losses.adversarial import PatchAdversarialLoss
from route2_swinjscc_gan.models.swinjscc_gan.generator import GeneratorOutput
from route2_swinjscc_gan.models.swinjscc_gan.training_stages import ProgressiveStageController, ProgressiveTrainingConfig
from route2_swinjscc_gan.trainers.trainer_swinjscc_gan import SwinJSCCGANTrainer


class DummyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.9))

    def sample_training_conditions(self, *, snr: int | None = None, rate: int | None = None) -> tuple[int, int]:
        return (10 if snr is None else snr, 96 if rate is None else rate)

    def forward(self, input_image: torch.Tensor, *, snr: int | None = None, rate: int | None = None) -> GeneratorOutput:
        selected_snr = 10 if snr is None else snr
        selected_rate = 96 if rate is None else rate
        reconstruction = torch.sigmoid(input_image * self.scale)
        return GeneratorOutput(
            reconstruction=reconstruction,
            cbr=0.5,
            snr=selected_snr,
            rate=selected_rate,
            noisy_feature=reconstruction,
            mask=None,
        )


class RecordingConditionalDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls: list[bool] = []

    def forward(self, source: torch.Tensor, candidate: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(candidate is None)
        paired_candidate = source if candidate is None else candidate
        merged = torch.cat([source, paired_candidate], dim=1)
        fine_logits = merged.mean(dim=1, keepdim=True) * self.scale
        coarse_logits = F.avg_pool2d(fine_logits, kernel_size=2, stride=2)
        return fine_logits, coarse_logits


class ZeroPerceptualLoss(nn.Module):
    def forward(self, reconstruction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return reconstruction.new_zeros(())


def test_trainer_passes_source_and_reconstruction_to_conditional_discriminator() -> None:
    generator = DummyGenerator()
    discriminator = RecordingConditionalDiscriminator()
    trainer = SwinJSCCGANTrainer(
        generator=generator,
        reconstruction_loss=nn.L1Loss(),
        stage_controller=ProgressiveStageController(
            ProgressiveTrainingConfig(
                total_epochs=3,
                phase1_epochs=1,
                phase2_epochs=1,
                adversarial_weight=0.1,
                adversarial_enabled=True,
            )
        ),
        discriminator=discriminator,
        perceptual_loss=ZeroPerceptualLoss(),
        adversarial_loss=PatchAdversarialLoss(),
    )
    batch = torch.rand(2, 3, 8, 8)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    trainer.train_step(
        batch,
        epoch=2,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
    )

    assert discriminator.calls == [True, False, False]

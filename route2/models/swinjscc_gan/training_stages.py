from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from route2_swinjscc_gan.common.checks import require, require_non_negative_int, require_positive_int


class StageName(str, Enum):
    STRUCTURAL = "structural_reconstruction"
    PERCEPTUAL = "perceptual_enhancement"
    ADVERSARIAL = "adversarial_refinement"


@dataclass(frozen=True)
class ActiveStage:
    name: StageName
    epoch: int
    ms_ssim_weight: float
    perceptual_weight: float
    adversarial_weight: float
    train_discriminator: bool


@dataclass(frozen=True)
class ProgressiveTrainingConfig:
    total_epochs: int
    phase1_epochs: int
    phase2_epochs: int
    perceptual_weight_max: float = 0.01
    adversarial_weight: float = 0.05
    adversarial_enabled: bool = True
    adversarial_ramp_epochs: int = 0

    def __post_init__(self) -> None:
        require_positive_int("total_epochs", self.total_epochs)
        require_non_negative_int("phase1_epochs", self.phase1_epochs)
        require_non_negative_int("phase2_epochs", self.phase2_epochs)
        if self.adversarial_enabled and self.adversarial_weight > 0.0:
            require(
                self.phase1_epochs + self.phase2_epochs < self.total_epochs,
                "Phase 1 and Phase 2 must leave at least one epoch for the adversarial stage.",
            )
        else:
            require(
                self.phase1_epochs + self.phase2_epochs <= self.total_epochs,
                "Phase 1 and Phase 2 exceed configured total epochs.",
            )
        require(self.perceptual_weight_max >= 0.0, "`perceptual_weight_max` must be non-negative.")
        require(self.adversarial_weight >= 0.0, "`adversarial_weight` must be non-negative.")
        require_non_negative_int("adversarial_ramp_epochs", self.adversarial_ramp_epochs)


class ProgressiveStageController:
    """Epoch-based progressive training controller following the paper."""

    def __init__(self, config: ProgressiveTrainingConfig) -> None:
        self.config = config

    def stage_at_epoch(self, epoch: int) -> ActiveStage:
        if epoch < 0 or epoch >= self.config.total_epochs:
            raise ValueError(
                f"Epoch {epoch} is out of range for configured total epochs {self.config.total_epochs}."
            )

        if epoch < self.config.phase1_epochs:
            return ActiveStage(
                name=StageName.STRUCTURAL,
                epoch=epoch,
                ms_ssim_weight=1.0,
                perceptual_weight=0.0,
                adversarial_weight=0.0,
                train_discriminator=False,
            )

        phase2_end = self.config.phase1_epochs + self.config.phase2_epochs
        if epoch < phase2_end:
            phase2_step = epoch - self.config.phase1_epochs
            if self.config.phase2_epochs == 1:
                perceptual_weight = self.config.perceptual_weight_max
            else:
                ratio = phase2_step / float(self.config.phase2_epochs - 1)
                perceptual_weight = ratio * self.config.perceptual_weight_max
            return ActiveStage(
                name=StageName.PERCEPTUAL,
                epoch=epoch,
                ms_ssim_weight=1.0,
                perceptual_weight=perceptual_weight,
                adversarial_weight=0.0,
                train_discriminator=False,
            )

        if not self.config.adversarial_enabled or self.config.adversarial_weight == 0.0:
            return ActiveStage(
                name=StageName.ADVERSARIAL,
                epoch=epoch,
                ms_ssim_weight=1.0,
                perceptual_weight=self.config.perceptual_weight_max,
                adversarial_weight=0.0,
                train_discriminator=False,
            )

        phase3_step = epoch - phase2_end
        adversarial_weight = self.config.adversarial_weight
        if self.config.adversarial_ramp_epochs > 0:
            ramp_ratio = min((phase3_step + 1) / float(self.config.adversarial_ramp_epochs), 1.0)
            adversarial_weight *= ramp_ratio

        return ActiveStage(
            name=StageName.ADVERSARIAL,
            epoch=epoch,
            ms_ssim_weight=1.0,
            perceptual_weight=self.config.perceptual_weight_max,
            adversarial_weight=adversarial_weight,
            train_discriminator=adversarial_weight > 0.0,
        )

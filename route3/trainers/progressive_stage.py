from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class StageName(str, Enum):
    STRUCTURAL = "structural_reconstruction"
    PERCEPTUAL = "perceptual_enhancement"
    ADVERSARIAL = "adversarial_refinement"


@dataclass(frozen=True)
class ActiveStage:
    name: StageName
    epoch: int
    perceptual_weight: float
    adversarial_weight: float
    train_discriminator: bool

    def asdict(self) -> dict[str, float | bool | str | int]:
        return asdict(self)


@dataclass(frozen=True)
class ProgressiveTrainingConfig:
    total_epochs: int
    phase1_epochs: int = 0
    phase2_epochs: int = 0
    perceptual_weight_max: float = 1.0
    adversarial_weight: float = 0.1
    adversarial_enabled: bool = True
    adversarial_ramp_epochs: int = 0

    def __post_init__(self) -> None:
        if self.total_epochs <= 0:
            raise ValueError(f"total_epochs must be positive, got {self.total_epochs}")
        if self.phase1_epochs < 0:
            raise ValueError(f"phase1_epochs must be non-negative, got {self.phase1_epochs}")
        if self.phase2_epochs < 0:
            raise ValueError(f"phase2_epochs must be non-negative, got {self.phase2_epochs}")
        if self.perceptual_weight_max < 0.0:
            raise ValueError(f"perceptual_weight_max must be non-negative, got {self.perceptual_weight_max}")
        if self.adversarial_weight < 0.0:
            raise ValueError(f"adversarial_weight must be non-negative, got {self.adversarial_weight}")
        if self.adversarial_ramp_epochs < 0:
            raise ValueError(f"adversarial_ramp_epochs must be non-negative, got {self.adversarial_ramp_epochs}")
        if self.phase1_epochs + self.phase2_epochs > self.total_epochs:
            raise ValueError(
                "phase1_epochs + phase2_epochs must not exceed total_epochs, "
                f"got {self.phase1_epochs} + {self.phase2_epochs} > {self.total_epochs}."
            )
        if self.adversarial_enabled and self.adversarial_weight > 0.0:
            if self.phase1_epochs + self.phase2_epochs >= self.total_epochs:
                raise ValueError(
                    "Progressive adversarial training must leave at least one epoch for the adversarial stage."
                )


class ProgressiveStageController:
    """Epoch-based stage controller adapted from Route 2 for Route 3 trainers."""

    def __init__(self, config: ProgressiveTrainingConfig):
        self.config = config

    def stage_at_epoch(self, epoch: int) -> ActiveStage:
        if epoch <= 0 or epoch > self.config.total_epochs:
            raise ValueError(
                f"epoch must be in [1, {self.config.total_epochs}], got {epoch}"
            )

        if epoch <= self.config.phase1_epochs:
            return ActiveStage(
                name=StageName.STRUCTURAL,
                epoch=epoch,
                perceptual_weight=0.0,
                adversarial_weight=0.0,
                train_discriminator=False,
            )

        phase2_end = self.config.phase1_epochs + self.config.phase2_epochs
        if epoch <= phase2_end:
            phase2_index = epoch - self.config.phase1_epochs
            if self.config.phase2_epochs <= 1:
                perceptual_weight = self.config.perceptual_weight_max
            else:
                # Avoid a "named stage but zero effect" epoch at the start of phase 2.
                ratio = phase2_index / float(self.config.phase2_epochs)
                perceptual_weight = ratio * self.config.perceptual_weight_max
            return ActiveStage(
                name=StageName.PERCEPTUAL,
                epoch=epoch,
                perceptual_weight=perceptual_weight,
                adversarial_weight=0.0,
                train_discriminator=False,
            )

        if not self.config.adversarial_enabled or self.config.adversarial_weight == 0.0:
            return ActiveStage(
                name=StageName.ADVERSARIAL,
                epoch=epoch,
                perceptual_weight=self.config.perceptual_weight_max,
                adversarial_weight=0.0,
                train_discriminator=False,
            )

        phase3_index = epoch - phase2_end
        adversarial_weight = self.config.adversarial_weight
        if self.config.adversarial_ramp_epochs > 0:
            ramp_ratio = min(phase3_index / float(self.config.adversarial_ramp_epochs), 1.0)
            adversarial_weight *= ramp_ratio

        return ActiveStage(
            name=StageName.ADVERSARIAL,
            epoch=epoch,
            perceptual_weight=self.config.perceptual_weight_max,
            adversarial_weight=adversarial_weight,
            train_discriminator=adversarial_weight > 0.0,
        )

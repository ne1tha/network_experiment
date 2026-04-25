from route2_swinjscc_gan.models.swinjscc_gan.training_stages import (
    ProgressiveStageController,
    ProgressiveTrainingConfig,
    StageName,
)


def test_progressive_stage_schedule_matches_paper() -> None:
    controller = ProgressiveStageController(
        ProgressiveTrainingConfig(
            total_epochs=10,
            phase1_epochs=3,
            phase2_epochs=4,
            perceptual_weight_max=0.01,
            adversarial_weight=0.05,
        )
    )

    assert controller.stage_at_epoch(0).name == StageName.STRUCTURAL
    assert controller.stage_at_epoch(2).perceptual_weight == 0.0
    assert controller.stage_at_epoch(3).name == StageName.PERCEPTUAL
    assert controller.stage_at_epoch(3).perceptual_weight == 0.0
    assert controller.stage_at_epoch(6).perceptual_weight == 0.01
    assert controller.stage_at_epoch(7).name == StageName.ADVERSARIAL
    assert controller.stage_at_epoch(7).train_discriminator is True
    assert controller.stage_at_epoch(7).adversarial_weight == 0.05


def test_progressive_stage_can_disable_adversarial_branch_for_ablation() -> None:
    controller = ProgressiveStageController(
        ProgressiveTrainingConfig(
            total_epochs=10,
            phase1_epochs=3,
            phase2_epochs=4,
            perceptual_weight_max=0.01,
            adversarial_weight=0.05,
            adversarial_enabled=False,
        )
    )

    stage = controller.stage_at_epoch(7)
    assert stage.name == StageName.ADVERSARIAL
    assert stage.perceptual_weight == 0.01
    assert stage.adversarial_weight == 0.0
    assert stage.train_discriminator is False


def test_progressive_stage_supports_adversarial_ramp() -> None:
    controller = ProgressiveStageController(
        ProgressiveTrainingConfig(
            total_epochs=12,
            phase1_epochs=3,
            phase2_epochs=4,
            perceptual_weight_max=0.01,
            adversarial_weight=0.05,
            adversarial_ramp_epochs=3,
        )
    )

    first_adv = controller.stage_at_epoch(7)
    second_adv = controller.stage_at_epoch(8)
    full_adv = controller.stage_at_epoch(9)

    assert first_adv.train_discriminator is True
    assert round(first_adv.adversarial_weight, 6) == round(0.05 / 3.0, 6)
    assert round(second_adv.adversarial_weight, 6) == round(0.05 * 2.0 / 3.0, 6)
    assert full_adv.adversarial_weight == 0.05


def test_phase12_only_pretrain_allows_total_epochs_to_match_phase_boundary() -> None:
    controller = ProgressiveStageController(
        ProgressiveTrainingConfig(
            total_epochs=7,
            phase1_epochs=3,
            phase2_epochs=4,
            adversarial_enabled=False,
        )
    )

    stage = controller.stage_at_epoch(6)
    assert stage.name == StageName.PERCEPTUAL
    assert stage.train_discriminator is False

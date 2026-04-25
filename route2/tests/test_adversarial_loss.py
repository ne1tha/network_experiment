import torch

from route2_swinjscc_gan.losses.adversarial import PatchAdversarialLoss


def test_multiscale_generator_loss_averages_patch_maps() -> None:
    loss_module = PatchAdversarialLoss()
    logits = (
        torch.full((2, 1, 15, 15), 2.0),
        torch.full((2, 1, 7, 7), -1.0),
    )

    loss = loss_module.generator_loss(logits)

    expected = torch.tensor((-2.0 + 1.0) / 2.0)
    assert torch.isclose(loss, expected)


def test_multiscale_discriminator_loss_matches_hinge_average() -> None:
    loss_module = PatchAdversarialLoss()
    real_logits = (
        torch.full((2, 1, 15, 15), 1.5),
        torch.full((2, 1, 7, 7), 0.5),
    )
    fake_logits = (
        torch.full((2, 1, 15, 15), -1.5),
        torch.full((2, 1, 7, 7), -0.5),
    )

    loss = loss_module.discriminator_loss(real_logits, fake_logits)

    scale0 = torch.tensor(0.0)
    scale1 = torch.tensor(0.5)
    expected = (scale0 + scale1) / 2.0
    assert torch.isclose(loss, expected)

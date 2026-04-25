# Route 2 Workspace: SwinJSCC-GAN Reproduction

This directory is the dedicated workspace for Route 2 in `工程计划/路线2-复现SwinJSCC-GAN.md`.

All future implementation, experiments, notes, configs, scripts, and artifacts for the Route 2 reproduction should stay under this directory.

## Scope

The target is a formal reproduction of `SwinJSCC-GAN`, not a simplified SwinJSCC baseline.

The implementation must include:

- SwinJSCC-based generator backbone
- VGG perceptual loss
- PatchGAN discriminator
- Three-stage progressive training
- Rayleigh fading training and evaluation
- Dynamic channel and rate adaptation
- PSNR / MS-SSIM / LPIPS evaluation
- Reconstruction visualization for comparison against Route 1

## Non-negotiable Rules

- Do not downgrade the target for convenience.
- Do not silently disable GAN, perceptual loss, or stage transitions.
- Do not fall back to identity channel or pixel-only loss without explicit failure.
- Raise errors on missing weights, broken training stages, invalid tensor shapes, or NaN losses.

## Current Workspace Layout

```text
route2_swinjscc_gan/
  artifacts/
  channels/
  configs/
  docs/
  evaluators/
  losses/
  models/
    swinjscc_gan/
  scripts/
  tests/
  trainers/
```

## Immediate Deliverables

The first implementation wave in this workspace should produce:

1. Formal module boundary documents for Route 2
2. Progressive training stage definition
3. Backbone inheritance plan from Route 1 SwinJSCC
4. Actual model / loss / trainer code under this workspace only

## Current Assessment

The top-level workspace currently contains plans and papers, but no existing codebase to extend directly.

That means Route 2 work should start from an isolated project skeleton here, while keeping the architecture compatible with the Route 1 plan.

## Formal Config Flow

Route 2 now supports a formal local-config workflow for real datasets and reproducible runs:

1. Fill dataset roots in `configs/datasets/route2_div2k_hr.local.json`
2. Pick a training or evaluation preset under `configs/experiments/`
3. Run `python scripts/check_experiment_config.py --config-json ...`
4. Run `python scripts/train_from_config.py --config-json ...`
5. Run `python scripts/eval_from_config.py --config-json ...`

Why this exists:

- keeps dataset paths and experiment hyperparameters versionable
- fails fast on bad paths, undersized crops, and evaluation-size mismatches
- removes reliance on long ad-hoc shell commands for formal reproduction

## Adversarial Ablation

Route 2 now exposes the adversarial branch as an explicit experiment surface instead of
hard-coding a single PatchGAN setup.

Supported config knobs:

- `adversarial.enabled`: turn the adversarial branch on or off
- `adversarial.loss_mode`: choose `hinge` or `bce`
- `adversarial.weight`: override the phase-3 adversarial weight
- `adversarial.ramp_epochs`: linearly ramp the adversarial weight after phase 3 begins
- `adversarial.discriminator_lr_scale`: scale discriminator LR without changing generator LR

Route 2 training also accepts a top-level config key `checkpoint_path`, which maps to
the runtime `training.checkpoint_path`, for resuming from a Route 2 checkpoint
produced by this workspace. This is intended for Route 2 continuation and ablation
experiments, not for importing Route 1 checkpoints.

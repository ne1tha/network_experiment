# Route 3 Workspace

This directory is the open-source mirror of the Route 3 codebase.

Route 3 is the "ultimate-system" line that combines:

- dual-path semantic/detail transmission
- `SwinJSCC`-style adaptive channel and rate modulation
- `SSDD`-style conditional main decoder
- perceptual enhancement plus conditional discriminator
- optional semantic distillation
- optional multi-user semantic sharing extensions

## Current public status

What is formally closed in this mirror:

- single-user training and evaluation code
- progressive three-stage training
- conditional enhancement output path
- single-user budget calibration tools
- semantic distillation export path
- multi-user core code path and tests

What is not claimed as fully closed:

- matched-budget formal Route 3 line
- multi-user second-formal long run
- full-resolution formal training pipeline

The strongest internal close line was the single-user open-quality progressive
300-epoch setup. This repository includes a public example config that mirrors
that schedule without embedding machine-specific paths.

## Included vs excluded

Included:

- source code
- tests
- sanitized example configs
- train / eval / preflight / teacher export / budget calibration scripts

Excluded intentionally:

- training artifacts
- internal docs and experiment reports
- machine-specific watchdogs
- historical one-off visualization scripts tied to local checkpoints

## Layout

```text
route3/
  channels/
  configs/
  datasets/
  evaluators/
  losses/
  models/
    ultimate/
  optim/
  scripts/
  tests/
  trainers/
  requirements.txt
```

## Minimal setup

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

If you want to run tests:

```bash
pip install pytest
pytest tests -q
```

## Dataset manifest flow

Route 3 uses a manifest JSON instead of hard-coding dataset roots.

Build a local manifest from your dataset root:

```bash
python scripts/build_route3_manifest.py \
  --dataset-root /path/to/datasets \
  --output datasets/route3_manifest.local.json
```

The generated `*.local.json` file is ignored by git.

## Suggested first commands

From inside `route3/`:

1. Dry-run preflight with the local bootstrap config:

```bash
python scripts/route3_preflight.py \
  --config configs/route3_single_user.bootstrap.json \
  --split train
```

2. Launch a bootstrap training run:

```bash
python scripts/route3_train.py \
  --config configs/route3_single_user.bootstrap.json
```

3. Evaluate a checkpoint on Kodak:

```bash
python scripts/route3_eval_single_user.py \
  --config configs/route3_single_user.bootstrap.json \
  --checkpoint outputs/bootstrap_single_user/checkpoints/latest.pt \
  --kodak-dir /path/to/Kodak
```

## Configs worth starting from

- `configs/route3_single_user.bootstrap.json`
  - easiest local smoke / sanity config
- `configs/route3_multi_user.bootstrap.json`
  - first local multi-user smoke config
- `configs/route3_single_user.example.json`
  - generic single-user placeholder example
- `configs/route3_multi_user.example.json`
  - generic multi-user placeholder example
- `configs/route3_single_user.stageA.progressive300_public.example.json`
  - public mirror of the final successful single-user training schedule

## Important note on paths

This mirror avoids embedding the original workstation paths, but Route 3's JSON
configs still expect you to provide valid dataset and checkpoint locations.

The bootstrap configs assume you run commands from inside `route3/` and that you
have already created:

- `datasets/route3_manifest.local.json`

## License / provenance note

This directory is a project-owned code mirror. Some Route 3 ideas were inspired
by external papers and public repositories, especially `SwinJSCC` and `SSDD`,
but the integrated Route 3 system here is the local project implementation.

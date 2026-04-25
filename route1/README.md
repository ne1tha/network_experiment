# Route 1 Workspace

This directory is the isolated workspace for the formal `SwinJSCC` reproduction task
defined in `../工程计划/路线1-复现SwinJSCC.md`.

All implementation work owned by this Codex should stay inside this directory,
including:

- source code
- configuration files
- dataset integrity scripts
- experiment scripts
- evaluation outputs
- reproduction notes

Top-level layout:

```text
codex_route1_neu/
  configs/
  datasets/
  models/
    swinjscc/
  channels/
  losses/
  trainers/
  evaluators/
  scripts/
  tests/
  docs/
  artifacts/
```

Execution rules for this workspace:

- no silent fallback
- no placeholder modules presented as finished implementations
- no structural downgrade from the original paper target
- fail fast on invalid shapes, configs, checkpoints, and dataset paths

Current status:

- route plan reviewed
- workspace initialized
- upstream reference integrated
- formal train and eval CLI entrypoints added
- JSON config driven check/train/eval entrypoints added
- real datasets downloaded and wired into local manifests
- AWGN `w/o SA/RA` baseline completed
- AWGN `SA/RA` training completed from compatible `w/o SA/RA` initialization
- detailed run record added in `docs/awgn_experiment_report.md`

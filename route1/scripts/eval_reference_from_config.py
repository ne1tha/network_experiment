from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.reference_loader import load_reference_experiment
from datasets.hr_datasets import build_hr_eval_loader
from evaluators.reference_evaluator import evaluate_reference_model
from models.swinjscc.upstream_reference import build_upstream_model
from support.runtime import build_logger, ensure_run_dirs, load_checkpoint_strict, save_config_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate route 1 using a JSON experiment config.")
    parser.add_argument("--config-json", type=Path, required=True, help="Experiment JSON config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment, _ = load_reference_experiment(args.config_json)
    experiment.training = False
    if experiment.checkpoint_path is None:
        raise ValueError("Evaluation config must define checkpoint_path.")

    model, bundle, runtime_config = build_upstream_model(experiment)
    device = runtime_config.device
    model = model.to(device)

    ensure_run_dirs(Path(experiment.run_paths.workdir), Path(experiment.run_paths.models_dir))
    logger = build_logger(
        f"route1.{experiment.run_name}.eval",
        log_path=Path(experiment.run_paths.log_path) if experiment.save_logs else None,
    )
    save_config_snapshot(experiment, Path(experiment.run_paths.workdir) / "config_snapshot.json")

    metadata = load_checkpoint_strict(
        model=model,
        checkpoint_path=Path(experiment.checkpoint_path),
        device=device,
    )
    logger.info("Loaded evaluation checkpoint from %s", experiment.checkpoint_path)
    logger.info("Checkpoint metadata: %s", metadata)

    eval_loader = build_hr_eval_loader(experiment)
    output_path = Path(experiment.run_paths.workdir) / "eval_results.json"
    results = evaluate_reference_model(
        model=model,
        eval_loader=eval_loader,
        experiment=experiment,
        bundle=bundle,
        device=device,
        logger=logger,
        output_path=output_path,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

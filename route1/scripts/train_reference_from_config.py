from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.reference_loader import load_reference_experiment
from trainers.reference_trainer import ReferenceTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train route 1 using a JSON experiment config.")
    parser.add_argument("--config-json", type=Path, required=True, help="Experiment JSON config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment, _ = load_reference_experiment(args.config_json)
    experiment.training = True
    trainer = ReferenceTrainer(experiment)
    summary = trainer.train()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

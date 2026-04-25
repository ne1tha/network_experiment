from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def export_semantic_teacher(source_checkpoint: str | Path, output_path: str | Path) -> dict[str, str]:
    source_checkpoint = Path(source_checkpoint).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Route-3 checkpoint not found: {source_checkpoint}")

    checkpoint = torch.load(source_checkpoint, map_location="cpu")
    model_state = checkpoint["model"]
    semantic_prefixes = (
        "single_user_model.encoder.semantic_branch.",
        "encoder.semantic_branch.",
        "semantic_branch.",
    )

    teacher_state = None
    for prefix in semantic_prefixes:
        extracted = {
            key[len(prefix):]: value
            for key, value in model_state.items()
            if key.startswith(prefix)
        }
        if extracted:
            teacher_state = extracted
            break

    if not teacher_state:
        raise RuntimeError("Failed to locate semantic_branch weights inside the provided route-3 checkpoint.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(teacher_state, output_path)
    return {
        "source_checkpoint": str(source_checkpoint),
        "teacher_checkpoint": str(output_path),
        "num_tensors": str(len(teacher_state)),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Export the semantic branch from a route-3 checkpoint as a frozen teacher checkpoint.")
    parser.add_argument("--source", required=True, help="Path to the source route-3 checkpoint.")
    parser.add_argument("--output", required=True, help="Path to write the semantic teacher checkpoint.")
    args = parser.parse_args()

    result = export_semantic_teacher(args.source, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    _main()

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "data"
DEFAULT_OUTPUT_PATH = WORKSPACE_ROOT / "datasets" / "route3_manifest.local.json"

SPLIT_SOURCES: dict[str, tuple[tuple[str, str], ...]] = {
    "train": (
        ("DIV2K/DIV2K_train_HR", "div2k_train"),
        ("CLIC/clic2020/train", "clic2020_train"),
    ),
    "val": (
        ("DIV2K/DIV2K_valid_HR", "div2k_val"),
        ("CLIC/clic2020/valid", "clic2020_val"),
    ),
    "test": (
        ("Kodak", "kodak"),
        ("CLIC/clic2021/test", "clic2021_test"),
        ("CLIC/clic2022/val", "clic2022_val"),
    ),
}


def _collect_split_samples(dataset_root: Path, split_name: str) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for relative_dir, user_prefix in SPLIT_SOURCES[split_name]:
        source_dir = dataset_root / relative_dir
        if not source_dir.exists():
            raise FileNotFoundError(f"Dataset source directory not found: {source_dir}")

        image_paths = sorted(
            path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            raise RuntimeError(f"No image files found under dataset source: {source_dir}")

        for image_path in image_paths:
            samples.append(
                {
                    "image": str(image_path.relative_to(dataset_root)).replace("\\", "/"),
                    "user_id": f"{user_prefix}_{image_path.stem}",
                }
            )
    return samples


def build_manifest(dataset_root: Path) -> dict[str, object]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    splits = {split_name: _collect_split_samples(dataset_root, split_name) for split_name in SPLIT_SOURCES}
    return {
        "root": str(dataset_root),
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the formal route-3 manifest from local image datasets.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Dataset root path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output manifest JSON path.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(dataset_root)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    split_counts = {split_name: len(samples) for split_name, samples in manifest["splits"].items()}
    print(json.dumps({"output": str(output_path), "dataset_root": str(dataset_root), "counts": split_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()

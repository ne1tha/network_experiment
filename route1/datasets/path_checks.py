from __future__ import annotations

from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def require_existing_paths(paths: Iterable[Path], *, label: str) -> list[Path]:
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise FileNotFoundError(f"No {label} paths were provided.")

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing {label} path(s): {joined}")
    return resolved


def require_images_present(paths: Iterable[Path], *, label: str) -> None:
    checked = require_existing_paths(paths, label=label)
    for path in checked:
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return

        if path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                    return

    raise FileNotFoundError(f"No image files found under {label} paths.")

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_KEYWORDS = ("div2k", "kodak", "clic", "clic2020", "clic2021", "clic2022", "flickr2k", "ffhq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search local disks for likely Route 2 image dataset roots.")
    parser.add_argument(
        "--search-root",
        nargs="+",
        default=("/mnt/nvme", "/home/neu"),
        help="Root directories to search.",
    )
    parser.add_argument(
        "--keyword",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help="Directory name keywords to match, case-insensitive.",
    )
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum directory depth below each search root.")
    parser.add_argument("--min-images", type=int, default=50, help="Minimum image count for a candidate to be reported.")
    parser.add_argument("--sample-limit", type=int, default=5, help="How many sample image paths to keep per candidate.")
    return parser.parse_args()


def _inspect_candidate(path: Path, *, sample_limit: int) -> dict[str, object]:
    count = 0
    samples: list[str] = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            candidate = Path(root) / filename
            if candidate.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            count += 1
            if len(samples) < sample_limit:
                samples.append(str(candidate))
    return {
        "path": str(path),
        "name": path.name,
        "num_images": count,
        "sample_images": samples,
    }


def _name_tokens(name: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", name.lower()) if token}


def _is_candidate_name(name: str, *, keywords: tuple[str, ...]) -> bool:
    lowered = name.lower()
    tokens = _name_tokens(lowered)
    return any(keyword == lowered or keyword in tokens for keyword in keywords)


def _walk_candidates(search_root: Path, *, keywords: tuple[str, ...], max_depth: int) -> list[Path]:
    matches: list[Path] = []
    seen: set[Path] = set()
    base_depth = len(search_root.parts)
    for root, dirnames, _ in os.walk(search_root, topdown=True):
        current = Path(root)
        try:
            depth = len(current.parts) - base_depth
        except Exception:
            continue
        if depth >= max_depth:
            dirnames[:] = []
        if _is_candidate_name(current.name, keywords=keywords) and current not in seen:
            seen.add(current)
            matches.append(current)
    return matches


def main() -> int:
    args = parse_args()
    keywords = tuple(keyword.lower() for keyword in args.keyword)
    results: list[dict[str, object]] = []

    for raw_root in args.search_root:
        search_root = Path(raw_root).expanduser().resolve()
        if not search_root.exists():
            continue
        try:
            candidates = _walk_candidates(search_root, keywords=keywords, max_depth=args.max_depth)
        except PermissionError:
            continue
        for candidate in candidates:
            try:
                info = _inspect_candidate(candidate, sample_limit=args.sample_limit)
            except PermissionError:
                continue
            if info["num_images"] < args.min_images:
                continue
            results.append(info)

    results.sort(key=lambda item: (-int(item["num_images"]), str(item["path"])))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

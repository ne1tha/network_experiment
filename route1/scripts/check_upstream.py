from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.swinjscc.upstream_reference import DEFAULT_UPSTREAM_ROOT, ensure_upstream_repo


def main() -> None:
    root = ensure_upstream_repo(DEFAULT_UPSTREAM_ROOT)
    summary = {
        "upstream_root": str(root),
        "readme": str(root / "README.md"),
        "main_entry": str(root / "main.py"),
        "network": str(root / "net" / "network.py"),
        "channel": str(root / "net" / "channel.py"),
        "encoder": str(root / "net" / "encoder.py"),
        "decoder": str(root / "net" / "decoder.py"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

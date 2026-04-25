from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from route2_swinjscc_gan.common.device import describe_cuda_environment
from route2_swinjscc_gan.models.swinjscc_gan.third_party_bridge import load_third_party_swinjscc


def check_dependency(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, getattr(module, "__version__", "unknown")


def main() -> int:
    dependencies = ("torch", "torchvision", "timm")
    failed = False

    for dependency in dependencies:
        ok, detail = check_dependency(dependency)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {dependency}: {detail}")
        failed = failed or (not ok)

    try:
        load_third_party_swinjscc()
    except Exception as exc:
        print(f"[FAIL] third_party/SwinJSCC import: {type(exc).__name__}: {exc}")
        failed = True
    else:
        print("[OK] third_party/SwinJSCC import")

    print("[INFO] cuda:", json.dumps(describe_cuda_environment(), ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

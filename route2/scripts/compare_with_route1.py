from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from route2_swinjscc_gan.common.io import save_json


METRIC_DIRECTIONS = {
    "psnr": "higher",
    "ms_ssim": "higher",
    "lpips": "lower",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Route 2 results against Route 1 results.")
    parser.add_argument("--route1", required=True, help="Route 1 metrics JSON.")
    parser.add_argument("--route2", required=True, help="Route 2 metrics JSON.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def load_metrics(path: str) -> dict[str, float]:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: float(payload[key]) for key in ("psnr", "ms_ssim", "lpips")}


def compare(route1: dict[str, float], route2: dict[str, float]) -> dict[str, dict[str, float | str]]:
    comparison: dict[str, dict[str, float | str]] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        delta = route2[metric] - route1[metric]
        improved = delta > 0 if direction == "higher" else delta < 0
        comparison[metric] = {
            "route1": route1[metric],
            "route2": route2[metric],
            "delta": delta,
            "preferred_direction": direction,
            "route2_better": "yes" if improved else "no",
        }
    return comparison


def main() -> int:
    args = parse_args()
    route1 = load_metrics(args.route1)
    route2 = load_metrics(args.route2)
    payload = compare(route1, route2)

    for metric, values in payload.items():
        print(
            f"{metric}: route1={values['route1']:.6f}, "
            f"route2={values['route2']:.6f}, delta={values['delta']:.6f}, "
            f"preferred={values['preferred_direction']}, route2_better={values['route2_better']}"
        )

    if args.output is not None:
        save_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

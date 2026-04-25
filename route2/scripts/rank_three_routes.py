from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from route2_swinjscc_gan.common.io import ensure_dir, save_json


DEFAULT_WEIGHTS = {
    "psnr": 0.4,
    "ms_ssim": 0.3,
    "lpips": 0.3,
}

DEFAULT_ANCHORS = {
    "psnr_min": 20.0,
    "psnr_max": 30.0,
    "ms_ssim_min": 0.75,
    "ms_ssim_max": 0.95,
    "lpips_best": 0.30,
    "lpips_worst": 0.60,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank route1/route2/route3 runs with a matched-cost primary board and open-cost secondary boards."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Benchmark manifest JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for ranking outputs.")
    return parser.parse_args()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)!r}.")
    return payload


def _infer_channel_type(summary: dict[str, Any], entry: dict[str, Any]) -> str | None:
    if entry.get("channel_type") is not None:
        return str(entry["channel_type"])
    if summary.get("channel_type") is not None:
        return str(summary["channel_type"])
    semantic = summary.get("semantic_channel_type")
    detail = summary.get("detail_channel_type")
    if semantic is not None and detail is not None and semantic == detail:
        return str(semantic)
    return None


def _infer_snr_db(summary: dict[str, Any], entry: dict[str, Any]) -> float | None:
    if entry.get("snr_db") is not None:
        return float(entry["snr_db"])
    if summary.get("snr_db") is not None:
        return float(summary["snr_db"])
    if summary.get("snr") is not None:
        return float(summary["snr"])
    return None


def _infer_effective_cbr(summary: dict[str, Any], entry: dict[str, Any]) -> float | None:
    if entry.get("effective_cbr") is not None:
        return float(entry["effective_cbr"])
    if summary.get("effective_cbr") is not None:
        return float(summary["effective_cbr"])

    rate = entry.get("rate", summary.get("rate"))
    if rate is None:
        return None
    downsample = int(entry.get("downsample", summary.get("downsample", 4)))
    denominator = float(2 * 3 * (2 ** (downsample * 2)))
    return float(rate) / denominator


def _quality_score(
    *,
    psnr: float,
    ms_ssim: float,
    lpips: float,
    anchors: dict[str, float],
    weights: dict[str, float],
) -> dict[str, float]:
    psnr_norm = _clamp01((psnr - anchors["psnr_min"]) / (anchors["psnr_max"] - anchors["psnr_min"]))
    ms_ssim_norm = _clamp01(
        (ms_ssim - anchors["ms_ssim_min"]) / (anchors["ms_ssim_max"] - anchors["ms_ssim_min"])
    )
    lpips_norm = _clamp01(
        (anchors["lpips_worst"] - lpips) / (anchors["lpips_worst"] - anchors["lpips_best"])
    )
    score = 100.0 * (
        weights["psnr"] * psnr_norm
        + weights["ms_ssim"] * ms_ssim_norm
        + weights["lpips"] * lpips_norm
    )
    return {
        "psnr_norm": psnr_norm,
        "ms_ssim_norm": ms_ssim_norm,
        "lpips_norm": lpips_norm,
        "quality_score": score,
    }


def _primary_eligibility_reasons(
    *,
    item: dict[str, Any],
    reference_channel_type: str | None,
    reference_snr_db: float | None,
    reference_cbr: float,
    cost_tolerance_ratio: float,
) -> list[str]:
    reasons: list[str] = []
    channel_type = item.get("channel_type")
    snr_db = item.get("snr_db")
    effective_cbr = item.get("effective_cbr")

    if item.get("include_in_primary", True) is False:
        reasons.append("include_in_primary=false")
    if reference_channel_type is not None and channel_type != reference_channel_type:
        reasons.append(f"channel_mismatch:{channel_type}")
    if reference_snr_db is not None:
        if snr_db is None:
            reasons.append("missing_snr")
        elif not math.isclose(float(snr_db), float(reference_snr_db), rel_tol=0.0, abs_tol=1e-6):
            reasons.append(f"snr_mismatch:{snr_db}")
    if effective_cbr is None:
        reasons.append("missing_effective_cbr")
    else:
        lower = reference_cbr * (1.0 - cost_tolerance_ratio)
        upper = reference_cbr * (1.0 + cost_tolerance_ratio)
        if effective_cbr < lower or effective_cbr > upper:
            reasons.append(f"cbr_out_of_window:{effective_cbr:.6f}")
    return reasons


def _build_markdown(
    *,
    benchmark_name: str,
    reference_channel_type: str | None,
    reference_snr_db: float | None,
    reference_cbr: float,
    cost_tolerance_ratio: float,
    entries: list[dict[str, Any]],
    primary: list[dict[str, Any]],
    open_quality: list[dict[str, Any]],
    efficiency: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {benchmark_name}",
        "",
        "## Primary Board",
        "",
        f"- reference channel: `{reference_channel_type}`" if reference_channel_type is not None else "- reference channel: `unspecified`",
        f"- reference snr_db: `{reference_snr_db}`" if reference_snr_db is not None else "- reference snr_db: `unspecified`",
        f"- reference cbr: `{reference_cbr:.6f}`",
        f"- cost tolerance: `±{cost_tolerance_ratio * 100:.1f}%`",
        "",
        "| Rank | Name | Route | PSNR | MS-SSIM | LPIPS | CBR | Quality Score |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if primary:
        for rank, item in enumerate(primary, start=1):
            lines.append(
                f"| {rank} | {item['name']} | {item['route_family']} | "
                f"{item['psnr']:.4f} | {item['ms_ssim']:.4f} | {item['lpips']:.4f} | "
                f"{item['effective_cbr']:.6f} | {item['quality_score']:.2f} |"
            )
    else:
        lines.append("| - | no eligible entry | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Open Quality Board",
            "",
            "| Rank | Name | Route | Channel | SNR | PSNR | MS-SSIM | LPIPS | CBR | Quality Score |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, item in enumerate(open_quality, start=1):
        channel_value = item["channel_type"] if item["channel_type"] is not None else "n/a"
        snr_value = f"{item['snr_db']:.2f}" if item["snr_db"] is not None else "n/a"
        cbr_value = f"{item['effective_cbr']:.6f}" if item["effective_cbr"] is not None else "n/a"
        lines.append(
            f"| {rank} | {item['name']} | {item['route_family']} | {channel_value} | {snr_value} | "
            f"{item['psnr']:.4f} | {item['ms_ssim']:.4f} | {item['lpips']:.4f} | {cbr_value} | {item['quality_score']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Efficiency Board",
            "",
            "| Rank | Name | Route | PSNR | MS-SSIM | LPIPS | CBR | Cost Ratio | Cost-Aware Score |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, item in enumerate(efficiency, start=1):
        cbr_value = f"{item['effective_cbr']:.6f}" if item["effective_cbr"] is not None else "n/a"
        cost_ratio = f"{item['cost_ratio']:.3f}" if item["cost_ratio"] is not None else "n/a"
        lines.append(
            f"| {rank} | {item['name']} | {item['route_family']} | "
            f"{item['psnr']:.4f} | {item['ms_ssim']:.4f} | {item['lpips']:.4f} | "
            f"{cbr_value} | {cost_ratio} | {item['cost_aware_score']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Eligibility Notes",
            "",
            "| Name | Eligible For Primary | Reasons |",
            "| --- | --- | --- |",
        ]
    )
    for item in entries:
        reasons = ", ".join(item["primary_eligibility_reasons"]) if item["primary_eligibility_reasons"] else "ok"
        lines.append(
            f"| {item['name']} | {'yes' if item['primary_eligible'] else 'no'} | {reasons} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    manifest = _load_json(args.manifest)

    benchmark_name = str(manifest.get("benchmark_name", args.manifest.stem))
    reference_channel_type = (
        str(manifest["reference_channel_type"])
        if manifest.get("reference_channel_type") is not None
        else None
    )
    reference_snr_db = (
        float(manifest["reference_snr_db"])
        if manifest.get("reference_snr_db") is not None
        else None
    )
    reference_cbr = float(manifest.get("reference_cbr", 0.0625))
    cost_tolerance_ratio = float(manifest.get("cost_tolerance_ratio", 0.05))
    anchors = dict(DEFAULT_ANCHORS)
    anchors.update({key: float(value) for key, value in manifest.get("anchors", {}).items()})
    weights = dict(DEFAULT_WEIGHTS)
    weights.update({key: float(value) for key, value in manifest.get("weights", {}).items()})

    entries_raw = manifest.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise ValueError("Manifest must contain a non-empty `entries` list.")

    normalized_entries: list[dict[str, Any]] = []
    for raw_entry in entries_raw:
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Each manifest entry must be an object, got {type(raw_entry)!r}.")
        name = str(raw_entry["name"])
        route_family = str(raw_entry.get("route_family", "custom"))
        summary_path = Path(raw_entry["summary_path"]).resolve()
        summary = _load_json(summary_path)

        psnr = float(summary["psnr"])
        ms_ssim = float(summary["ms_ssim"])
        lpips = float(summary["lpips"])
        channel_type = _infer_channel_type(summary, raw_entry)
        snr_db = _infer_snr_db(summary, raw_entry)
        effective_cbr = _infer_effective_cbr(summary, raw_entry)
        quality_payload = _quality_score(
            psnr=psnr,
            ms_ssim=ms_ssim,
            lpips=lpips,
            anchors=anchors,
            weights=weights,
        )

        cost_ratio = (
            float(effective_cbr) / reference_cbr
            if effective_cbr is not None and reference_cbr > 0
            else None
        )
        cost_aware_score = (
            quality_payload["quality_score"] / max(cost_ratio, 1.0)
            if cost_ratio is not None
            else quality_payload["quality_score"]
        )
        reasons = _primary_eligibility_reasons(
            item={
                "channel_type": channel_type,
                "snr_db": snr_db,
                "effective_cbr": effective_cbr,
                "include_in_primary": raw_entry.get("include_in_primary", True),
            },
            reference_channel_type=reference_channel_type,
            reference_snr_db=reference_snr_db,
            reference_cbr=reference_cbr,
            cost_tolerance_ratio=cost_tolerance_ratio,
        )
        normalized_entries.append(
            {
                "name": name,
                "route_family": route_family,
                "summary_path": str(summary_path),
                "psnr": psnr,
                "ms_ssim": ms_ssim,
                "lpips": lpips,
                "channel_type": channel_type,
                "snr_db": snr_db,
                "effective_cbr": effective_cbr,
                "cost_ratio": cost_ratio,
                "cost_aware_score": cost_aware_score,
                "include_in_primary": bool(raw_entry.get("include_in_primary", True)),
                "primary_eligible": len(reasons) == 0,
                "primary_eligibility_reasons": reasons,
                "notes": raw_entry.get("notes"),
                **quality_payload,
            }
        )

    primary = sorted(
        [item for item in normalized_entries if item["primary_eligible"]],
        key=lambda item: item["quality_score"],
        reverse=True,
    )
    open_quality = sorted(
        normalized_entries,
        key=lambda item: item["quality_score"],
        reverse=True,
    )
    efficiency = sorted(
        normalized_entries,
        key=lambda item: item["cost_aware_score"],
        reverse=True,
    )

    output_dir = ensure_dir(args.output_dir)
    payload = {
        "benchmark_name": benchmark_name,
        "reference_channel_type": reference_channel_type,
        "reference_snr_db": reference_snr_db,
        "reference_cbr": reference_cbr,
        "cost_tolerance_ratio": cost_tolerance_ratio,
        "anchors": anchors,
        "weights": weights,
        "entries": normalized_entries,
        "primary_board": primary,
        "open_quality_board": open_quality,
        "efficiency_board": efficiency,
    }
    save_json(output_dir / "ranking_summary.json", payload)

    markdown = _build_markdown(
        benchmark_name=benchmark_name,
        reference_channel_type=reference_channel_type,
        reference_snr_db=reference_snr_db,
        reference_cbr=reference_cbr,
        cost_tolerance_ratio=cost_tolerance_ratio,
        entries=normalized_entries,
        primary=primary,
        open_quality=open_quality,
        efficiency=efficiency,
    )
    (output_dir / "ranking_report.md").write_text(markdown, encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

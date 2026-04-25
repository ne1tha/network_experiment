from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Route3ImageManifestDataset
from evaluators import compute_effective_cbr_tensor
from optim import move_to_device, select_torch_device
from scripts.route3_train import load_training_config
from scripts.route3_preflight import _resolve_dataset_resize_hw, _resolve_dataset_transform_mode, build_runtime_model


@dataclass(frozen=True)
class BudgetCalibrationCandidate:
    sem_rate_ratio: float
    det_rate_ratio: float
    semantic_active_channels: int
    detail_active_channels: int
    semantic_symbol_count: int
    detail_symbol_count: int
    total_symbol_count: int
    effective_cbr: float
    cbr_absolute_gap: float
    cbr_relative_gap: float
    within_target_tolerance: bool


def _load_reference_image(config, split: str) -> torch.Tensor:
    dataset = Route3ImageManifestDataset(
        manifest_path=config.base.dataset.manifest_path,
        split=split,
        resize_hw=_resolve_dataset_resize_hw(config.base, split),
        transform_mode=_resolve_dataset_transform_mode(config.base, split),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset split '{split}' is empty.")
    image, _ = dataset[0]
    return image.unsqueeze(0)


def _build_calibration_runtime(config):
    runtime = replace(
        config.base.runtime,
        enable_distillation=False,
        enable_perceptual=False,
        enable_adversarial=False,
    )
    return replace(config.base, runtime=runtime)


@torch.no_grad()
def _extract_single_user_shapes(config, split: str) -> dict[str, Any]:
    base_config = _build_calibration_runtime(config)
    device = select_torch_device(base_config.runtime.device)
    model = build_runtime_model(base_config)
    model.to(device)
    model.eval()

    source = move_to_device(_load_reference_image(config, split=split), device)
    output = model(
        source,
        snr_db=base_config.runtime.snr_db,
        sem_rate_ratio=1.0,
        det_rate_ratio=1.0,
        decode_stochastic=False,
        run_enhancement=False,
        compute_enhancement_discriminator_loss=False,
    )
    return {
        "input_image": source.detach().cpu(),
        "semantic_channels": int(output.semantic.tx.shape[1]),
        "detail_channels": int(output.detail.tx.shape[1]),
        "semantic_shape": output.semantic.tx.shape,
        "detail_shape": output.detail.tx.shape,
        "semantic_latent_hw": [int(output.semantic.tx.shape[-2]), int(output.semantic.tx.shape[-1])],
        "detail_latent_hw": [int(output.detail.tx.shape[-2]), int(output.detail.tx.shape[-1])],
    }


def enumerate_budget_candidates(
    *,
    input_image: torch.Tensor,
    semantic_channels: int,
    detail_channels: int,
    semantic_shape: torch.Size,
    detail_shape: torch.Size,
    target_effective_cbr: float,
    tolerance: float,
    min_sem_active: int = 1,
    min_det_active: int = 1,
    max_sem_active: int | None = None,
    max_det_active: int | None = None,
) -> list[BudgetCalibrationCandidate]:
    max_sem_active = semantic_channels if max_sem_active is None else min(max_sem_active, semantic_channels)
    max_det_active = detail_channels if max_det_active is None else min(max_det_active, detail_channels)
    candidates: list[BudgetCalibrationCandidate] = []
    device = input_image.device
    dtype = input_image.dtype
    semantic_spatial = int(semantic_shape[-2]) * int(semantic_shape[-1])
    detail_spatial = int(detail_shape[-2]) * int(detail_shape[-1])

    for semantic_active in range(min_sem_active, max_sem_active + 1):
        semantic_tensor = torch.tensor([semantic_active], device=device, dtype=dtype)
        for detail_active in range(min_det_active, max_det_active + 1):
            detail_tensor = torch.tensor([detail_active], device=device, dtype=dtype)
            effective_cbr = float(
                compute_effective_cbr_tensor(
                    input_image=input_image,
                    semantic_active_channels=semantic_tensor,
                    detail_active_channels=detail_tensor,
                    semantic_shape=semantic_shape,
                    detail_shape=detail_shape,
                ).item()
            )
            absolute_gap = abs(effective_cbr - target_effective_cbr)
            relative_gap = absolute_gap / max(abs(target_effective_cbr), 1e-12)
            semantic_symbol_count = int(semantic_active * semantic_spatial)
            detail_symbol_count = int(detail_active * detail_spatial)
            candidates.append(
                BudgetCalibrationCandidate(
                    sem_rate_ratio=float(semantic_active) / float(semantic_channels),
                    det_rate_ratio=float(detail_active) / float(detail_channels),
                    semantic_active_channels=semantic_active,
                    detail_active_channels=detail_active,
                    semantic_symbol_count=semantic_symbol_count,
                    detail_symbol_count=detail_symbol_count,
                    total_symbol_count=semantic_symbol_count + detail_symbol_count,
                    effective_cbr=effective_cbr,
                    cbr_absolute_gap=absolute_gap,
                    cbr_relative_gap=relative_gap,
                    within_target_tolerance=relative_gap <= tolerance + 1e-12,
                )
            )

    candidates.sort(
        key=lambda item: (
            0 if item.within_target_tolerance else 1,
            item.cbr_absolute_gap,
            item.total_symbol_count,
            item.semantic_active_channels,
            item.detail_active_channels,
        )
    )
    return candidates


def extract_candidate_frontier(
    candidates: list[BudgetCalibrationCandidate],
    *,
    gap_tolerance: float = 1e-12,
) -> list[BudgetCalibrationCandidate]:
    """Return the feasibility/gap frontier without pretending it is quality-ranked."""
    if not candidates:
        return []

    first = candidates[0]
    frontier: list[BudgetCalibrationCandidate] = []
    for candidate in candidates:
        if candidate.within_target_tolerance != first.within_target_tolerance:
            break
        if abs(candidate.cbr_absolute_gap - first.cbr_absolute_gap) > gap_tolerance:
            break
        frontier.append(candidate)
    return frontier


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Route-3 single-user rate ratios for a target effective CBR.")
    parser.add_argument("--config", required=True, help="Path to the route-3 training config JSON.")
    parser.add_argument("--target-effective-cbr", type=float, required=True, help="Target effective CBR.")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Relative tolerance, default 0.05.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split used to probe shapes.")
    parser.add_argument("--top-k", type=int, default=20, help="How many candidates to keep in the report.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--device", default=None, help="Optional runtime device override, e.g. cpu or cuda:0.")
    args = parser.parse_args()

    config = load_training_config(args.config)
    if config.base.runtime.mode != "single_user":
        raise ValueError(f"Expected single_user config, got {config.base.runtime.mode}")
    if args.target_effective_cbr <= 0.0:
        raise ValueError(f"--target-effective-cbr must be positive, got {args.target_effective_cbr}")
    if args.tolerance < 0.0 or args.tolerance >= 1.0:
        raise ValueError(f"--tolerance must be in [0, 1), got {args.tolerance}")

    if args.device is not None:
        runtime = replace(config.base.runtime, device=args.device)
        config = replace(config, base=replace(config.base, runtime=runtime))

    shapes = _extract_single_user_shapes(config, split=args.split)
    candidates = enumerate_budget_candidates(
        input_image=shapes["input_image"],
        semantic_channels=shapes["semantic_channels"],
        detail_channels=shapes["detail_channels"],
        semantic_shape=shapes["semantic_shape"],
        detail_shape=shapes["detail_shape"],
        target_effective_cbr=float(args.target_effective_cbr),
        tolerance=float(args.tolerance),
    )

    top_candidates = candidates[: max(1, int(args.top_k))]
    frontier_candidates = extract_candidate_frontier(candidates)
    recommended_candidate = frontier_candidates[0] if len(frontier_candidates) == 1 else None
    payload = {
        "config_path": str(Path(args.config).resolve()),
        "device": config.base.runtime.device,
        "split": args.split,
        "target_effective_cbr": float(args.target_effective_cbr),
        "tolerance": float(args.tolerance),
        "semantic_channels": shapes["semantic_channels"],
        "detail_channels": shapes["detail_channels"],
        "semantic_latent_hw": shapes["semantic_latent_hw"],
        "detail_latent_hw": shapes["detail_latent_hw"],
        "num_candidates": len(candidates),
        "num_within_tolerance": sum(1 for candidate in candidates if candidate.within_target_tolerance),
        "candidate_ordering_policy": "feasible_first_then_gap_only_with_deterministic_tiebreak",
        "frontier_is_quality_ambiguous": len(frontier_candidates) > 1,
        "frontier_size": len(frontier_candidates),
        "frontier_candidates": [asdict(candidate) for candidate in frontier_candidates],
        "recommended_candidate": asdict(recommended_candidate) if recommended_candidate is not None else None,
        "best_candidate": asdict(top_candidates[0]),
        "best_candidate_quality_ranked": recommended_candidate is not None,
        "top_candidates": [asdict(candidate) for candidate in top_candidates],
    }

    if args.output is not None:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

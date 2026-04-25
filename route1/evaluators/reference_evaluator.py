from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch

from configs.route1_reference import Route1ExperimentConfig
from models.swinjscc.upstream_reference import UpstreamImportBundle
from support.runtime import (
    AverageMeter,
    build_ms_ssim_metric,
    compute_ms_ssim_value,
    compute_psnr_from_mse,
    save_json,
)


@torch.no_grad()
def evaluate_reference_model(
    *,
    model: torch.nn.Module,
    eval_loader: Any,
    experiment: Route1ExperimentConfig,
    bundle: UpstreamImportBundle,
    device: torch.device,
    logger: Any | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    metric = build_ms_ssim_metric(bundle.MS_SSIM, device, experiment.trainset)
    snr_values = experiment.snr_values
    channel_values = experiment.channel_values

    result_grid: dict[str, list[list[float]]] = {
        "snr": [],
        "cbr": [],
        "psnr": [],
        "ms_ssim": [],
    }

    for snr in snr_values:
        snr_row: list[float] = []
        cbr_row: list[float] = []
        psnr_row: list[float] = []
        ms_ssim_row: list[float] = []

        for rate in channel_values:
            elapsed_meter = AverageMeter()
            cbr_meter = AverageMeter()
            snr_meter = AverageMeter()
            psnr_meter = AverageMeter()
            ms_ssim_meter = AverageMeter()

            for sample_index, batch in enumerate(eval_loader):
                if experiment.max_eval_samples is not None and sample_index >= experiment.max_eval_samples:
                    break

                if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                    raise ValueError("Evaluation batch must contain (image_tensor, name).")
                inputs, names = batch
                inputs = inputs.to(device)

                start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                cpu_start = time.perf_counter()
                if start is not None and end is not None:
                    start.record()

                recon_image, cbr, used_snr, mse, _ = model(inputs, given_SNR=snr, given_rate=rate)

                if start is not None and end is not None:
                    end.record()
                    torch.cuda.synchronize(device)
                    elapsed_seconds = start.elapsed_time(end) / 1000.0
                else:
                    elapsed_seconds = time.perf_counter() - cpu_start

                mse_value = float(mse.item())
                psnr_value = compute_psnr_from_mse(mse_value)
                ms_ssim_value = compute_ms_ssim_value(metric, inputs, recon_image)

                elapsed_meter.update(elapsed_seconds)
                cbr_meter.update(float(cbr))
                snr_meter.update(float(used_snr))
                psnr_meter.update(psnr_value)
                ms_ssim_meter.update(ms_ssim_value)

            if cbr_meter.count == 0:
                raise RuntimeError(
                    f"Evaluation produced no samples for snr={snr}, rate={rate}."
                )

            if logger is not None:
                logger.info(
                    "Eval summary | channel=%s | snr=%s | rate=%s | cbr=%.6f | psnr=%.4f | ms-ssim=%.6f",
                    experiment.channel_type,
                    snr,
                    rate,
                    cbr_meter.avg,
                    psnr_meter.avg,
                    ms_ssim_meter.avg,
                )

            snr_row.append(snr_meter.avg)
            cbr_row.append(cbr_meter.avg)
            psnr_row.append(psnr_meter.avg)
            ms_ssim_row.append(ms_ssim_meter.avg)

        result_grid["snr"].append(snr_row)
        result_grid["cbr"].append(cbr_row)
        result_grid["psnr"].append(psnr_row)
        result_grid["ms_ssim"].append(ms_ssim_row)

    results = {
        "channel_type": experiment.channel_type,
        "model": experiment.model,
        "snr_values": snr_values,
        "channel_values": channel_values,
        "results": result_grid,
    }
    if output_path is not None:
        save_json(results, output_path)
    return results

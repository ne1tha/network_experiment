from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch

from configs.route1_reference import Route1ExperimentConfig
from datasets.hr_datasets import build_hr_dataloaders
from evaluators.reference_evaluator import evaluate_reference_model
from models.swinjscc.upstream_reference import build_upstream_model
from support.runtime import (
    AverageMeter,
    build_logger,
    build_ms_ssim_metric,
    compute_ms_ssim_value,
    compute_psnr_from_mse,
    ensure_run_dirs,
    load_checkpoint_compatible,
    load_checkpoint_strict,
    save_checkpoint,
    save_config_snapshot,
)


class ReferenceTrainer:
    def __init__(self, experiment: Route1ExperimentConfig) -> None:
        self.experiment = experiment
        self.model, self.bundle, self.runtime_config = build_upstream_model(experiment)
        self.device = self.runtime_config.device
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment.learning_rate)
        self.train_loader, self.eval_loader = build_hr_dataloaders(experiment)
        self.ms_ssim_metric = build_ms_ssim_metric(self.bundle.MS_SSIM, self.device, experiment.trainset)

        run_paths = experiment.run_paths
        workdir = Path(run_paths.workdir)
        models_dir = Path(run_paths.models_dir)
        ensure_run_dirs(workdir, models_dir)
        log_path = Path(run_paths.log_path) if experiment.save_logs else None
        self.logger = build_logger(f"route1.{experiment.run_name}", log_path=log_path)
        self.runtime_config.logger = self.logger
        self.logger.info("Initialized reference trainer for %s", experiment.run_name)
        save_config_snapshot(experiment, workdir / "config_snapshot.json")

        self.global_step = 0
        self.start_epoch = 0
        if experiment.checkpoint_path is not None:
            if experiment.checkpoint_load_mode == "compatible":
                metadata = load_checkpoint_compatible(
                    model=self.model,
                    checkpoint_path=Path(experiment.checkpoint_path),
                    device=self.device,
                )
            else:
                metadata = load_checkpoint_strict(
                    model=self.model,
                    checkpoint_path=Path(experiment.checkpoint_path),
                    device=self.device,
                )
            self.logger.info("Loaded model checkpoint from %s", experiment.checkpoint_path)
            self.logger.info("Checkpoint metadata: %s", metadata)

        if experiment.resume_path is not None:
            metadata = load_checkpoint_strict(
                model=self.model,
                checkpoint_path=Path(experiment.resume_path),
                device=self.device,
                optimizer=self.optimizer,
                require_optimizer=True,
            )
            self.start_epoch = metadata["epoch"]
            self.global_step = metadata["global_step"]
            self.logger.info("Resumed training from %s", experiment.resume_path)
            self.logger.info("Resume metadata: %s", metadata)

    def _move_train_batch(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            if len(batch) != 1:
                raise ValueError(f"Unexpected train batch structure: {type(batch)!r} len={len(batch)}")
            batch = batch[0]
        if not torch.is_tensor(batch):
            raise TypeError(f"Train batch must be a tensor, got {type(batch)!r}")
        return batch.to(self.device)

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        elapsed_meter = AverageMeter()
        loss_meter = AverageMeter()
        cbr_meter = AverageMeter()
        snr_meter = AverageMeter()
        psnr_meter = AverageMeter()
        ms_ssim_meter = AverageMeter()

        for batch_index, batch in enumerate(self.train_loader):
            if self.experiment.max_train_steps is not None and self.global_step >= self.experiment.max_train_steps:
                break

            inputs = self._move_train_batch(batch)
            self.global_step += 1

            if self.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                start = end = None
            cpu_start = time.perf_counter()

            recon_image, cbr, used_snr, mse, loss = self.model(inputs)
            loss_value = float(loss.item())
            if not torch.isfinite(loss):
                raise ValueError(f"Training loss became non-finite at step {self.global_step}: {loss_value}")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if start is not None and end is not None:
                end.record()
                torch.cuda.synchronize(self.device)
                elapsed_seconds = start.elapsed_time(end) / 1000.0
            else:
                elapsed_seconds = time.perf_counter() - cpu_start

            mse_value = float(mse.item())
            psnr_value = compute_psnr_from_mse(mse_value)
            ms_ssim_value = compute_ms_ssim_value(self.ms_ssim_metric, inputs, recon_image)

            elapsed_meter.update(elapsed_seconds)
            loss_meter.update(loss_value)
            cbr_meter.update(float(cbr))
            snr_meter.update(float(used_snr))
            psnr_meter.update(psnr_value)
            ms_ssim_meter.update(ms_ssim_value)

            if self.global_step % self.experiment.print_step == 0:
                self.logger.info(
                    "Train | epoch=%s | step=%s | time=%.4f | loss=%.6f | cbr=%.6f | snr=%.2f | psnr=%.4f | ms-ssim=%.6f | lr=%.6g",
                    epoch + 1,
                    self.global_step,
                    elapsed_meter.val,
                    loss_meter.val,
                    cbr_meter.val,
                    snr_meter.val,
                    psnr_meter.val,
                    ms_ssim_meter.val,
                    self.optimizer.param_groups[0]["lr"],
                )

        if loss_meter.count == 0:
            raise RuntimeError(
                "Training epoch produced no batches. Check dataset size versus batch_size."
            )

        return {
            "loss": loss_meter.avg,
            "cbr": cbr_meter.avg,
            "snr": snr_meter.avg,
            "psnr": psnr_meter.avg,
            "ms_ssim": ms_ssim_meter.avg,
        }

    def train(self) -> dict[str, Any]:
        final_summary: dict[str, Any] = {}
        for epoch in range(self.start_epoch, self.experiment.total_epochs):
            if self.experiment.max_train_steps is not None and self.global_step >= self.experiment.max_train_steps:
                break

            train_summary = self._train_one_epoch(epoch)
            self.logger.info("Epoch summary | epoch=%s | %s", epoch + 1, train_summary)
            final_summary = {"epoch": epoch + 1, "train": train_summary}

            if (epoch + 1) % self.experiment.save_model_freq == 0:
                checkpoint_path = Path(self.experiment.run_paths.models_dir) / f"{self.experiment.run_name}_ep{epoch + 1}.pt"
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    global_step=self.global_step,
                    experiment=self.experiment,
                    output_path=checkpoint_path,
                )
                self.logger.info("Saved checkpoint to %s", checkpoint_path)

                eval_output = Path(self.experiment.run_paths.workdir) / f"eval_epoch_{epoch + 1}.json"
                eval_summary = evaluate_reference_model(
                    model=self.model,
                    eval_loader=self.eval_loader,
                    experiment=self.experiment,
                    bundle=self.bundle,
                    device=self.device,
                    logger=self.logger,
                    output_path=eval_output,
                )
                final_summary["eval"] = eval_summary

        return final_summary

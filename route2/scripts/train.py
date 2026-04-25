from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import replace
import json
from pathlib import Path
import random
import sys

import numpy as np
PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import torch

from route2_swinjscc_gan.common.device import prepare_runtime_device, resolve_runtime_device
from route2_swinjscc_gan.common.io import ensure_dir, save_json
from route2_swinjscc_gan.common.schedulers import build_warmup_cosine_scheduler
from route2_swinjscc_gan.configs.defaults import Route2ExperimentConfig, build_default_experiment_config
from route2_swinjscc_gan.datasets import build_test_loader, build_train_val_loaders
from route2_swinjscc_gan.evaluators import ImageQualityMetricSuite, SwinJSCCGANEvaluator
from route2_swinjscc_gan.losses.adversarial import AdversarialLossConfig, PatchAdversarialLoss
from route2_swinjscc_gan.losses.perceptual_vgg import VGGPerceptualConfig, VGGPerceptualLoss
from route2_swinjscc_gan.losses.reconstruction import ReconstructionLossConfig, build_reconstruction_loss
from route2_swinjscc_gan.models.swinjscc_gan.discriminator_patchgan import PatchGANConfig, PatchGANDiscriminator
from route2_swinjscc_gan.models.swinjscc_gan.generator import SwinJSCCGenerator
from route2_swinjscc_gan.models.swinjscc_gan.training_stages import ProgressiveStageController, ProgressiveTrainingConfig
from route2_swinjscc_gan.trainers.trainer_swinjscc_gan import SwinJSCCGANTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Route 2 SwinJSCC-GAN reproduction.")
    parser.add_argument("--train-roots", nargs="+", required=True, help="Training image directories.")
    parser.add_argument("--val-roots", nargs="*", default=(), help="Optional validation image directories.")
    parser.add_argument("--test-roots", nargs="*", default=(), help="Optional evaluation image directories.")
    parser.add_argument("--output-dir", default="route2_swinjscc_gan/artifacts/train_run", help="Run output directory.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-variant", default="SwinJSCC_w/_SAandRA")
    parser.add_argument("--model-size", default="base", choices=["small", "base", "large"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--channel-type", default="rayleigh", choices=["awgn", "rayleigh"])
    parser.add_argument("--multiple-snr", default="1,4,7,10,13")
    parser.add_argument("--channel-numbers", default="32,64,96,128,192")
    parser.add_argument("--total-epochs", type=int, default=300)
    parser.add_argument("--phase1-epochs", type=int, default=100)
    parser.add_argument("--phase2-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-eval-size-adjustment", action="store_true")
    parser.add_argument("--vgg-weights-path", default=None)
    parser.add_argument("--resume-checkpoint", default=None, help="Optional Route 2 checkpoint used to resume training.")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional Route 2 checkpoint used to initialize weights only for fresh adaptation training.",
    )
    parser.add_argument("--eval-every-epochs", type=int, default=10)
    parser.add_argument("--save-every-epochs", type=int, default=10)
    parser.add_argument("--eval-snr", type=int, default=None, help="Optional evaluation SNR override.")
    parser.add_argument("--eval-rate", type=int, default=None, help="Optional evaluation rate override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional early-stop step limit for smoke runs.")
    parser.add_argument("--disable-adversarial", action="store_true", help="Disable the adversarial branch for ablation.")
    parser.add_argument("--adv-loss-mode", choices=["hinge", "bce"], default=None)
    parser.add_argument("--adv-weight", type=float, default=None)
    parser.add_argument("--adv-ramp-epochs", type=int, default=None)
    parser.add_argument("--disc-lr-scale", type=float, default=None)
    return parser.parse_args()


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"Expected a non-empty comma-separated integer list, got `{raw}`.")
    return values


def build_config(args: argparse.Namespace) -> Route2ExperimentConfig:
    default = build_default_experiment_config()
    data = replace(
        default.data,
        train_roots=tuple(args.train_roots),
        val_roots=tuple(args.val_roots),
        test_roots=tuple(args.test_roots),
        crop_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        allow_eval_size_adjustment=args.allow_eval_size_adjustment,
    )
    model = replace(
        default.model,
        model_variant=args.model_variant,
        model_size=args.model_size,
        image_size=args.image_size,
        channel_type=args.channel_type,
        multiple_snr=_parse_csv_ints(args.multiple_snr),
        channel_numbers=_parse_csv_ints(args.channel_numbers),
        device=args.device,
        vgg_weights_path=args.vgg_weights_path,
    )
    training = replace(
        default.training,
        total_epochs=args.total_epochs,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        output_dir=args.output_dir,
        save_every_epochs=args.save_every_epochs,
        eval_every_epochs=args.eval_every_epochs,
        checkpoint_path=args.resume_checkpoint,
        init_checkpoint_path=args.init_checkpoint,
        max_steps=args.max_steps,
    )
    adversarial = replace(
        default.adversarial,
        enabled=False if args.disable_adversarial else default.adversarial.enabled,
        loss_mode=args.adv_loss_mode if args.adv_loss_mode is not None else default.adversarial.loss_mode,
        weight=args.adv_weight if args.adv_weight is not None else default.adversarial.weight,
        ramp_epochs=args.adv_ramp_epochs if args.adv_ramp_epochs is not None else default.adversarial.ramp_epochs,
        discriminator_lr_scale=(
            args.disc_lr_scale if args.disc_lr_scale is not None else default.adversarial.discriminator_lr_scale
        ),
    )
    evaluation = replace(
        default.evaluation,
        snr=args.eval_snr if args.eval_snr is not None else int(model.multiple_snr[0]),
        rate=args.eval_rate if args.eval_rate is not None else int(model.channel_numbers[0]),
    )
    return Route2ExperimentConfig(
        data=data,
        model=model,
        optimizer=default.optimizer,
        discriminator=default.discriminator,
        adversarial=adversarial,
        training=training,
        evaluation=evaluation,
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    config: Route2ExperimentConfig,
    generator: SwinJSCCGenerator,
    discriminator: PatchGANDiscriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    generator_scheduler: torch.optim.lr_scheduler.LambdaLR,
    discriminator_scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "data_config": asdict(config.data),
            "model_config": asdict(config.model),
            "optimizer_config": asdict(config.optimizer),
            "discriminator_config": asdict(config.discriminator),
            "adversarial_config": asdict(config.adversarial),
            "training_config": asdict(config.training),
            "evaluation_config": asdict(config.evaluation),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "generator_scheduler": generator_scheduler.state_dict(),
            "discriminator_scheduler": discriminator_scheduler.state_dict(),
        },
        path,
    )


def build_perceptual_loss(config: Route2ExperimentConfig, device: torch.device) -> VGGPerceptualLoss:
    return VGGPerceptualLoss(
        VGGPerceptualConfig(weights_path=config.model.vgg_weights_path)
    ).to(device)


def configure_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _fast_forward_scheduler(scheduler: torch.optim.lr_scheduler.LambdaLR, *, steps: int) -> None:
    for _ in range(steps):
        scheduler.step()


def _checkpoint_discriminator_config(checkpoint: dict[str, object]) -> PatchGANConfig:
    payload = checkpoint.get("discriminator_config")
    if payload is None:
        return PatchGANConfig()
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint discriminator_config must be a JSON object.")
    return PatchGANConfig(**payload)


def _should_resume_discriminator_state(
    current_config: PatchGANConfig,
    checkpoint_config: PatchGANConfig,
) -> bool:
    if current_config == checkpoint_config:
        return True
    if checkpoint_config.kind == "legacy_patchgan":
        return False
    raise ValueError(
        "Configured discriminator does not match the checkpoint discriminator. "
        f"Current={current_config.kind}, checkpoint={checkpoint_config.kind}. "
        "Use the matching discriminator config or resume from a compatible checkpoint."
    )


def _resume_route2_checkpoint(
    *,
    checkpoint_path: str | None,
    device: torch.device,
    total_epochs: int,
    steps_per_epoch: int,
    discriminator_config: PatchGANConfig,
    generator: SwinJSCCGenerator,
    discriminator: PatchGANDiscriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    generator_scheduler: torch.optim.lr_scheduler.LambdaLR,
    discriminator_scheduler: torch.optim.lr_scheduler.LambdaLR,
    discriminator_lr: float,
) -> tuple[int, int]:
    if checkpoint_path is None:
        return 0, 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    required_keys = {
        "epoch",
        "generator",
        "discriminator",
        "generator_optimizer",
        "discriminator_optimizer",
    }
    missing = sorted(required_keys.difference(checkpoint.keys()))
    if missing:
        raise KeyError(
            f"Route 2 resume checkpoint is missing required keys {missing}: {checkpoint_path}"
        )

    checkpoint_discriminator_config = _checkpoint_discriminator_config(checkpoint)
    generator.load_state_dict(checkpoint["generator"], strict=True)
    generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
    should_resume_discriminator = _should_resume_discriminator_state(
        discriminator_config,
        checkpoint_discriminator_config,
    )
    if should_resume_discriminator:
        discriminator.load_state_dict(checkpoint["discriminator"], strict=True)
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])

    start_epoch = int(checkpoint["epoch"]) + 1
    if start_epoch >= total_epochs:
        raise ValueError(
            f"Resume checkpoint {checkpoint_path} is already at epoch {start_epoch}, "
            f"which does not leave work for configured total_epochs={total_epochs}."
        )

    global_step = int(checkpoint.get("global_step", start_epoch * steps_per_epoch))
    generator_scheduler_state = checkpoint.get("generator_scheduler")
    discriminator_scheduler_state = checkpoint.get("discriminator_scheduler")
    if generator_scheduler_state is not None:
        generator_scheduler.load_state_dict(generator_scheduler_state)
    else:
        _fast_forward_scheduler(generator_scheduler, steps=global_step)
    if should_resume_discriminator and discriminator_scheduler_state is not None:
        discriminator_scheduler.load_state_dict(discriminator_scheduler_state)
    elif should_resume_discriminator:
        _fast_forward_scheduler(discriminator_scheduler, steps=global_step)
    for param_group in discriminator_optimizer.param_groups:
        param_group["lr"] = discriminator_lr
        if "initial_lr" in param_group:
            param_group["initial_lr"] = discriminator_lr

    return start_epoch, global_step


def _initialize_route2_from_checkpoint(
    *,
    checkpoint_path: str | None,
    device: torch.device,
    discriminator_config: PatchGANConfig,
    generator: SwinJSCCGenerator,
    discriminator: PatchGANDiscriminator,
) -> None:
    if checkpoint_path is None:
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    required_keys = {"generator"}
    missing = sorted(required_keys.difference(checkpoint.keys()))
    if missing:
        raise KeyError(
            f"Route 2 init checkpoint is missing required keys {missing}: {checkpoint_path}"
        )

    generator.load_state_dict(checkpoint["generator"], strict=True)
    checkpoint_discriminator_config = _checkpoint_discriminator_config(checkpoint)
    should_init_discriminator = _should_resume_discriminator_state(
        discriminator_config,
        checkpoint_discriminator_config,
    )
    if should_init_discriminator and "discriminator" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator"], strict=True)


def run_training(config: Route2ExperimentConfig) -> int:
    resolved_device = resolve_runtime_device(config.model.device)
    if resolved_device != config.model.device:
        config = replace(config, model=replace(config.model, device=resolved_device))
    configure_global_seed(config.data.seed)
    device = prepare_runtime_device(config.model.device)

    train_loader, val_loader = build_train_val_loaders(config.data)
    test_loader = build_test_loader(config.data) if config.data.test_roots else None

    generator = SwinJSCCGenerator(config.model.build_generator_config()).to(device)
    discriminator = PatchGANDiscriminator(config.discriminator).to(device)
    reconstruction_loss = build_reconstruction_loss(ReconstructionLossConfig(metric="ms-ssim")).to(device)
    perceptual_loss = None
    adversarial_loss = PatchAdversarialLoss(AdversarialLossConfig(mode=config.adversarial.loss_mode))
    stage_controller = ProgressiveStageController(
        ProgressiveTrainingConfig(
            total_epochs=config.training.total_epochs,
            phase1_epochs=config.training.phase1_epochs,
            phase2_epochs=config.training.phase2_epochs,
            adversarial_weight=config.adversarial.weight,
            adversarial_enabled=config.adversarial.enabled,
            adversarial_ramp_epochs=config.adversarial.ramp_epochs,
        )
    )
    trainer = SwinJSCCGANTrainer(
        generator=generator,
        reconstruction_loss=reconstruction_loss,
        stage_controller=stage_controller,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        adversarial_loss=adversarial_loss,
    )

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config.optimizer.generator_lr,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.optimizer.discriminator_lr * config.adversarial.discriminator_lr_scale,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
    )

    total_steps = config.training.total_epochs * len(train_loader)
    warmup_steps = config.optimizer.warmup_epochs * len(train_loader)
    min_lr_ratio = config.optimizer.min_lr / config.optimizer.generator_lr
    generator_scheduler = build_warmup_cosine_scheduler(
        generator_optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )
    discriminator_scheduler = build_warmup_cosine_scheduler(
        discriminator_optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )

    run_dir = ensure_dir(config.training.output_path)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    validation_dir = ensure_dir(run_dir / "validation")
    save_json(run_dir / "experiment_config.json", asdict(config))
    metric_suite = ImageQualityMetricSuite(lpips_network=config.evaluation.lpips_network).to(device)
    evaluator = SwinJSCCGANEvaluator(generator, metric_suite)
    existing_log_path = run_dir / "training_log.json"
    log_records: list[dict[str, float | int | str]] = []
    if config.training.checkpoint_path is not None and existing_log_path.exists():
        with existing_log_path.open("r", encoding="utf-8") as handle:
            existing_payload = json.load(handle)
        existing_records = existing_payload.get("records", []) if isinstance(existing_payload, dict) else []
        if isinstance(existing_records, list):
            log_records = list(existing_records)
    _initialize_route2_from_checkpoint(
        checkpoint_path=config.training.init_checkpoint_path,
        device=device,
        discriminator_config=config.discriminator,
        generator=generator,
        discriminator=discriminator,
    )
    start_epoch, global_step = _resume_route2_checkpoint(
        checkpoint_path=config.training.checkpoint_path,
        device=device,
        total_epochs=config.training.total_epochs,
        steps_per_epoch=len(train_loader),
        discriminator_config=config.discriminator,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_scheduler=generator_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        discriminator_lr=config.optimizer.discriminator_lr * config.adversarial.discriminator_lr_scale,
    )
    should_stop = False

    for epoch in range(start_epoch, config.training.total_epochs):
        active_stage = stage_controller.stage_at_epoch(epoch)
        if active_stage.perceptual_weight > 0.0 and perceptual_loss is None:
            perceptual_loss = build_perceptual_loss(config, device)
            trainer.perceptual_loss = perceptual_loss

        generator.train()
        discriminator.train()

        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            result = trainer.train_step(
                batch,
                epoch=epoch,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
            )
            generator_scheduler.step()
            if result.discriminator_loss is not None:
                discriminator_scheduler.step()
            global_step += 1

            if global_step % config.training.log_every_steps == 0:
                record = {
                    "epoch": epoch,
                    "step": global_step,
                    "stage": result.stage.name.value,
                    "generator_loss": float(result.generator_loss.item()),
                    "reconstruction_loss": float(result.reconstruction_loss.item()),
                    "cbr": result.cbr,
                    "snr": result.snr,
                    "rate": result.rate,
                }
                if result.perceptual_loss is not None:
                    record["perceptual_loss"] = float(result.perceptual_loss.item())
                if result.adversarial_loss is not None:
                    record["adversarial_loss"] = float(result.adversarial_loss.item())
                if result.discriminator_loss is not None:
                    record["discriminator_loss"] = float(result.discriminator_loss.item())
                log_records.append(record)

            if config.training.max_steps is not None and global_step >= config.training.max_steps:
                should_stop = True
                break

        if (epoch + 1) % config.training.save_every_epochs == 0:
            save_checkpoint(
                checkpoints_dir / f"epoch_{epoch + 1:04d}.pt",
                epoch=epoch,
                global_step=global_step,
                config=config,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                generator_scheduler=generator_scheduler,
                discriminator_scheduler=discriminator_scheduler,
            )

        if (epoch + 1) % config.training.eval_every_epochs == 0:
            summary = evaluator.evaluate(
                _wrap_validation_loader(val_loader),
                device=device,
                snr=config.evaluation.snr,
                rate=config.evaluation.rate,
                output_dir=validation_dir / f"epoch_{epoch + 1:04d}",
                max_saved_images=config.evaluation.max_saved_images if config.evaluation.save_images else 0,
            )
            log_records.append(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "stage": "validation",
                    "psnr": summary.psnr,
                    "ms_ssim": summary.ms_ssim,
                    "lpips": summary.lpips,
                }
            )

        if should_stop:
            break

    save_checkpoint(
        checkpoints_dir / "last.pt",
        epoch=epoch,
        global_step=global_step,
        config=config,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_scheduler=generator_scheduler,
        discriminator_scheduler=discriminator_scheduler,
    )
    save_json(run_dir / "training_log.json", {"records": log_records})

    if test_loader is not None:
        summary = evaluator.evaluate(
            test_loader,
            device=device,
            snr=config.evaluation.snr,
            rate=config.evaluation.rate,
            output_dir=run_dir / "test_eval",
            max_saved_images=config.evaluation.max_saved_images if config.evaluation.save_images else 0,
        )
        save_json(
            run_dir / "test_summary.json",
            {
                "psnr": summary.psnr,
                "ms_ssim": summary.ms_ssim,
                "lpips": summary.lpips,
                "num_samples": summary.num_samples,
            },
        )

    return 0


def main() -> int:
    args = parse_args()
    config = build_config(args)
    return run_training(config)


def _wrap_validation_loader(loader: torch.utils.data.DataLoader[torch.Tensor]) -> torch.utils.data.DataLoader[tuple[torch.Tensor, list[str]]]:
    class _ValidationWrapper:
        def __iter__(self_inner):
            for index, batch in enumerate(loader):
                names = [f"val_{index:06d}_{sample_idx}" for sample_idx in range(batch.shape[0])]
                yield batch, names

        def __len__(self_inner):
            return len(loader)

    return _ValidationWrapper()  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())

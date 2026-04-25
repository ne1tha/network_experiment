from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from channels.specs import validate_channel_type
from configs.route1_reference import Route1ExperimentConfig
from datasets.path_checks import require_existing_paths, require_images_present


DEFAULT_UPSTREAM_ROOT = Path("/mnt/nvme/jiwang/third_party/SwinJSCC")


@dataclass(slots=True)
class UpstreamImportBundle:
    upstream_root: Path
    SwinJSCC: type[nn.Module]
    get_loader: object | None
    AverageMeter: object
    MS_SSIM: type[nn.Module]
    logger_configuration: object
    save_model: object
    seed_torch: object


class Route1RuntimeConfig(SimpleNamespace):
    pass


def _patch_upstream_cpu_compatibility() -> None:
    from net import channel as channel_module
    from net import decoder as decoder_module
    from net import encoder as encoder_module

    if getattr(encoder_module, "_route1_cpu_patch_applied", False):
        return

    def patched_update_mask(self) -> None:
        if self.shift_size <= 0:
            self.attn_mask = None
            return

        H, W = self.input_resolution
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = encoder_module.window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        self.attn_mask = attn_mask.to(self.attn.relative_position_bias_table.device)

    def patched_encoder_forward(self, x, snr, rate, model):
        B, _, _, _ = x.size()
        device = x.device
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        latent_length = x.size(1)

        if model == "SwinJSCC_w/o_SAandRA":
            x = self.head_list(x)
            return x

        if model == "SwinJSCC_w/_SA":
            snr_tensor = torch.tensor(snr, dtype=torch.float, device=device)
            snr_batch = snr_tensor.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                temp = self.sm_list[i](x.detach() if i == 0 else temp)
                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, latent_length, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            x = self.head_list(x)
            return x

        if model == "SwinJSCC_w/_RA":
            rate_tensor = torch.tensor(rate, dtype=torch.float, device=device)
            rate_batch = rate_tensor.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                temp = self.sm_list[i](x.detach() if i == 0 else temp)
                bm = self.bm_list[i](rate_batch).unsqueeze(1).expand(-1, latent_length, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            mask = torch.sum(mod_val, dim=1)
            _, indices = mask.sort(dim=1, descending=True)
            c_indices = indices[:, :rate]
            add = torch.arange(0, B * x.size(2), x.size(2), device=device, dtype=torch.int64)
            add = add.unsqueeze(1).repeat(1, rate)
            c_indices = c_indices + add
            mask = torch.zeros(mask.size(), device=device).reshape(-1)
            mask[c_indices.reshape(-1)] = 1
            mask = mask.reshape(B, x.size(2))
            mask = mask.unsqueeze(1).expand(-1, latent_length, -1)
            x = x * mask
            return x, mask

        if model == "SwinJSCC_w/_SAandRA":
            snr_tensor = torch.tensor(snr, dtype=torch.float, device=device)
            rate_tensor = torch.tensor(rate, dtype=torch.float, device=device)
            snr_batch = snr_tensor.unsqueeze(0).expand(B, -1)
            rate_batch = rate_tensor.unsqueeze(0).expand(B, -1)

            for i in range(self.layer_num):
                temp = self.sm_list1[i](x.detach() if i == 0 else temp)
                bm = self.bm_list1[i](snr_batch).unsqueeze(1).expand(-1, latent_length, -1)
                temp = temp * bm
            mod_val1 = self.sigmoid1(self.sm_list1[-1](temp))
            x = x * mod_val1

            for i in range(self.layer_num):
                temp = self.sm_list[i](x.detach() if i == 0 else temp)
                bm = self.bm_list[i](rate_batch).unsqueeze(1).expand(-1, latent_length, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            mask = torch.sum(mod_val, dim=1)
            _, indices = mask.sort(dim=1, descending=True)
            c_indices = indices[:, :rate]
            add = torch.arange(0, B * x.size(2), x.size(2), device=device, dtype=torch.int64)
            add = add.unsqueeze(1).repeat(1, rate)
            c_indices = c_indices + add
            mask = torch.zeros(mask.size(), device=device).reshape(-1)
            mask[c_indices.reshape(-1)] = 1
            mask = mask.reshape(B, x.size(2))
            mask = mask.unsqueeze(1).expand(-1, latent_length, -1)
            x = x * mask
            return x, mask

        raise ValueError(f"Unsupported upstream model variant: {model!r}")

    def patched_decoder_forward(self, x, snr, model):
        if model == "SwinJSCC_w/o_SAandRA":
            x = self.head_list(x)
            for layer in self.layers:
                x = layer(x)
            B, L, N = x.shape
            return x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)

        if model == "SwinJSCC_w/_SA":
            B, L, _ = x.size()
            device = x.device
            x = self.head_list(x)
            snr_tensor = torch.tensor(snr, dtype=torch.float, device=device)
            snr_batch = snr_tensor.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                temp = self.sm_list[i](x.detach() if i == 0 else temp)
                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            for layer in self.layers:
                x = layer(x)
            B, L, N = x.shape
            return x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)

        if model == "SwinJSCC_w/_RA":
            for layer in self.layers:
                x = layer(x)
            B, L, N = x.shape
            return x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)

        if model == "SwinJSCC_w/_SAandRA":
            B, L, _ = x.size()
            device = x.device
            snr_tensor = torch.tensor(snr, dtype=torch.float, device=device)
            snr_batch = snr_tensor.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                temp = self.sm_list[i](x.detach() if i == 0 else temp)
                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            for layer in self.layers:
                x = layer(x)
            B, L, N = x.shape
            return x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)

        raise ValueError(f"Unsupported upstream model variant: {model!r}")

    def patched_gaussian_noise_layer(self, input_layer, std, name=None):
        device = input_layer.device
        noise_real = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        return input_layer + (noise_real + 1j * noise_imag)

    def patched_rayleigh_noise_layer(self, input_layer, std, name=None):
        device = input_layer.device
        noise_real = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise = noise_real + 1j * noise_imag
        h = torch.sqrt(
            torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=device) ** 2
            + torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=device) ** 2
        ) / np.sqrt(2)
        return input_layer * h + noise

    encoder_module.SwinTransformerBlock.update_mask = patched_update_mask
    encoder_module.SwinJSCC_Encoder.forward = patched_encoder_forward
    decoder_module.SwinJSCC_Decoder.forward = patched_decoder_forward
    channel_module.Channel.gaussian_noise_layer = patched_gaussian_noise_layer
    channel_module.Channel.rayleigh_noise_layer = patched_rayleigh_noise_layer
    encoder_module._route1_cpu_patch_applied = True


def ensure_upstream_repo(upstream_root: Path | None = None) -> Path:
    root = Path(upstream_root or DEFAULT_UPSTREAM_ROOT).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Upstream SwinJSCC repo not found: {root}")

    required = [
        root / "main.py",
        root / "net" / "network.py",
        root / "data" / "datasets.py",
        root / "loss" / "distortion.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Upstream repo is incomplete, missing: {joined}")
    return root


def load_upstream_bundle(
    upstream_root: Path | None = None,
    *,
    include_dataloader: bool = False,
) -> UpstreamImportBundle:
    root = ensure_upstream_repo(upstream_root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    _patch_upstream_cpu_compatibility()
    from loss.distortion import MS_SSIM
    from net.network import SwinJSCC
    from utils import AverageMeter, logger_configuration, save_model, seed_torch
    get_loader = None
    if include_dataloader:
        from data.datasets import get_loader as upstream_get_loader

        get_loader = upstream_get_loader

    return UpstreamImportBundle(
        upstream_root=root,
        SwinJSCC=SwinJSCC,
        get_loader=get_loader,
        AverageMeter=AverageMeter,
        MS_SSIM=MS_SSIM,
        logger_configuration=logger_configuration,
        save_model=save_model,
        seed_torch=seed_torch,
    )


def build_runtime_config(
    experiment: Route1ExperimentConfig,
    *,
    validate_datasets: bool = True,
) -> Route1RuntimeConfig:
    experiment.validate()
    validate_channel_type(experiment.channel_type)
    if validate_datasets:
        require_existing_paths(experiment.dataset_paths.train_dirs, label="training dataset")
        require_existing_paths(experiment.dataset_paths.test_dirs, label="evaluation dataset")
        require_images_present(experiment.dataset_paths.train_dirs, label="training dataset")
        require_images_present(experiment.dataset_paths.test_dirs, label="evaluation dataset")

    use_cuda = experiment.device.startswith("cuda")
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            f"Configured device {experiment.device!r} requires CUDA, but CUDA is unavailable."
        )

    if experiment.trainset == "CIFAR10":
        image_dims = (3, 32, 32)
        downsample = 2
        channel_number = experiment.channel_values[0]
        batch_size = 128
        save_model_freq = 5
        encoder_kwargs = dict(
            model=experiment.model,
            img_size=(image_dims[1], image_dims[2]),
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 256],
            depths=[2, 4],
            num_heads=[4, 8],
            C=channel_number,
            window_size=2,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        decoder_kwargs = dict(
            model=experiment.model,
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128],
            depths=[4, 2],
            num_heads=[8, 4],
            C=channel_number,
            window_size=2,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
    else:
        image_dims = (3, 256, 256)
        downsample = 4
        batch_size = experiment.batch_size
        save_model_freq = experiment.save_model_freq
        if experiment.model in {"SwinJSCC_w/o_SAandRA", "SwinJSCC_w/_SA"}:
            channel_number = experiment.channel_values[0]
        else:
            channel_number = None

        size_map = {
            "small": ([128, 192, 256, 320], [2, 2, 2, 2], [4, 6, 8, 10]),
            "base": ([128, 192, 256, 320], [2, 2, 6, 2], [4, 6, 8, 10]),
            "large": ([128, 192, 256, 320], [2, 2, 18, 2], [4, 6, 8, 10]),
        }
        embed_dims, encoder_depths, encoder_heads = size_map[experiment.model_size]
        decoder_depths = list(reversed(encoder_depths))
        decoder_heads = list(reversed(encoder_heads))
        decoder_dims = list(reversed(embed_dims))

        encoder_kwargs = dict(
            model=experiment.model,
            img_size=(image_dims[1], image_dims[2]),
            patch_size=2,
            in_chans=3,
            embed_dims=embed_dims,
            depths=encoder_depths,
            num_heads=encoder_heads,
            C=channel_number,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        decoder_kwargs = dict(
            model=experiment.model,
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=decoder_dims,
            depths=decoder_depths,
            num_heads=decoder_heads,
            C=channel_number,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )

    train_dirs, test_dirs = experiment.dataset_paths.as_strings()
    run_paths = experiment.run_paths

    return Route1RuntimeConfig(
        seed=experiment.seed,
        pass_channel=experiment.pass_channel,
        CUDA=use_cuda,
        device=torch.device(experiment.device if use_cuda else "cpu"),
        norm=experiment.norm,
        print_step=experiment.print_step,
        plot_step=experiment.plot_step,
        filename=run_paths.run_name,
        workdir=str(run_paths.workdir),
        log=str(run_paths.log_path),
        samples=str(run_paths.samples_dir),
        models=str(run_paths.models_dir),
        logger=None,
        normalize=False,
        learning_rate=experiment.learning_rate,
        tot_epoch=experiment.total_epochs,
        save_model_freq=save_model_freq,
        image_dims=image_dims,
        train_data_dir=train_dirs,
        test_data_dir=test_dirs,
        batch_size=batch_size,
        downsample=downsample,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
    )


def build_upstream_model(
    experiment: Route1ExperimentConfig,
    *,
    upstream_root: Path | None = None,
    include_dataloader: bool = False,
    validate_datasets: bool = True,
) -> tuple[nn.Module, object, Route1RuntimeConfig]:
    bundle = load_upstream_bundle(upstream_root, include_dataloader=include_dataloader)
    args = experiment.to_upstream_args()
    runtime_config = build_runtime_config(experiment, validate_datasets=validate_datasets)
    model = bundle.SwinJSCC(args, runtime_config)
    return model, bundle, runtime_config

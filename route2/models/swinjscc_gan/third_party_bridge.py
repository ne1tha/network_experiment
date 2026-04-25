from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType

import torch

from route2_swinjscc_gan.common.checks import require


THIRD_PARTY_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "SwinJSCC"


@dataclass(frozen=True)
class ThirdPartySwinJSCCModules:
    encoder_module: ModuleType
    decoder_module: ModuleType
    channel_module: ModuleType


def _ensure_third_party_root() -> None:
    require(
        THIRD_PARTY_ROOT.is_dir(),
        f"Expected original SwinJSCC repository at {THIRD_PARTY_ROOT}, but it does not exist.",
    )
    root_str = str(THIRD_PARTY_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _patch_encoder_module(encoder_module: ModuleType) -> None:
    if getattr(encoder_module, "_route2_device_safe_patch", False):
        return

    def patched_update_mask(self):
        if self.shift_size <= 0:
            return
        height, width = self.input_resolution
        img_mask = torch.zeros((1, height, width, 1), device=self.norm1.weight.device)
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
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1

        mask_windows = encoder_module.window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.attn_mask = attn_mask.to(self.norm1.weight.device)

    def patched_encoder_forward(self, x, snr, rate, model):
        batch_size, _, _, _ = x.size()
        device = x.device
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        token_count = x.shape[1]

        if model == "SwinJSCC_w/o_SAandRA":
            return self.head_list(x)

        if model == "SwinJSCC_w/_SA":
            snr_tensor = torch.tensor(snr, dtype=torch.float32, device=device)
            snr_batch = snr_tensor.unsqueeze(0).expand(batch_size, -1)
            for index in range(self.layer_num):
                if index == 0:
                    temp = self.sm_list[index](x.detach())
                else:
                    temp = self.sm_list[index](temp)
                bm = self.bm_list[index](snr_batch).unsqueeze(1).expand(-1, token_count, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            return self.head_list(x)

        rate_int = int(rate)
        rate_tensor = torch.tensor(rate_int, dtype=torch.float32, device=device)
        rate_batch = rate_tensor.unsqueeze(0).expand(batch_size, -1)

        if model == "SwinJSCC_w/_RA":
            for index in range(self.layer_num):
                if index == 0:
                    temp = self.sm_list[index](x.detach())
                else:
                    temp = self.sm_list[index](temp)
                bm = self.bm_list[index](rate_batch).unsqueeze(1).expand(-1, token_count, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
        elif model == "SwinJSCC_w/_SAandRA":
            snr_tensor = torch.tensor(snr, dtype=torch.float32, device=device)
            snr_batch = snr_tensor.unsqueeze(0).expand(batch_size, -1)
            for index in range(self.layer_num):
                if index == 0:
                    temp = self.sm_list1[index](x.detach())
                else:
                    temp = self.sm_list1[index](temp)
                bm = self.bm_list1[index](snr_batch).unsqueeze(1).expand(-1, token_count, -1)
                temp = temp * bm
            mod_val1 = self.sigmoid1(self.sm_list1[-1](temp))
            x = x * mod_val1

            for index in range(self.layer_num):
                if index == 0:
                    temp = self.sm_list[index](x.detach())
                else:
                    temp = self.sm_list[index](temp)
                bm = self.bm_list[index](rate_batch).unsqueeze(1).expand(-1, token_count, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
        else:
            raise ValueError(f"Unsupported encoder model variant `{model}`.")

        channel_scores = torch.sum(mod_val, dim=1)
        if rate_int <= 0 or rate_int > x.size(2):
            raise ValueError(f"Requested rate {rate_int} is invalid for latent dimension {x.size(2)}.")
        topk_indices = channel_scores.topk(k=rate_int, dim=1, largest=True).indices
        base_mask = torch.zeros(batch_size, x.size(2), device=device, dtype=x.dtype)
        base_mask.scatter_(1, topk_indices, 1.0)
        mask = base_mask.unsqueeze(1).expand(-1, token_count, -1)
        x = x * mask
        return x, mask

    encoder_module.SwinTransformerBlock.update_mask = patched_update_mask
    encoder_module.SwinJSCC_Encoder.forward = patched_encoder_forward
    encoder_module._route2_device_safe_patch = True


def _patch_decoder_module(decoder_module: ModuleType) -> None:
    if getattr(decoder_module, "_route2_device_safe_patch", False):
        return

    def patched_decoder_forward(self, x, snr, model):
        if model == "SwinJSCC_w/o_SAandRA":
            x = self.head_list(x)
            for layer in self.layers:
                x = layer(x)
            batch_size, _, channels = x.shape
            return x.reshape(batch_size, self.H, self.W, channels).permute(0, 3, 1, 2)

        if model == "SwinJSCC_w/_RA":
            for layer in self.layers:
                x = layer(x)
            batch_size, _, channels = x.shape
            return x.reshape(batch_size, self.H, self.W, channels).permute(0, 3, 1, 2)

        batch_size, token_count, _ = x.size()
        device = x.device
        if model == "SwinJSCC_w/_SA":
            x = self.head_list(x)
        snr_tensor = torch.tensor(snr, dtype=torch.float32, device=device)
        snr_batch = snr_tensor.unsqueeze(0).expand(batch_size, -1)
        for index in range(self.layer_num):
            if index == 0:
                temp = self.sm_list[index](x.detach())
            else:
                temp = self.sm_list[index](temp)
            bm = self.bm_list[index](snr_batch).unsqueeze(1).expand(-1, token_count, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        x = x * mod_val
        for layer in self.layers:
            x = layer(x)
        batch_size, _, channels = x.shape
        return x.reshape(batch_size, self.H, self.W, channels).permute(0, 3, 1, 2)

    decoder_module.SwinJSCC_Decoder.forward = patched_decoder_forward
    decoder_module._route2_device_safe_patch = True


def load_third_party_swinjscc() -> ThirdPartySwinJSCCModules:
    _ensure_third_party_root()
    try:
        encoder_module = import_module("net.encoder")
        decoder_module = import_module("net.decoder")
        channel_module = import_module("net.channel")
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise RuntimeError(
            "Failed to import `third_party/SwinJSCC`. Missing dependency "
            f"`{missing}`. Route 2 must inherit the original backbone and must not "
            "silently downgrade to a simplified implementation."
        ) from exc

    _patch_encoder_module(encoder_module)
    _patch_decoder_module(decoder_module)

    return ThirdPartySwinJSCCModules(
        encoder_module=encoder_module,
        decoder_module=decoder_module,
        channel_module=channel_module,
    )

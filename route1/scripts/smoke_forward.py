from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.route1_reference import build_div2k_reference_config
from models.swinjscc.upstream_reference import build_upstream_model


def main() -> None:
    config = build_div2k_reference_config(
        workspace_root=REPO_ROOT,
        train_dirs=[REPO_ROOT],
        test_dirs=[REPO_ROOT],
        testset="kodak",
        model="SwinJSCC_w/o_SAandRA",
        channel_type="awgn",
        channels_csv="96",
        snrs_csv="10",
        model_size="base",
        run_name="smoke_forward",
    )
    config.device = "cpu"
    config.pass_channel = False

    model, _, runtime_config = build_upstream_model(config, validate_datasets=False)
    device = runtime_config.device
    model = model.to(device)
    model.eval()
    model.H = 256
    model.W = 256

    dummy = torch.rand(1, 3, 256, 256, device=device)
    with torch.no_grad():
        recon, cbr, snr, mse, loss = model(dummy, given_SNR=10, given_rate=96)

    print(f"input_shape={tuple(dummy.shape)}")
    print(f"recon_shape={tuple(recon.shape)}")
    print(f"cbr={float(cbr):.6f}")
    print(f"snr={float(snr):.2f}")
    print(f"mse={float(mse):.6f}")
    print(f"loss={float(loss):.6f}")


if __name__ == "__main__":
    main()

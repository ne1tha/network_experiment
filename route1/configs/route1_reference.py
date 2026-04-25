from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence


SUPPORTED_TRAINSETS = {"CIFAR10", "DIV2K"}
SUPPORTED_TESTSETS = {"kodak", "CLIC21", "ffhq"}
SUPPORTED_DISTORTION_METRICS = {"MSE", "MS-SSIM"}
SUPPORTED_MODELS = {
    "SwinJSCC_w/o_SAandRA",
    "SwinJSCC_w/_SA",
    "SwinJSCC_w/_RA",
    "SwinJSCC_w/_SAandRA",
}
SUPPORTED_CHANNELS = {"awgn", "rayleigh"}
SUPPORTED_MODEL_SIZES = {"small", "base", "large"}
SUPPORTED_CHECKPOINT_LOAD_MODES = {"strict", "compatible"}


def parse_csv_ints(value: str) -> list[int]:
    if not value or not value.strip():
        raise ValueError("CSV integer string must not be empty.")

    parsed: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            raise ValueError(f"Invalid CSV integer string {value!r}: empty item detected.")
        parsed.append(int(item))
    return parsed


@dataclass(slots=True)
class DatasetPaths:
    train_dirs: list[Path]
    test_dirs: list[Path]

    def as_strings(self) -> tuple[list[str], list[str]]:
        return ([str(path) for path in self.train_dirs], [str(path) for path in self.test_dirs])


@dataclass(slots=True)
class RunPaths:
    workspace_root: Path
    run_name: str = "route1_reference"
    histories_dirname: str = "artifacts/history"
    checkpoints_dirname: str = "artifacts/checkpoints"

    @property
    def workdir(self) -> Path:
        return self.workspace_root / self.histories_dirname / self.run_name

    @property
    def log_path(self) -> Path:
        return self.workdir / f"{self.run_name}.log"

    @property
    def samples_dir(self) -> Path:
        return self.workdir / "samples"

    @property
    def models_dir(self) -> Path:
        return self.workspace_root / self.checkpoints_dirname / self.run_name


@dataclass(slots=True)
class Route1ExperimentConfig:
    workspace_root: Path
    dataset_paths: DatasetPaths
    trainset: str = "DIV2K"
    testset: str = "kodak"
    distortion_metric: str = "MSE"
    model: str = "SwinJSCC_w/_SAandRA"
    channel_type: str = "awgn"
    channels_csv: str = "32,64,96,128,192"
    snrs_csv: str = "1,4,7,10,13"
    model_size: str = "base"
    training: bool = False
    pass_channel: bool = True
    norm: bool = False
    learning_rate: float = 1e-4
    save_model_freq: int = 100
    batch_size: int = 16
    total_epochs: int = 10_000_000
    num_workers: int = 0
    pin_memory: bool = False
    print_step: int = 100
    plot_step: int = 10_000
    run_name: str = "route1_reference"
    device: str = "cuda:0"
    seed: int = 42
    save_logs: bool = True
    checkpoint_path: Path | None = None
    checkpoint_load_mode: str = "strict"
    resume_path: Path | None = None
    max_train_steps: int | None = None
    max_eval_samples: int | None = None
    allow_eval_size_adjustment: bool = False

    def validate(self) -> None:
        if self.trainset not in SUPPORTED_TRAINSETS:
            raise ValueError(f"Unsupported trainset {self.trainset!r}.")
        if self.testset not in SUPPORTED_TESTSETS:
            raise ValueError(f"Unsupported testset {self.testset!r}.")
        if self.distortion_metric not in SUPPORTED_DISTORTION_METRICS:
            raise ValueError(f"Unsupported distortion metric {self.distortion_metric!r}.")
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model variant {self.model!r}.")
        if self.channel_type not in SUPPORTED_CHANNELS:
            raise ValueError(f"Unsupported channel type {self.channel_type!r}.")
        if self.model_size not in SUPPORTED_MODEL_SIZES:
            raise ValueError(f"Unsupported model size {self.model_size!r}.")
        if self.checkpoint_load_mode not in SUPPORTED_CHECKPOINT_LOAD_MODES:
            raise ValueError(
                f"Unsupported checkpoint load mode {self.checkpoint_load_mode!r}."
            )

        channels = parse_csv_ints(self.channels_csv)
        snrs = parse_csv_ints(self.snrs_csv)
        if any(channel <= 0 for channel in channels):
            raise ValueError(f"Channel counts must be positive, got {channels!r}.")
        if len(snrs) == 0:
            raise ValueError("At least one SNR value is required.")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}.")
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive when provided.")
        if self.max_eval_samples is not None and self.max_eval_samples <= 0:
            raise ValueError("max_eval_samples must be positive when provided.")
        if self.checkpoint_path is not None and self.resume_path is not None:
            raise ValueError("checkpoint_path and resume_path cannot be set at the same time.")
        if self.checkpoint_path is not None and not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {self.checkpoint_path}")
        if self.resume_path is not None and not Path(self.resume_path).exists():
            raise FileNotFoundError(f"Resume path does not exist: {self.resume_path}")

    @property
    def run_paths(self) -> RunPaths:
        return RunPaths(workspace_root=self.workspace_root, run_name=self.run_name)

    @property
    def channel_values(self) -> list[int]:
        return parse_csv_ints(self.channels_csv)

    @property
    def snr_values(self) -> list[int]:
        return parse_csv_ints(self.snrs_csv)

    def to_upstream_args(self) -> SimpleNamespace:
        self.validate()
        return SimpleNamespace(
            training=self.training,
            trainset=self.trainset,
            testset=self.testset,
            distortion_metric=self.distortion_metric,
            model=self.model,
            channel_type=self.channel_type,
            C=self.channels_csv,
            multiple_snr=self.snrs_csv,
            model_size=self.model_size,
        )


def ensure_existing_dirs(paths: Sequence[Path], *, label: str) -> None:
    if not paths:
        raise FileNotFoundError(f"No paths provided for {label}.")

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing {label} path(s): {joined}")


def build_div2k_reference_config(
    workspace_root: Path,
    train_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    *,
    testset: str = "kodak",
    model: str = "SwinJSCC_w/_SAandRA",
    channel_type: str = "awgn",
    channels_csv: str = "32,64,96,128,192",
    snrs_csv: str = "1,4,7,10,13",
    model_size: str = "base",
    run_name: str = "route1_reference",
    training: bool = False,
    distortion_metric: str = "MSE",
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
    total_epochs: int = 10_000_000,
    learning_rate: float = 1e-4,
    save_model_freq: int = 100,
    print_step: int = 100,
    device: str = "cuda:0",
    checkpoint_path: Path | None = None,
    checkpoint_load_mode: str = "strict",
    resume_path: Path | None = None,
    save_logs: bool = True,
    max_train_steps: int | None = None,
    max_eval_samples: int | None = None,
    allow_eval_size_adjustment: bool = False,
) -> Route1ExperimentConfig:
    dataset_paths = DatasetPaths(
        train_dirs=[Path(path) for path in train_dirs],
        test_dirs=[Path(path) for path in test_dirs],
    )
    ensure_existing_dirs(dataset_paths.train_dirs, label="training dataset")
    ensure_existing_dirs(dataset_paths.test_dirs, label="evaluation dataset")

    config = Route1ExperimentConfig(
        workspace_root=Path(workspace_root),
        dataset_paths=dataset_paths,
        trainset="DIV2K",
        testset=testset,
        distortion_metric=distortion_metric,
        model=model,
        channel_type=channel_type,
        channels_csv=channels_csv,
        snrs_csv=snrs_csv,
        model_size=model_size,
        training=training,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        total_epochs=total_epochs,
        learning_rate=learning_rate,
        save_model_freq=save_model_freq,
        print_step=print_step,
        run_name=run_name,
        device=device,
        checkpoint_path=checkpoint_path,
        checkpoint_load_mode=checkpoint_load_mode,
        resume_path=resume_path,
        save_logs=save_logs,
        max_train_steps=max_train_steps,
        max_eval_samples=max_eval_samples,
        allow_eval_size_adjustment=allow_eval_size_adjustment,
    )
    config.validate()
    return config

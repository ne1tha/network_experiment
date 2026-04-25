from __future__ import annotations

from pathlib import Path

from route2_swinjscc_gan.scripts.watch_training_until_done import (
    CHECKPOINT_PATTERN,
    VALIDATION_PATTERN,
    is_run_completed,
    latest_epoch,
    scheduled_epochs,
)


def test_scheduled_epochs_uses_interval_until_total() -> None:
    assert scheduled_epochs(25, 10) == (10, 20)


def test_latest_epoch_parses_checkpoint_and_validation_names(tmp_path: Path) -> None:
    checkpoint_paths = [
        tmp_path / "epoch_0010.pt",
        tmp_path / "epoch_0040.pt",
        tmp_path / "epoch_0020.pt",
    ]
    validation_paths = [
        tmp_path / "epoch_0010",
        tmp_path / "epoch_0030",
    ]

    for path in checkpoint_paths + validation_paths:
        path.touch()

    assert latest_epoch(checkpoint_paths, CHECKPOINT_PATTERN) == 40
    assert latest_epoch(validation_paths, VALIDATION_PATTERN) == 30


def test_is_run_completed_requires_test_summary_when_test_set_exists() -> None:
    assert not is_run_completed(
        has_last_checkpoint=True,
        has_training_log=True,
        has_test_summary=False,
        require_test_summary=True,
    )
    assert is_run_completed(
        has_last_checkpoint=True,
        has_training_log=True,
        has_test_summary=False,
        require_test_summary=False,
    )

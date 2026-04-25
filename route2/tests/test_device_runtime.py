from __future__ import annotations

import pytest

from route2_swinjscc_gan.common.device import resolve_runtime_device


def test_resolve_runtime_device_accepts_cpu() -> None:
    assert resolve_runtime_device("cpu") == "cpu"


def test_resolve_runtime_device_rejects_cuda_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="torch.cuda.is_available\\(\\) is False"):
        resolve_runtime_device("cuda:0")


def test_resolve_runtime_device_normalizes_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)

    assert resolve_runtime_device("cuda") == "cuda:0"
    assert resolve_runtime_device("cuda:1") == "cuda:1"


def test_resolve_runtime_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert resolve_runtime_device("auto") == "cpu"

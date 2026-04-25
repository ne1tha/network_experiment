from __future__ import annotations

from typing import Any

import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_in(name: str, value: Any, choices: set[str]) -> None:
    if value not in choices:
        raise ValueError(f"Invalid {name} `{value}`. Expected one of {sorted(choices)}.")


def require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer, got {value!r}.")


def require_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"`{name}` must be a non-negative integer, got {value!r}.")


def require_non_empty_sequence(name: str, values: tuple[Any, ...] | list[Any]) -> None:
    if len(values) == 0:
        raise ValueError(f"`{name}` must not be empty.")


def require_shape(
    tensor: torch.Tensor,
    *,
    name: str,
    ndim: int | None = None,
    channels: int | None = None,
) -> None:
    if ndim is not None and tensor.ndim != ndim:
        raise RuntimeError(f"`{name}` must be a {ndim}D tensor, got shape {tuple(tensor.shape)}.")
    if channels is not None:
        require(
            tensor.ndim >= 2 and tensor.shape[1] == channels,
            f"`{name}` must have {channels} channels, got shape {tuple(tensor.shape)}.",
        )


def require_same_shape(name_a: str, tensor_a: torch.Tensor, name_b: str, tensor_b: torch.Tensor) -> None:
    if tuple(tensor_a.shape) != tuple(tensor_b.shape):
        raise RuntimeError(
            f"`{name_a}` shape {tuple(tensor_a.shape)} does not match "
            f"`{name_b}` shape {tuple(tensor_b.shape)}."
        )


def require_finite(value: torch.Tensor | float | int, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"`{name}` contains NaN or Inf.")
        return
    if value != value or value in (float("inf"), float("-inf")):
        raise FloatingPointError(f"`{name}` contains NaN or Inf.")


def require_has_gradients(module: torch.nn.Module, name: str) -> None:
    has_grad = False
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            continue
        require_finite(parameter.grad, f"{name}.grad")
        has_grad = True
    if not has_grad:
        raise RuntimeError(f"`{name}` produced no gradients.")

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


TensorPyramid = Tuple[torch.Tensor, ...]


def assert_valid_image_size(height: int, width: int) -> None:
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(
            "Route 3 expects image height and width divisible by 16; "
            f"received {(height, width)}."
        )


def _validate_pyramid(name: str, pyramid: TensorPyramid, batch_size: int) -> None:
    if not isinstance(pyramid, tuple) or not pyramid:
        raise ValueError(f"{name} must be a non-empty tuple of tensors.")

    prev_hw = None
    for idx, tensor in enumerate(pyramid):
        if tensor.ndim != 4:
            raise ValueError(f"{name}[{idx}] must be a 4D tensor, got {tensor.ndim}D.")
        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"{name}[{idx}] batch size mismatch: expected {batch_size}, got {tensor.shape[0]}."
            )
        hw = tensor.shape[-2:]
        if prev_hw is not None and (hw[0] > prev_hw[0] or hw[1] > prev_hw[1]):
            raise ValueError(
                f"{name} spatial resolutions must be non-increasing, got {prev_hw} -> {hw}."
            )
        prev_hw = hw


@dataclass(frozen=True)
class DualPathEncoderOutput:
    z_sem: torch.Tensor
    z_det: torch.Tensor
    sem_pyramid: TensorPyramid
    det_pyramid: TensorPyramid
    input_size: Tuple[int, int]

    def validate(self) -> "DualPathEncoderOutput":
        input_h, input_w = self.input_size
        assert_valid_image_size(input_h, input_w)

        if self.z_sem.ndim != 4:
            raise ValueError(f"z_sem must be 4D, got {self.z_sem.ndim}D.")
        if self.z_det.ndim != 4:
            raise ValueError(f"z_det must be 4D, got {self.z_det.ndim}D.")
        if self.z_sem.shape[0] != self.z_det.shape[0]:
            raise ValueError(
                "z_sem and z_det batch sizes must match, "
                f"got {self.z_sem.shape[0]} and {self.z_det.shape[0]}."
            )
        if self.z_sem.data_ptr() == self.z_det.data_ptr():
            raise ValueError("z_sem and z_det must come from independent branch tensors.")

        expected_sem_hw = (input_h // 16, input_w // 16)
        expected_det_hw = (input_h // 4, input_w // 4)

        if self.z_sem.shape[-2:] != expected_sem_hw:
            raise ValueError(
                "z_sem must have H/16 x W/16 spatial size; "
                f"expected {expected_sem_hw}, got {self.z_sem.shape[-2:]}."
            )
        if self.z_det.shape[-2:] != expected_det_hw:
            raise ValueError(
                "z_det must have H/4 x W/4 spatial size; "
                f"expected {expected_det_hw}, got {self.z_det.shape[-2:]}."
            )

        _validate_pyramid("sem_pyramid", self.sem_pyramid, self.z_sem.shape[0])
        _validate_pyramid("det_pyramid", self.det_pyramid, self.z_det.shape[0])

        if self.sem_pyramid[-1].shape[-2:] != self.z_sem.shape[-2:]:
            raise ValueError(
                "The final semantic pyramid feature must match z_sem spatial size, "
                f"got {self.sem_pyramid[-1].shape[-2:]} vs {self.z_sem.shape[-2:]}."
            )
        if self.det_pyramid[-1].shape[-2:] != self.z_det.shape[-2:]:
            raise ValueError(
                "The final detail pyramid feature must match z_det spatial size, "
                f"got {self.det_pyramid[-1].shape[-2:]} vs {self.z_det.shape[-2:]}."
            )

        return self


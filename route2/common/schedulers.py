from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from route2_swinjscc_gan.common.checks import require_positive_int


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    require_positive_int("total_steps", total_steps)
    require_positive_int("warmup_steps", max(1, warmup_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / float(warmup_steps), min_lr_ratio)

        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


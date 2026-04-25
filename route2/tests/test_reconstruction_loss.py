from __future__ import annotations

import torch

from route2_swinjscc_gan.losses.reconstruction import weighted_geometric_product


def test_weighted_geometric_product_matches_torch_prod_on_cpu() -> None:
    stacked = torch.tensor(
        [
            [0.8, 0.9],
            [0.7, 0.6],
            [0.5, 0.4],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([[0.2], [0.3], [0.5]], dtype=torch.float32)

    expected = torch.prod(stacked.pow(weights), dim=0)
    actual = weighted_geometric_product(stacked, weights)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

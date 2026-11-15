"""Tests for hlcnet.rocket."""

from __future__ import annotations

import torch

from hlcnet.models import InputChannels
from hlcnet.rocket import RocketFeatures, rocket_transform


def test_rocket_shape_and_determinism() -> None:
    x = torch.randn(10, 480, 2, generator=torch.Generator().manual_seed(0))
    chans = InputChannels(use_derivative=True, deriv_std=0.1)
    r1 = RocketFeatures(3, 480, n_kernels=50, seed=1)
    r2 = RocketFeatures(3, 480, n_kernels=50, seed=1)
    f1 = rocket_transform(x, r1, chans, torch.device("cpu"), batch_size=4)
    f2 = rocket_transform(x, r2, chans, torch.device("cpu"), batch_size=4)
    assert f1.shape == (10, 100)
    assert torch.equal(f1, f2)
    assert torch.isfinite(f1).all()


def test_rocket_on_zero_input_returns_biases() -> None:
    # with a zero input the convolution output is the bias everywhere:
    # PPV is 1 for positive biases and 0 otherwise, the max equals the bias
    r = RocketFeatures(3, 480, n_kernels=40, seed=0)
    f = r(torch.zeros(2, 3, 480))
    col = 0
    for i in range(len(r.specs)):
        b = getattr(r, f"b{i}")
        k = b.numel()
        assert torch.equal(f[0, col : col + k], (b > 0).float())
        assert torch.allclose(f[0, col + k : col + 2 * k], b)
        col += 2 * k

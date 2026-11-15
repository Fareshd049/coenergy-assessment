"""Tests for hlcnet.losses."""

from __future__ import annotations

import pytest
import torch

from hlcnet.data import Normalizer, SimData
from hlcnet.losses import LossConfig, MultiTaskLoss, gaussian_nll, target_weights
from hlcnet.models import ModelOutput


def fake_normalizer(n: int = 500) -> tuple[Normalizer, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    ext = 110 + 17 * torch.rand(n, generator=g)
    mit = 120 + 26 * torch.rand(n, generator=g)
    tau_n = 8 + 4 * torch.rand(n, generator=g)
    tau_inf = tau_n + 1 + 20 * torch.rand(n, generator=g)
    tau_b = tau_inf + 1 + 20 * torch.rand(n, generator=g)
    y = torch.stack([ext + mit, ext, mit, tau_b, tau_n, tau_inf], dim=1)
    x = torch.randn(n, 10, 2, generator=g)
    data = SimData(x, y, torch.zeros(n, 1), ["m"], torch.zeros(n, dtype=torch.long))
    return Normalizer.fit(data), y


def test_family_weights_balance_families() -> None:
    w = target_weights({"HLC": 1.0, "Hb": 1.0, "CDT": 1.0})
    assert w.sum().item() == pytest.approx(1.0)
    assert w[0].item() == pytest.approx(1 / 3)  # HLC alone
    assert w[1].item() == pytest.approx(1 / 6)  # each Hb
    assert w[3].item() == pytest.approx(1 / 9)  # each tau


def test_perfect_prediction_gives_zero_mse() -> None:
    norm, y = fake_normalizer()
    loss_fn = MultiTaskLoss(LossConfig(kind="mse"), norm)
    loss, logs = loss_fn(ModelOutput(y, None, None), y)
    assert loss.item() == pytest.approx(0.0, abs=1e-10)
    assert logs["mse_HLC"] == pytest.approx(0.0, abs=1e-10)


def test_nll_with_unit_variance_is_half_mse() -> None:
    g = torch.Generator().manual_seed(0)
    r = torch.randn(1000, 6, generator=g)
    nll = gaussian_nll(r, torch.zeros_like(r), beta=0.0)
    assert torch.allclose(nll, 0.5 * r.pow(2))


def test_nll_is_minimized_at_the_true_variance() -> None:
    r = torch.full((1,), 2.0)  # residual of 2 -> best variance 4
    grid = torch.linspace(-3, 4, 7001)
    vals = torch.stack(
        [gaussian_nll(r, torch.full((1,), float(v))) for v in grid]
    ).squeeze()
    best = grid[vals.argmin()].item()
    assert best == pytest.approx(torch.log(torch.tensor(4.0)).item(), abs=1e-2)


def test_beta_nll_rescales_by_variance() -> None:
    r = torch.tensor([1.0])
    lv = torch.tensor([2.0])
    plain = gaussian_nll(r, lv, beta=0.0)
    scaled = gaussian_nll(r, lv, beta=0.5)
    assert scaled.item() == pytest.approx(plain.item() * torch.exp(lv).sqrt().item())


def test_nll_requires_log_var() -> None:
    norm, y = fake_normalizer()
    loss_fn = MultiTaskLoss(LossConfig(kind="nll"), norm)
    with pytest.raises(ValueError):
        loss_fn(ModelOutput(y, None, None), y)


def test_loss_is_differentiable() -> None:
    norm, y = fake_normalizer()
    pred = (y * 1.05).requires_grad_(True)
    log_var = torch.zeros_like(y, requires_grad=True)
    for kind in ("mse", "huber", "nll"):
        loss_fn = MultiTaskLoss(LossConfig(kind=kind), norm)  # type: ignore[arg-type]
        loss, _ = loss_fn(ModelOutput(pred, log_var, None), y)
        loss.backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()
        pred.grad = None

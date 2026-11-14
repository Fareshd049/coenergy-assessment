"""Tests for hlcnet.metrics."""

from __future__ import annotations

import pytest
import torch

from hlcnet.metrics import (
    constraint_violations,
    family_summary,
    interval_coverage,
    mean_predictor,
    metrics_by_group,
    regression_metrics,
)
from hlcnet.targets import TARGET_NAMES


def physical_targets(n: int = 2000, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ext = 110 + 17 * torch.randn(n, generator=g)
    mit = 120 + 26 * torch.randn(n, generator=g)
    tau_n = 11 + 2 * torch.rand(n, generator=g)
    tau_inf = tau_n + 20 * torch.rand(n, generator=g)
    tau_b = tau_inf + 20 * torch.rand(n, generator=g)
    return torch.stack([ext + mit, ext, mit, tau_b, tau_n, tau_inf], dim=1)


def test_perfect_prediction() -> None:
    y = physical_targets()
    m = regression_metrics(y, y)
    assert torch.allclose(
        torch.tensor(m["rmse"].values), torch.zeros(6, dtype=torch.float64)
    )
    assert (m["r2"] == 1.0).all()


def test_known_values() -> None:
    true = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    pred = true + torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])
    m = regression_metrics(pred, true, names=["a"]).loc["a"]
    assert m["rmse"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(1.0)
    assert m["bias"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1 - 4 / 5)


def test_mean_predictor_has_zero_r2_and_unit_nrmse() -> None:
    y = physical_targets()
    m = regression_metrics(mean_predictor(y, len(y)), y)
    assert m["r2"].abs().max() < 1e-9
    assert (m["nrmse"] - 1).abs().max() < 1e-9


def test_family_summary_has_three_families() -> None:
    y = physical_targets()
    fam = family_summary(regression_metrics(y + 1, y))
    assert list(fam.index) == ["HLC", "Hb", "CDT"]
    assert fam.loc["Hb", "n_indicateurs"] == 2


def test_metrics_by_group_skips_small_groups() -> None:
    y = physical_targets(n=100)
    groups = torch.tensor([0] * 90 + [1] * 10)
    out = metrics_by_group(y + 1, y, groups, metric="mae", min_size=20)
    assert list(out.index) == [0]
    assert out.loc[0, "n"] == 90
    assert set(TARGET_NAMES) <= set(out.columns)


def test_constraint_violations() -> None:
    y = physical_targets()
    assert constraint_violations(y).loc["part |HLC - (ext + mit)| > tol"] == 0.0
    assert constraint_violations(y).loc["part ordre tau violé"] == 0.0
    bad = y.clone()
    bad[:500, 0] += 5.0  # break HLC = ext + mit on a quarter of rows
    bad[:200, 4] = bad[:200, 3] + 1  # tau_n > tau_b on 10% of rows
    v = constraint_violations(bad)
    assert v.loc["part |HLC - (ext + mit)| > tol"] == pytest.approx(0.25)
    assert v.loc["part ordre tau violé"] == pytest.approx(0.10)


def test_interval_coverage_is_calibrated_for_gaussian_errors() -> None:
    g = torch.Generator().manual_seed(0)
    n = 20000
    mean = torch.zeros(n, 6)
    std = torch.full((n, 6), 2.0)
    true = mean + std * torch.randn(n, 6, generator=g)
    cov = interval_coverage(true, mean, std, levels=(0.5, 0.9, 0.95))
    assert (cov["couverture_95%"] - 0.95).abs().max() < 0.01
    assert (cov["couverture_50%"] - 0.50).abs().max() < 0.015
    # an over-confident model (std too small) under-covers
    over = interval_coverage(true, mean, std / 2, levels=(0.95,))
    assert (over["couverture_95%"] < 0.75).all()

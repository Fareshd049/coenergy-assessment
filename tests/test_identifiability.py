"""Tests for hlcnet.identifiability."""

from __future__ import annotations

import torch

from hlcnet.identifiability import (
    RidgeProbe,
    group_identical_excitations,
    information_timeline,
    pca,
    r2_score,
    ridge_information_curve,
)
from hlcnet.targets import TARGET_NAMES


def test_group_identical_excitations_sorted_by_size() -> None:
    a, b, c = torch.zeros(5), torch.ones(5), torch.arange(5.0)
    u = torch.stack([a, b, a, c, a, b])
    group, sizes = group_identical_excitations(u)
    assert sizes.tolist() == [3, 2, 1]
    assert group[[0, 2, 4]].tolist() == [0, 0, 0]
    assert group[[1, 5]].tolist() == [1, 1]
    assert group[3].item() == 2


def test_information_timeline_detects_signal_and_constant_step() -> None:
    g = torch.Generator().manual_seed(0)
    n, t = 500, 10
    y = 1 + torch.rand(n, len(TARGET_NAMES), generator=g)
    temp = torch.zeros(n, t)
    temp[:, 1:] = -y[:, :1] * torch.arange(1, t).float()  # depends only on HLC (col 0)
    df = information_timeline(temp, y)
    assert df.shape == (t, len(TARGET_NAMES))
    assert df.iloc[0].isna().all()  # constant first step
    assert (df["HLC"].iloc[1:] < -0.99).all()
    assert df["tau_b"].iloc[1:].abs().max() < 0.2


def test_ridge_recovers_linear_relation() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2000, 8, generator=g)
    w = torch.randn(8, 2, generator=g)
    y = x @ w + 0.01 * torch.randn(2000, 2, generator=g)
    probe = RidgeProbe.fit(x[:1500], y[:1500], alpha=1e-6)
    assert (r2_score(y[1500:], probe.predict(x[1500:])) > 0.99).all()


def test_ridge_information_curve_shape() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(600, 120, 2, generator=g)
    y = 1 + torch.rand(600, len(TARGET_NAMES), generator=g)
    df = ridge_information_curve(x[:500], y[:500], x[500:], y[500:], hours=(0.5, 1))
    assert list(df.index) == [0.5, 1]
    assert set(TARGET_NAMES) <= set(df.columns)


def test_pca_explained_variance() -> None:
    g = torch.Generator().manual_seed(0)
    base = torch.randn(1, 50, generator=g)
    curves = torch.randn(300, 1, generator=g) * base + 0.01 * torch.randn(
        300, 50, generator=g
    )
    res = pca(curves, k=5)
    assert res.explained[0] > 0.95
    assert res.scores.shape == (300, 5)

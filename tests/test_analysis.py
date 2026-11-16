"""Tests for hlcnet.analysis."""

from __future__ import annotations

import torch

from hlcnet.analysis import (
    excitation_map,
    lognormal_mean_correction,
    reliability_table,
    subgroup_metrics,
    unseen_excitation_mask,
)
from hlcnet.data import Normalizer, SimData


def targets(n: int = 600, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ext = 110 + 17 * torch.rand(n, generator=g)
    mit = 120 + 26 * torch.rand(n, generator=g)
    tau_n = 8 + 4 * torch.rand(n, generator=g)
    tau_inf = tau_n + 1 + 20 * torch.rand(n, generator=g)
    tau_b = tau_inf + 1 + 20 * torch.rand(n, generator=g)
    return torch.stack([ext + mit, ext, mit, tau_b, tau_n, tau_inf], dim=1)


def test_unseen_excitation_mask() -> None:
    a, b, c = torch.zeros(10), torch.ones(10) * 7000, torch.arange(10.0) % 2 * 7000
    u_train = torch.stack([a, b])
    u_eval = torch.stack([a, c, b, c])
    assert unseen_excitation_mask(u_train, u_eval).tolist() == [
        False,
        True,
        False,
        True,
    ]


def test_reliability_is_calibrated_for_gaussian_errors() -> None:
    g = torch.Generator().manual_seed(0)
    n = 20000
    z_pred = torch.zeros(n, 6)
    log_var = torch.full((n, 6), 0.5)
    z_true = z_pred + torch.exp(0.5 * log_var) * torch.randn(n, 6, generator=g)
    rel = reliability_table(z_true, z_pred, log_var, levels=(0.5, 0.9))
    assert (rel.loc[0.5] - 0.5).abs().max() < 0.015
    assert (rel.loc[0.9] - 0.9).abs().max() < 0.01


def test_lognormal_correction_increases_taus_only() -> None:
    y = targets()
    data = SimData(
        torch.randn(len(y), 4, 2),
        y,
        torch.zeros(len(y), 1),
        ["m"],
        torch.zeros(len(y), dtype=torch.long),
    )
    norm = Normalizer.fit(data, log_cdt=True)
    log_var = torch.zeros_like(y)
    corrected = lognormal_mean_correction(y, log_var, norm)
    assert torch.allclose(corrected[:, :3], y[:, :3])
    assert (corrected[:, 3:] > y[:, 3:]).all()
    # factor = exp(s^2 / 2) with s = y_std of log tau
    expected = y[:, 4] * torch.exp(0.5 * norm.y_std[4] ** 2)
    assert torch.allclose(corrected[:, 4], expected, rtol=1e-5)


def test_subgroup_metrics_splits_never_heated() -> None:
    y = targets(400)
    u = torch.zeros(400, 120)
    u[200:, 30:] = 7000.0
    out = subgroup_metrics(y + 1, y, u, metric="rmse")
    assert out.loc["jamais chauffées", "n"] == 200
    assert out.loc["chauffées", "n"] == 200
    assert abs(out.loc["toutes", "HLC"] - 1.0) < 1e-6


def test_excitation_map_shape() -> None:
    y = targets(3000)
    g = torch.Generator().manual_seed(0)
    names = ["prbs_probability_on", "prbs_min_duration_hours"]
    meta = torch.stack(
        [0.2 + 0.7 * torch.rand(3000, generator=g), 2 * torch.rand(3000, generator=g)],
        dim=1,
    )
    table = excitation_map(y + 2, y, meta, names, target="HLC", n_bins=3, min_count=10)
    assert table.shape == (3, 3)
    assert ((table - 2.0).abs() < 1e-6).all().all()

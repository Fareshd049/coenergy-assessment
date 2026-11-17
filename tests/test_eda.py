"""Tests for hlcnet.eda feature helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hlcnet.data import SimData
from hlcnet.eda import excitation_features


def test_excitation_features() -> None:
    u = torch.zeros(3, 120)
    u[1, 60:] = 7000.0  # one hour on, starting at 1 h
    u[2, ::2] = 7000.0  # alternating
    f = excitation_features(u)
    assert f["never_on"].tolist() == [True, False, False]
    assert f["first_on_h"].tolist() == [2.0, 1.0, 0.0]
    assert abs(f["energy_kwh"][1].item() - 7.0) < 1e-5
    assert f["duty"][2].item() == 0.5
    assert f["n_switch"].tolist() == [0, 1, 119]


def test_conditional_correlation_recovers_sign() -> None:
    from hlcnet.eda import conditional_correlation

    g = torch.Generator().manual_seed(0)
    cond = torch.rand(2000, generator=g)
    hlc = torch.rand(2000, generator=g)
    rise = 10 * cond - 3 * hlc  # rise grows with energy, falls with HLC
    table = conditional_correlation(cond, rise, hlc, n_bins=5)
    assert len(table) == 5
    assert (table["corr"] < -0.5).all()


def _fake_sim(n: int, seed: int) -> SimData:
    g = torch.Generator().manual_seed(seed)
    u = (torch.rand(n, 48, generator=g) > 0.5).float() * 7000
    temp = 24.62 + torch.cumsum(u, dim=1) / 1e5
    ext = 80 + 60 * torch.rand(n, generator=g)
    mit = 60 + 100 * torch.rand(n, generator=g)
    tau_n = 6 + 10 * torch.rand(n, generator=g)
    tau_inf = tau_n + 5
    y = torch.stack([ext + mit, ext, mit, tau_inf + 5, tau_n, tau_inf], dim=1)
    meta = torch.rand(n, 2, generator=g)
    names = ["prbs_probability_on", "prbs_min_duration_hours"]
    return SimData(
        torch.stack([temp, u], dim=-1), y, meta, names, torch.full((n,), seed)
    )


def test_excitation_hash_matches_exact_grouping() -> None:
    from hlcnet.eda import excitation_hash

    g = torch.Generator().manual_seed(0)
    u = (torch.rand(300, 480, generator=g) > 0.5).float() * 7000
    u[10] = u[3]
    u[20] = 0.0
    u[21] = 0.0
    h = excitation_hash(u)
    assert h[10] == h[3] and h[20] == h[21]
    assert torch.unique(h).numel() == torch.unique(u, dim=0).shape[0]


def test_scan_dataset_one_row_per_simulation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hlcnet.eda as eda

    monkeypatch.setattr(eda, "list_batches", lambda folder: [0, 1])
    monkeypatch.setattr(
        eda, "load_batches", lambda folder, batch_ids: _fake_sim(30, batch_ids[0])
    )
    cache = tmp_path / "scan.pkl"
    scan = eda.scan_dataset(tmp_path, cache=cache)
    assert len(scan) == 60 and cache.exists()
    assert set(scan["batch"]) == {0, 1}
    checks = eda.full_consistency_checks(scan)
    assert checks["simulations avec NaN"] == 0
    assert checks["part tau_b >= tau_inf >= tau_n"] == 1.0
    assert eda.scan_targets(scan).shape == (60, 6)
    # the cache is read back without touching the data
    monkeypatch.setattr(eda, "load_batches", None)
    assert len(eda.scan_dataset(tmp_path, cache=cache)) == 60

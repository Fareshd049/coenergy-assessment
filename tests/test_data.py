"""Tests for hlcnet.data and hlcnet.targets (no real data needed)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hlcnet.data import Normalizer, SimData, SplitSpec, add_sensor_noise, list_batches
from hlcnet.targets import CDT_IDX, TARGET_NAMES, assemble_targets, hlc_consistency_gap


def fake_data(n: int = 64, t: int = 48, seed: int = 0) -> SimData:
    g = torch.Generator().manual_seed(seed)
    x = torch.stack(
        [
            24.6 + torch.rand(n, t, generator=g) * 10,
            (torch.rand(n, t, generator=g) > 0.5).float() * 7000,
        ],
        dim=-1,
    )
    ext = 80 + 60 * torch.rand(n, generator=g)
    mit = 60 + 100 * torch.rand(n, generator=g)
    tau_n = 6 + 10 * torch.rand(n, generator=g)
    tau_inf = tau_n + 15 * torch.rand(n, generator=g)
    tau_b = tau_inf + 20 * torch.rand(n, generator=g)
    y = torch.stack([ext + mit, ext, mit, tau_b, tau_n, tau_inf], dim=1)
    return SimData(
        x,
        y,
        torch.zeros(n, 12),
        [f"f{i}" for i in range(12)],
        torch.zeros(n, dtype=torch.long),
    )


def test_assemble_targets_reorders_by_label() -> None:
    hlc = torch.tensor([10.0])
    hb = torch.tensor([[4.0, 6.0]])  # stored as [mit, ext]
    cdt = torch.tensor([[50.0, 10.0, 30.0]])
    y = assemble_targets(hlc, hb, ["HLC_mit", "HLC_ext"], cdt)
    assert y.shape == (1, len(TARGET_NAMES))
    assert y[0].tolist() == [10.0, 6.0, 4.0, 50.0, 10.0, 30.0]
    assert hlc_consistency_gap(y).item() == 0.0


def test_assemble_targets_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError):
        assemble_targets(
            torch.zeros(1), torch.zeros(1, 2), ["a", "b"], torch.ones(1, 3)
        )


def test_list_batches_natural_order(tmp_path: Path) -> None:
    for i in (1, 2, 10, 20):
        (tmp_path / f"batch_{i}.pt").touch()
    assert [p.name for p in list_batches(tmp_path)] == [
        "batch_1.pt",
        "batch_2.pt",
        "batch_10.pt",
        "batch_20.pt",
    ]


def test_split_spec_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        SplitSpec(train=(0, 1), val=(1,), test=(2,))


@pytest.mark.parametrize("log_cdt", [True, False])
def test_normalizer_roundtrip(log_cdt: bool) -> None:
    d = fake_data()
    norm = Normalizer.fit(d, log_cdt=log_cdt)
    z = norm.transform_y(d.y)
    assert torch.allclose(z.mean(0), torch.zeros(6), atol=1e-4)
    assert torch.allclose(norm.inverse_y(z), d.y, rtol=1e-4, atol=1e-3)
    xz = norm.transform_x(d.x)
    assert torch.allclose(xz.mean(dim=(0, 1)), torch.zeros(2), atol=1e-4)


def test_normalizer_state_dict_roundtrip() -> None:
    d = fake_data()
    norm = Normalizer.fit(d)
    norm2 = Normalizer.from_state_dict(norm.state_dict())
    assert torch.equal(norm.transform_y(d.y), norm2.transform_y(d.y))


def test_subset_and_concat() -> None:
    d = fake_data(n=10)
    parts = [d.subset(torch.arange(0, 4)), d.subset(torch.arange(4, 10))]
    back = SimData.concat(parts)
    assert torch.equal(back.x, d.x) and torch.equal(back.y, d.y)


def test_sensor_noise_only_on_temperature() -> None:
    d = fake_data()
    noisy = add_sensor_noise(
        d.x, temp_std=0.2, u_std=0.0, generator=torch.Generator().manual_seed(0)
    )
    assert torch.equal(noisy[..., 1], d.x[..., 1])
    assert (noisy[..., 0] - d.x[..., 0]).std() == pytest.approx(0.2, rel=0.1)


def test_fake_taus_are_ordered() -> None:
    y = fake_data().y
    tau_b, tau_n, tau_inf = (y[:, i] for i in CDT_IDX)
    assert bool(((tau_b >= tau_inf) & (tau_inf >= tau_n)).all())

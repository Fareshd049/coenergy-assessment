"""Tests for hlcnet.specialist and the ensemble helpers of hlcnet.analysis."""

from __future__ import annotations

import torch

from hlcnet.analysis import ensemble_mean, mixture_std_z
from hlcnet.metrics import constraint_violations
from hlcnet.specialist import FeatureScaler, constant_excitation_masks, fit_mlp


def constrained(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ext = 100 + 20 * torch.rand(n, generator=g)
    mit = 110 + 30 * torch.rand(n, generator=g)
    tau_n = 8 + 4 * torch.rand(n, generator=g)
    tau_inf = tau_n + 20 * torch.rand(n, generator=g)
    tau_b = tau_inf + 20 * torch.rand(n, generator=g)
    return torch.stack([ext + mit, ext, mit, tau_b, tau_n, tau_inf], dim=1)


def test_constant_excitation_masks() -> None:
    u = torch.zeros(3, 10)
    u[1] = 7000.0
    u[2, 5:] = 7000.0
    m = constant_excitation_masks(u)
    assert m["jamais chauffé"].tolist() == [True, False, False]
    assert m["chauffe continue"].tolist() == [False, True, False]


def test_ensemble_preserves_physical_constraints() -> None:
    ens = ensemble_mean([constrained(500, s) for s in range(4)])
    v = constraint_violations(ens, tol=1e-3)
    assert v.loc["part |HLC - (ext + mit)| > tol"] == 0.0
    assert v.loc["part ordre tau violé"] == 0.0


def test_mixture_std() -> None:
    z = torch.zeros(10, 6)
    lv = torch.zeros(10, 6)  # unit variance
    mean, std = mixture_std_z([z, z], [lv, lv])
    assert torch.allclose(std, torch.ones(10, 6, dtype=torch.float64))
    # two members at -1 and +1 with unit variance: total variance 1 + 1 = 2
    mean, std = mixture_std_z([z - 1, z + 1], [lv, lv])
    assert torch.allclose(mean, torch.zeros(10, 6, dtype=torch.float64))
    assert torch.allclose(std, torch.full((10, 6), 2.0, dtype=torch.float64).sqrt())


def test_fit_mlp_learns_a_simple_function() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2000, 20, generator=g)
    z = torch.stack([x[:, 0] + 0.5 * x[:, 1], x[:, 2] ** 2 - 1], dim=1)
    scaler = FeatureScaler.fit(x[:1600])
    model = fit_mlp(
        scaler(x[:1600]), z[:1600], scaler(x[1600:]), z[1600:], epochs=40, seed=0
    )
    with torch.no_grad():
        mse = (model(scaler(x[1600:])) - z[1600:]).pow(2).mean().item()
    assert mse < 0.2 * z[1600:].var(dim=0).mean().item()

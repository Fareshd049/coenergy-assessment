"""Tests for hlcnet.models."""

from __future__ import annotations

import pytest
import torch

from hlcnet.metrics import constraint_violations
from hlcnet.models import (
    ModelConfig,
    ModelStats,
    build_model,
    count_parameters,
    params_to_targets,
    targets_to_params,
)


def physical_targets(n: int = 256, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ext = 110 + 17 * torch.rand(n, generator=g)
    mit = 120 + 26 * torch.rand(n, generator=g)
    tau_n = 8 + 4 * torch.rand(n, generator=g)
    tau_inf = tau_n + 1 + 20 * torch.rand(n, generator=g)
    tau_b = tau_inf + 1 + 20 * torch.rand(n, generator=g)
    return torch.stack([ext + mit, ext, mit, tau_b, tau_n, tau_inf], dim=1)


def fake_inputs(n: int = 8, t: int = 480, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, t, 2, generator=g)


@pytest.mark.parametrize("head", ["share", "sum", "free"])
def test_params_roundtrip(head: str) -> None:
    y = physical_targets().double()
    back = params_to_targets(targets_to_params(y, head), head)  # type: ignore[arg-type]
    assert torch.allclose(back, y, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("arch", ["cnn_gru", "tcn_gru", "tcn", "film_gru", "mlp"])
@pytest.mark.parametrize("head", ["share", "sum"])
def test_constraints_hold_for_any_input(arch: str, head: str) -> None:
    y = physical_targets()
    x = fake_inputs()
    cfg = ModelConfig(arch=arch, head=head)  # type: ignore[arg-type]
    model = build_model(cfg, ModelStats.from_data(x, y, cfg.head))
    # random weights, so predictions are arbitrary, but physics must hold
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.5)
    model.eval()
    out = model(x * 10)
    v = constraint_violations(
        out.pred.detach(), tol=1e-3 * out.pred[:, 0].abs().max().item()
    )
    assert v.loc["part |HLC - (ext + mit)| > tol"] == 0.0
    assert v.loc["part ordre tau violé"] == 0.0
    assert v.loc["part valeurs négatives"] == 0.0


@pytest.mark.parametrize("arch", ["cnn_gru", "tcn_gru", "tcn", "film_gru", "mlp"])
def test_forward_shapes_and_initial_prediction(arch: str) -> None:
    y = physical_targets()
    x = fake_inputs()
    cfg = ModelConfig(arch=arch)  # type: ignore[arg-type]
    model = build_model(cfg, ModelStats.from_data(x, y, cfg.head)).eval()
    out = model(x)
    assert out.pred.shape == (8, 6)
    assert out.log_var is not None and out.log_var.shape == (8, 6)
    # at initialization the model predicts close to the typical building
    typical = y.double().log().mean(0).exp()[0].item()  # geometric mean of HLC
    assert out.pred[:, 0].sub(typical).abs().max().item() < 0.05 * typical
    if arch != "mlp":
        assert out.attention is not None
        assert torch.allclose(out.attention.sum(1), torch.ones(8), atol=1e-5)
    assert count_parameters(model) > 0


def test_gradients_flow() -> None:
    y = physical_targets()
    x = fake_inputs()
    cfg = ModelConfig()
    model = build_model(cfg, ModelStats.from_data(x, y, cfg.head))
    out = model(x)
    assert out.log_var is not None
    (out.pred.sum() + out.log_var.sum()).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_without_derivative_or_uncertainty() -> None:
    y = physical_targets()
    x = fake_inputs()
    cfg = ModelConfig(use_derivative=False, predict_uncertainty=False)
    model = build_model(cfg, ModelStats.from_data(x, y, cfg.head)).eval()
    out = model(x)
    assert out.log_var is None
    assert out.pred.shape == (8, 6)


def test_tcn_receptive_field_covers_the_series() -> None:
    cfg = ModelConfig(arch="tcn", tcn_levels=6, tcn_kernel=5)
    rf = 1 + 2 * (cfg.tcn_kernel - 1) * (2**cfg.tcn_levels - 1)
    assert rf >= cfg.seq_len


def test_film_starts_as_identity() -> None:
    from hlcnet.models import TCNRegressor

    y = physical_targets()
    x = fake_inputs()
    cfg = ModelConfig(arch="film_gru")
    model = build_model(cfg, ModelStats.from_data(x, y, cfg.head))
    assert isinstance(model, TCNRegressor)
    for proj in model.film_proj:
        assert proj.weight.abs().sum().item() == 0.0

"""End-to-end tests of the training loop on small synthetic data."""

from __future__ import annotations

from pathlib import Path

import torch

from hlcnet.config import ExperimentConfig, TrainConfig
from hlcnet.data import Normalizer, SimData, make_loaders
from hlcnet.losses import LossConfig
from hlcnet.models import ModelConfig, ModelStats, build_model
from hlcnet.train import fit, load_checkpoint, predict, save_checkpoint


def synthetic(n: int, seed: int) -> SimData:
    """Temperature rise driven by U and inversely proportional to HLC."""
    g = torch.Generator().manual_seed(seed)
    t = 480
    ext = 110 + 17 * torch.rand(n, generator=g)
    mit = 120 + 26 * torch.rand(n, generator=g)
    hlc = ext + mit
    tau_n = 8 + 4 * torch.rand(n, generator=g)
    tau_inf = tau_n + 1 + 20 * torch.rand(n, generator=g)
    tau_b = tau_inf + 1 + 20 * torch.rand(n, generator=g)
    y = torch.stack([hlc, ext, mit, tau_b, tau_n, tau_inf], dim=1)
    on = torch.rand(n, 1, generator=g) * t
    u = (torch.arange(t).float()[None, :] > on).float() * 7000
    tint = 24.6 + torch.cumsum(u, dim=1) / (60 * hlc[:, None])
    x = torch.stack([tint, u], dim=-1)
    return SimData(x, y, torch.zeros(n, 1), ["m"], torch.zeros(n, dtype=torch.long))


def tiny_setup(tmp_path: Path, kind: str = "mse") -> tuple:
    train, val = synthetic(256, 0), synthetic(128, 1)
    norm = Normalizer.fit(train)
    loaders = make_loaders({"train": train, "val": val}, norm, batch_size=64)
    mcfg = ModelConfig(conv_channels=8, gru_hidden=8, gru_layers=1, bidirectional=False)
    stats = ModelStats.from_data(norm.transform_x(train.x), train.y, mcfg.head)
    model = build_model(mcfg, stats)
    tcfg = TrainConfig(
        epochs=4, batch_size=64, lr=3e-3, mse_warmup_epochs=1, patience=10, amp=False
    )
    lcfg = LossConfig(kind=kind)  # type: ignore[arg-type]
    return model, loaders, norm, lcfg, tcfg, mcfg, stats


def test_fit_runs_and_keeps_best_epoch(tmp_path: Path) -> None:
    model, loaders, norm, lcfg, tcfg, _, _ = tiny_setup(tmp_path, kind="nll")
    hist = fit(
        model,
        loaders,
        norm,
        lcfg,
        tcfg,
        torch.device("cpu"),
        out_dir=tmp_path,
        verbose=False,
    )
    frame = hist.to_frame()
    assert len(frame) == 4
    assert hist.best_epoch >= 2  # warm-up epoch is not eligible with the NLL loss
    assert (tmp_path / "best_weights.pt").exists()
    assert torch.isfinite(torch.tensor(frame["val_score"].values)).all()


def test_training_improves_on_initial_prediction() -> None:
    # MLP and HLC-only loss: a simple, fast learning problem (rise ~ 1 / HLC)
    torch.manual_seed(0)
    train, val = synthetic(512, 0), synthetic(256, 1)
    norm = Normalizer.fit(train)
    loaders = make_loaders({"train": train, "val": val}, norm, batch_size=64)
    mcfg = ModelConfig(arch="mlp", mlp_hidden=64, predict_uncertainty=False)
    model = build_model(
        mcfg, ModelStats.from_data(norm.transform_x(train.x), train.y, mcfg.head)
    )
    lcfg = LossConfig(kind="mse", family_weights={"HLC": 1.0, "Hb": 0.0, "CDT": 0.0})
    tcfg = TrainConfig(
        epochs=15, batch_size=64, lr=3e-3, mse_warmup_epochs=0, patience=20, amp=False
    )
    before = predict(model, loaders["val"], norm, torch.device("cpu"))
    err_before = (before.pred[:, 0] - before.true[:, 0]).pow(2).mean()
    fit(model, loaders, norm, lcfg, tcfg, torch.device("cpu"), verbose=False)
    after = predict(model, loaders["val"], norm, torch.device("cpu"))
    err_after = (after.pred[:, 0] - after.true[:, 0]).pow(2).mean()
    assert err_after < err_before


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model, loaders, norm, _, _, mcfg, stats = tiny_setup(tmp_path)
    cfg = ExperimentConfig(name="t", model=mcfg)
    save_checkpoint(tmp_path / "model.pt", model, cfg, norm, stats)
    model2, cfg2, norm2 = load_checkpoint(tmp_path / "model.pt")
    x, _ = next(iter(loaders["val"]))
    model.eval()
    assert torch.allclose(model(x).pred, model2(x).pred)
    assert cfg2.model == mcfg
    assert torch.equal(norm2.y_mean, norm.y_mean)


def test_config_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = ExperimentConfig(name="abc")
    cfg.train.epochs = 7
    cfg.save(tmp_path / "c.yaml")
    back = ExperimentConfig.from_yaml(tmp_path / "c.yaml")
    assert back.name == "abc" and back.train.epochs == 7
    assert back.loss.family_weights == cfg.loss.family_weights

"""Training loop, prediction and checkpoints.

Key choices:

* AdamW with a linear warm-up then cosine decay of the learning rate.
* ``mse_warmup_epochs``: the first epochs use the squared error only, so the
  model learns to predict before learning its uncertainty (otherwise the
  likelihood can inflate the variances while the means are still poor).
* Early stopping on a criterion independent of the loss: the weighted mean
  squared error in the normalized space on the validation set, so models
  trained with different losses are selected the same way.
* Optional Gaussian noise on ``Tint`` during training (sensor noise).
* Mixed precision on GPU; the physics head runs in float32.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from hlcnet.config import ExperimentConfig, TrainConfig
from hlcnet.data import Normalizer
from hlcnet.losses import LossConfig, MultiTaskLoss
from hlcnet.models import ModelConfig, ModelStats, build_model


@dataclass
class History:
    """Per-epoch training log."""

    rows: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_score: float = math.inf

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


@dataclass
class Predictions:
    """Model outputs on a dataset, in physical units."""

    pred: Tensor
    true: Tensor
    log_var: Tensor | None
    attention: Tensor | None


def make_scheduler(
    optimizer: torch.optim.Optimizer, steps_per_epoch: int, cfg: TrainConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up for ``warmup_epochs``, then cosine decay to 1 % of the peak."""
    total = max(1, cfg.epochs * steps_per_epoch)
    warm = max(1, int(cfg.warmup_epochs * steps_per_epoch))

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total - warm))
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, ...]],
    normalizer: Normalizer,
    device: torch.device,
    keep_attention: bool = False,
) -> Predictions:
    """Run the model on a loader of normalized ``(x, z)`` pairs.

    Returns predictions and targets in physical units (W/K, h).
    """
    model.eval()
    preds, trues, lvs, atts = [], [], [], []
    for x, z in loader:
        out = model(x.to(device, non_blocking=True))
        preds.append(out.pred.float().cpu())
        trues.append(normalizer.inverse_y(z.float()))
        if out.log_var is not None:
            lvs.append(out.log_var.float().cpu())
        if keep_attention and out.attention is not None:
            atts.append(out.attention.float().cpu())
    return Predictions(
        pred=torch.cat(preds),
        true=torch.cat(trues),
        log_var=torch.cat(lvs) if lvs else None,
        attention=torch.cat(atts) if atts else None,
    )


def selection_score(
    pred: Tensor, true: Tensor, normalizer: Normalizer, weights: Tensor
) -> float:
    """Weighted mean squared error in the normalized space (early-stopping criterion)."""
    r = normalizer.transform_y(pred.double()) - normalizer.transform_y(true.double())
    return float((r.pow(2).mean(0) * weights.double()).sum())


def fit(
    model: nn.Module,
    loaders: dict[str, DataLoader[tuple[Tensor, ...]]],
    normalizer: Normalizer,
    loss_cfg: LossConfig,
    cfg: TrainConfig,
    device: torch.device,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> History:
    """Train ``model`` and keep the weights of the best validation epoch.

    Args:
        loaders: ``"train"`` and ``"val"`` loaders of normalized ``(x, z)``.
        out_dir: if given, the best weights are saved to ``out_dir/best.pt``.

    Returns:
        The training history; the model is left with the best weights loaded.
    """
    model.to(device)
    main_loss = MultiTaskLoss(loss_cfg, normalizer).to(device)
    mse_loss = MultiTaskLoss(
        LossConfig(kind="mse", family_weights=loss_cfg.family_weights), normalizer
    ).to(device)
    weights: Tensor = main_loss.weights.cpu()  # type: ignore[assignment]

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = make_scheduler(opt, len(loaders["train"]), cfg)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    noise_scale = cfg.noise_temp_std / float(
        normalizer.x_std[0]
    )  # °C -> normalized units

    history = History()
    best_state: dict[str, Tensor] | None = None
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        loss_fn = mse_loss if epoch < cfg.mse_warmup_epochs else main_loss
        t0, total, n = time.perf_counter(), 0.0, 0
        for x, z in loaders["train"]:
            x, z = x.to(device, non_blocking=True), z.to(device, non_blocking=True)
            if noise_scale > 0:
                x = x.clone()
                x[..., 0] += noise_scale * torch.randn_like(x[..., 0])
            y = normalizer.inverse_y(z)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
            loss, _ = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * x.shape[0]
            n += x.shape[0]

        val = predict(model, loaders["val"], normalizer, device)
        score = selection_score(val.pred, val.true, normalizer, weights)
        row = {
            "epoch": epoch + 1,
            "loss_kind": 0.0 if loss_fn is mse_loss else 1.0,
            "train_loss": total / max(n, 1),
            "val_score": score,
            "lr": opt.param_groups[0]["lr"],
            "seconds": time.perf_counter() - t0,
        }
        history.rows.append(row)

        # warm-up epochs are not eligible: the variance head is not trained yet
        eligible = epoch >= cfg.mse_warmup_epochs or loss_cfg.kind != "nll"
        improved = eligible and score < history.best_score - 1e-6
        if improved:
            history.best_score, history.best_epoch = score, epoch + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
            if out_dir is not None:
                torch.save(best_state, out_dir / "best_weights.pt")
        elif eligible:
            bad_epochs += 1

        if verbose:
            flag = " *" if improved else ""
            print(
                f"epoch {epoch + 1:3d} | {'mse' if loss_fn is mse_loss else loss_cfg.kind:5s} | "
                f"train {row['train_loss']:.4f} | val score {score:.4f} | "
                f"lr {row['lr']:.2e} | {row['seconds']:.0f} s{flag}"
            )
        if bad_epochs >= cfg.patience:
            if verbose:
                print(f"early stopping: no improvement for {cfg.patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: ExperimentConfig,
    normalizer: Normalizer,
    stats: ModelStats,
) -> None:
    """Save everything needed to rebuild the model and use it on new data."""
    torch.save(
        {
            "config": cfg.to_dict(),
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "normalizer": normalizer.state_dict(),
            "stats": {
                "loc": stats.loc,
                "scale": stats.scale,
                "deriv_std": stats.deriv_std,
            },
        },
        path,
    )


def load_checkpoint(
    path: Path, device: torch.device | None = None
) -> tuple[nn.Module, ExperimentConfig, Normalizer]:
    """Rebuild a trained model saved with :func:`save_checkpoint`."""
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    stats = ModelStats(**ckpt["stats"])
    model = build_model(ModelConfig(**ckpt["config"]["model"]), stats)
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model.to(device)
    return model.eval(), cfg, Normalizer.from_state_dict(ckpt["normalizer"])


def save_json(obj: Any, path: Path) -> None:
    """Write a JSON file (pandas objects are converted)."""

    def default(o: Any) -> Any:
        if isinstance(o, (pd.DataFrame, pd.Series)):
            return json.loads(o.to_json(force_ascii=False))
        if isinstance(o, Tensor):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=default), encoding="utf-8"
    )

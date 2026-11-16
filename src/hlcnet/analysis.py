"""Critical error analysis of a trained model (report sections 6 and 7).

All functions work in physical units (W/K, h) unless stated otherwise:

* :func:`predict_physical` runs a model on raw inputs, optionally adding
  sensor noise to ``Tint`` (then rounding to the sensor resolution);
* :func:`noise_robustness` gives R² per indicator for several noise levels;
* :func:`subgroup_metrics` splits errors by heating status and energy decile;
* :func:`excitation_map` maps the HLC error over the two PRBS parameters;
* :func:`unseen_excitation_mask` flags validation simulations whose
  excitation never appears in the training set;
* :func:`reliability_table` and :func:`plot_reliability` compare nominal and
  observed coverage of the predicted intervals (calibration) and their width
  (sharpness);
* :func:`lognormal_mean_correction` removes the retransformation bias of the
  time constants: a model of ``log tau`` predicts its median ``exp(mu)``,
  while the mean is ``exp(mu + sigma^2 / 2)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import Tensor, nn

from hlcnet.data import TEMP_DECIMALS, Normalizer
from hlcnet.eda import excitation_features
from hlcnet.metrics import regression_metrics
from hlcnet.targets import CDT_IDX, HLC_IDX, TARGET_NAMES


@dataclass
class PhysicalPredictions:
    """Predictions in physical units and log-variances in the loss space."""

    pred: Tensor
    log_var: Tensor | None


# --------------------------------------------------------------------------
# Prediction with optional sensor noise
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_physical(
    model: nn.Module,
    normalizer: Normalizer,
    x: Tensor,
    device: torch.device,
    noise_std: float = 0.0,
    batch_size: int = 1024,
    seed: int = 0,
) -> PhysicalPredictions:
    """Predict from physical inputs ``(N, T, 2)``.

    Args:
        noise_std: standard deviation (°C) of i.i.d. Gaussian noise added to
            ``Tint`` at every minute, after which ``Tint`` is rounded to the
            sensor resolution (0.01 °C), as in the data.
    """
    model.eval().to(device)
    g = torch.Generator().manual_seed(seed)
    preds, lvs = [], []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].clone()
        if noise_std > 0:
            xb[..., 0] = (
                xb[..., 0] + noise_std * torch.randn(xb[..., 0].shape, generator=g)
            ).round(decimals=TEMP_DECIMALS)
        out = model(normalizer.transform_x(xb).to(device))
        preds.append(out.pred.float().cpu())
        if out.log_var is not None:
            lvs.append(out.log_var.float().cpu())
    return PhysicalPredictions(torch.cat(preds), torch.cat(lvs) if lvs else None)


def noise_robustness(
    model: nn.Module,
    normalizer: Normalizer,
    x: Tensor,
    y: Tensor,
    device: torch.device,
    levels: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.25),
    metric: str = "r2",
) -> pd.DataFrame:
    """One row per noise level (°C), one column per indicator."""
    rows = {}
    for lv in levels:
        p = predict_physical(model, normalizer, x, device, noise_std=lv)
        rows[lv] = regression_metrics(p.pred, y)[metric]
    out = pd.DataFrame(rows).T
    out.index.name = "bruit (°C)"
    return out


def noise_coverage(
    model: nn.Module,
    normalizer: Normalizer,
    x: Tensor,
    y: Tensor,
    device: torch.device,
    levels: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.25),
    nominal: float = 0.95,
) -> pd.DataFrame:
    """Observed coverage of the nominal intervals when ``Tint`` is noisy.

    Shows whether the model *knows* that noisy inputs make it less accurate:
    a coverage far below ``nominal`` means over-confidence.
    """
    z = float(
        torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.5 + nominal / 2))
    )
    zt = normalizer.transform_y(y.double())
    rows = {}
    for lv in levels:
        p = predict_physical(model, normalizer, x, device, noise_std=lv)
        if p.log_var is None:
            return pd.DataFrame()
        err = (zt - normalizer.transform_y(p.pred.double())).abs()
        inside = err <= z * torch.exp(0.5 * p.log_var.double())
        rows[lv] = inside.double().mean(0).numpy()
    out = pd.DataFrame(rows, index=list(TARGET_NAMES)).T
    out.index.name = "bruit (°C)"
    return out


def plot_noise_robustness(
    curves: dict[str, pd.DataFrame], metric: str = "R²"
) -> Figure:
    """One panel per indicator, one line per model."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, name in zip(axes.ravel(), TARGET_NAMES, strict=True):
        for label, df in curves.items():
            ax.plot(df.index, df[name], marker="o", ms=4, label=label)
        ax.set_xscale("symlog", linthresh=0.02)
        ax.set_title(name, fontsize=10)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("écart-type du bruit sur Tint (°C)")
    for ax in axes[:, 0]:
        ax.set_ylabel(metric)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Sub-groups
# --------------------------------------------------------------------------
def subgroup_metrics(
    pred: Tensor, true: Tensor, u: Tensor, metric: str = "r2"
) -> pd.DataFrame:
    """Metric per indicator for never-heated / heated simulations and energy deciles.

    Args:
        u: ``(N, T)`` heating power, used to compute the injected energy.
    """
    feats = excitation_features(u)
    never = feats["never_on"]
    rows = {}
    for label, m in (
        ("toutes", torch.ones_like(never)),
        ("jamais chauffées", never),
        ("chauffées", ~never),
    ):
        if m.sum() >= 20:
            rows[label] = {
                "n": int(m.sum()),
                **regression_metrics(pred[m], true[m])[metric].to_dict(),
            }

    heated = ~never
    energy = feats["energy_kwh"]
    edges = torch.quantile(energy[heated], torch.linspace(0, 1, 11))
    for k in range(10):
        lo, hi = edges[k], edges[k + 1]
        m = heated & (energy >= lo) & ((energy < hi) if k < 9 else (energy <= hi))
        if m.sum() >= 20:
            rows[f"énergie D{k + 1} [{lo:.0f}-{hi:.0f} kWh]"] = {
                "n": int(m.sum()),
                **regression_metrics(pred[m], true[m])[metric].to_dict(),
            }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------
# Excitation design
# --------------------------------------------------------------------------
def excitation_map(
    pred: Tensor,
    true: Tensor,
    meta: Tensor,
    meta_names: Sequence[str],
    target: str = "HLC",
    n_bins: int = 6,
    min_count: int = 30,
) -> pd.DataFrame:
    """RMSE of one indicator over bins of the two PRBS parameters.

    Returns:
        Pivot table indexed by ``prbs_probability_on`` bins, columns
        ``prbs_min_duration_hours`` bins; cells with fewer than ``min_count``
        simulations are NaN.
    """
    names = list(meta_names)
    p_on = meta[:, names.index("prbs_probability_on")].numpy()
    dur = meta[:, names.index("prbs_min_duration_hours")].numpy()
    j = list(TARGET_NAMES).index(target)
    err2 = (pred[:, j] - true[:, j]).double().pow(2).numpy()
    df = pd.DataFrame(
        {
            "p_on": pd.cut(
                p_on,
                np.linspace(p_on.min(), p_on.max() + 1e-9, n_bins + 1),
                include_lowest=True,
            ),
            "durée": pd.cut(
                dur,
                np.linspace(dur.min(), dur.max() + 1e-9, n_bins + 1),
                include_lowest=True,
            ),
            "err2": err2,
        }
    )
    grouped = df.groupby(["p_on", "durée"], observed=False)["err2"]
    rmse = grouped.mean().pow(0.5).where(grouped.count() >= min_count)
    return rmse.unstack("durée")


def plot_excitation_map(
    table: pd.DataFrame, target: str = "HLC", unit: str = "W/K"
) -> Figure:
    """Heatmap of :func:`excitation_map`, best cell highlighted."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    vals = table.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap="viridis_r", aspect="auto", origin="lower")
    ax.set_xticks(
        range(vals.shape[1]), [f"{iv.mid:.2f}" for iv in table.columns], rotation=45
    )
    ax.set_yticks(range(vals.shape[0]), [f"{iv.mid:.2f}" for iv in table.index])
    ax.set_xlabel("durée minimale des créneaux PRBS (h)")
    ax.set_ylabel("probabilité d'allumage PRBS")
    for i in range(vals.shape[0]):
        for k in range(vals.shape[1]):
            if np.isfinite(vals[i, k]):
                ax.text(
                    k,
                    i,
                    f"{vals[i, k]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="w",
                )
    if np.isfinite(vals).any():
        i, k = np.unravel_index(np.nanargmin(vals), vals.shape)
        ax.add_patch(
            plt.Rectangle((k - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=2)
        )
    fig.colorbar(im, ax=ax, label=f"RMSE {target} ({unit})")
    ax.set_title(
        f"Erreur sur {target} selon les paramètres de l'excitation", fontsize=10
    )
    fig.tight_layout()
    return fig


def unseen_excitation_mask(u_train: Tensor, u_eval: Tensor) -> Tensor:
    """``True`` for evaluation simulations whose excitation is absent from training."""

    def keys(u: Tensor) -> list[bytes]:
        return [row.tobytes() for row in (u > 0).numpy().astype(np.uint8)]

    seen = set(keys(u_train))
    return torch.tensor([k not in seen for k in keys(u_eval)])


# --------------------------------------------------------------------------
# Calibration and sharpness
# --------------------------------------------------------------------------
def reliability_table(
    z_true: Tensor,
    z_pred: Tensor,
    log_var: Tensor,
    levels: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
) -> pd.DataFrame:
    """Observed coverage of central Gaussian intervals, in the loss space.

    Coverage is invariant to the monotone maps between physical and loss
    spaces, so computing it on standardized (log for the taus) values is exact.
    """
    std = torch.exp(0.5 * log_var.double())
    dist = torch.distributions.Normal(0.0, 1.0)
    rows = {}
    for lv in levels:
        z = float(dist.icdf(torch.tensor(0.5 + lv / 2)))
        inside = (z_true.double() - z_pred.double()).abs() <= z * std
        rows[lv] = inside.double().mean(0).numpy()
    out = pd.DataFrame(rows, index=list(TARGET_NAMES)).T
    out.index.name = "niveau nominal"
    return out


def plot_reliability(table: pd.DataFrame) -> Figure:
    """Reliability diagram: observed vs nominal coverage per indicator."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls="--", label="calibration parfaite")
    for col in table.columns:
        ax.plot(table.index, table[col], marker="o", ms=3, label=col)
    ax.set_xlabel("couverture nominale")
    ax.set_ylabel("couverture observée")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def interval_width_physical(
    pred: Tensor, log_var: Tensor, normalizer: Normalizer, level: float = 0.95
) -> pd.Series:
    """Median width of the central interval in physical units (W/K, h)."""
    z = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.5 + level / 2)))
    zp = normalizer.transform_y(pred.double())
    s = torch.exp(0.5 * log_var.double())
    lo = normalizer.inverse_y(zp - z * s)
    hi = normalizer.inverse_y(zp + z * s)
    return pd.Series((hi - lo).median(0).values.numpy(), index=list(TARGET_NAMES))


# --------------------------------------------------------------------------
# Retransformation bias of the time constants
# --------------------------------------------------------------------------
def lognormal_mean_correction(
    pred: Tensor, log_var: Tensor, normalizer: Normalizer
) -> Tensor:
    """Replace each predicted median ``exp(mu)`` of a time constant by the mean ``exp(mu + s^2 / 2)``.

    ``s`` is the predicted standard deviation of ``log tau``: the loss-space
    standard deviation times the standardization scale of that column.
    """
    if not normalizer.log_cdt:
        return pred.clone()
    out = pred.clone().double()
    idx = list(CDT_IDX)
    s_log = torch.exp(0.5 * log_var[:, idx].double()) * normalizer.y_std[idx].double()
    out[:, idx] = out[:, idx] * torch.exp(0.5 * s_log.pow(2))
    return out.float()


def ensemble_mean(preds: Sequence[Tensor]) -> Tensor:
    """Average of physical predictions from several models.

    Averaging preserves every *linear* constraint and inequality satisfied by
    each member: ``HLC = HLC_ext + HLC_mit``, ``tau_n <= tau_inf <= tau_b``
    and positivity hold for the ensemble as soon as they hold for each model.
    """
    return torch.stack([p.double() for p in preds]).mean(dim=0).float()


def mixture_std_z(
    z_preds: Sequence[Tensor], log_vars: Sequence[Tensor]
) -> tuple[Tensor, Tensor]:
    """Mean and standard deviation of an equal-weight Gaussian mixture (loss space).

    ``var = mean(sigma_i^2 + mu_i^2) - mean(mu_i)^2``: the average predicted
    variance plus the disagreement between members.
    """
    mu = torch.stack([z.double() for z in z_preds])
    var = torch.stack([torch.exp(lv.double()) for lv in log_vars])
    mean = mu.mean(dim=0)
    total = (var + mu.pow(2)).mean(dim=0) - mean.pow(2)
    return mean, total.clamp_min(1e-12).sqrt()


def hlc_error_vs_energy(pred: Tensor, true: Tensor, u: Tensor) -> Figure:
    """Absolute HLC error against injected energy (binned medians and IQR)."""
    feats = excitation_features(u)
    e = feats["energy_kwh"].numpy()
    err = (pred[:, HLC_IDX] - true[:, HLC_IDX]).abs().numpy()
    bins = np.linspace(0, e.max() + 1e-9, 15)
    idx = np.digitize(e, bins) - 1
    mids, med, q1, q3 = [], [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= 30:
            mids.append(0.5 * (bins[b] + bins[b + 1]))
            med.append(np.median(err[m]))
            q1.append(np.quantile(err[m], 0.25))
            q3.append(np.quantile(err[m], 0.75))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(mids, q1, q3, alpha=0.3, label="écart interquartile")
    ax.plot(mids, med, marker="o", label="médiane")
    ax.set_xlabel("énergie injectée (kWh)")
    ax.set_ylabel("|erreur HLC| (W/K)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

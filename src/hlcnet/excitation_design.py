"""Which excitation should be imposed on site, and does the model know it?

The dataset contains only 2 798 distinct excitations, each received by
hundreds of simulated buildings. This module evaluates a trained model
**excitation by excitation** (section 7.4 b of the report):

* realized RMSE per excitation (needs the true indicators);
* predicted uncertainty per excitation, ``sigma = sqrt(mean(predicted variance))``,
  which needs **no** true indicator. For a calibrated model, ``E[error²] =
  E[sigma²]``, so ``sigma`` and the RMSE should agree excitation by excitation.

If they do, the model itself can rank candidate excitations from simulated
responses alone: the off-site excitation design of coEnergy's paper becomes a
forward pass of the trained network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from scipy.stats import spearmanr
from torch import Tensor

from hlcnet.data import Normalizer
from hlcnet.eda import DT_MINUTES, excitation_features
from hlcnet.targets import CDT_IDX, TARGET_NAMES

#: Two-sided 95 % quantile of the standard normal distribution.
Z95: float = 1.959964


def physical_std(
    log_var: Tensor, normalizer: Normalizer, names: Sequence[str] = ("HLC", "HLC_ext")
) -> np.ndarray:
    """Predicted standard deviation in physical units (W/K).

    Only for indicators that are standardized without a log transform, for
    which ``sigma_phys = sigma_z * y_std``.
    """
    idx = [TARGET_NAMES.index(n) for n in names]
    if normalizer.log_cdt and any(i in CDT_IDX for i in idx):
        raise ValueError(
            "time constants are log-transformed: no symmetric physical std"
        )
    sigma_z = torch.exp(0.5 * log_var[:, idx].double())
    return (sigma_z * normalizer.y_std[idx].double()).numpy()


def per_excitation_table(
    hashes: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    std: np.ndarray,
    names: Sequence[str],
    min_count: int = 50,
) -> pd.DataFrame:
    """Error, predicted uncertainty and coverage for each excitation.

    Args:
        hashes: ``(N,)`` excitation fingerprints (:func:`hlcnet.eda.excitation_hash`).
        true, pred, std: ``(N, k)`` true values, predictions and predicted
            standard deviations, in physical units, columns in ``names`` order.
        names: The ``k`` indicator names.
        min_count: Excitations received by fewer simulations are dropped.

    Returns:
        One row per excitation (index ``exc_hash``): ``n``, then for each
        indicator ``rmse_<name>``, ``sigma_<name>`` (root mean predicted
        variance) and ``couverture95_<name>``.
    """
    err = pred - true
    df = pd.DataFrame({"exc_hash": hashes})
    for j, n in enumerate(names):
        df[f"se_{n}"] = err[:, j] ** 2
        df[f"var_{n}"] = std[:, j] ** 2
        df[f"in95_{n}"] = np.abs(err[:, j]) <= Z95 * std[:, j]
    g = df.groupby("exc_hash")
    out = pd.DataFrame({"n": g.size()})
    for n in names:
        out[f"rmse_{n}"] = np.sqrt(g[f"se_{n}"].mean())
        out[f"sigma_{n}"] = np.sqrt(g[f"var_{n}"].mean())
        out[f"couverture95_{n}"] = g[f"in95_{n}"].mean()
    return out[out["n"] >= min_count]


def first_signal_per_excitation(hashes: np.ndarray, u: Tensor) -> dict[int, np.ndarray]:
    """One ``U(t)`` signal per excitation fingerprint."""
    _, first = np.unique(hashes, return_index=True)
    return {int(hashes[i]): u[i].numpy() for i in first}


def excitation_descriptors(signals: Mapping[int, np.ndarray]) -> pd.DataFrame:
    """Duty cycle, energy, number of switches and first heating time of each excitation."""
    keys = list(signals)
    feats = excitation_features(torch.from_numpy(np.stack([signals[k] for k in keys])))
    return pd.DataFrame(
        {
            "taux de chauffe": feats["duty"].numpy(),
            "énergie (kWh)": feats["energy_kwh"].numpy(),
            "commutations": feats["n_switch"].numpy(),
            "première chauffe (h)": feats["first_on_h"].numpy(),
        },
        index=pd.Index(keys, name="exc_hash"),
    )


def join_splits(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Excitations present in both splits, columns suffixed ``_val`` / ``_test``."""
    return val.join(test, lsuffix="_val", rsuffix="_test", how="inner")


def ranking_summary(merged: pd.DataFrame, name: str) -> pd.Series:
    """Is the ranking of excitations reproducible, and does the model know it?

    Rankings are made on validation; every reported error is a **test** error,
    so the choice of an excitation is never made on the data used to judge it.
    """
    rmse_val, rmse_test = merged[f"rmse_{name}_val"], merged[f"rmse_{name}_test"]
    sigma_val = merged[f"sigma_{name}_val"]
    by_rmse, by_sigma = rmse_val.idxmin(), sigma_val.idxmin()
    worst_sigma = sigma_val.idxmax()
    return pd.Series(
        {
            "excitations comparées": len(merged),
            "Spearman RMSE validation / RMSE test": spearmanr(
                rmse_val, rmse_test
            ).statistic,
            "Spearman sigma prédit (validation) / RMSE test": spearmanr(
                sigma_val, rmse_test
            ).statistic,
            "RMSE test médian sur les excitations": rmse_test.median(),
            "RMSE test, meilleure excitation selon l'erreur de validation": rmse_test[
                by_rmse
            ],
            "RMSE test, meilleure excitation selon le sigma prédit": rmse_test[
                by_sigma
            ],
            "RMSE test, pire excitation selon le sigma prédit": rmse_test[worst_sigma],
        }
    )


def plot_excitation_ranking(merged: pd.DataFrame, names: Sequence[str]) -> Figure:
    """Per excitation: validation vs test RMSE, and predicted sigma vs test RMSE."""
    fig, axes = plt.subplots(
        len(names), 2, figsize=(12, 4.6 * len(names)), squeeze=False
    )
    size = 4 + 30 * merged["n_val"] / merged["n_val"].max()
    for row, name in enumerate(names):
        y = merged[f"rmse_{name}_test"]
        for col, (col_name, label) in enumerate(
            (
                (f"rmse_{name}_val", f"RMSE validation {name} (W/K)"),
                (
                    f"sigma_{name}_val",
                    f"σ prédit par le modèle, validation {name} (W/K)",
                ),
            )
        ):
            ax = axes[row, col]
            x = merged[col_name]
            ax.scatter(x, y, s=size, alpha=0.5)
            lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
            ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls="--", label="y = x")
            rho = spearmanr(x, y).statistic
            ax.set_title(f"{name} : Spearman = {rho:.2f}", fontsize=10)
            ax.set_xlabel(label)
            ax.set_ylabel(f"RMSE test {name} (W/K)")
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_excitation_signals(
    merged: pd.DataFrame,
    signals: Mapping[int, np.ndarray],
    name: str = "HLC",
    k: int = 3,
) -> Figure:
    """``U(t)`` of the ``k`` excitations with the lowest and highest predicted sigma."""
    order = merged[f"sigma_{name}_val"].sort_values()
    chosen = [("meilleure", h) for h in order.index[:k]] + [
        ("pire", h) for h in order.index[-k:]
    ]
    fig, axes = plt.subplots(2, k, figsize=(4.2 * k, 5.2), sharex=True, sharey=True)
    for ax, (kind, h) in zip(axes.ravel(), chosen, strict=True):
        u = signals[int(h)]
        hours = np.arange(u.shape[0]) * DT_MINUTES / 60.0
        ax.step(
            hours,
            u / 1000.0,
            where="post",
            color="tab:green" if kind == "meilleure" else "tab:red",
        )
        ax.set_title(
            f"{kind} : σ prédit {merged.loc[h, f'sigma_{name}_val']:.1f}, "
            f"RMSE test {merged.loc[h, f'rmse_{name}_test']:.1f} W/K",
            fontsize=8,
        )
    for ax in axes[-1]:
        ax.set_xlabel("temps (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("U (kW)")
    fig.suptitle(
        f"Excitations classées par l'incertitude prédite sur {name} (validation)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig

"""Evaluation metrics, computed in physical units (W/K for HLC and Hb, h for CDT).

Three kinds of evaluation:

* **Accuracy** per indicator (:func:`regression_metrics`) and per family
  (:func:`family_summary`).
* **Physical consistency** of predictions (:func:`constraint_violations`).
* **Calibration** of predicted uncertainties (:func:`interval_coverage`).

:func:`metrics_by_group` breaks any of the accuracy metrics down by sub-group
(e.g. never-heated vs heated, energy deciles) for the critical error analysis.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import torch
from torch import Tensor

from hlcnet.targets import CDT_IDX, FAMILIES, HB_IDX, HLC_IDX, TARGET_NAMES, UNITS

#: Standard normal quantiles for central intervals.
_Z: dict[float, float] = {
    0.5: 0.6745,
    0.8: 1.2816,
    0.9: 1.6449,
    0.95: 1.9600,
    0.99: 2.5758,
}


# --------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------
def regression_metrics(
    pred: Tensor, true: Tensor, names: Sequence[str] = TARGET_NAMES
) -> pd.DataFrame:
    """Accuracy metrics for each indicator.

    Args:
        pred: ``(N, K)`` predictions in physical units.
        true: ``(N, K)`` ground truth in physical units.
        names: Column names, defaults to the canonical 6 indicators.

    Returns:
        DataFrame indexed by indicator with columns ``rmse``, ``mae``,
        ``bias`` (mean of pred - true), ``mape_%``, ``r2``, ``p95_abs`` and
        ``p99_abs`` (95th / 99th percentile of the absolute error), ``max_abs``
        and ``nrmse`` (RMSE divided by the standard deviation of the truth).
        P95 alone is blind to rare large errors (< 5 % of cases): compare
        RMSE with MAE and look at P99 / max to detect them.
    """
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(true.shape)}")
    p, t = pred.double(), true.double()
    err = p - t
    std = t.std(dim=0, correction=0)
    rmse = err.pow(2).mean(0).sqrt()
    df = pd.DataFrame(
        {
            "rmse": rmse.numpy(),
            "mae": err.abs().mean(0).numpy(),
            "bias": err.mean(0).numpy(),
            "mape_%": (100.0 * (err.abs() / t.abs().clamp_min(1e-12)).mean(0)).numpy(),
            "r2": (1.0 - err.pow(2).sum(0) / (t - t.mean(0)).pow(2).sum(0)).numpy(),
            "p95_abs": torch.quantile(err.abs(), 0.95, dim=0).numpy(),
            "p99_abs": torch.quantile(err.abs(), 0.99, dim=0).numpy(),
            "max_abs": err.abs().max(0).values.numpy(),
            "nrmse": (rmse / std.clamp_min(1e-12)).numpy(),
        },
        index=pd.Index(list(names), name="indicateur"),
    )
    return df


def family_summary(per_target: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-indicator metrics into the three families (HLC, Hb, CDT).

    Scale-free metrics (``nrmse``, ``r2``, ``mape_%``) are averaged over the
    family's indicators; scale-dependent ones are averaged only within a family
    that shares a unit, which is true for all three families here.
    """
    rows = {}
    for fam, idx in FAMILIES.items():
        sub = per_target.iloc[list(idx)]
        rows[fam] = {
            "unité": UNITS[fam],
            "n_indicateurs": len(idx),
            "rmse_moy": sub["rmse"].mean(),
            "p95_abs_moy": sub["p95_abs"].mean(),
            "p99_abs_moy": sub["p99_abs"].mean(),
            "mape_%_moy": sub["mape_%"].mean(),
            "r2_moy": sub["r2"].mean(),
            "nrmse_moy": sub["nrmse"].mean(),
        }
    return pd.DataFrame(rows).T


def metrics_by_group(
    pred: Tensor,
    true: Tensor,
    groups: Tensor | Sequence[str],
    metric: str = "rmse",
    names: Sequence[str] = TARGET_NAMES,
    min_size: int = 20,
) -> pd.DataFrame:
    """One metric for each indicator inside each sub-group.

    Args:
        pred, true: ``(N, K)`` tensors in physical units.
        groups: ``(N,)`` group label per sample (int tensor or strings).
        metric: Any column of :func:`regression_metrics`.
        min_size: Groups smaller than this are skipped (too noisy).

    Returns:
        DataFrame indexed by group with a column ``n`` and one column per indicator.
    """
    labels = pd.Series(groups.tolist() if isinstance(groups, Tensor) else list(groups))
    rows = {}
    for g, idx in labels.groupby(labels).groups.items():
        ii = torch.as_tensor(list(idx))
        if ii.numel() < min_size:
            continue
        m = regression_metrics(pred[ii], true[ii], names)[metric]
        rows[g] = {"n": ii.numel(), **m.to_dict()}
    return pd.DataFrame(rows).T


def mean_predictor(train_true: Tensor, n: int) -> Tensor:
    """Dummy baseline: always predict the training mean of each indicator."""
    return train_true.double().mean(0, keepdim=True).expand(n, -1).clone()


# --------------------------------------------------------------------------
# Physical consistency
# --------------------------------------------------------------------------
def constraint_violations(pred: Tensor, tol: float = 1e-3) -> pd.Series:
    """How often predictions break the physics observed in the data.

    Args:
        pred: ``(N, 6)`` predictions in physical units, canonical order.
        tol: Tolerance on ``|HLC - (ext + mit)|`` in W/K.

    Returns:
        Shares (between 0 and 1) of predictions violating each rule, plus
        the mean absolute HLC gap.
    """
    p = pred.double()
    gap = (p[:, HLC_IDX] - p[:, list(HB_IDX)].sum(1)).abs()
    tau_b, tau_n, tau_inf = (p[:, i] for i in CDT_IDX)
    return pd.Series(
        {
            "part |HLC - (ext + mit)| > tol": (gap > tol).double().mean().item(),
            "écart moyen HLC - (ext + mit) (W/K)": gap.mean().item(),
            "part ordre tau violé": (~((tau_b >= tau_inf) & (tau_inf >= tau_n)))
            .double()
            .mean()
            .item(),
            "part valeurs négatives": (p < 0).any(dim=1).double().mean().item(),
        }
    )


# --------------------------------------------------------------------------
# Calibration of uncertainties
# --------------------------------------------------------------------------
def interval_coverage(
    true: Tensor,
    mean: Tensor,
    std: Tensor,
    levels: Sequence[float] = (0.5, 0.9, 0.95),
    names: Sequence[str] = TARGET_NAMES,
) -> pd.DataFrame:
    """Empirical coverage of Gaussian central intervals ``mean ± z * std``.

    A well-calibrated model has coverage close to the nominal level: lower
    means over-confident, higher means over-cautious.

    Returns:
        DataFrame indexed by indicator, one column per level, plus the mean
        interval width at the highest level.
    """
    t, m, s = true.double(), mean.double(), std.double()
    out = {}
    for lv in levels:
        if lv not in _Z:
            raise ValueError(f"unsupported level {lv}; choose among {sorted(_Z)}")
        inside = (t - m).abs() <= _Z[lv] * s
        out[f"couverture_{int(lv * 100)}%"] = inside.double().mean(0).numpy()
    top = max(levels)
    out[f"largeur_moy_{int(top * 100)}%"] = (2 * _Z[top] * s).mean(0).numpy()
    return pd.DataFrame(out, index=pd.Index(list(names), name="indicateur"))

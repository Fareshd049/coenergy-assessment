"""Helpers for the report notebook: read saved results, never retrain.

Every training run writes ``artifacts/runs/<name>/val_metrics.json`` and every
evaluation writes CSV/PNG files under ``artifacts/eval*/<split>/``. The
notebook only assembles these files into tables and figures, so it runs in
seconds and its numbers are exactly those of the logged experiments.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from hlcnet.targets import TARGET_NAMES

#: Experiments of the report, in display order: run folder -> label.
RUN_LABELS: dict[str, str] = {
    "cnn_gru_share_nll": "CNN-GRU (référence)",
    "abl_head_sum": "tête « sum » (ext + mit)",
    "abl_head_free": "tête libre (sans contrainte)",
    "abl_no_deriv": "sans canal dérivée",
    "abl_mse": "perte MSE (sans incertitude)",
    "abl_noise": "entraîné avec bruit 0,25 °C",
    "abl_tcn_gru": "TCN-GRU",
    "abl_tcn": "TCN (sans GRU)",
    "abl_film_gru": "FiLM-GRU",
    "abl_mlp": "MLP (non séquentiel)",
    "rocket_ridge": "ROCKET + ridge",
}

_VIOLATION_KEY = "part |HLC - (ext + mit)| > tol"


def load_metrics(run_dir: Path) -> dict[str, Any]:
    """Read ``val_metrics.json`` of a run."""
    with open(Path(run_dir) / "val_metrics.json", encoding="utf-8") as f:
        return json.load(f)


def comparison_table(
    runs_dir: Path, runs: Mapping[str, str] = RUN_LABELS, metric: str = "r2"
) -> pd.DataFrame:
    """One row per run: ``metric`` for each indicator, constraint violations, data size.

    Runs whose folder or metrics file is missing are skipped.
    """
    rows: dict[str, dict[str, float]] = {}
    for name, label in runs.items():
        path = Path(runs_dir) / name / "val_metrics.json"
        if not path.exists():
            continue
        m = load_metrics(path.parent)
        row = {t: float(m["per_target"][metric][t]) for t in TARGET_NAMES}
        violations = m.get("constraints", {}).get(_VIOLATION_KEY, math.nan)
        row["violations HLC = ext + mit (%)"] = 100.0 * float(violations)
        row["simulations d'entraînement"] = float(m.get("n_train", math.nan))
        rows[label] = row
    return pd.DataFrame(rows).T


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV written by ``scripts/evaluate.py`` (first column = index)."""
    return pd.read_csv(path, index_col=0)


def val_test_table(
    run_dir: Path,
    eval_dir: Path,
    metrics: tuple[str, ...] = ("r2", "mape_%", "p95_abs"),
) -> pd.DataFrame:
    """Validation metrics of a run next to its test metrics.

    Args:
        run_dir: Run folder holding ``val_metrics.json`` (written at training).
        eval_dir: Folder holding the test ``per_target.csv`` of ``scripts/evaluate.py``.
        metrics: Metric names present in both files.

    Returns:
        DataFrame indexed by indicator, columns ``(metric, "validation" | "test")``.
    """
    val = load_metrics(run_dir)["per_target"]
    test = read_table(Path(eval_dir) / "per_target.csv")
    cols = {}
    for m in metrics:
        cols[(m, "validation")] = pd.Series(val[m], dtype=float).reindex(TARGET_NAMES)
        cols[(m, "test")] = test[m].reindex(TARGET_NAMES).astype(float)
    df = pd.DataFrame(cols)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["métrique", "jeu"])
    df.index.name = "indicateur"
    return df


def bias_corrected_r2(eval_dir: Path) -> pd.DataFrame:
    """R² of the time constants with the median ``exp(mu)`` and the mean ``exp(mu + s²/2)``.

    Uses ``R² = 1 - (RMSE / std)²`` with ``std = RMSE / nRMSE`` read from
    ``per_target.csv``; the corrected RMSE comes from ``retransformation_bias.csv``.
    """
    per_target = read_table(Path(eval_dir) / "per_target.csv")
    bias = read_table(Path(eval_dir) / "retransformation_bias.csv")
    std = per_target["rmse"] / per_target["nrmse"]
    rows = {}
    for t in bias.index:
        rows[t] = {
            "R² médiane exp(mu)": 1.0 - (bias.loc[t, "rmse médiane"] / std[t]) ** 2,
            "R² moyenne exp(mu + s²/2)": 1.0 - (bias.loc[t, "rmse moyenne"] / std[t]) ** 2,
        }
    return pd.DataFrame(rows).T

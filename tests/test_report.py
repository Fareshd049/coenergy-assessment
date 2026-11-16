"""Tests for hlcnet.report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlcnet.report import comparison_table
from hlcnet.targets import TARGET_NAMES


def write_run(root: Path, name: str, r2: float, violations: float) -> None:
    d = root / name
    d.mkdir(parents=True)
    metrics = {
        "n_train": 1000,
        "per_target": {"r2": {t: r2 for t in TARGET_NAMES}},
        "constraints": {"part |HLC - (ext + mit)| > tol": violations},
    }
    (d / "val_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_comparison_table_reads_runs_and_skips_missing(tmp_path: Path) -> None:
    write_run(tmp_path, "a", 0.4, 0.0)
    write_run(tmp_path, "b", 0.3, 0.5)
    table = comparison_table(tmp_path, {"a": "A", "b": "B", "missing": "M"})
    assert list(table.index) == ["A", "B"]
    assert table.loc["A", "HLC"] == pytest.approx(0.4)
    assert table.loc["B", "violations HLC = ext + mit (%)"] == pytest.approx(50.0)


def test_val_test_table_aligns_splits(tmp_path: Path) -> None:
    import pandas as pd

    from hlcnet.report import val_test_table

    write_run(tmp_path, "run", 0.4, 0.0)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame(
        {"r2": [0.3] * 6}, index=pd.Index(list(TARGET_NAMES), name="indicateur")
    ).to_csv(eval_dir / "per_target.csv")
    table = val_test_table(tmp_path / "run", eval_dir, metrics=("r2",))
    assert table.loc["HLC", ("r2", "validation")] == pytest.approx(0.4)
    assert table.loc["tau_b", ("r2", "test")] == pytest.approx(0.3)


def test_bias_corrected_r2_matches_definition(tmp_path: Path) -> None:
    import pandas as pd

    from hlcnet.report import bias_corrected_r2

    idx = pd.Index(["tau_b"], name="indicateur")
    pd.DataFrame({"rmse": [5.0], "nrmse": [0.5]}, index=idx).to_csv(
        tmp_path / "per_target.csv"
    )
    pd.DataFrame({"rmse médiane": [5.0], "rmse moyenne": [4.0]}, index=idx).to_csv(
        tmp_path / "retransformation_bias.csv"
    )
    table = bias_corrected_r2(tmp_path)
    # std = 10 -> R² = 1 - 0.25 and 1 - 0.16
    assert table.loc["tau_b", "R² médiane exp(mu)"] == pytest.approx(0.75)
    assert table.loc["tau_b", "R² moyenne exp(mu + s²/2)"] == pytest.approx(0.84)

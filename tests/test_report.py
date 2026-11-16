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

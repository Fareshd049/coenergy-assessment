"""Tests for hlcnet.excitation_design."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hlcnet.excitation_design import (
    Z95,
    join_splits,
    per_excitation_table,
    ranking_summary,
)


def test_per_excitation_table_rmse_sigma_and_coverage() -> None:
    hashes = np.array([1, 1, 2, 2, 2, 3])
    true = np.zeros((6, 1))
    pred = np.array([[3.0], [-4.0], [1.0], [1.0], [1.0], [0.0]])
    std = np.array([[1.0], [1.0], [2.0], [2.0], [2.0], [1.0]])
    table = per_excitation_table(hashes, true, pred, std, ["HLC"], min_count=2)
    assert list(table.index) == [1, 2]  # excitation 3 has a single simulation
    assert table.loc[1, "rmse_HLC"] == pytest.approx(np.sqrt(12.5))
    assert table.loc[2, "sigma_HLC"] == pytest.approx(2.0)
    assert table.loc[1, "couverture95_HLC"] == pytest.approx(0.0)  # |3|, |4| > 1.96
    assert table.loc[2, "couverture95_HLC"] == pytest.approx(1.0)
    assert Z95 == pytest.approx(1.96, abs=1e-3)


def test_ranking_summary_judges_validation_choice_on_test() -> None:
    idx = pd.Index([10, 20, 30], name="exc_hash")
    val = pd.DataFrame(
        {"n": [100] * 3, "rmse_HLC": [1.0, 2.0, 3.0], "sigma_HLC": [3.0, 2.0, 1.0]},
        index=idx,
    )
    test = pd.DataFrame(
        {"n": [100] * 3, "rmse_HLC": [1.5, 2.5, 3.5], "sigma_HLC": [3.0, 2.0, 1.0]},
        index=idx,
    )
    s = ranking_summary(join_splits(val, test), "HLC")
    assert s["Spearman RMSE validation / RMSE test"] == pytest.approx(1.0)
    assert s["Spearman sigma prédit (validation) / RMSE test"] == pytest.approx(-1.0)
    assert s[
        "RMSE test, meilleure excitation selon l'erreur de validation"
    ] == pytest.approx(1.5)
    assert s["RMSE test, meilleure excitation selon le sigma prédit"] == pytest.approx(
        3.5
    )

"""Tests for hlcnet.eda feature helpers."""

from __future__ import annotations

import torch

from hlcnet.eda import excitation_features


def test_excitation_features() -> None:
    u = torch.zeros(3, 120)
    u[1, 60:] = 7000.0  # one hour on, starting at 1 h
    u[2, ::2] = 7000.0  # alternating
    f = excitation_features(u)
    assert f["never_on"].tolist() == [True, False, False]
    assert f["first_on_h"].tolist() == [2.0, 1.0, 0.0]
    assert abs(f["energy_kwh"][1].item() - 7.0) < 1e-5
    assert f["duty"][2].item() == 0.5
    assert f["n_switch"].tolist() == [0, 1, 119]

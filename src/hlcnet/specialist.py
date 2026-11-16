"""Specialist models trained on a single excitation (coEnergy's Phase 2 setting).

In the paper's Phase 2, one model is trained per test, on simulations that all
share the excitation actually applied on site. Our dataset contains two
constant excitations shared by many simulations: **never heated** (``U = 0``)
and **heated continuously** (``U = 7 kW``, a step response). Training
specialists on them and comparing with the generalist model *on the same
houses* measures directly whether the generalist leaves information unused:
if a specialist does not beat it, the generalist already extracts what that
excitation reveals.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from hlcnet.data import SimData, load_batches


def constant_excitation_masks(u: Tensor) -> dict[str, Tensor]:
    """Masks of the never-heated and continuously heated simulations, ``u``: ``(N, T)``."""
    on = u > 0
    return {"jamais chauffé": ~on.any(dim=1), "chauffe continue": on.all(dim=1)}


def load_constant_excitation_groups(
    folder: Path, batch_ids: Sequence[int], round_inputs: bool = True
) -> dict[str, SimData]:
    """Load every simulation of the two constant-excitation groups, one file at a time.

    Peak memory stays around one file: each file is loaded, filtered, freed.
    """
    parts: dict[str, list[SimData]] = {}
    for i in batch_ids:
        d = load_batches(folder, [i], per_batch=None, round_inputs=round_inputs)
        for name, mask in constant_excitation_masks(d.x[..., 1]).items():
            if mask.any():
                parts.setdefault(name, []).append(
                    d.subset(torch.nonzero(mask).squeeze(1))
                )
        del d
    return {name: SimData.concat(p) for name, p in parts.items()}


@dataclass
class FeatureScaler:
    """Per-feature standardization fitted on the training rows."""

    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, x: Tensor) -> FeatureScaler:
        return cls(x.mean(dim=0), x.std(dim=0).clamp_min(1e-6))

    def __call__(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


def fit_mlp(
    x_train: Tensor,
    z_train: Tensor,
    x_val: Tensor,
    z_val: Tensor,
    hidden: Sequence[int] = (256, 128),
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    batch_size: int = 256,
    patience: int = 10,
    seed: int = 0,
) -> nn.Module:
    """Small MLP regressor trained with MSE and early stopping on ``(x_val, z_val)``.

    Inputs and targets are expected standardized. Returns the best model (eval mode).
    """
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    d = x_train.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, z_train.shape[1]))
    model = nn.Sequential(*layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)

    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(x_train.shape[0], generator=gen)
        for start in range(0, perm.numel(), batch_size):
            idx = perm[start : start + batch_size]
            loss = (model(x_train[idx]) - z_train[idx]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = (model(x_val) - z_val).pow(2).mean().item()
        if val_loss < best - 1e-6:
            best, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()

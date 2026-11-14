"""Understand the excitation signals: types, time step, uninformative cases.

Usage:
    uv run python scripts/inspect_excitation.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from hlcnet import util_assessement as ua
from hlcnet.data import list_batches


def main() -> None:
    b = ua.load_single_batch(list_batches(Path("data/quentin_house"))[0])
    u = b.indoor_U.float()
    meta, names = b.solicitation.float(), list(b.solicitation_field_names)
    col = {n: meta[:, i] for i, n in enumerate(names)}

    print("== Solicitation parameters of sims 0-5 ==")
    for i in range(6):
        print(
            f"sim {i}:", {n: round(meta[i, j].item(), 3) for j, n in enumerate(names)}
        )

    print("\n== Parameter ranges ==")
    for n in names:
        c = col[n]
        print(
            f"{n:26s} min={c.min():9.3f} max={c.max():9.3f} frac_zero={(c == 0).float().mean():.2f}"
        )

    # Switches per series and unique power levels hint at PRBS vs sinusoidal
    switches = (u.diff(dim=1) != 0).sum(dim=1)
    n_levels = torch.tensor([row.unique().numel() for row in u[:2000]])
    print("\n== Shape of U ==")
    print(
        "switches per sim: quantiles 0/25/50/75/100% =",
        torch.quantile(
            switches.float(), torch.tensor([0, 0.25, 0.5, 0.75, 1.0])
        ).tolist(),
    )
    print(
        "sims (first 2000) with <=2 distinct power levels:",
        (n_levels <= 2).float().mean().item(),
    )

    # Uninformative excitations
    never_on = u.max(dim=1).values == 0
    energy = u.sum(dim=1)
    print("\n== Weak excitations ==")
    print(
        f"sims with U == 0 everywhere: {never_on.sum().item()} ({never_on.float().mean():.2%})"
    )
    print(
        "total injected energy (U summed over steps), quantiles 0/5/50%:",
        torch.quantile(energy, torch.tensor([0, 0.05, 0.5])).tolist(),
    )

    # Time step: compare PRBS block length in steps to prbs_min_duration_hours
    if "prbs_min_duration_hours" in col:
        i = int(switches.argmax())
        row = u[i]
        change = torch.nonzero(row.diff() != 0).squeeze(1)
        runs = change.diff()
        print("\n== Time step estimate (most-switching sim) ==")
        print(
            f"sim {i}: min run length = {runs.min().item()} steps, "
            f"prbs_min_duration_hours = {col['prbs_min_duration_hours'][i]:.3f}, "
            f"prbs_max_duration_hours = {col['prbs_max_duration_hours'][i]:.3f}"
        )
        print(
            "-> if min run = prbs_min_duration, dt (minutes) ~",
            round(
                col["prbs_min_duration_hours"][i].item()
                * 60
                / max(runs.min().item(), 1),
                2,
            ),
        )


if __name__ == "__main__":
    main()

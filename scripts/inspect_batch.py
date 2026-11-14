"""Sanity checks and quick plots on one simulation batch.

Usage:
    uv run python scripts/inspect_batch.py
    uv run python scripts/inspect_batch.py --folder data/quentin_house --n-plot 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import torch

from hlcnet import util_assessement as ua  # registers the pickle package


def describe(name: str, t: torch.Tensor) -> None:
    """Print shape, dtype, NaN/inf flags and range of a tensor."""
    t_f = t.float()
    print(
        f"{name:20s} shape={tuple(t.shape)!s:16s} dtype={t.dtype!s:14s} "
        f"nan={t_f.isnan().any().item()} inf={t_f.isinf().any().item()} "
        f"min={t_f.min().item():10.3f} max={t_f.max().item():10.3f} "
        f"mean={t_f.mean().item():10.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=Path("data/quentin_house"))
    parser.add_argument("--n-plot", type=int, default=4)
    args = parser.parse_args()

    files = sorted(args.folder.glob("*.pt"))
    print(f"Found {len(files)} .pt files:", [f.name for f in files][:5], "...")
    f = files[0]
    print(f"\nFile: {f.name}  size={f.stat().st_size / 1e6:.1f} MB")

    b = ua.load_single_batch(f)
    print("Object type:", type(b).__name__)
    print("Attributes:", [a for a in vars(b)] if hasattr(b, "__dict__") else dir(b))

    # ---- 1. Shapes, dtypes, ranges -------------------------------------
    print("\n== Tensors ==")
    for name in ("indoor_temperature", "indoor_U", "HLC", "Hb", "CDT", "solicitation"):
        if hasattr(b, name):
            describe(name, getattr(b, name))
    print("Hb_labels:", getattr(b, "Hb_labels", None))
    print("solicitation fields:", getattr(b, "solicitation_field_names", None))

    temp, u = b.indoor_temperature.float(), b.indoor_U.float()
    hlc, hb, cdt = b.HLC.float(), b.Hb.float(), b.CDT.float()
    n, t_len = temp.shape[0], temp.shape[-1]
    print(f"\nN simulations = {n}, T time steps = {t_len}")
    mem = sum(x.element_size() * x.nelement() for x in (b.indoor_temperature, b.indoor_U))
    print(f"Memory of (Tint, U) for this batch: {mem / 1e6:.1f} MB")

    # ---- 2. Physical constraint ----------------------------------------
    print("\n== HLC == sum(Hb) ? ==")
    gap = (hlc - hb.sum(dim=1)).abs()
    print(f"max |HLC - sum(Hb)| = {gap.max().item():.2e}, mean = {gap.mean().item():.2e}")

    # ---- 3. Targets ----------------------------------------------------
    print("\n== Targets ==")
    print(f"HLC   mean={hlc.mean():.2f} std={hlc.std():.2f} min={hlc.min():.2f} max={hlc.max():.2f}")
    for j, lab in enumerate(getattr(b, "Hb_labels", range(hb.shape[1]))):
        c = hb[:, j]
        print(f"{lab!s:6s} mean={c.mean():.2f} std={c.std():.2f} min={c.min():.2f} max={c.max():.2f}")
    for j, lab in enumerate(["tau_b", "tau_n", "tau_inf"]):
        c = cdt[:, j]
        print(f"{lab:7s} mean={c.mean():.2f} std={c.std():.2f} min={c.min():.2f} max={c.max():.2f}")
    print("Fraction tau_b >= tau_inf:", (cdt[:, 0] >= cdt[:, 2]).float().mean().item())
    print("Fraction tau_inf >= tau_n:", (cdt[:, 2] >= cdt[:, 1]).float().mean().item())
    print("Target correlation matrix [HLC, Hb..., taus]:")
    y = torch.cat([hlc.unsqueeze(1), hb, cdt], dim=1)
    print(torch.corrcoef(y.T).round(decimals=2))

    # ---- 4. Inputs -----------------------------------------------------
    print("\n== Inputs ==")
    print(f"Tint(t=0): mean={temp[:, 0].mean():.2f} std={temp[:, 0].std():.3f}")
    print(f"Tint rise (max - t0): mean={(temp.max(1).values - temp[:, 0]).mean():.2f}")
    print(f"U < 0 anywhere? {(u < 0).any().item()}  | fraction of steps with U == 0: "
          f"{(u == 0).float().mean().item():.2f}")
    print(f"U max per sim: mean={u.max(1).values.mean():.1f} "
          f"min={u.max(1).values.min():.1f} max={u.max(1).values.max():.1f}")
    print(f"Unique values in U of sim 0 (first 10): {u[0].unique()[:10].tolist()}")

    # ---- 5. Plots ------------------------------------------------------
    out = Path("artifacts/figures")
    out.mkdir(parents=True, exist_ok=True)
    k = args.n_plot
    fig, axes = plt.subplots(k, 2, figsize=(12, 2.5 * k), sharex=True)
    for i in range(k):
        axes[i, 0].plot(temp[i])
        axes[i, 0].set_ylabel("Tint")
        axes[i, 0].set_title(f"sim {i} | HLC={hlc[i]:.1f} W/K", fontsize=9)
        axes[i, 1].plot(u[i], color="tab:orange")
        axes[i, 1].set_ylabel("U")
    axes[-1, 0].set_xlabel("time step")
    axes[-1, 1].set_xlabel("time step")
    fig.tight_layout()
    fig.savefig(out / "example_curves.png", dpi=120)

    fig, axes = plt.subplots(1, y.shape[1], figsize=(3 * y.shape[1], 2.8))
    names = ["HLC", *map(str, getattr(b, "Hb_labels", ["Hb1", "Hb2"])), "tau_b", "tau_n", "tau_inf"]
    for j, ax in enumerate(axes):
        ax.hist(y[:, j].numpy(), bins=60)
        ax.set_title(names[j], fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "target_hists.png", dpi=120)
    print(f"\nSaved {out / 'example_curves.png'} and {out / 'target_hists.png'}")


if __name__ == "__main__":
    main()
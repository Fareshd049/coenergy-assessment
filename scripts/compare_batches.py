"""Compare the 20 batch files to check they come from the same distribution.

Loads one file at a time (peak memory ~ one batch) and writes:

* ``artifacts/eda/batch_summary.csv``: one row per file (targets, excitations, checks).
* ``artifacts/eda/batch_target_quantiles.png``: per-file target medians and IQR.
* ``artifacts/eda/batch_hlc_ecdf.png``: HLC empirical CDF of every file.

Usage:
    uv run python scripts/compare_batches.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy.stats import ks_2samp

from hlcnet import util_assessement as ua
from hlcnet.data import list_batches
from hlcnet.targets import TARGET_NAMES, assemble_targets, hlc_consistency_gap

SINE_FIELDS = ("start_amplitude_wat", "end_amplitude_wat", "start_period_hours")


def excitation_kind(meta: torch.Tensor, names: list[str]) -> torch.Tensor:
    """Label each simulation: 0 = PRBS, 1 = sinusoidal, 2 = both/unknown."""
    col = {n: meta[:, i] for i, n in enumerate(names)}
    is_sine = torch.zeros(meta.shape[0], dtype=torch.bool)
    for f in SINE_FIELDS:
        if f in col:
            is_sine |= col[f] > 0
    is_prbs = (
        col["prbs_min_duration_hours"] > 0
        if "prbs_min_duration_hours" in col
        else ~is_sine
    )
    kind = torch.full((meta.shape[0],), 2, dtype=torch.long)
    kind[is_prbs & ~is_sine] = 0
    kind[is_sine & ~is_prbs] = 1
    return kind


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=Path("data/quentin_house"))
    parser.add_argument("--ks-sample", type=int, default=5000)
    args = parser.parse_args()

    files = list_batches(args.folder)
    print(f"{len(files)} files: {files[0].name} ... {files[-1].name}")

    rows: list[dict[str, float | int | str]] = []
    ref_y: torch.Tensor | None = None
    hlc_samples: dict[str, torch.Tensor] = {}
    quantiles: dict[str, torch.Tensor] = {}
    gen = torch.Generator().manual_seed(0)
    q = torch.tensor([0.25, 0.5, 0.75])

    for i, f in enumerate(files):
        b = ua.load_single_batch(f)
        y = assemble_targets(b.HLC, b.Hb, b.Hb_labels, b.CDT)
        temp, u = b.indoor_temperature.float(), b.indoor_U.float()
        meta, names = b.solicitation.float(), list(b.solicitation_field_names)
        kind = excitation_kind(meta, names)
        col = {n: meta[:, j] for j, n in enumerate(names)}

        idx = torch.randperm(y.shape[0], generator=gen)[: args.ks_sample]
        ys = y[idx]
        if ref_y is None:
            ref_y = ys
        ks = {
            f"ks_p_{name}": float(
                ks_2samp(ys[:, j].numpy(), ref_y[:, j].numpy()).pvalue
            )
            for j, name in enumerate(TARGET_NAMES)
        }

        row: dict[str, float | int | str] = {
            "file": f.name,
            "n": y.shape[0],
            "T": temp.shape[1],
            "size_mb": round(f.stat().st_size / 1e6, 1),
            "frac_prbs": (kind == 0).float().mean().item(),
            "frac_sine": (kind == 1).float().mean().item(),
            "frac_other": (kind == 2).float().mean().item(),
            "frac_never_on": (u.max(dim=1).values == 0).float().mean().item(),
            "mean_energy_kwh": (u.sum(dim=1) / 60 / 1000).mean().item(),  # 1-min steps
            "prob_on_mean": col.get("prbs_probability_on", torch.zeros(1))
            .mean()
            .item(),
            "prbs_min_dur_mean": col.get("prbs_min_duration_hours", torch.zeros(1))
            .mean()
            .item(),
            "tint0_std": temp[:, 0].std().item(),
            "u_min": u.min().item(),
            "u_max": u.max().item(),
            "max_hlc_gap": hlc_consistency_gap(y).max().item(),
            "frac_tau_order_ok": ((y[:, 3] >= y[:, 5]) & (y[:, 5] >= y[:, 4]))
            .float()
            .mean()
            .item(),
            "any_nan": bool(temp.isnan().any() or u.isnan().any() or y.isnan().any()),
        }
        for j, name in enumerate(TARGET_NAMES):
            row[f"mean_{name}"] = y[:, j].mean().item()
            row[f"std_{name}"] = y[:, j].std().item()
        row.update(ks)
        rows.append(row)

        hlc_samples[f.stem] = ys[:, 0]
        quantiles[f.stem] = torch.quantile(y, q, dim=0)  # (3, 6)
        print(
            f"[{i + 1:2d}/{len(files)}] {f.name:12s} T={row['T']} "
            f"prbs={row['frac_prbs']:.2f} sine={row['frac_sine']:.2f} "
            f"never_on={row['frac_never_on']:.3f} HLC={row['mean_HLC']:.1f} "
            f"min KS p={min(ks.values()):.3f}"
        )
        del b, temp, u

    df = pd.DataFrame(rows)
    out = Path("artifacts/eda")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "batch_summary.csv", index=False)

    # ---- Figures -------------------------------------------------------
    stems = list(quantiles)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    for j, (ax, name) in enumerate(zip(axes.ravel(), TARGET_NAMES, strict=True)):
        med = [quantiles[s][1, j].item() for s in stems]
        lo = [quantiles[s][0, j].item() for s in stems]
        hi = [quantiles[s][2, j].item() for s in stems]
        x = range(len(stems))
        ax.fill_between(x, lo, hi, alpha=0.3, label="IQR")
        ax.plot(x, med, marker="o", ms=3, label="median")
        ax.set_title(name, fontsize=10)
    for ax in axes[-1]:
        ax.set_xticks(range(len(stems)))
        ax.set_xticklabels([s.replace("batch_", "") for s in stems], fontsize=7)
        ax.set_xlabel("batch")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "batch_target_quantiles.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7, 4))
    for h in hlc_samples.values():
        ax.plot(h.sort().values, torch.linspace(0, 1, h.numel()), lw=0.8, alpha=0.7)
    ax.set_xlabel("HLC (W/K)")
    ax.set_ylabel("ECDF")
    ax.set_title("HLC distribution per batch")
    fig.tight_layout()
    fig.savefig(out / "batch_hlc_ecdf.png", dpi=120)

    # ---- Verdict -------------------------------------------------------
    print("\n== Summary over files ==")
    cols = [
        "frac_prbs",
        "frac_sine",
        "frac_never_on",
        "mean_energy_kwh",
        "mean_HLC",
        "mean_tau_b",
        "max_hlc_gap",
        "frac_tau_order_ok",
        "tint0_std",
    ]
    print(df[cols].describe().loc[["mean", "std", "min", "max"]].round(4).to_string())
    ks_cols = [c for c in df.columns if c.startswith("ks_p_")]
    n_low = int((df[ks_cols].iloc[1:] < 0.01).sum().sum())
    print(
        f"\nKS tests vs {files[0].name} with p < 0.01: {n_low} / {len(ks_cols) * (len(df) - 1)}"
    )
    print(f"T values seen: {sorted(df['T'].unique().tolist())}")
    print("Saved artifacts/eda/batch_summary.csv and two figures")


if __name__ == "__main__":
    main()

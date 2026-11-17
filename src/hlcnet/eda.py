"""Exploratory data analysis helpers used by the report notebook.

Every function returns a matplotlib ``Figure`` or a table, so the notebook only
orchestrates and comments. Time axis: 1 step = 1 minute (480 steps = 8 h).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import Tensor

from hlcnet.data import SimData, list_batches, load_batches
from hlcnet.targets import CDT_IDX, HLC_IDX, TARGET_NAMES, hlc_consistency_gap

DT_MINUTES: float = 1.0
UNITS: dict[str, str] = {
    "HLC": "W/K",
    "HLC_ext": "W/K",
    "HLC_mit": "W/K",
    "tau_b": "h",
    "tau_n": "h",
    "tau_inf": "h",
}


# --------------------------------------------------------------------------
# Tables and features
# --------------------------------------------------------------------------
def target_table(y: Tensor) -> pd.DataFrame:
    """Descriptive statistics of the 6 targets (one row per target)."""
    df = pd.DataFrame(y.numpy(), columns=list(TARGET_NAMES))
    out = df.describe(percentiles=[0.05, 0.5, 0.95]).T
    out["skew"] = df.skew()
    out.insert(0, "unité", [UNITS[n] for n in TARGET_NAMES])
    return out.round(2)


def consistency_checks(data: SimData) -> pd.Series:
    """Physical sanity checks on a :class:`SimData`."""
    y = data.y
    tau_b, tau_n, tau_inf = (y[:, i] for i in CDT_IDX)
    return pd.Series(
        {
            "max |HLC - (ext + mit)| (W/K)": hlc_consistency_gap(y).max().item(),
            "part tau_b >= tau_inf >= tau_n": ((tau_b >= tau_inf) & (tau_inf >= tau_n))
            .float()
            .mean()
            .item(),
            "Tint(t=0) moyenne (°C)": data.x[:, 0, 0].mean().item(),
            "Tint(t=0) écart-type (°C)": data.x[:, 0, 0].std().item(),
            "U min (W)": data.x[..., 1].min().item(),
            "U max (W)": data.x[..., 1].max().item(),
            "NaN présents": bool(data.x.isnan().any() or y.isnan().any()),
        }
    )


def excitation_features(u: Tensor) -> dict[str, Tensor]:
    """Summary features of the heating signal ``u`` of shape ``(N, T)``.

    Returns:
        ``duty`` (fraction of time on), ``energy_kwh`` (1-min steps),
        ``n_switch`` (number of on/off changes), ``never_on`` (bool) and
        ``first_on_h`` (time of first heating, ``T`` if never on).
    """
    on = u > 0
    t_len = u.shape[1]
    first = torch.where(on.any(dim=1), on.float().argmax(dim=1), torch.tensor(t_len))
    return {
        "duty": on.float().mean(dim=1),
        "energy_kwh": u.sum(dim=1) * DT_MINUTES / 60.0 / 1000.0,
        "n_switch": (u.diff(dim=1) != 0).sum(dim=1),
        "never_on": ~on.any(dim=1),
        "first_on_h": first.float() * DT_MINUTES / 60.0,
    }


def _hours(t_len: int) -> np.ndarray:
    return np.arange(t_len) * DT_MINUTES / 60.0


def _subsample(n: int, k: int, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=g)[: min(k, n)]


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_examples(data: SimData, indices: Sequence[int]) -> Figure:
    """Tint and U for a few simulations, with their targets in the title."""
    k = len(indices)
    fig, axes = plt.subplots(k, 2, figsize=(12, 2.3 * k), sharex=True, squeeze=False)
    t = _hours(data.x.shape[1])
    for row, i in enumerate(indices):
        y = data.y[i]
        axes[row, 0].plot(t, data.x[i, :, 0])
        axes[row, 0].set_ylabel("Tint (°C)")
        axes[row, 0].set_title(
            f"sim {i} | HLC={y[0]:.0f} W/K (ext {y[1]:.0f}, mit {y[2]:.0f}) | "
            f"τb={y[3]:.0f} h, τn={y[4]:.1f} h, τ∞={y[5]:.0f} h",
            fontsize=8,
        )
        axes[row, 1].step(t, data.x[i, :, 1], where="post", color="tab:orange")
        axes[row, 1].set_ylabel("U (W)")
    axes[-1, 0].set_xlabel("temps (h)")
    axes[-1, 1].set_xlabel("temps (h)")
    fig.tight_layout()
    return fig


def plot_target_distributions(y: Tensor) -> Figure:
    """Histograms of the 6 targets (taus also shown on a log axis)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for j, (ax, name) in enumerate(zip(axes.ravel(), TARGET_NAMES, strict=True)):
        v = y[:, j].numpy()
        ax.hist(v, bins=80, color="tab:blue", alpha=0.8)
        ax.axvline(np.median(v), color="k", lw=1, ls="--")
        ax.set_title(f"{name} ({UNITS[name]})", fontsize=10)
        if j in CDT_IDX:
            ax2 = ax.inset_axes((0.58, 0.5, 0.4, 0.45))
            ax2.hist(np.log(v), bins=60, color="tab:green", alpha=0.8)
            ax2.set_title("log", fontsize=7)
            ax2.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def plot_correlation(y: Tensor) -> Figure:
    """Pearson correlation matrix of the targets."""
    c = torch.corrcoef(y.T).numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(TARGET_NAMES)), TARGET_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(TARGET_NAMES)), TARGET_NAMES)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Corrélations entre indicateurs")
    fig.tight_layout()
    return fig


def plot_physical_relations(y: Tensor, k: int = 5000) -> Figure:
    """Taus vs HLC, the HLC decomposition and tau ordering."""
    idx = _subsample(y.shape[0], k)
    ys = y[idx].numpy()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, j in zip(axes[:3], CDT_IDX, strict=True):
        ax.scatter(ys[:, HLC_IDX], ys[:, j], s=2, alpha=0.3)
        ax.set_yscale("log")
        ax.set_xlabel("HLC (W/K)")
        ax.set_ylabel(f"{TARGET_NAMES[j]} (h, log)")
    axes[3].scatter(ys[:, 1], ys[:, 2], s=2, alpha=0.3, c=ys[:, 0], cmap="viridis")
    axes[3].set_xlabel("HLC_ext (W/K)")
    axes[3].set_ylabel("HLC_mit (W/K)")
    axes[3].set_title("couleur = HLC", fontsize=9)
    fig.tight_layout()
    return fig


def plot_excitation_summary(feats: dict[str, Tensor]) -> Figure:
    """Distributions of duty cycle, injected energy, switches and first heating time."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.5))
    axes[0].hist(feats["duty"].numpy(), bins=50)
    axes[0].set_xlabel("taux de chauffe (fraction du temps U > 0)")
    axes[1].hist(feats["energy_kwh"].numpy(), bins=50)
    axes[1].set_xlabel("énergie injectée (kWh)")
    axes[2].hist(feats["n_switch"].clamp(max=40).numpy(), bins=41)
    axes[2].set_xlabel("nombre de commutations (tronqué à 40)")
    axes[3].hist(feats["first_on_h"].numpy(), bins=50)
    axes[3].set_xlabel("instant de première chauffe (h, 8 = jamais)")
    for ax in axes:
        ax.set_ylabel("simulations")
    fig.tight_layout()
    return fig


def plot_informativeness(data: SimData, k: int = 8000) -> Figure:
    """How much the temperature response tells about HLC.

    Left: final temperature rise vs injected energy, colored by HLC. For a
    given energy, a leakier building (high HLC) warms less.
    Right: simulations without heating, final drift vs HLC.
    """
    feats = excitation_features(data.x[..., 1])
    rise = data.x[:, -1, 0] - data.x[:, 0, 0]
    hlc = data.y[:, HLC_IDX]
    idx = _subsample(len(data), k)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sc = axes[0].scatter(
        feats["energy_kwh"][idx], rise[idx], c=hlc[idx], s=3, cmap="viridis", alpha=0.6
    )
    axes[0].set_xlabel("énergie injectée (kWh)")
    axes[0].set_ylabel("Tint(8 h) − Tint(0) (K)")
    fig.colorbar(sc, ax=axes[0], label="HLC (W/K)")

    off = feats["never_on"]
    if off.any():
        axes[1].scatter(hlc[off], rise[off], s=3, alpha=0.4, color="tab:red")
        r = np.corrcoef(hlc[off].numpy(), rise[off].numpy())[0, 1]
        axes[1].set_title(
            f"sans chauffe (n={int(off.sum())}), corr = {r:.2f}", fontsize=9
        )
    axes[1].set_xlabel("HLC (W/K)")
    axes[1].set_ylabel("Tint(8 h) − Tint(0) (K)")
    fig.tight_layout()
    return fig


def plot_batch_homogeneity(summary: pd.DataFrame) -> Figure:
    """Per-file mean ± std of HLC and tau_b, and share of never-heated sims."""
    x = np.arange(len(summary))
    labels = [f.replace("batch_", "").replace(".pt", "") for f in summary["file"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))
    for ax, name in zip(axes[:2], ("HLC", "tau_b"), strict=True):
        ax.errorbar(
            x, summary[f"mean_{name}"], yerr=summary[f"std_{name}"], fmt="o", ms=3
        )
        ax.set_title(f"{name} : moyenne ± écart-type par fichier", fontsize=9)
    axes[2].bar(x, 100 * summary["frac_never_on"])
    axes[2].set_title("% de simulations sans chauffe", fontsize=9)
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_xlabel("batch")
    fig.tight_layout()
    return fig


def conditional_correlation(
    cond: Tensor, a: Tensor, b: Tensor, n_bins: int = 10
) -> pd.DataFrame:
    """Pearson correlation of ``a`` and ``b`` inside quantile bins of ``cond``.

    Used to measure the effect of HLC on the temperature response once the
    injected energy is held (approximately) fixed.
    """
    edges = torch.quantile(cond.float(), torch.linspace(0, 1, n_bins + 1))
    rows = []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        m = (cond >= lo) & ((cond < hi) if k < n_bins - 1 else (cond <= hi))
        if m.sum() < 3:
            continue
        r = np.corrcoef(a[m].numpy(), b[m].numpy())[0, 1]
        rows.append(
            {
                "bin": k + 1,
                "min": lo.item(),
                "max": hi.item(),
                "n": int(m.sum()),
                "corr": r,
            }
        )
    return pd.DataFrame(rows).round(3)


# --------------------------------------------------------------------------
# Full-dataset scan (1 000 000 simulations)
# --------------------------------------------------------------------------
#: Excitation parameters that vary across simulations (the others are constant).
PRBS_PARAMS: tuple[str, ...] = ("prbs_probability_on", "prbs_min_duration_hours")


def excitation_hash(u: Tensor) -> Tensor:
    """64-bit fingerprint of each excitation signal ``u`` of shape ``(N, T)``.

    Signals are compared at the 0.1 W resolution of the data. Two identical
    signals always get the same hash; two different ones collide with
    probability about ``2**-60``, negligible for a million signals.
    """
    q = torch.round(u.double() * 10.0).to(torch.int64)
    mask, base, p, w = (1 << 64) - 1, 1_000_003, 1, []
    for _ in range(q.shape[1]):
        w.append(p - (1 << 64) if p >= 1 << 63 else p)  # signed int64
        p = (p * base) & mask
    return (q * torch.tensor(w, dtype=torch.int64)).sum(dim=1)  # wraps modulo 2**64


def scan_dataset(folder: Path, cache: Path | None = None) -> pd.DataFrame:
    """One row per simulation of the whole dataset, read one file at a time.

    Columns: the 6 targets, the 2 varying PRBS parameters, the excitation
    features of :func:`excitation_features`, the response summary (``t0``,
    ``rise`` = Tint(8 h) - Tint(0), ``rise_max``), ``u_min``/``u_max``,
    ``has_nan``, the excitation fingerprint ``exc_hash`` and the file index
    ``batch``. Inputs are rounded as in coEnergy's loading example.

    Args:
        folder: Folder containing the 20 batch files.
        cache: Optional pickle file: read if it exists, written otherwise.
    """
    if cache is not None and Path(cache).exists():
        return pd.read_pickle(cache)
    parts = []
    for i in range(len(list_batches(folder))):
        d = load_batches(folder, batch_ids=[i])
        temp, u = d.x[..., 0], d.x[..., 1]
        cols: dict[str, Tensor] = {n: d.y[:, j] for j, n in enumerate(TARGET_NAMES)}
        for name in PRBS_PARAMS:
            cols[name] = d.meta[:, d.meta_names.index(name)]
        cols.update(excitation_features(u))
        cols["t0"] = temp[:, 0]
        cols["rise"] = temp[:, -1] - temp[:, 0]
        cols["rise_max"] = temp.max(dim=1).values - temp[:, 0]
        cols["u_min"], cols["u_max"] = u.min(dim=1).values, u.max(dim=1).values
        cols["has_nan"] = d.x.isnan().any(dim=(1, 2)) | d.y.isnan().any(dim=1)
        cols["exc_hash"] = excitation_hash(u)
        cols["batch"] = d.batch_id
        parts.append(pd.DataFrame({k: v.numpy() for k, v in cols.items()}))
        del d, temp, u
    scan = pd.concat(parts, ignore_index=True)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        scan.to_pickle(cache)
    return scan


def scan_targets(scan: pd.DataFrame) -> Tensor:
    """``(N, 6)`` target tensor of a :func:`scan_dataset` table."""
    return torch.from_numpy(
        scan[list(TARGET_NAMES)].to_numpy(dtype=np.float32, copy=True)
    )


def scan_features(scan: pd.DataFrame) -> dict[str, Tensor]:
    """Excitation features of a :func:`scan_dataset` table, as tensors."""
    keys = ("duty", "energy_kwh", "n_switch", "never_on", "first_on_h")
    return {k: torch.from_numpy(scan[k].to_numpy(copy=True)) for k in keys}


def full_consistency_checks(scan: pd.DataFrame) -> pd.Series:
    """Physical and data-quality checks on every simulation of the dataset."""
    gap = (scan["HLC"] - scan["HLC_ext"] - scan["HLC_mit"]).abs()
    ordered = (scan["tau_b"] >= scan["tau_inf"]) & (scan["tau_inf"] >= scan["tau_n"])
    return pd.Series(
        {
            "simulations": len(scan),
            "simulations avec NaN": int(scan["has_nan"].sum()),
            "max |HLC - (ext + mit)| (W/K)": float(gap.max()),
            "part tau_b >= tau_inf >= tau_n": float(ordered.mean()),
            "Tint(t=0) min / max (°C)": f"{scan['t0'].min():.2f} / {scan['t0'].max():.2f}",
            "U min / max (W)": f"{scan['u_min'].min():.1f} / {scan['u_max'].max():.1f}",
            "HLC min / max (W/K)": f"{scan['HLC'].min():.0f} / {scan['HLC'].max():.0f}",
            "part sans chauffe": float(scan["never_on"].mean()),
        },
        dtype=object,
    )


def sample_vs_population(scan: pd.DataFrame, sample: SimData) -> pd.DataFrame:
    """Is the EDA sample representative? Means, std and a two-sample KS test.

    Compares the sample used for curve-level analyses with all simulations,
    on the targets and on two excitation features.
    """
    from scipy.stats import ks_2samp

    feats = excitation_features(sample.x[..., 1])
    sample_cols = {n: sample.y[:, j].numpy() for j, n in enumerate(TARGET_NAMES)}
    sample_cols.update({k: feats[k].numpy() for k in ("energy_kwh", "duty")})
    rows = {}
    for name, v in sample_cols.items():
        pop = scan[name].to_numpy()
        ks = ks_2samp(v, pop)
        rows[name] = {
            "moyenne (1 M)": pop.mean(),
            "moyenne (échantillon)": v.mean(),
            "écart-type (1 M)": pop.std(),
            "écart-type (échantillon)": v.std(),
            "KS p-value": ks.pvalue,
        }
    return pd.DataFrame(rows).T

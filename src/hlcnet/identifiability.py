"""How much information do the curves carry about each indicator, and when?

Three complementary, model-free analyses used in section 3.6 of the report:

1. **Natural experiments.** Simulations that received *exactly* the same
   excitation ``U(t)`` differ only by the building, so within such a group every
   difference in ``Tint`` is caused by the building. We use them to compute an
   *information timeline*: the correlation between ``Tint(t)`` and each
   indicator, minute by minute.
2. **Linear information ceiling.** A ridge regression from the raw sequence
   (optionally truncated to its first hours) to each indicator gives a
   held-out R² per indicator: a difficulty ranking and a reference that the
   neural model must beat.
3. **PCA of the responses.** Effective dimension of the temperature curves and
   which components relate to which indicator.

Section 3.7 adds the ill-posedness diagnostics: the share of the dataset's
excitations that are shared between splits, and nearest-neighbour "twins"
(near-identical curves, different buildings) with the Delta test, a lower
estimate of the best achievable R².
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
from torch import Tensor

from hlcnet.data import SplitSpec
from hlcnet.eda import DT_MINUTES, excitation_features
from hlcnet.targets import HLC_IDX, TARGET_NAMES


# --------------------------------------------------------------------------
# 1. Natural experiments
# --------------------------------------------------------------------------
def group_identical_excitations(u: Tensor) -> tuple[Tensor, Tensor]:
    """Assign a group id to simulations whose excitation is identical.

    Args:
        u: ``(N, T)`` heating power. Signals are compared exactly.

    Returns:
        ``group`` of shape ``(N,)`` (ids sorted by decreasing group size, so
        group 0 is the largest) and ``sizes`` with the size of each group.
    """
    _, inverse, counts = torch.unique(u, dim=0, return_inverse=True, return_counts=True)
    order = torch.argsort(counts, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel())
    return rank[inverse], counts[order]


def excitation_groups_table(
    u: Tensor, group: Tensor, sizes: Tensor, top: int = 10
) -> pd.DataFrame:
    """Describe the largest groups of identical excitations."""
    feats = excitation_features(u)
    rows = []
    for g in range(min(top, sizes.numel())):
        i = int(torch.nonzero(group == g)[0])
        duty = feats["duty"][i].item()
        kind = (
            "jamais chauffé"
            if duty == 0
            else ("chauffe continue" if duty == 1 else "PRBS")
        )
        rows.append(
            {
                "groupe": g,
                "taille": int(sizes[g]),
                "type": kind,
                "taux de chauffe": round(duty, 3),
                "commutations": int(feats["n_switch"][i]),
                "énergie (kWh)": round(feats["energy_kwh"][i].item(), 1),
            }
        )
    return pd.DataFrame(rows)


def pick_groups(
    groups_tbl: pd.DataFrame, sizes: Tensor, min_size: int = 30
) -> dict[str, int]:
    """Largest group of each kind (never heated, continuous heating, PRBS).

    Groups smaller than ``min_size`` are dropped: correlations and regressions
    need enough buildings.
    """
    labels = {
        "jamais chauffé": "Jamais chauffé",
        "chauffe continue": "Chauffe continue (réponse indicielle)",
        "PRBS": "PRBS le plus fréquent",
    }
    chosen = {}
    for kind, label in labels.items():
        rows = groups_tbl[groups_tbl["type"] == kind]
        if len(rows) and int(sizes[int(rows["groupe"].iloc[0])]) >= min_size:
            chosen[label] = int(rows["groupe"].iloc[0])
    return chosen


def excitation_coverage(scan: pd.DataFrame, split: SplitSpec) -> pd.DataFrame:
    """How many distinct excitations each split has, and how many it shares with train.

    Args:
        scan: Table of :func:`hlcnet.eda.scan_dataset` (``exc_hash``, ``batch``).
        split: File assignment to train / val / test.
    """
    sets = {"train": split.train, "val": split.val, "test": split.test}
    train_hashes = set(scan.loc[scan["batch"].isin(split.train), "exc_hash"])
    rows = {}
    for name, ids in sets.items():
        h = scan.loc[scan["batch"].isin(ids), "exc_hash"]
        seen = h.isin(train_hashes)
        rows[name] = {
            "simulations": len(h),
            "excitations distinctes": h.nunique(),
            "simulations dont l'excitation existe en train (%)": 100.0 * seen.mean(),
            "excitations absentes du train": h[~seen].nunique(),
        }
    rows["total"] = {
        "simulations": len(scan),
        "excitations distinctes": scan["exc_hash"].nunique(),
        "simulations dont l'excitation existe en train (%)": float("nan"),
        "excitations absentes du train": float("nan"),
    }
    return pd.DataFrame(rows).T


def information_timeline(temp: Tensor, y: Tensor) -> pd.DataFrame:
    """Pearson correlation between ``Tint(t)`` and each indicator, for every t.

    Should be applied to a group with a single excitation, so that the
    correlation reflects the building only. Time steps where ``Tint`` is
    constant across the group (e.g. ``t = 0``) are reported as NaN.

    Args:
        temp: ``(n, T)`` temperatures of the group.
        y: ``(n, 6)`` indicators of the group.

    Returns:
        DataFrame indexed by time in hours, one column per indicator.
    """
    t64, y64 = temp.double(), y.double()
    tc = t64 - t64.mean(dim=0)
    yc = y64 - y64.mean(dim=0)
    cov = tc.T @ yc / (temp.shape[0] - 1)  # (T, 6)
    std_t = tc.std(dim=0, correction=1)
    std_y = yc.std(dim=0, correction=1)
    corr = cov / (std_t[:, None] * std_y[None, :])
    corr[std_t < 1e-9] = float("nan")
    hours = np.arange(temp.shape[1]) * DT_MINUTES / 60.0
    return pd.DataFrame(
        corr.numpy(),
        index=pd.Index(hours, name="temps (h)"),
        columns=list(TARGET_NAMES),
    )


def plot_group_responses(
    temp: Tensor, y: Tensor, title: str, max_curves: int = 400
) -> Figure:
    """Temperature curves of one excitation group, colored by HLC."""
    n = min(max_curves, temp.shape[0])
    hours = np.arange(temp.shape[1]) * DT_MINUTES / 60.0
    hlc = y[:n, HLC_IDX].numpy()
    norm = plt.Normalize(hlc.min(), hlc.max())
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in np.argsort(hlc)[::-1]:
        ax.plot(hours, temp[i].numpy(), color=cmap(norm(hlc[i])), lw=0.6, alpha=0.7)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="HLC (W/K)")
    ax.set_xlabel("temps (h)")
    ax.set_ylabel("Tint (°C)")
    ax.set_title(f"{title} (n = {temp.shape[0]})", fontsize=10)
    fig.tight_layout()
    return fig


def plot_information_timelines(timelines: Mapping[str, pd.DataFrame]) -> Figure:
    """One panel per excitation group, one line per indicator."""
    k = len(timelines)
    fig, axes = plt.subplots(1, k, figsize=(6 * k, 4), sharey=True, squeeze=False)
    for ax, (name, df) in zip(axes[0], timelines.items(), strict=True):
        for col in df.columns:
            ax.plot(df.index, df[col], label=col, lw=1.2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("temps (h)")
    axes[0, 0].set_ylabel("corr(Tint(t), indicateur)")
    axes[0, -1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 2. Linear information ceiling (ridge regression)
# --------------------------------------------------------------------------
@dataclass
class RidgeProbe:
    """Multi-output ridge regression on standardized features and targets."""

    alpha: float
    x_mean: Tensor
    x_std: Tensor
    y_mean: Tensor
    y_std: Tensor
    weight: Tensor  # (d, k)

    @classmethod
    def fit(cls, x: Tensor, y: Tensor, alpha: float) -> RidgeProbe:
        """Closed-form fit; ``alpha`` is scaled by the number of samples."""
        x, y = x.double(), y.double()
        x_mean, x_std = x.mean(0), x.std(0).clamp_min(1e-9)
        y_mean, y_std = y.mean(0), y.std(0).clamp_min(1e-9)
        xs, ys = (x - x_mean) / x_std, (y - y_mean) / y_std
        gram = xs.T @ xs + alpha * xs.shape[0] * torch.eye(xs.shape[1], dtype=xs.dtype)
        weight = torch.linalg.solve(gram, xs.T @ ys)
        return cls(alpha, x_mean, x_std, y_mean, y_std, weight)

    def predict(self, x: Tensor) -> Tensor:
        """Predict in the original target units."""
        xs = (x.double() - self.x_mean) / self.x_std
        return xs @ self.weight * self.y_std + self.y_mean


def r2_score(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Coefficient of determination per column."""
    y_true, y_pred = y_true.double(), y_pred.double()
    sse = ((y_true - y_pred) ** 2).sum(0)
    sst = ((y_true - y_true.mean(0)) ** 2).sum(0)
    return 1.0 - sse / sst


def fit_ridge_cv(
    x: Tensor,
    y: Tensor,
    alphas: Sequence[float] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    seed: int = 0,
) -> RidgeProbe:
    """Choose ``alpha`` on an inner 80/20 holdout (mean R²), then refit on all of ``x``."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    cut = int(0.8 * x.shape[0])
    tr, va = perm[:cut], perm[cut:]
    scores = [
        r2_score(y[va], RidgeProbe.fit(x[tr], y[tr], a).predict(x[va])).mean().item()
        for a in alphas
    ]
    return RidgeProbe.fit(x, y, alphas[int(np.argmax(scores))])


def sequence_features(x: Tensor, hours: float) -> Tensor:
    """Flatten the first ``hours`` of ``(N, T, 2)`` inputs into ``(N, 2 * steps)``."""
    steps = round(hours * 60.0 / DT_MINUTES)
    return x[:, :steps].reshape(x.shape[0], -1)


def ridge_information_curve(
    x_train: Tensor,
    y_train: Tensor,
    x_eval: Tensor,
    y_eval: Tensor,
    hours: Sequence[float] = (0.5, 1, 2, 4, 6, 8),
    log_cdt: bool = True,
) -> pd.DataFrame:
    """Held-out R² of a ridge probe for each indicator and observation window.

    The time constants are fitted on their log (as in the main model), and R²
    is computed in physical units.

    Returns:
        DataFrame indexed by window length (h), one column per indicator.
    """
    cdt = [i for i, n in enumerate(TARGET_NAMES) if n.startswith("tau")]

    def fwd(y: Tensor) -> Tensor:
        out = y.double().clone()
        if log_cdt:
            out[:, cdt] = out[:, cdt].log()
        return out

    def inv(z: Tensor) -> Tensor:
        out = z.clone()
        if log_cdt:
            out[:, cdt] = out[:, cdt].exp()
        return out

    rows = []
    for h in hours:
        probe = fit_ridge_cv(sequence_features(x_train, h), fwd(y_train))
        pred = inv(probe.predict(sequence_features(x_eval, h)))
        r2 = r2_score(y_eval, pred).numpy()
        rows.append(
            {
                "fenêtre (h)": h,
                "alpha": probe.alpha,
                **dict(zip(TARGET_NAMES, r2, strict=True)),
            }
        )
    return pd.DataFrame(rows).set_index("fenêtre (h)").round(3)


def plot_information_curve(df: pd.DataFrame, title: str = "") -> Figure:
    """R² of the linear probe against the observation window, per indicator."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in TARGET_NAMES:
        ax.plot(df.index, df[col], marker="o", ms=4, label=col)
    ax.set_xlabel("durée d'observation (h)")
    ax.set_ylabel("R² (jeu de validation)")
    ax.set_ylim(min(0.0, float(df[list(TARGET_NAMES)].min().min())) - 0.02, 1.0)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 3. PCA of the responses
# --------------------------------------------------------------------------
@dataclass
class PCAResult:
    """Principal components of a set of curves."""

    explained: Tensor  # (k,) explained variance ratio
    scores: Tensor  # (n, k)
    components: Tensor  # (k, T)


def pca(curves: Tensor, k: int = 20) -> PCAResult:
    """PCA by SVD of the centered ``(n, T)`` matrix."""
    c = curves.double() - curves.double().mean(0)
    u, s, vh = torch.linalg.svd(c, full_matrices=False)
    var = s**2
    k = min(k, s.numel())
    return PCAResult((var / var.sum())[:k], (u[:, :k] * s[:k]), vh[:k])


def plot_pca(res: PCAResult, y: Tensor, title: str, n_corr: int = 6) -> Figure:
    """Cumulative explained variance, first components and |corr| with indicators."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    cum = res.explained.cumsum(0).numpy()
    axes[0].plot(np.arange(1, cum.size + 1), cum, marker="o", ms=3)
    axes[0].axhline(0.99, color="k", lw=0.6, ls="--")
    axes[0].set_xlabel("nombre de composantes")
    axes[0].set_ylabel("variance expliquée cumulée")
    axes[0].set_title(title, fontsize=10)

    hours = np.arange(res.components.shape[1]) * DT_MINUTES / 60.0
    for j in range(min(3, res.components.shape[0])):
        axes[1].plot(
            hours,
            res.components[j].numpy(),
            label=f"CP{j + 1} ({res.explained[j]:.1%})",
        )
    axes[1].set_xlabel("temps (h)")
    axes[1].set_title("profils temporels des composantes", fontsize=10)
    axes[1].legend(fontsize=8)

    m = min(n_corr, res.scores.shape[1])
    s = res.scores[:, :m]
    yy = y.double()
    c = torch.corrcoef(torch.cat([s, yy], dim=1).T)[:m, m:].abs().numpy()
    im = axes[2].imshow(c, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_yticks(range(m), [f"CP{j + 1}" for j in range(m)])
    axes[2].set_xticks(range(len(TARGET_NAMES)), TARGET_NAMES, rotation=45, ha="right")
    for i in range(m):
        for j in range(len(TARGET_NAMES)):
            axes[2].text(
                j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w"
            )
    axes[2].set_title("|corrélation| composantes / indicateurs", fontsize=10)
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 4. Ill-posedness: nearest-neighbour twins and the Delta test
# --------------------------------------------------------------------------
def nearest_neighbours(
    curves: Tensor, k: int = 1, chunk: int = 1024
) -> tuple[Tensor, Tensor]:
    """RMS distance (same unit as ``curves``) to the ``k`` nearest other curves.

    Computed in float64 on centered curves, so that near-identical curves get
    an accurate distance (the float32 ``|a|² + |b|² - 2ab`` trick loses about
    0.1 °C of precision on 480-step temperature curves).

    Args:
        curves: ``(n, T)`` curves, ``n >= 2``.
        k: Number of neighbours.
        chunk: Rows processed at once (memory ``chunk x n``).

    Returns:
        ``dist`` ``(n, k)`` sorted increasingly and ``idx`` ``(n, k)``.
    """
    n, t = curves.shape
    if n < 2:
        raise ValueError("need at least two curves")
    k = min(k, n - 1)
    c = curves.double() - curves.double().mean(dim=0)
    sq = (c * c).sum(dim=1)
    dists, idxs = [], []
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d2 = (sq[a:b, None] + sq[None, :] - 2.0 * c[a:b] @ c.T).clamp_min(0.0)
        rows = torch.arange(a, b)
        d2[rows - a, rows] = float("inf")  # exclude the curve itself
        val, ind = torch.topk(d2, k, dim=1, largest=False)
        dists.append((val / t).sqrt())
        idxs.append(ind)
    return torch.cat(dists).float(), torch.cat(idxs)


def delta_test_r2(y: Tensor, nn: Tensor) -> Tensor:
    """Share of variance explained by curve resemblance alone (Delta test).

    If a curve and its nearest neighbour were the same input,
    ``E[(y_i - y_nn(i))²] = 2 Var(y | x)`` (Pi & Peterson, 1994), so
    ``1 - mean((y_i - y_nn)²) / (2 Var(y))`` would estimate the best achievable
    R². Neighbours are chosen by overall (Euclidean) resemblance, which ignores
    low-variance shape details: a model exploiting those details can do better,
    so the value is a lower estimate.
    """
    y64 = y.double()
    noise_var = (y64 - y64[nn]).pow(2).mean(dim=0) / 2.0
    return (1.0 - noise_var / y64.var(dim=0, correction=0)).float()


def ill_posedness_table(
    groups: Mapping[str, tuple[Tensor, Tensor]], threshold: float = 0.05, seed: int = 0
) -> pd.DataFrame:
    """Twins and Delta-test R² for each excitation group.

    Args:
        groups: ``name -> (curves (n, T), targets (n, 6))``, one excitation each.
        threshold: Distance (°C RMS) below which two curves are called twins,
            e.g. the sensor noise level.
        seed: Seed of the random pairs used as a reference.
    """
    hlc = HLC_IDX
    rows = {}
    for name, (curves, y) in groups.items():
        dist, idx = nearest_neighbours(curves, k=1)
        d, nn = dist[:, 0], idx[:, 0]
        twins = d < threshold
        dy = (y[:, hlc] - y[nn, hlc]).abs()
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(y), generator=g)
        partner = torch.empty_like(perm)
        partner[perm] = perm.roll(1)
        r2 = delta_test_r2(y, nn)
        rows[name] = {
            "n": len(y),
            "distance au plus proche voisin, médiane (°C)": d.median().item(),
            f"part de maisons ayant une jumelle < {threshold} °C (%)": 100.0
            * twins.float().mean().item(),
            "|ΔHLC| médian entre jumelles (W/K)": dy[twins].median().item()
            if twins.any()
            else float("nan"),
            "|ΔHLC| médian entre maisons au hasard (W/K)": (y[:, hlc] - y[partner, hlc])
            .abs()
            .median()
            .item(),
            **{
                f"R² test Delta {TARGET_NAMES[j]}": r2[j].item()
                for j in range(len(TARGET_NAMES))
            },
        }
    return pd.DataFrame(rows).T


def plot_twins(
    curves: Tensor, y: Tensor, title: str, threshold: float = 0.05
) -> Figure:
    """Nearest-neighbour distance vs HLC gap, and the most striking pair of twins.

    Left: one point per building, RMS distance to its most similar curve
    versus the HLC difference. Middle and right: among pairs closer than
    ``threshold``, the one with the largest HLC gap, and the difference
    between its two curves.
    """
    dist, idx = nearest_neighbours(curves, k=1)
    d, nn = dist[:, 0], idx[:, 0]
    dy = (y[:, HLC_IDX] - y[nn, HLC_IDX]).abs()
    hours = np.arange(curves.shape[1]) * DT_MINUTES / 60.0
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    axes[0].scatter(d.clamp_min(1e-3).numpy(), dy.numpy(), s=4, alpha=0.4)
    axes[0].set_xscale("log")
    axes[0].xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axes[0].xaxis.set_major_formatter(FormatStrFormatter("%g"))
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].axvline(threshold, color="tab:red", ls="--", lw=1, label=f"{threshold} °C")
    axes[0].set_xlabel("distance au plus proche voisin (°C RMS, log)")
    axes[0].set_ylabel("|ΔHLC| avec ce voisin (W/K)")
    axes[0].legend(fontsize=8)

    close = d < threshold
    pool = torch.nonzero(close).squeeze(1) if close.any() else torch.argsort(d)[:1]
    i = int(pool[torch.argmax(dy[pool])])
    j = int(nn[i])
    for k, color in ((i, "tab:blue"), (j, "tab:orange")):
        axes[1].plot(
            hours,
            curves[k].numpy(),
            color=color,
            lw=1.2,
            label=f"HLC = {y[k, HLC_IDX]:.0f} W/K",
        )
    axes[1].set_title(
        f"distance RMS {d[i]:.3f} °C, |ΔHLC| = {dy[i]:.0f} W/K", fontsize=9
    )
    axes[1].set_xlabel("temps (h)")
    axes[1].set_ylabel("Tint (°C)")
    axes[1].legend(fontsize=8)

    axes[2].plot(hours, (curves[i] - curves[j]).numpy(), color="k", lw=1)
    axes[2].axhspan(
        -threshold, threshold, color="tab:red", alpha=0.15, label=f"± {threshold} °C"
    )
    axes[2].set_xlabel("temps (h)")
    axes[2].set_ylabel("écart entre les deux courbes (°C)")
    axes[2].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig

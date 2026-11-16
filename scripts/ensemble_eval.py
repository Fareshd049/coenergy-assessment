"""Ensemble of already-trained models on validation (no new training).

Averages the physical predictions of several runs that share the same
validation set (the 140k-simulation runs) and compares the ensemble with each
member. Averaging preserves the physical constraints; the ensemble
uncertainty is the equal-weight Gaussian mixture (average predicted variance
plus disagreement between members).

Usage (CPU, a few minutes):
    uv run python scripts/ensemble_eval.py --data-folder data/quentin_house
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from hlcnet.analysis import ensemble_mean, mixture_std_z, predict_physical
from hlcnet.data import load_batches
from hlcnet.metrics import constraint_violations, interval_coverage, regression_metrics
from hlcnet.train import load_checkpoint
from hlcnet.utils import timer

DEFAULT_RUNS = [
    "cnn_gru_share_nll",
    "abl_head_sum",
    "abl_tcn",
    "abl_film_gru",
    "abl_tcn_gru",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval/ensemble"))
    args = parser.parse_args()

    device = torch.device("cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    _, cfg, norm = load_checkpoint(args.runs_dir / args.runs[0] / "model.pt")
    with timer("loading validation data"):
        data = load_batches(
            Path(args.data_folder),
            cfg.data.val_batches,
            cfg.data.per_batch_eval,
            seed=cfg.train.seed + 1,
            round_inputs=cfg.data.round_inputs,
        )
    print(f"validation: {len(data)} simulations")

    preds, log_vars, rows = [], [], {}
    for name in args.runs:
        model, _, model_norm = load_checkpoint(
            args.runs_dir / name / "model.pt", device
        )
        with timer(name):
            p = predict_physical(model, model_norm, data.x, device)
        preds.append(p.pred)
        log_vars.append(p.log_var)
        rows[name] = regression_metrics(p.pred, data.y)["r2"]

    ens = ensemble_mean(preds)
    rows[f"ensemble ({len(preds)} modèles)"] = regression_metrics(ens, data.y)["r2"]
    table = pd.DataFrame(rows).T
    table.to_csv(args.out / "ensemble_r2.csv")
    print("\n== R² on validation ==\n", table.round(3).to_string())
    print(
        "\n== Ensemble physical constraints ==\n",
        constraint_violations(ens).to_string(),
    )

    if all(lv is not None for lv in log_vars):
        z_true = norm.transform_y(data.y.double())
        z_preds = [norm.transform_y(p.double()) for p in preds]
        mean, std = mixture_std_z(z_preds, log_vars)  # type: ignore[arg-type]
        cov = interval_coverage(z_true, mean, std, levels=(0.5, 0.9, 0.95))
        cov.to_csv(args.out / "ensemble_coverage.csv")
        print("\n== Ensemble coverage (loss space) ==\n", cov.round(3).to_string())


if __name__ == "__main__":
    main()

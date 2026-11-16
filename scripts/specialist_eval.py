"""Generalist vs specialist models on the two constant excitations (validation).

For the never-heated and the continuously heated simulations, trains two
specialists on the houses of the *training* files that received exactly that
excitation (as in coEnergy's Phase 2), then compares them with the generalist
models on the houses of the *validation* files that received it. The test
files are not used.

Specialists (input: the 480-minute Tint curve; targets: standardized, log for
the time constants):
* ridge regression (penalty chosen on an inner holdout);
* MLP 480-256-128-6 with early stopping on 10 % of the training houses.

Usage (CPU, about 10-20 min, mostly file loading):
    uv run python scripts/specialist_eval.py --runs artifacts/runs/final_700k artifacts/runs/final_700k_noise005 --data-folder data/quentin_house
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from hlcnet.analysis import predict_physical
from hlcnet.data import Normalizer
from hlcnet.identifiability import fit_ridge_cv
from hlcnet.metrics import regression_metrics
from hlcnet.specialist import FeatureScaler, fit_mlp, load_constant_excitation_groups
from hlcnet.train import load_checkpoint
from hlcnet.utils import set_seed, timer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", type=Path, nargs="+", required=True, help="generalist run folders"
    )
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval/specialists"))
    args = parser.parse_args()

    set_seed(0)
    device = torch.device("cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    _, cfg, _ = load_checkpoint(args.runs[0] / "model.pt")
    folder = Path(args.data_folder)

    with timer("loading constant-excitation groups"):
        train_groups = load_constant_excitation_groups(folder, cfg.data.train_batches)
        val_groups = load_constant_excitation_groups(folder, cfg.data.val_batches)

    generalists = {
        run.name: load_checkpoint(run / "model.pt", device) for run in args.runs
    }
    all_tables = {}
    for group, train in train_groups.items():
        if group not in val_groups:
            continue
        val = val_groups[group]
        # 10 % of the training houses for early stopping / ridge selection
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(train), generator=g)
        cut = int(0.9 * len(train))
        fit_part, stop_part = train.subset(perm[:cut]), train.subset(perm[cut:])
        print(
            f"\n### {group}: {len(fit_part)} houses to fit, {len(stop_part)} for early stopping, {len(val)} evaluated"
        )

        norm = Normalizer.fit(fit_part)  # target transform (log taus + standardization)
        x_fit, x_stop, x_val = (
            fit_part.x[..., 0],
            stop_part.x[..., 0],
            val.x[..., 0],
        )  # U is constant here
        scaler = FeatureScaler.fit(x_fit)
        z_fit, z_stop = norm.transform_y(fit_part.y), norm.transform_y(stop_part.y)

        rows: dict[str, pd.Series] = {}
        with timer(f"{group}: ridge specialist"):
            probe = fit_ridge_cv(x_fit, z_fit)
            rows["spécialiste ridge"] = regression_metrics(
                norm.inverse_y(probe.predict(x_val).float()), val.y
            )["r2"]
        with timer(f"{group}: MLP specialist"):
            mlp = fit_mlp(scaler(x_fit), z_fit, scaler(x_stop), z_stop)
            with torch.no_grad():
                pred = norm.inverse_y(mlp(scaler(x_val)))
            rows["spécialiste MLP"] = regression_metrics(pred, val.y)["r2"]
        for name, (model, _, model_norm) in generalists.items():
            p = predict_physical(model, model_norm, val.x, device)
            rows[f"généraliste {name}"] = regression_metrics(p.pred, val.y)["r2"]

        table = pd.DataFrame(rows).T
        table.insert(0, "n évalués", len(val))
        table.to_csv(
            args.out / f"specialists_{'never' if 'jamais' in group else 'step'}.csv"
        )
        all_tables[group] = table
        print(f"\n== R² on validation houses, {group} ==\n", table.round(3).to_string())

    print(f"\nTables saved in {args.out}")


if __name__ == "__main__":
    main()

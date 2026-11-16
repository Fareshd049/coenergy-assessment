"""Critical evaluation of trained models on validation (or, once, on test).

The first run given is analysed in depth; all runs are compared on the noise
robustness curve.

Usage (CPU is fine, about a few minutes):
    uv run python scripts/evaluate.py --runs artifacts/runs/cnn_gru_share_nll artifacts/runs/abl_noise
    uv run python scripts/evaluate.py --runs artifacts/runs/final_700k --split test   # final, once

Outputs in ``artifacts/eval/<split>/``: CSV tables and PNG figures used by the
report notebook, plus ``summary.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hlcnet.analysis import (
    excitation_map,
    hlc_error_vs_energy,
    interval_width_physical,
    lognormal_mean_correction,
    noise_coverage,
    noise_robustness,
    plot_excitation_map,
    plot_noise_robustness,
    plot_reliability,
    predict_physical,
    reliability_table,
    subgroup_metrics,
    unseen_excitation_mask,
)
from hlcnet.data import load_batches
from hlcnet.metrics import constraint_violations, family_summary, regression_metrics
from hlcnet.train import load_checkpoint, save_json
from hlcnet.utils import get_device, timer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="run folders containing model.pt",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--data-folder", type=str, default=None)
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1, 0.25]
    )
    parser.add_argument(
        "--no-unseen", action="store_true", help="skip the unseen-excitation analysis"
    )
    parser.add_argument(
        "--only-noise",
        action="store_true",
        help="only the noise curves (fast comparison of models)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval"))
    args = parser.parse_args()

    device = get_device(args.device)
    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)

    model, cfg, norm = load_checkpoint(args.runs[0] / "model.pt", device)
    folder = Path(args.data_folder or cfg.data.folder)
    batches = cfg.data.val_batches if args.split == "val" else cfg.data.test_batches
    seed = cfg.train.seed + (1 if args.split == "val" else 2)
    with timer(f"loading {args.split} data"):
        data = load_batches(
            folder,
            batches,
            cfg.data.per_batch_eval,
            seed=seed,
            round_inputs=cfg.data.round_inputs,
        )
    print(
        f"{args.split}: {len(data)} simulations | primary model: {args.runs[0].name} on {device}"
    )
    u = data.x[..., 1]

    summary: dict[str, object] = {
        "run": args.runs[0].name,
        "split": args.split,
        "n": len(data),
    }
    if not args.only_noise:
        # ---- 1. Accuracy, constraints, bias correction ----------------------
        with timer("prediction"):
            p = predict_physical(model, norm, data.x, device)
        per_target = regression_metrics(p.pred, data.y)
        summary.update(
            {
                "per_target": per_target,
                "families": family_summary(per_target),
                "constraints": constraint_violations(p.pred),
            }
        )
        per_target.to_csv(out / "per_target.csv")
        print("\n== Accuracy ==\n", per_target.round(3).to_string())

        if p.log_var is not None:
            corrected = lognormal_mean_correction(p.pred, p.log_var, norm)
            bias = pd.DataFrame(
                {
                    "biais médiane exp(mu)": regression_metrics(p.pred, data.y)["bias"],
                    "biais moyenne exp(mu + s²/2)": regression_metrics(
                        corrected, data.y
                    )["bias"],
                    "rmse médiane": regression_metrics(p.pred, data.y)["rmse"],
                    "rmse moyenne": regression_metrics(corrected, data.y)["rmse"],
                }
            ).loc[["tau_b", "tau_n", "tau_inf"]]
            bias.to_csv(out / "retransformation_bias.csv")
            summary["retransformation_bias"] = bias
            print(
                "\n== Retransformation bias of the time constants ==\n",
                bias.round(3).to_string(),
            )

            # ---- 2. Calibration and sharpness -------------------------------
            z_true, z_pred = (
                norm.transform_y(data.y.double()),
                norm.transform_y(p.pred.double()),
            )
            rel = reliability_table(z_true, z_pred, p.log_var)
            rel.to_csv(out / "reliability.csv")
            plot_reliability(rel).savefig(out / "reliability.png", dpi=130)
            widths = interval_width_physical(p.pred, p.log_var, norm, level=0.95)
            widths.to_csv(out / "interval_width_95.csv")
            summary["reliability"] = rel
            summary["median_width_95"] = widths
            print("\n== Coverage (rows: nominal level) ==\n", rel.round(3).to_string())
            print(
                "\n== Median width of 95% intervals (W/K, h) ==\n",
                widths.round(2).to_string(),
            )

        # ---- 3. Sub-groups and energy ---------------------------------------
        sub = subgroup_metrics(p.pred, data.y, u, metric="r2")
        sub_rmse = subgroup_metrics(p.pred, data.y, u, metric="rmse")
        sub.to_csv(out / "subgroups_r2.csv")
        sub_rmse.to_csv(out / "subgroups_rmse.csv")
        hlc_error_vs_energy(p.pred, data.y, u).savefig(
            out / "hlc_error_vs_energy.png", dpi=130
        )
        summary["subgroups_r2"] = sub
        print("\n== R² by sub-group ==\n", sub.round(3).to_string())

        # ---- 4. Excitation design -------------------------------------------
        for target, unit in (("HLC", "W/K"), ("HLC_ext", "W/K")):
            emap = excitation_map(
                p.pred, data.y, data.meta, data.meta_names, target=target
            )
            emap.to_csv(out / f"excitation_map_{target}.csv")
            plot_excitation_map(emap, target, unit).savefig(
                out / f"excitation_map_{target}.png", dpi=130
            )
        print("\nExcitation maps saved.")

        # ---- 5. Unseen excitations ------------------------------------------
        if not args.no_unseen:
            with timer("loading training excitations"):
                train = load_batches(
                    folder,
                    cfg.data.train_batches,
                    cfg.data.per_batch_train,
                    seed=cfg.train.seed,
                )
            unseen = unseen_excitation_mask(train.x[..., 1], u)
            del train
            rows = {}
            for label, m in (
                ("excitations vues", ~unseen),
                ("excitations inédites", unseen),
            ):
                if m.sum() >= 20:
                    rows[label] = {
                        "n": int(m.sum()),
                        **regression_metrics(p.pred[m], data.y[m])["r2"].to_dict(),
                    }
            unseen_df = pd.DataFrame(rows).T
            unseen_df.to_csv(out / "unseen_excitations_r2.csv")
            summary["unseen_excitations_r2"] = unseen_df
            print(
                "\n== R² on seen vs unseen excitations ==\n",
                unseen_df.round(3).to_string(),
            )

    # ---- 6. Noise robustness (all runs) ---------------------------------
    curves = {}
    for run in args.runs:
        m, _, n = load_checkpoint(run / "model.pt", device)
        with timer(f"noise robustness {run.name}"):
            curves[run.name] = noise_robustness(
                m, n, data.x, data.y, device, levels=args.noise_levels
            )
        curves[run.name].to_csv(out / f"noise_robustness_{run.name}.csv")
        print(
            f"\n== R² vs noise: {run.name} ==\n", curves[run.name].round(3).to_string()
        )
        cov = noise_coverage(m, n, data.x, data.y, device, levels=args.noise_levels)
        if not cov.empty:
            cov.to_csv(out / f"noise_coverage95_{run.name}.csv")
            print(
                f"\n== 95% coverage vs noise: {run.name} ==\n", cov.round(3).to_string()
            )
    plot_noise_robustness(curves).savefig(out / "noise_robustness.png", dpi=130)
    summary["noise_robustness"] = curves

    save_json(summary, out / "summary.json")
    print(f"\nAll outputs in {out}")


if __name__ == "__main__":
    main()

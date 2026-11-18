"""Evaluate the final model excitation by excitation (report, section 7.4 b).

For every excitation received by at least ``--min-count`` simulations in both
the validation and the test files, computes on HLC and HLC_ext the realized
RMSE, the uncertainty predicted by the model and the 95 % coverage. Excitations
are ranked on validation; the ranking is judged on test.

Usage (CPU, about 10-15 minutes: 300 000 simulations are predicted):
    uv run python scripts/excitation_eval.py --data-folder data/quentin_house
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hlcnet.analysis import predict_physical
from hlcnet.data import load_batches
from hlcnet.eda import excitation_hash
from hlcnet.excitation_design import (
    excitation_descriptors,
    first_signal_per_excitation,
    join_splits,
    per_excitation_table,
    physical_std,
    plot_excitation_ranking,
    plot_excitation_signals,
    ranking_summary,
)
from hlcnet.targets import TARGET_NAMES
from hlcnet.train import load_checkpoint
from hlcnet.utils import get_device, timer

NAMES = ("HLC", "HLC_ext")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=Path, default=Path("artifacts/runs/final_700k_noise005")
    )
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval/excitations"))
    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    model, cfg, norm = load_checkpoint(args.run / "model.pt", device)
    idx = [TARGET_NAMES.index(n) for n in NAMES]

    tables: dict[str, pd.DataFrame] = {}
    overall: dict[str, dict[str, float]] = {}
    signals: dict[int, np.ndarray] = {}
    for split, ids in (("val", cfg.data.val_batches), ("test", cfg.data.test_batches)):
        with timer(f"loading {split}"):
            data = load_batches(
                Path(args.data_folder),
                ids,
                per_batch=None,
                round_inputs=cfg.data.round_inputs,
            )
        with timer(f"prediction {split} ({len(data)} simulations)"):
            p = predict_physical(model, norm, data.x, device)
        if p.log_var is None:
            raise ValueError("the model must predict its uncertainty")
        hashes = excitation_hash(data.x[..., 1]).numpy()
        true, pred = data.y[:, idx].numpy(), p.pred[:, idx].numpy()
        tables[split] = per_excitation_table(
            hashes,
            true,
            pred,
            physical_std(p.log_var, norm, NAMES),
            NAMES,
            args.min_count,
        )
        overall[split] = {
            n: float(np.sqrt(((pred[:, j] - true[:, j]) ** 2).mean()))
            for j, n in enumerate(NAMES)
        }
        if split == "val":
            signals = first_signal_per_excitation(hashes, data.x[..., 1])
        del data, p

    merged = join_splits(tables["val"], tables["test"]).join(
        excitation_descriptors(signals)
    )
    merged.to_csv(args.out / "excitations.csv")

    summary = pd.DataFrame({n: ranking_summary(merged, n) for n in NAMES})
    summary.loc["RMSE test, toutes simulations (référence)"] = overall["test"]
    summary.to_csv(args.out / "ranking_summary.csv")
    print(
        "\n== Excitations, ranked on validation, judged on test ==\n",
        summary.round(3).to_string(),
    )

    cols = [
        "n_val",
        "n_test",
        "sigma_HLC_val",
        "rmse_HLC_test",
        "couverture95_HLC_test",
        "taux de chauffe",
        "énergie (kWh)",
        "commutations",
        "première chauffe (h)",
    ]
    ranked = merged.sort_values("sigma_HLC_val")[cols]
    print(
        "\n== 5 lowest predicted sigma (HLC) ==\n", ranked.head(5).round(2).to_string()
    )
    print(
        "\n== 5 highest predicted sigma (HLC) ==\n", ranked.tail(5).round(2).to_string()
    )

    plot_excitation_ranking(merged, NAMES).savefig(
        args.out / "excitation_ranking.png", dpi=130
    )
    plot_excitation_signals(merged, signals, "HLC").savefig(
        args.out / "excitation_signals.png", dpi=130
    )
    print(f"\nOutputs in {args.out}")


if __name__ == "__main__":
    main()

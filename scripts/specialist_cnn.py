"""CNN-GRU specialists: train the final architecture on ONE informative excitation.

Protocol (report, section 7.4 c):

1. **Choose** the excitations with the lowest predicted uncertainty on HLC,
   on validation (``artifacts/eval/excitations/excitations.csv``, written by
   ``scripts/excitation_eval.py``).
2. **Train** a CNN-GRU with the final configuration (same architecture, loss
   and 0.05 °C noise) only on the training-file simulations that received
   this excitation; early stopping on the validation-file ones.
3. **Judge** on the test-file simulations that received it, against the
   generalist trained on all 700 000 simulations.

The specialist reuses the generalist's normalizer (fitted on the training
files), so both models see identically scaled inputs.

Usage (CPU, about 15-25 minutes):
    uv run python scripts/specialist_cnn.py --data-folder data/quentin_house
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd
import torch

from hlcnet.analysis import predict_physical
from hlcnet.data import SimData, load_batches, make_loaders
from hlcnet.eda import excitation_hash
from hlcnet.metrics import regression_metrics
from hlcnet.models import ModelStats, build_model
from hlcnet.train import fit, load_checkpoint
from hlcnet.utils import get_device, set_seed, timer


def collect(
    folder: Path, ids: list[int], hashes: list[int], round_inputs: bool
) -> dict[int, SimData]:
    """Simulations of files ``ids`` whose excitation fingerprint is in ``hashes``."""
    parts: dict[int, list[SimData]] = {h: [] for h in hashes}
    for i in ids:
        d = load_batches(folder, [i], round_inputs=round_inputs)
        fp = excitation_hash(d.x[..., 1])
        for h in hashes:
            m = torch.nonzero(fp == h).squeeze(1)
            if m.numel():
                parts[h].append(d.subset(m))
        del d
    return {h: SimData.concat(p) for h, p in parts.items() if p}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generalist", type=Path, default=Path("artifacts/runs/final_700k_noise005")
    )
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument(
        "--excitations",
        type=Path,
        default=Path("artifacts/eval/excitations/excitations.csv"),
    )
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/eval/specialists_cnn")
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    generalist, cfg, norm = load_checkpoint(args.generalist / "model.pt", device)
    folder = Path(args.data_folder)

    exc = pd.read_csv(args.excitations, index_col=0)
    chosen = exc.sort_values("sigma_HLC_val").head(args.top)
    hashes = [int(h) for h in chosen.index]
    print("Chosen on validation (lowest predicted sigma on HLC):")
    print(
        chosen[["n_val", "n_test", "sigma_HLC_val", "taux de chauffe", "énergie (kWh)"]]
        .round(2)
        .to_string()
    )

    with timer("collecting the simulations of the chosen excitations"):
        splits = {
            name: collect(folder, ids, hashes, cfg.data.round_inputs)
            for name, ids in (
                ("train", cfg.data.train_batches),
                ("val", cfg.data.val_batches),
                ("test", cfg.data.test_batches),
            )
        }

    train_cfg = dataclasses.replace(
        cfg.train,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        num_workers=0,
    )
    rows = {}
    for rank, h in enumerate(hashes, start=1):
        train, val, test = splits["train"][h], splits["val"][h], splits["test"][h]
        label = f"excitation {rank} ({chosen.loc[h, 'énergie (kWh)']:.1f} kWh)"
        print(f"\n### {label}: train {len(train)}, val {len(val)}, test {len(test)}")
        set_seed(cfg.train.seed)
        loaders = make_loaders(
            {"train": train, "val": val}, norm, batch_size=train_cfg.batch_size
        )
        stats = ModelStats.from_data(norm.transform_x(train.x), train.y, cfg.model.head)
        specialist = build_model(cfg.model, stats)
        with timer(f"training the specialist on {len(train)} simulations"):
            history = fit(
                specialist, loaders, norm, cfg.loss, train_cfg, device, verbose=True
            )
        print(f"best epoch {history.best_epoch}")

        for name, model in (
            ("spécialiste CNN-GRU", specialist),
            ("généraliste (700 000 sim.)", generalist),
        ):
            p = predict_physical(model, norm, test.x, device)
            m = regression_metrics(p.pred, test.y)
            rows[(label, name)] = {
                "simulations d'entraînement": len(train) if "spéc" in name else 700_000,
                "simulations de test": len(test),
                **{f"RMSE {t}": m.loc[t, "rmse"] for t in ("HLC", "HLC_ext", "tau_n")},
                **{f"R² {t}": m.loc[t, "r2"] for t in ("HLC", "HLC_ext", "tau_n")},
            }

    table = pd.DataFrame(rows).T
    table.index.names = ["excitation", "modèle"]
    table.to_csv(args.out / "specialists_cnn.csv")
    print(
        "\n== Test, simulations that received each excitation ==\n",
        table.round(3).to_string(),
    )
    print(f"\nOutputs in {args.out}")


if __name__ == "__main__":
    main()

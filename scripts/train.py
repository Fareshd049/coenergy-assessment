"""Train one experiment described by a YAML config, then evaluate it on validation.

Usage:
    uv run python scripts/train.py --config configs/debug.yaml
    uv run python scripts/train.py --config configs/full.yaml --data-folder /kaggle/input/quentin-house

Outputs (in ``artifacts/runs/<name>/``): ``config.yaml``, ``model.pt``
(checkpoint), ``history.csv``, ``val_metrics.json`` and ``val_predictions.pt``.
The test set is never touched here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hlcnet.config import ExperimentConfig
from hlcnet.data import Normalizer, load_batches, make_loaders
from hlcnet.metrics import (
    constraint_violations,
    family_summary,
    interval_coverage,
    regression_metrics,
)
from hlcnet.models import ModelStats, build_model, count_parameters
from hlcnet.train import fit, predict, save_checkpoint, save_json, selection_score
from hlcnet.utils import get_device, set_seed, timer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-folder", type=str, default=None, help="overrides data.folder"
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.data_folder:
        cfg.data.folder = args.data_folder
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    out_dir = args.out / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.yaml")
    print(f"Experiment '{cfg.name}' on {device} -> {out_dir}")

    # ---- Data ----------------------------------------------------------
    folder = Path(cfg.data.folder)
    with timer("loading data"):
        train = load_batches(
            folder,
            cfg.data.train_batches,
            cfg.data.per_batch_train,
            seed=cfg.train.seed,
        )
        val = load_batches(
            folder,
            cfg.data.val_batches,
            cfg.data.per_batch_eval,
            seed=cfg.train.seed + 1,
        )
    print(f"train: {len(train)} simulations | val: {len(val)} simulations")

    normalizer = Normalizer.fit(train, log_cdt=cfg.data.log_cdt)
    loaders = make_loaders(
        {"train": train, "val": val},
        normalizer,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        seed=cfg.train.seed,
    )

    # ---- Model ---------------------------------------------------------
    stats = ModelStats.from_data(
        normalizer.transform_x(train.x), train.y, cfg.model.head
    )
    model = build_model(cfg.model, stats)
    print(
        f"model: {cfg.model.arch} / head {cfg.model.head} | {count_parameters(model):,} parameters"
    )

    # ---- Training ------------------------------------------------------
    with timer("training"):
        history = fit(
            model, loaders, normalizer, cfg.loss, cfg.train, device, out_dir=out_dir
        )
    history.to_frame().to_csv(out_dir / "history.csv", index=False)
    save_checkpoint(out_dir / "model.pt", model, cfg, normalizer, stats)
    print(f"best epoch: {history.best_epoch} (val score {history.best_score:.4f})")

    # ---- Validation evaluation ----------------------------------------
    res = predict(model, loaders["val"], normalizer, device, keep_attention=False)
    per_target = regression_metrics(res.pred, res.true)
    summary = family_summary(per_target)
    violations = constraint_violations(res.pred)
    report: dict[str, object] = {
        "name": cfg.name,
        "n_train": len(train),
        "n_val": len(val),
        "best_epoch": history.best_epoch,
        "val_score": selection_score(res.pred, res.true, normalizer, torch.ones(6) / 6),
        "per_target": per_target,
        "families": summary,
        "constraints": violations,
    }
    if res.log_var is not None:
        z_pred = normalizer.transform_y(res.pred.double())
        z_true = normalizer.transform_y(res.true.double())
        report["coverage"] = interval_coverage(
            z_true, z_pred, torch.exp(0.5 * res.log_var.double())
        )
    save_json(report, out_dir / "val_metrics.json")
    torch.save(
        {
            "pred": res.pred,
            "true": res.true,
            "log_var": res.log_var,
            "meta": val.meta,
            "batch_id": val.batch_id,
        },
        out_dir / "val_predictions.pt",
    )

    print("\n== Validation, per indicator ==")
    print(per_target.round(3).to_string())
    print("\n== Validation, per family ==")
    print(summary.to_string())
    print("\n== Physical constraints ==")
    print(violations.to_string())
    if "coverage" in report:
        print("\n== Interval coverage (should be close to the nominal level) ==")
        print(report["coverage"].round(3).to_string())  # type: ignore[union-attr]


if __name__ == "__main__":
    main()

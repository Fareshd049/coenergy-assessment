"""ROCKET + ridge baseline, evaluated on validation like the neural models.

Uses the ``data`` section of a YAML config (same training / validation
simulations as the neural runs), random convolutional features and a ridge
regression whose penalty is chosen on an inner holdout. Time constants are
regressed on their log, as for the neural models.

Usage:
    uv run python scripts/rocket_baseline.py --config configs/cnn_gru_share_nll.yaml --n-kernels 2000

Outputs in ``artifacts/runs/rocket_ridge/``: ``val_metrics.json`` and
``val_predictions.pt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hlcnet.config import ExperimentConfig
from hlcnet.data import Normalizer, load_batches
from hlcnet.identifiability import fit_ridge_cv
from hlcnet.metrics import constraint_violations, family_summary, regression_metrics
from hlcnet.models import (
    InputChannels,
    ModelStats,
    params_to_targets,
    targets_to_params,
)
from hlcnet.rocket import RocketFeatures, rocket_transform
from hlcnet.train import save_json
from hlcnet.utils import get_device, set_seed, timer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-folder", type=str, default=None)
    parser.add_argument("--n-kernels", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.data_folder:
        cfg.data.folder = args.data_folder
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    out_dir = args.out / "rocket_ridge"
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(
        f"train: {len(train)} | val: {len(val)} | {args.n_kernels} kernels on {device}"
    )

    norm = Normalizer.fit(train)
    xtr, xva = norm.transform_x(train.x), norm.transform_x(val.x)
    stats = ModelStats.from_data(xtr, train.y, "free")
    channels = InputChannels(use_derivative=True, deriv_std=stats.deriv_std)
    rocket = RocketFeatures(
        channels.n_channels,
        train.x.shape[1],
        n_kernels=args.n_kernels,
        seed=cfg.train.seed,
    )

    with timer("ROCKET features"):
        ftr = rocket_transform(xtr, rocket, channels, device)
        fva = rocket_transform(xva, rocket, channels, device)

    # Same target space as the "free" head: HLC, ext, mit, log tau_b, log tau_n, log tau_inf
    with timer("ridge fit"):
        probe = fit_ridge_cv(
            ftr, targets_to_params(train.y, "free"), alphas=(1e-4, 1e-3, 1e-2, 1e-1)
        )
    pred = params_to_targets(probe.predict(fva), "free").float()
    print(f"selected alpha: {probe.alpha}")

    per_target = regression_metrics(pred, val.y)
    report = {
        "name": "rocket_ridge",
        "n_train": len(train),
        "n_val": len(val),
        "n_kernels": args.n_kernels,
        "alpha": probe.alpha,
        "per_target": per_target,
        "families": family_summary(per_target),
        "constraints": constraint_violations(pred),
    }
    save_json(report, out_dir / "val_metrics.json")
    torch.save(
        {"pred": pred, "true": val.y, "meta": val.meta, "batch_id": val.batch_id},
        out_dir / "val_predictions.pt",
    )

    print("\n== Validation, per indicator ==")
    print(per_target.round(3).to_string())
    print("\n== Physical constraints ==")
    print(report["constraints"].to_string())  # type: ignore[union-attr]


if __name__ == "__main__":
    main()

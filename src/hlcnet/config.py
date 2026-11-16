"""Experiment configuration, loaded from a YAML file.

One YAML file fully describes one experiment (data subset, model, loss and
training), which makes every result in the report reproducible::

    data:  {per_batch_train: 10000, ...}
    model: {arch: cnn_gru, head: share, ...}
    loss:  {kind: nll, beta: 0.5}
    train: {epochs: 30, lr: 0.002, ...}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from hlcnet.data import SplitSpec
from hlcnet.losses import LossConfig
from hlcnet.models import ModelConfig


@dataclass
class DataConfig:
    """Which simulations are used."""

    folder: str = "data/quentin_house"
    train_batches: list[int] = field(default_factory=lambda: list(SplitSpec().train))
    val_batches: list[int] = field(default_factory=lambda: list(SplitSpec().val))
    test_batches: list[int] = field(default_factory=lambda: list(SplitSpec().test))
    per_batch_train: int | None = 10_000
    per_batch_eval: int | None = 5_000
    log_cdt: bool = True
    round_inputs: bool = True

    def split_spec(self) -> SplitSpec:
        return SplitSpec(
            tuple(self.train_batches), tuple(self.val_batches), tuple(self.test_batches)
        )


@dataclass
class TrainConfig:
    """Optimization settings."""

    epochs: int = 30
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    mse_warmup_epochs: int = 2
    grad_clip: float = 1.0
    patience: int = 6
    noise_temp_std: float = 0.0
    noise_temp_max: float = (
        0.0  # if > 0: per-sample std drawn uniformly in [0, noise_temp_max]
    )
    amp: bool = True
    device: str = "auto"
    num_workers: int = 0
    seed: int = 0


@dataclass
class ExperimentConfig:
    """Full description of one experiment."""

    name: str = "experiment"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Build from a nested dict; unknown keys raise an error (typos are caught)."""

        def build(kind: type, values: dict[str, Any] | None) -> Any:
            values = values or {}
            allowed = {f.name for f in fields(kind)}
            unknown = set(values) - allowed
            if unknown:
                raise ValueError(f"unknown keys for {kind.__name__}: {sorted(unknown)}")
            return kind(**values)

        return cls(
            name=d.get("name", "experiment"),
            data=build(DataConfig, d.get("data")),
            model=build(ModelConfig, d.get("model")),
            loss=build(LossConfig, d.get("loss")),
            train=build(TrainConfig, d.get("train")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

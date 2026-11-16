"""Memory-safe data loading, splitting and normalization.

Facts established on ``batch_1.pt`` (see ``scripts/inspect_batch.py``):

* 20 files of 50 000 simulations, ~196 MB each, 480 time steps per series.
* Inputs ``indoor_temperature`` and ``indoor_U`` are float32, no NaN/inf.
* ``Tint(t=0)`` is identical for every simulation (24.62 °C).
* ``U`` lies in ``[0, 7000]`` W and is never negative.
* ``HLC == HLC_ext + HLC_mit`` up to float32 rounding (max gap 3e-5 W/K).

Batches are loaded one at a time, optionally subsampled, and stripped of what
is not needed, so peak memory stays around one batch (~400 MB).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from hlcnet.targets import CDT_IDX, assemble_targets

#: Sensor resolution applied in coEnergy's loading example.
TEMP_DECIMALS: int = 2
U_DECIMALS: int = 1


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------
@dataclass
class SimData:
    """A set of simulations held in memory.

    Attributes:
        x: ``(N, T, 2)`` inputs, channels ``[Tint, U]``.
        y: ``(N, 6)`` targets in :data:`hlcnet.targets.TARGET_NAMES` order.
        meta: ``(N, M)`` excitation parameters (``solicitation``), kept for
            error analysis only, never used as model input.
        meta_names: Names of the ``meta`` columns.
        batch_id: ``(N,)`` index of the source file (position in sorted order).
    """

    x: Tensor
    y: Tensor
    meta: Tensor
    meta_names: list[str]
    batch_id: Tensor

    def __len__(self) -> int:
        return self.x.shape[0]

    def subset(self, idx: Tensor) -> SimData:
        """Return the rows selected by ``idx``."""
        return SimData(
            self.x[idx],
            self.y[idx],
            self.meta[idx],
            self.meta_names,
            self.batch_id[idx],
        )

    @staticmethod
    def concat(parts: Sequence[SimData]) -> SimData:
        """Concatenate several :class:`SimData` along the sample axis."""
        if not parts:
            raise ValueError("Nothing to concatenate")
        return SimData(
            torch.cat([p.x for p in parts]),
            torch.cat([p.y for p in parts]),
            torch.cat([p.meta for p in parts]),
            parts[0].meta_names,
            torch.cat([p.batch_id for p in parts]),
        )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def list_batches(folder: Path) -> list[Path]:
    """List ``*.pt`` files sorted by the number in their name.

    Plain ``sorted`` would put ``batch_10`` before ``batch_2``.
    """

    def key(p: Path) -> tuple[int, str]:
        nums = re.findall(r"\d+", p.stem)
        return (int(nums[-1]) if nums else -1, p.name)

    files = sorted(Path(folder).glob("*.pt"), key=key)
    if not files:
        raise FileNotFoundError(f"No .pt files in {folder}")
    return files


def _read_batch(path: Path) -> object:
    """Load one raw batch (imports coEnergy's helper lazily)."""
    from hlcnet import util_assessement as ua  # registers the pickle package

    return ua.load_single_batch(path)


def load_batches(
    folder: Path,
    batch_ids: Sequence[int] | None = None,
    per_batch: int | None = None,
    time_stride: int = 1,
    seed: int = 0,
    round_inputs: bool = True,
) -> SimData:
    """Load a subset of the simulations, one file at a time.

    Args:
        folder: Folder containing the batch files.
        batch_ids: Positions in :func:`list_batches` order (0-based);
            ``None`` loads every file.
        per_batch: Number of simulations sampled per file; ``None`` keeps all.
        time_stride: Keep one time step out of ``time_stride``.
        seed: Seed of the per-file subsampling.
        round_inputs: Round ``Tint`` to 0.01 °C and ``U`` to 0.1 W (sensor
            resolution, as in coEnergy's loading example). ``False`` keeps the
            full simulator precision, to measure the effect of quantization.

    Returns:
        A :class:`SimData` with float32 tensors.
    """
    files = list_batches(folder)
    ids = list(range(len(files))) if batch_ids is None else list(batch_ids)
    gen = torch.Generator().manual_seed(seed)
    parts: list[SimData] = []

    for i in ids:
        b = _read_batch(files[i])
        n = b.HLC.shape[0]
        idx = (
            torch.randperm(n, generator=gen)[:per_batch]
            if per_batch
            else torch.arange(n)
        )

        temp, u = b.indoor_temperature[idx], b.indoor_U[idx]
        if round_inputs:
            temp = temp.round(decimals=TEMP_DECIMALS)
            u = u.round(decimals=U_DECIMALS)
        x = torch.stack([temp, u], dim=-1)[:, ::time_stride].float().contiguous()
        y = assemble_targets(b.HLC[idx], b.Hb[idx], b.Hb_labels, b.CDT[idx])
        parts.append(
            SimData(
                x=x,
                y=y,
                meta=b.solicitation[idx].float().clone(),
                meta_names=list(b.solicitation_field_names),
                batch_id=torch.full((len(idx),), i, dtype=torch.long),
            )
        )
        del b  # free the full batch before reading the next one

    return SimData.concat(parts)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitSpec:
    """Assignment of batch files to train / val / test.

    Splitting by file guarantees that no simulation appears in two sets.
    """

    train: tuple[int, ...] = tuple(range(14))
    val: tuple[int, ...] = (14, 15, 16)
    test: tuple[int, ...] = (17, 18, 19)

    def __post_init__(self) -> None:
        sets = [set(self.train), set(self.val), set(self.test)]
        if (sets[0] & sets[1]) or (sets[0] & sets[2]) or (sets[1] & sets[2]):
            raise ValueError("train/val/test batch ids overlap")


#: 14 / 3 / 3 files (700k / 150k / 150k simulations available).
DEFAULT_SPLIT = SplitSpec()


def load_splits(
    folder: Path,
    spec: SplitSpec = DEFAULT_SPLIT,
    per_batch_train: int | None = 10_000,
    per_batch_eval: int | None = 5_000,
    time_stride: int = 1,
    seed: int = 0,
) -> dict[str, SimData]:
    """Load the train / val / test sets defined by ``spec``."""
    return {
        "train": load_batches(folder, spec.train, per_batch_train, time_stride, seed),
        "val": load_batches(folder, spec.val, per_batch_eval, time_stride, seed + 1),
        "test": load_batches(folder, spec.test, per_batch_eval, time_stride, seed + 2),
    }


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
@dataclass
class Normalizer:
    """Standardizes inputs and targets with statistics fitted on train only.

    Inputs are standardized per channel. Targets are optionally
    log-transformed for the time constants (positive, right-skewed), then
    standardized per column.
    """

    x_mean: Tensor
    x_std: Tensor
    y_mean: Tensor
    y_std: Tensor
    log_cdt: bool = True

    @classmethod
    def fit(cls, data: SimData, log_cdt: bool = True, eps: float = 1e-6) -> Normalizer:
        """Fit the statistics on ``data`` (the training set)."""
        y = _log_cdt(data.y) if log_cdt else data.y
        return cls(
            x_mean=data.x.mean(dim=(0, 1)),
            x_std=data.x.std(dim=(0, 1)) + eps,
            y_mean=y.mean(dim=0),
            y_std=y.std(dim=0) + eps,
            log_cdt=log_cdt,
        )

    def transform_x(self, x: Tensor) -> Tensor:
        """Physical inputs -> model inputs."""
        return (x - self.x_mean.to(x)) / self.x_std.to(x)

    def transform_y(self, y: Tensor) -> Tensor:
        """Physical targets -> standardized targets."""
        y = _log_cdt(y) if self.log_cdt else y
        return (y - self.y_mean.to(y)) / self.y_std.to(y)

    def inverse_y(self, z: Tensor) -> Tensor:
        """Standardized targets -> physical units (W/K, h)."""
        y = z * self.y_std.to(z) + self.y_mean.to(z)
        if self.log_cdt:
            y = y.clone()
            y[:, list(CDT_IDX)] = y[:, list(CDT_IDX)].exp()
        return y

    def state_dict(self) -> dict[str, Tensor | bool]:
        """Serializable state, saved alongside model checkpoints."""
        return {
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "log_cdt": self.log_cdt,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor | bool]) -> Normalizer:
        """Rebuild a normalizer saved with :meth:`state_dict`."""
        return cls(**state)  # type: ignore[arg-type]


def _log_cdt(y: Tensor) -> Tensor:
    out = y.clone()
    out[:, list(CDT_IDX)] = out[:, list(CDT_IDX)].log()
    return out


# --------------------------------------------------------------------------
# Augmentation and loaders
# --------------------------------------------------------------------------
def add_sensor_noise(
    x: Tensor,
    temp_std: float = 0.1,
    u_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Add Gaussian noise to physical inputs to mimic sensor errors.

    The paper calibrates noise on the real sensors (±0.25 °C typical for the
    temperature probe); ``temp_std`` is in °C and ``u_std`` in W.
    """
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    scale = torch.tensor([temp_std, u_std], device=x.device, dtype=x.dtype)
    return x + noise * scale


def make_loaders(
    splits: dict[str, SimData],
    normalizer: Normalizer,
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 0,
) -> dict[str, DataLoader[tuple[Tensor, ...]]]:
    """Wrap normalized splits in DataLoaders (train is shuffled)."""
    loaders: dict[str, DataLoader[tuple[Tensor, ...]]] = {}
    for name, d in splits.items():
        ds = TensorDataset(normalizer.transform_x(d.x), normalizer.transform_y(d.y))
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed) if name == "train" else None,
            drop_last=False,
        )
    return loaders

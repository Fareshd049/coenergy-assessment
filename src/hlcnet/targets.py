"""Target layout and indicator families.

The canonical target vector used everywhere in the project is::

    y = [HLC, HLC_ext, HLC_mit, tau_b, tau_n, tau_inf]

The raw files store ``Hb`` with labels ``Hb_labels`` (observed order:
``['HLC_mit', 'HLC_ext']``); :func:`assemble_targets` reorders by label name so
the canonical layout never depends on the storage order.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

TARGET_NAMES: tuple[str, ...] = (
    "HLC",
    "HLC_ext",
    "HLC_mit",
    "tau_b",
    "tau_n",
    "tau_inf",
)
HB_NAMES: tuple[str, ...] = ("HLC_ext", "HLC_mit")
CDT_NAMES: tuple[str, ...] = ("tau_b", "tau_n", "tau_inf")

HLC_IDX: int = 0
HB_IDX: tuple[int, ...] = (1, 2)
CDT_IDX: tuple[int, ...] = (3, 4, 5)

#: Indicator families used for reporting metrics (unit in the key's comment).
FAMILIES: dict[str, tuple[int, ...]] = {
    "HLC": (HLC_IDX,),  # W/K
    "Hb": HB_IDX,  # W/K
    "CDT": CDT_IDX,  # h
}
UNITS: dict[str, str] = {"HLC": "W/K", "Hb": "W/K", "CDT": "h"}


def assemble_targets(
    hlc: Tensor, hb: Tensor, hb_labels: Sequence[str], cdt: Tensor
) -> Tensor:
    """Build the canonical ``(N, 6)`` target matrix from the raw fields.

    Args:
        hlc: ``(N,)`` global heat loss coefficient.
        hb: ``(N, 2)`` boundary decomposition, columns ordered as ``hb_labels``.
        hb_labels: Names of the ``hb`` columns, e.g. ``['HLC_mit', 'HLC_ext']``.
        cdt: ``(N, 3)`` time constants ordered ``(tau_b, tau_n, tau_inf)``.

    Returns:
        ``(N, 6)`` float32 tensor in :data:`TARGET_NAMES` order.
    """
    labels = list(hb_labels)
    missing = set(HB_NAMES) - set(labels)
    if missing:
        raise ValueError(f"Hb_labels {labels} is missing {sorted(missing)}")
    cols = [hb[:, labels.index(name)] for name in HB_NAMES]
    return torch.stack([hlc, *cols, *cdt.unbind(dim=1)], dim=1).float()


def hlc_consistency_gap(y: Tensor) -> Tensor:
    """Absolute violation of ``HLC = HLC_ext + HLC_mit`` for each row."""
    return (y[:, HLC_IDX] - y[:, list(HB_IDX)].sum(dim=1)).abs()

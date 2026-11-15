"""Multi-task loss, computed in the normalized target space.

The model predicts physical indicators; they are mapped with the same
:class:`~hlcnet.data.Normalizer` as the targets (standardization, log for the
time constants) before computing the loss, so every indicator contributes on a
comparable scale.

Three kinds of loss:

* ``"mse"``: squared error, i.e. Gaussian maximum likelihood with a constant
  variance (homoscedastic, like ordinary least squares);
* ``"huber"``: robust variant, quadratic near 0 and linear beyond ``delta``;
* ``"nll"``: heteroscedastic Gaussian negative log-likelihood, with a variance
  predicted for each sample and indicator. Like weighted least squares, each
  residual is weighted by ``1 / sigma^2``; the ``log sigma^2`` term prevents
  the model from claiming a large uncertainty for free. The beta-NLL variant
  (Seitzer et al., ICLR 2022) multiplies each term by ``stopgrad(sigma^(2 beta))``
  so that hard samples are not ignored by the mean.

Families (HLC, Hb, CDT) receive equal total weight by default, whatever their
number of indicators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hlcnet.data import Normalizer
from hlcnet.models import ModelOutput
from hlcnet.targets import FAMILIES, TARGET_NAMES

LossKind = Literal["mse", "huber", "nll"]


@dataclass
class LossConfig:
    """Loss hyper-parameters."""

    kind: LossKind = "nll"
    beta: float = 0.5
    huber_delta: float = 1.0
    family_weights: dict[str, float] = field(
        default_factory=lambda: {"HLC": 1.0, "Hb": 1.0, "CDT": 1.0}
    )


def target_weights(family_weights: dict[str, float]) -> Tensor:
    """Per-indicator weights: each family's weight is split among its members,
    then weights are normalized to sum to 1."""
    w = torch.zeros(len(TARGET_NAMES))
    for fam, idx in FAMILIES.items():
        w[list(idx)] = family_weights.get(fam, 0.0) / len(idx)
    if w.sum() <= 0:
        raise ValueError("at least one family must have a positive weight")
    return w / w.sum()


def gaussian_nll(residual: Tensor, log_var: Tensor, beta: float = 0.0) -> Tensor:
    """Element-wise (beta-)Gaussian negative log-likelihood, up to a constant.

    Args:
        residual: prediction minus target, in the loss space.
        log_var: predicted log-variance, same shape.
        beta: 0 for the plain likelihood, 0.5 recommended, 1 ~ MSE on the mean.
    """
    nll = 0.5 * (log_var + residual.pow(2) * torch.exp(-log_var))
    if beta > 0:
        nll = nll * torch.exp(log_var).detach().pow(beta)
    return nll


class MultiTaskLoss(nn.Module):
    """Weighted multi-indicator loss in the normalized space."""

    def __init__(self, cfg: LossConfig, normalizer: Normalizer) -> None:
        super().__init__()
        self.cfg = cfg
        self.normalizer = normalizer
        self.register_buffer("weights", target_weights(cfg.family_weights))

    def forward(
        self, out: ModelOutput, y_true: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """Return the scalar loss and per-indicator diagnostics.

        Args:
            out: model output, ``pred`` in physical units.
            y_true: ``(B, 6)`` targets in physical units.

        Returns:
            ``(loss, logs)`` where ``logs`` holds the mean squared error of each
            indicator in the normalized space (``mse_<name>``) and, for the
            likelihood loss, the mean predicted standard deviation (``sigma_<name>``).
        """
        z = self.normalizer.transform_y(out.pred)
        zt = self.normalizer.transform_y(y_true)
        r = z - zt

        if self.cfg.kind == "mse":
            per = r.pow(2)
        elif self.cfg.kind == "huber":
            per = F.huber_loss(z, zt, reduction="none", delta=self.cfg.huber_delta)
        elif self.cfg.kind == "nll":
            if out.log_var is None:
                raise ValueError(
                    "loss 'nll' needs a model with predict_uncertainty=True"
                )
            per = gaussian_nll(r, out.log_var, self.cfg.beta)
        else:
            raise ValueError(f"unknown loss {self.cfg.kind!r}")

        weights: Tensor = self.weights  # type: ignore[assignment]
        loss = (per.mean(dim=0) * weights).sum()

        with torch.no_grad():
            logs = {
                f"mse_{n}": v
                for n, v in zip(TARGET_NAMES, r.pow(2).mean(0).tolist(), strict=True)
            }
            if out.log_var is not None:
                sig = torch.exp(0.5 * out.log_var).mean(0).tolist()
                logs.update(
                    {f"sigma_{n}": v for n, v in zip(TARGET_NAMES, sig, strict=True)}
                )
        return loss, logs

"""Neural models with physics-constrained outputs.

Design (each choice is justified by a finding of the EDA, section 3):

* **Inputs.** ``Tint`` and ``U`` are both read (3.6 b: a linear model only
  works once the excitation is fixed). An optional third channel is the
  minute-to-minute change ``dTint/dt``, which highlights the fast air response
  after each switch (3.6 a).
* **Encoder.** Convolutions at full one-minute resolution detect local
  patterns, a single stride-2 layer halves the length, then a bidirectional
  GRU follows the slow dynamics over 8 h. Early aggressive pooling is avoided
  because the information on ``HLC_ext`` lies in small shape differences in
  the first hour (3.6 c).
* **Attention pooling.** The model learns *which minutes* matter (3.6 a: the
  information arrives right after switch-on). The weights are returned for
  interpretation.
* **Physics-constrained head.** Every prediction satisfies
  ``HLC = HLC_ext + HLC_mit`` and ``tau_n <= tau_inf <= tau_b`` with positive
  values, by construction (section 2). Three variants allow an ablation:

  - ``"share"`` (default): predict ``HLC`` and the share ``r = HLC_ext / HLC``.
    HLC is the best-identified indicator and the split the hardest (3.6), so
    uncertainty on the split does not contaminate HLC;
  - ``"sum"``: predict ``HLC_ext`` and ``HLC_mit``, ``HLC`` is their sum;
  - ``"free"``: six unconstrained outputs (reference for the ablation).

* **Anchored log-normal parameterization.** Positive quantities are written
  ``exp(loc + scale * a)`` where ``loc, scale`` are the mean and standard
  deviation of their log on the training set; the share uses a logit. The
  network output ``a`` is therefore a standardized quantity and the model
  starts from the typical building.
* **Uncertainty.** An optional head predicts a log-variance per indicator, in
  the space where the loss is computed (standardized, log for the time
  constants), trained with a Gaussian likelihood (see ``losses.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn

HeadKind = Literal["share", "sum", "free"]
ArchKind = Literal["cnn_gru", "mlp"]

_EPS = 1e-6
_N_TARGETS = 6
_N_PARAMS: dict[str, int] = {"share": 5, "sum": 5, "free": 6}


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Hyper-parameters of a model (all variants used in the report)."""

    arch: ArchKind = "cnn_gru"
    head: HeadKind = "share"
    use_derivative: bool = True
    predict_uncertainty: bool = True
    seq_len: int = 480
    conv_channels: int = 64
    gru_hidden: int = 64
    gru_layers: int = 2
    bidirectional: bool = True
    mlp_hidden: int = 256
    dropout: float = 0.1


class ModelOutput(NamedTuple):
    """What a model returns.

    Attributes:
        pred: ``(B, 6)`` indicators in physical units, canonical order.
        log_var: ``(B, 6)`` log-variance in the loss space, or ``None``.
        attention: ``(B, T')`` attention weights over time, or ``None``.
    """

    pred: Tensor
    log_var: Tensor | None
    attention: Tensor | None


# --------------------------------------------------------------------------
# Physical parameterization of the outputs
# --------------------------------------------------------------------------
def targets_to_params(y: Tensor, head: HeadKind) -> Tensor:
    """Map physical targets ``(N, 6)`` to the unconstrained head parameters.

    ``share``: ``[log HLC, logit r, log tau_n, log(tau_inf - tau_n), log(tau_b - tau_inf)]``
    ``sum``:   ``[log ext, log mit, log tau_n, log(tau_inf - tau_n), log(tau_b - tau_inf)]``
    ``free``:  ``[HLC, ext, mit, log tau_b, log tau_n, log tau_inf]``
    """
    y = y.double()
    hlc, ext, mit, tau_b, tau_n, tau_inf = y.unbind(dim=1)
    gap_inf = (tau_inf - tau_n).clamp_min(0) + _EPS
    gap_b = (tau_b - tau_inf).clamp_min(0) + _EPS
    if head == "share":
        r = (ext / hlc).clamp(_EPS, 1 - _EPS)
        cols = [hlc.log(), torch.logit(r), tau_n.log(), gap_inf.log(), gap_b.log()]
    elif head == "sum":
        cols = [ext.log(), mit.log(), tau_n.log(), gap_inf.log(), gap_b.log()]
    elif head == "free":
        cols = [hlc, ext, mit, tau_b.log(), tau_n.log(), tau_inf.log()]
    else:
        raise ValueError(f"unknown head {head!r}")
    return torch.stack(cols, dim=1)


def params_to_targets(p: Tensor, head: HeadKind) -> Tensor:
    """Inverse of :func:`targets_to_params`; guarantees the constraints."""
    if head == "free":
        hlc, ext, mit = p[:, 0], p[:, 1], p[:, 2]
        tau_b, tau_n, tau_inf = p[:, 3].exp(), p[:, 4].exp(), p[:, 5].exp()
        return torch.stack([hlc, ext, mit, tau_b, tau_n, tau_inf], dim=1)

    q = p.clamp(-30.0, 30.0)  # keeps exp() finite early in training
    if head == "share":
        hlc = q[:, 0].exp()
        ext = torch.sigmoid(q[:, 1]) * hlc
        mit = hlc - ext
    elif head == "sum":
        ext, mit = q[:, 0].exp(), q[:, 1].exp()
        hlc = ext + mit
    else:
        raise ValueError(f"unknown head {head!r}")
    tau_n = q[:, 2].exp()
    tau_inf = tau_n + q[:, 3].exp()
    tau_b = tau_inf + q[:, 4].exp()
    return torch.stack([hlc, ext, mit, tau_b, tau_n, tau_inf], dim=1)


@dataclass
class ModelStats:
    """Training-set statistics the model needs (saved with checkpoints).

    Attributes:
        loc, scale: Mean and std of the head parameters on the training set.
        deriv_std: Std of the minute-to-minute change of normalized ``Tint``.
    """

    loc: Tensor
    scale: Tensor
    deriv_std: float

    @classmethod
    def from_data(cls, x_norm: Tensor, y: Tensor, head: HeadKind) -> ModelStats:
        """Compute the statistics from normalized inputs and physical targets."""
        p = targets_to_params(y, head)
        d = x_norm[..., 0].diff(dim=1)
        return cls(
            loc=p.mean(0).float(),
            scale=p.std(0).clamp_min(1e-6).float(),
            deriv_std=float(d.std().clamp_min(1e-8)),
        )


class PhysicsHead(nn.Module):
    """Maps a feature vector to constrained indicators and log-variances."""

    def __init__(
        self, in_dim: int, head: HeadKind, stats: ModelStats, uncertainty: bool
    ) -> None:
        super().__init__()
        self.head: HeadKind = head
        k = _N_PARAMS[head]
        if stats.loc.numel() != k:
            raise ValueError(
                f"stats computed for {stats.loc.numel()} params, head {head!r} needs {k}"
            )
        self.out = nn.Linear(in_dim, k)
        nn.init.normal_(self.out.weight, std=1e-3)  # start at the typical building
        nn.init.zeros_(self.out.bias)
        self.register_buffer("loc", stats.loc.clone())
        self.register_buffer("scale", stats.scale.clone())
        self.log_var = nn.Linear(in_dim, _N_TARGETS) if uncertainty else None
        if self.log_var is not None:
            nn.init.zeros_(self.log_var.weight)
            nn.init.zeros_(self.log_var.bias)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor | None]:
        a = self.out(h)
        params = self.loc + self.scale * a
        pred = params_to_targets(params, self.head)
        log_var = (
            self.log_var(h).clamp(-12.0, 6.0) if self.log_var is not None else None
        )
        return pred, log_var


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """Weighted average over time with learned weights (additive attention)."""

    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """``h``: ``(B, T, D)`` -> context ``(B, D)`` and weights ``(B, T)``."""
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)
        return (w.unsqueeze(-1) * h).sum(dim=1), w


def _conv_block(
    c_in: int, c_out: int, k: int, stride: int = 1, dilation: int = 1
) -> nn.Sequential:
    pad = dilation * (k - 1) // 2
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, stride=stride, padding=pad, dilation=dilation),
        nn.BatchNorm1d(c_out),
        nn.GELU(),
    )


class InputChannels(nn.Module):
    """Builds the input channels ``[Tint, (dTint/dt), U]`` from ``(B, T, 2)``."""

    def __init__(self, use_derivative: bool, deriv_std: float) -> None:
        super().__init__()
        self.use_derivative = use_derivative
        self.register_buffer("deriv_std", torch.tensor(float(deriv_std)))

    @property
    def n_channels(self) -> int:
        return 3 if self.use_derivative else 2

    def forward(self, x: Tensor) -> Tensor:
        tint, u = x[..., 0], x[..., 1]
        chans = [tint]
        if self.use_derivative:
            d = torch.diff(tint, dim=1, prepend=tint[:, :1]) / self.deriv_std
            chans.append(d)
        chans.append(u)
        return torch.stack(chans, dim=-1)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class CNNGRURegressor(nn.Module):
    """CNN + bidirectional GRU + attention pooling + physics-constrained head."""

    def __init__(self, cfg: ModelConfig, stats: ModelStats) -> None:
        super().__init__()
        self.cfg = cfg
        self.inputs = InputChannels(cfg.use_derivative, stats.deriv_std)
        c = cfg.conv_channels
        self.conv = nn.Sequential(
            _conv_block(self.inputs.n_channels, c, k=7),  # ~7 min, full resolution
            _conv_block(c, c, k=5, stride=2),  # halves the length (480 -> 240)
            _conv_block(c, c, k=5, dilation=2),  # wider receptive field (~25 min)
        )
        self.gru = nn.GRU(
            c,
            cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        d = cfg.gru_hidden * (2 if cfg.bidirectional else 1)
        self.pool = AttentionPooling(d)
        self.trunk = nn.Sequential(
            nn.Linear(2 * d, 128), nn.GELU(), nn.Dropout(cfg.dropout)
        )
        self.head = PhysicsHead(128, cfg.head, stats, cfg.predict_uncertainty)

    def forward(self, x: Tensor) -> ModelOutput:
        """``x``: ``(B, T, 2)`` normalized ``[Tint, U]``."""
        z = self.conv(self.inputs(x).transpose(1, 2)).transpose(1, 2)  # (B, T', C)
        h, _ = self.gru(z)  # (B, T', D)
        context, w = self.pool(h)
        feats = torch.cat([context, h.mean(dim=1)], dim=1)
        pred, log_var = self.head(self.trunk(feats))
        return ModelOutput(pred, log_var, w)


class MLPRegressor(nn.Module):
    """Non-sequential baseline: an MLP on the (time-averaged by 4) flattened series.

    Same inputs and same physics head as :class:`CNNGRURegressor`, so the
    comparison isolates the benefit of the sequential architecture.
    """

    def __init__(self, cfg: ModelConfig, stats: ModelStats) -> None:
        super().__init__()
        self.cfg = cfg
        self.inputs = InputChannels(cfg.use_derivative, stats.deriv_std)
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=4)
        in_dim = self.inputs.n_channels * (cfg.seq_len // 4)
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.mlp_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.head = PhysicsHead(128, cfg.head, stats, cfg.predict_uncertainty)

    def forward(self, x: Tensor) -> ModelOutput:
        z = self.downsample(self.inputs(x).transpose(1, 2)).flatten(1)
        pred, log_var = self.head(self.net(z))
        return ModelOutput(pred, log_var, None)


def build_model(cfg: ModelConfig, stats: ModelStats) -> nn.Module:
    """Factory used by the training script and the notebook."""
    if cfg.arch == "cnn_gru":
        return CNNGRURegressor(cfg, stats)
    if cfg.arch == "mlp":
        return MLPRegressor(cfg, stats)
    raise ValueError(f"unknown arch {cfg.arch!r}")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

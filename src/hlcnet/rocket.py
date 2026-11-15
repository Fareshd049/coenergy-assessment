"""ROCKET features (Dempster et al., 2020) for time series extrinsic regression.

ROCKET convolves each series with thousands of *random, untrained* kernels
(random length, weights, bias, dilation and padding) and keeps two numbers per
kernel: the proportion of positive values (PPV) and the maximum. A ridge
regression on these features was the most accurate method of the first Time
Series Extrinsic Regression benchmark (Tan et al., 2021), which makes it a
recognized literature baseline for our task.

Multivariate version: each kernel spans all input channels.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hlcnet.models import InputChannels


class RocketFeatures(nn.Module):
    """Random convolutional features, ``2 * n_kernels`` per series.

    Kernels sharing the same (length, dilation, padding) are applied together
    in one ``conv1d`` call, so the transform is fast on GPU.
    """

    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        n_kernels: int = 2000,
        lengths: tuple[int, ...] = (7, 9, 11),
        seed: int = 0,
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        groups: dict[tuple[int, int, int], list[tuple[Tensor, float]]] = {}
        for _ in range(n_kernels):
            length = lengths[int(torch.randint(len(lengths), (1,), generator=g))]
            w = torch.randn(n_channels, length, generator=g)
            w = w - w.mean()
            bias = float(torch.rand(1, generator=g) * 2 - 1)
            max_exp = math.log2((seq_len - 1) / (length - 1))
            dilation = int(2 ** (float(torch.rand(1, generator=g)) * max_exp))
            padding = (
                ((length - 1) * dilation) // 2
                if bool(torch.rand(1, generator=g) < 0.5)
                else 0
            )
            groups.setdefault((length, dilation, padding), []).append((w, bias))

        self.specs: list[tuple[int, int]] = []
        for i, ((_, dilation, padding), kernels) in enumerate(sorted(groups.items())):
            self.register_buffer(
                f"w{i}", torch.stack([k for k, _ in kernels])
            )  # (K, C, L)
            self.register_buffer(f"b{i}", torch.tensor([b for _, b in kernels]))
            self.specs.append((dilation, padding))
        self.n_features = 2 * n_kernels

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """``x``: ``(B, C, T)`` -> features ``(B, 2 * n_kernels)``."""
        feats = []
        for i, (dilation, padding) in enumerate(self.specs):
            out = F.conv1d(
                x,
                getattr(self, f"w{i}"),
                getattr(self, f"b{i}"),
                padding=padding,
                dilation=dilation,
            )
            feats.append((out > 0).float().mean(dim=-1))  # PPV
            feats.append(out.amax(dim=-1))  # max
        return torch.cat(feats, dim=1)


@torch.no_grad()
def rocket_transform(
    x_norm: Tensor,
    rocket: RocketFeatures,
    channels: InputChannels,
    device: torch.device,
    batch_size: int = 2048,
) -> Tensor:
    """Apply :class:`RocketFeatures` to normalized ``(N, T, 2)`` inputs, by batches."""
    rocket.to(device)
    channels.to(device)
    out = []
    for start in range(0, x_norm.shape[0], batch_size):
        xb = x_norm[start : start + batch_size].to(device)
        out.append(rocket(channels(xb).transpose(1, 2)).cpu())
    return torch.cat(out)

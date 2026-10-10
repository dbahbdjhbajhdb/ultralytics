# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Grouped channel attention using a deterministic orthogonal spatial descriptor bank."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .block import C3k2

__all__ = ("C3k2GOA", "GroupedOrthogonalAttention")


def _positive_integer(name: str, value: int) -> int:
    """Validate integer dimensions without silently truncating configuration values."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _dct_basis(grid_size: int, num_filters: int) -> torch.Tensor:
    """Create low-frequency, orthonormal 2D DCT-II bases on a fixed square grid."""
    positions = torch.arange(grid_size, dtype=torch.float32) + 0.5
    frequencies = sorted(
        ((u, v) for u in range(grid_size) for v in range(grid_size)),
        key=lambda frequency: (sum(frequency), frequency[0], frequency[1]),
    )[:num_filters]
    bank = []
    for u, v in frequencies:
        alpha_u = math.sqrt((1 if u == 0 else 2) / grid_size)
        alpha_v = math.sqrt((1 if v == 0 else 2) / grid_size)
        vertical = alpha_u * torch.cos(math.pi * u * positions / grid_size)
        horizontal = alpha_v * torch.cos(math.pi * v * positions / grid_size)
        bank.append(vertical[:, None] * horizontal[None, :])
    return torch.stack(bank)


class GroupedOrthogonalAttention(nn.Module):
    """Gate channel groups from multiple mutually orthogonal spatial projections.

    For normalized DCT bases phi_k, the discrete inner product obeys
    sum(phi_i * phi_j) = delta_ij, up to floating-point rounding. Orthogonality
    describes this fixed spatial filter bank, rather than the learned MLP weights
    or the channel groups. Adaptive pooling puts input features on its fixed grid.

    Each channel is projected onto every basis. Independent group MLPs combine
    those coefficients into channel gates. The final projection starts at zero,
    so the module initially returns its input exactly. The multiplicative gate
    is bounded by 1 +/- residual_scale. No labels are needed during inference.

    This is an experimental design, not a reproduction of a particular paper.
    """

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        num_filters: int = 8,
        grid_size: int = 7,
        reduction: int = 4,
        residual_scale: float = 0.5,
    ):
        """Initialize grouped projections and an identity-initialized residual gate."""
        super().__init__()
        self.channels = _positive_integer("channels", channels)
        requested_groups = _positive_integer("groups", groups)
        self.num_filters = _positive_integer("num_filters", num_filters)
        self.grid_size = _positive_integer("grid_size", grid_size)
        reduction = _positive_integer("reduction", reduction)
        if self.num_filters > self.grid_size**2:
            raise ValueError("num_filters cannot exceed grid_size squared for an orthogonal spatial basis bank")
        self.residual_scale = float(residual_scale)
        if not math.isfinite(self.residual_scale) or not 0 < self.residual_scale <= 1:
            raise ValueError("residual_scale must be finite and in (0, 1]")

        # Width scaling can change channels; use the largest compatible divisor.
        self.groups = math.gcd(self.channels, requested_groups)
        channels_per_group = self.channels // self.groups
        hidden_channels = self.groups * max(1, channels_per_group // reduction)
        self.register_buffer("dct_basis", _dct_basis(self.grid_size, self.num_filters), persistent=False)
        self.projection = nn.Sequential(
            nn.Conv1d(self.channels * self.num_filters, hidden_channels, 1, groups=self.groups),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_channels, self.channels, 1, groups=self.groups),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return same-shaped, gated features using only image-derived descriptors."""
        pooled = F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))
        basis = self.dct_basis.to(dtype=pooled.dtype)
        coefficients = torch.einsum("bchw,khw->bck", pooled, basis) / self.grid_size
        descriptors = coefficients.reshape(x.shape[0], self.channels * self.num_filters, 1)
        logits = self.projection(descriptors).reshape(x.shape[0], self.channels, 1, 1)
        gate = 1 + self.residual_scale * torch.tanh(logits)
        return x * gate.to(dtype=x.dtype)


class C3k2GOA(C3k2):
    """C3k2 followed by grouped orthogonal attention, retaining existing weight keys."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        attention_groups: int = 8,
        num_filters: int = 8,
        grid_size: int = 7,
        reduction: int = 4,
    ):
        """Preserve C3k2's constructor prefix for model parsing and weight transfer."""
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.attention = GroupedOrthogonalAttention(c2, attention_groups, num_filters, grid_size, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention after the original C3k2 feature fusion."""
        return self.attention(super().forward(x))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve attention when exporters select C2f's split-based forward path."""
        return self.attention(super().forward_split(x))

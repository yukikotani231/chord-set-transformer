"""Set Transformer building blocks.

Based on "Set Transformer: A Framework for Attention-based
Permutation-Invariant Neural Networks" (Lee et al., 2019)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiheadAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch_size = query.size(0)

        # Linear projections
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        context = torch.matmul(attn, v)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output: Tensor = self.w_o(context)

        return output


class MAB(nn.Module):
    """Multihead Attention Block.

    MAB(X, Y) = LayerNorm(H + rFF(H))
    where H = LayerNorm(X + Attention(X, Y, Y))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ff_dim: int | None = None,
    ):
        super().__init__()

        self.attention = MultiheadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        ff_dim = ff_dim or d_model * 4
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        h = self.norm1(x + self.attention(x, y, y, mask))
        out: Tensor = self.norm2(h + self.ff(h))
        return out


class SAB(nn.Module):
    """Set Attention Block.

    SAB(X) = MAB(X, X)
    Self-attention over the set.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mab = MAB(d_model, num_heads, dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        out: Tensor = self.mab(x, x, mask)
        return out


class ISAB(nn.Module):
    """Induced Set Attention Block.

    Uses inducing points to reduce complexity from O(n^2) to O(nm).
    ISAB(X) = MAB(X, MAB(I, X))
    where I is a set of m learnable inducing points.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_inducing_points: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing_points, d_model))
        self.mab1 = MAB(d_model, num_heads, dropout)
        self.mab2 = MAB(d_model, num_heads, dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size = x.size(0)
        inducing = self.inducing_points.expand(batch_size, -1, -1)

        h = self.mab1(inducing, x, mask)
        out: Tensor = self.mab2(x, h)
        return out


class PMA(nn.Module):
    """Pooling by Multihead Attention.

    Aggregates set elements into k output vectors using attention.
    PMA(X) = MAB(S, X)
    where S is a set of k learnable seed vectors.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_seeds: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, d_model))
        self.mab = MAB(d_model, num_heads, dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size = x.size(0)
        seeds = self.seeds.expand(batch_size, -1, -1)
        out: Tensor = self.mab(seeds, x, mask)
        return out

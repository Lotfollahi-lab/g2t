"""
Cross-attention velocity network for flow matching.

LUNA-style architecture: a transformer encoder operating across cells
within a slice. Each layer applies multi-head self-attention over all
cells, followed by a position-wise feed-forward block, with pre-norm +
residuals.

Why this exists (and why the per-cell MLP ``VelocityNetwork`` did not work)
--------------------------------------------------------------------------

The MLP variant predicts each cell's velocity independently given its own
state plus a single scalar ``section_embed`` summarizing the slice. With
no information flow between cells, the model can only learn marginal
positions ("this gene expression usually implies y ≈ −0.3"), not the
spatial *arrangement* (which cells are neighbours). The generated
trajectories therefore stay close to noise.

This module fixes that: every cell's hidden state at every denoising
step depends on every other cell's hidden state via multi-head
attention. That's the same architectural ingredient LUNA's
``models/transformer.TransformerLayer`` provides.

The math (flow-matching MSE on conditional velocities) is unchanged —
only the function approximator that produces the velocity field is
upgraded.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Time embedding (shared design with velocity_net.SinusoidalTimeEmbedding;
# duplicated here to keep this module independent of the legacy MLP
# velocity net while we still ship both)
# ---------------------------------------------------------------------------


class _SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the flow time t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb
        )
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


# ---------------------------------------------------------------------------
# A single pre-norm transformer encoder layer
# ---------------------------------------------------------------------------


class _TransformerLayer(nn.Module):
    """Pre-norm transformer block: LN → MHA → residual → LN → FFN → residual.

    LUNA's ``models/transformer.TransformerLayer`` is structurally the
    same (with their PositionsMLP / PositionNorm extras on top). We omit
    those refinements here — the core "self-attention over cells +
    feed-forward" is what makes flow matching actually work.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (1, n_cells, hidden_dim). The "batch" dim is 1 because all
                cells in one forward call belong to the same slice;
                attention runs over the n_cells dimension.
        """
        a = self.norm1(x)
        a_out, _ = self.attn(a, a, a, need_weights=False)
        x = x + self.attn_dropout(a_out)
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# The velocity network proper
# ---------------------------------------------------------------------------


class CrossAttentionVelocityNetwork(nn.Module):
    """LUNA-style velocity network with self-attention across cells.

    Architecture (per denoising step):
      1. Concatenate per-cell features:
            [z_t  ‖  t_emb  ‖  cell_embed  ‖  k_emb]
      2. Linear projection to ``hidden_dim``.
      3. ``n_layers`` of pre-norm transformer blocks
         (multi-head self-attention over the cells of the slice + FFN).
      4. Final LayerNorm + Linear → velocity ∈ ℝ^{spatial_dim}.

    The output Linear is zero-initialised so the velocity field starts
    at exactly ``v(z, t) = 0`` (well-conditioned start, same convention
    as our MLP velocity net).

    ``section_embed`` is accepted for API parity with
    ``VelocityNetwork.forward`` but ignored: self-attention provides
    full cell-to-cell context, making the scalar section summary
    redundant. This matches LUNA's setup where there's no separate
    section-level embedding either.

    Args:
        spatial_dim: 2 for (x, y).
        cell_embed_dim: dim of per-cell gene-expression embedding.
        hidden_dim: width of the transformer.
        n_layers: number of transformer blocks (LUNA default: 8).
        n_heads: heads in each MHA (LUNA default: 16).
        time_embed_dim: dim of the sinusoidal time embedding.
        k_embed_dim: dim of the learned k-target embedding.
        k_max: max k value the embedding table accepts.
        dropout: attention + FFN dropout.
        ff_mult: FFN expansion factor (hidden_dim * ff_mult).
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        cell_embed_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 8,
        n_heads: int = 16,
        time_embed_dim: int = 64,
        k_embed_dim: int = 16,
        k_max: int = 50,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim

        # Time + k embeddings (same design as the MLP variant)
        self.time_embed = _SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
        )
        # k=0 reserved for "no control"; +1 to span [0, k_max] inclusive.
        self.k_embed = nn.Embedding(k_max + 1, k_embed_dim)

        # Per-cell input projection.
        in_dim = spatial_dim + time_embed_dim + cell_embed_dim + k_embed_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        # Stack of cross-attention transformer blocks.
        self.layers = nn.ModuleList([
            _TransformerLayer(hidden_dim, n_heads, ff_mult=ff_mult, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output head: LN + zero-init Linear so v(z, t) starts at 0.
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, spatial_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_embed: torch.Tensor,
        section_embed: Optional[torch.Tensor] = None,  # accepted, ignored
        k_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_t: (n_cells, spatial_dim) current cell locations at flow time t.
            t: (n_cells,) flow time in [0, 1].
            cell_embed: (n_cells, cell_embed_dim) gene-expression embedding.
            section_embed: ignored (cross-attention supplies global context).
            k_target: (n_cells,) long tensor of target k values, or None.

        Returns:
            velocity: (n_cells, spatial_dim).
        """
        n = z_t.shape[0]
        device = z_t.device

        # Time embedding
        t_emb = self.time_mlp(self.time_embed(t))                    # (n, time_dim)

        # k embedding (default to k=0 = no control if None)
        if k_target is None:
            k_target = torch.zeros(n, dtype=torch.long, device=device)
        k_emb = self.k_embed(k_target)                                # (n, k_dim)

        # Per-cell feature stack
        x = torch.cat([z_t, t_emb, cell_embed, k_emb], dim=-1)        # (n, in_dim)
        h = self.in_proj(x)                                            # (n, hidden)

        # Treat the n cells as one sequence of length n that gets
        # self-attended. Add a leading "batch" dim of 1 for MHA's
        # batch_first=True convention.
        h = h.unsqueeze(0)                                             # (1, n, hidden)
        for layer in self.layers:
            h = layer(h)
        h = self.out_norm(h).squeeze(0)                                # (n, hidden)

        return self.out_proj(h)                                        # (n, spatial_dim)

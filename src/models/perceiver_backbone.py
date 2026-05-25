"""Perceiver-style backbone — Architectural extension #3.

Implements Jaegle et al. 2021, "Perceiver IO" (arXiv:2107.14795).
The key idea: replace O(N²) cell-to-cell attention with attention
routed through K << N learnable **anchor tokens** (the "latent
array" in Perceiver terminology). Three sub-modules:

1. **Cells → anchors** (cross-attention): anchors query cells; reads
   tissue context into the anchor stream.
2. **Anchors ↔ anchors** (self-attention, repeated): anchors mix
   among themselves. Cost is O(K²) per block, ignorable for K=32.
3. **Anchors → cells** (cross-attention): cells query anchors;
   reads tissue context back into the cell stream for output.

Attention cost drops from N²·D to N·K·D + K²·D. For cortex (N≈7k,
K=32) that's ~7k × 32 = 224k vs 7k² = 49M operations per layer —
~200× cheaper. The K-token bottleneck doubles as implicit
regularisation: the model has to summarise tissue context through K
dimensions, which is exactly the "less overfitting on 6 slices"
inductive bias we like.

Output interface matches ``models.model.Model``: DataHolder in →
DataHolder out, with ``pred.node_features`` of width
``output_features_to_pos_dims`` and ``pred.positions`` of width 2.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data.dataholder import DataHolder

# Reuse the DiT helpers (timestep embedder + adaLN-Zero modulation)
# to keep the time-conditioning behaviour consistent across the two
# new backbones. They live in dit_backbone.py.
from models.dit_backbone import (
    TimestepEmbedder,
    _modulate,
)


# ---------------------------------------------------------------------------
# Cross-attention block (queries → keys/values)
# ---------------------------------------------------------------------------


class _CrossAttentionBlock(nn.Module):
    """One cross-attention layer with pre-LN and a feedforward sub-layer.

    The queries Q come from one stream, the keys/values KV from
    another. Output has the same shape as Q. Uses PyTorch's
    ``MultiheadAttention`` with ``batch_first=True`` so input shape
    is (B, N, D).

    No time conditioning here — keep cross-attention as a pure
    relational primitive. Time enters via the anchor self-attention
    blocks (which DO use adaLN) so the model can still modulate its
    behaviour per FM step.
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        # If q_dim != kv_dim, PyTorch's MultiheadAttention with
        # kdim/vdim handles the projection internally. We use that
        # so cells and anchors can have different widths if desired
        # (config defaults them to equal but the option is here).
        self.attn = nn.MultiheadAttention(
            embed_dim=q_dim,
            num_heads=n_heads,
            kdim=kv_dim,
            vdim=kv_dim,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(q_dim)
        ff_hidden = int(q_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(q_dim, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, q_dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q:  (B, M, q_dim) — query tokens (the stream we're updating).
            kv: (B, N, kv_dim) — key/value tokens.
            kv_key_padding_mask: (B, N) bool, True at PAD positions of kv.

        Returns:
            (B, M, q_dim) — updated query tokens.
        """
        # Cross-attention with residual.
        h = self.norm_q(q)
        k = self.norm_kv(kv)
        attn_out, _ = self.attn(
            h, k, k,
            key_padding_mask=kv_key_padding_mask,
            need_weights=False,
        )
        q = q + attn_out
        # FFN with residual.
        q = q + self.ff(self.norm_ff(q))
        return q


# ---------------------------------------------------------------------------
# Anchor self-attention block with adaLN-Zero time conditioning
# ---------------------------------------------------------------------------


class _AnchorSelfAttnBlock(nn.Module):
    """One anchor↔anchor self-attention block. Same pattern as the
    DiT block (pre-LN + adaLN-Zero modulation + MSA + FFN + gates),
    but operating on the K-sized anchor stream so the O(K²) cost is
    trivial. Time conditioning via the same six modulation vectors.
    """

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"anchor_dim {hidden_dim} must be divisible by n_heads {n_heads}"
            )

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        ff_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, hidden_dim),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, anchors: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = _modulate(self.norm1(anchors), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        anchors = anchors + gate_msa.unsqueeze(1) * attn_out

        h = _modulate(self.norm2(anchors), shift_mlp, scale_mlp)
        anchors = anchors + gate_mlp.unsqueeze(1) * self.mlp(h)
        return anchors


# ---------------------------------------------------------------------------
# Perceiver backbone — public interface
# ---------------------------------------------------------------------------


class PerceiverBackbone(nn.Module):
    """Drop-in replacement for ``models.model.Model`` using a
    Perceiver-IO-style anchored attention architecture. Same
    DataHolder-in / DataHolder-out contract.

    Architecture:
        cells = embed(gene + position)        # (B, N, cell_dim)
        anchors = anchor_tokens.expand(B, ...) # (B, K, anchor_dim)
        anchors = cross_attn_input(anchors_q=anchors, cells_kv=cells)
        for block in anchor_blocks:
            anchors = block(anchors, time_emb)         # K² self-attn + adaLN
        cells = cross_attn_output(cells_q=cells, anchors_kv=anchors)
        out_features, out_positions = final(cells, time_emb)

    Drops cell-cell attention's O(N²) cost to O(N·K) per cross-attention
    plus O(K²) per anchor self-attention. For cortex N≈7k, K=32 that's
    ~250× cheaper than full attention.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        n_layers: int,
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        output_dims: Dict[str, int],
        perceiver_cfg=None,
    ):
        super().__init__()
        self.out_features_dim = int(hidden_dims["output_features_to_pos_dims"])

        def _g(k, default):
            if perceiver_cfg is None:
                return default
            return (
                perceiver_cfg.get(k, default)
                if hasattr(perceiver_cfg, "get")
                else getattr(perceiver_cfg, k, default)
            )

        self.n_anchors = int(_g("n_anchors", 32))
        self.anchor_dim = int(_g("anchor_dim", 256))
        self.cell_dim = int(_g("cell_dim", 256))
        # Default n_anchor_blocks comes from the perceiver-specific
        # config; cfg.model.n_layers (the global one) is ignored here
        # because perceiver's effective depth is mostly in the anchor
        # blocks. We still take n_layers as the positional default for
        # the rare case where perceiver_cfg is None.
        self.n_anchor_blocks = int(_g("n_anchor_blocks", n_layers))
        self.n_heads = int(_g("n_heads", 8))
        self.mlp_ratio = float(_g("mlp_ratio", 4))
        self.time_embed_dim = int(_g("time_embed_dim", 256))

        gene_in = int(input_dims["node_features_dimensions"])

        # Cell-stream input projections (gene + position → cell token).
        self.gene_embed = nn.Sequential(
            nn.Linear(gene_in, self.cell_dim),
            nn.SiLU(),
            nn.Linear(self.cell_dim, self.cell_dim),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(2, self.cell_dim),
            nn.SiLU(),
            nn.Linear(self.cell_dim, self.cell_dim),
        )

        # K learnable anchor tokens. Drawn from a small-variance
        # Gaussian to break symmetry without producing huge initial
        # cross-attention scores.
        self.anchor_tokens = nn.Parameter(
            torch.randn(self.n_anchors, self.anchor_dim) * 0.02
        )

        # Time embedding (fed into the anchor self-attention blocks
        # for adaLN-Zero conditioning).
        self.t_embed = TimestepEmbedder(self.anchor_dim, self.time_embed_dim)

        # Encoder cross-attention: anchors query cells.
        self.cross_in = _CrossAttentionBlock(
            q_dim=self.anchor_dim,
            kv_dim=self.cell_dim,
            n_heads=self.n_heads,
            mlp_ratio=self.mlp_ratio,
        )

        # Anchor self-attention stack.
        self.anchor_blocks = nn.ModuleList([
            _AnchorSelfAttnBlock(self.anchor_dim, self.n_heads, self.mlp_ratio)
            for _ in range(self.n_anchor_blocks)
        ])

        # Decoder cross-attention: cells query anchors.
        self.cross_out = _CrossAttentionBlock(
            q_dim=self.cell_dim,
            kv_dim=self.anchor_dim,
            n_heads=self.n_heads,
            mlp_ratio=self.mlp_ratio,
        )

        # Output head: project the updated cell tokens to
        # (features, positions). NOT zero-init — see the matching
        # comment in dit_backbone.py for why (cdist's gradient at
        # x=y is degenerate, so zero-init poisons the gradient path
        # through pairwise-distance losses).
        self.head = nn.Linear(self.cell_dim, self.out_features_dim + 2)

    def forward(self, data: DataHolder, **_unused_kwargs) -> DataHolder:
        node_mask = data.node_mask
        B = data.node_features.shape[0]

        # Build per-cell tokens.
        cells = self.gene_embed(data.node_features) + self.pos_embed(data.positions)
        cells = cells * node_mask.unsqueeze(-1).to(cells.dtype)

        # Anchor tokens, replicated per slice.
        anchors = self.anchor_tokens.unsqueeze(0).expand(B, -1, -1)   # (B, K, D)
        # We DON'T add time embedding to anchors directly — time enters
        # via the adaLN-Zero modulation inside each anchor block.

        # Time conditioning vector for the anchor blocks.
        t = data.diffusion_time
        if t.dim() > 1:
            t = t.view(-1)
        c = self.t_embed(t)                                            # (B, anchor_dim)

        # ---- Encoder: cells → anchors via cross-attention ----
        # Anchors query cells; padding cells excluded via key mask.
        kv_pad = ~node_mask                                            # (B, N)
        anchors = self.cross_in(anchors, cells, kv_key_padding_mask=kv_pad)

        # ---- Anchor self-attention with time conditioning ----
        for block in self.anchor_blocks:
            anchors = block(anchors, c)

        # ---- Decoder: cells → anchors (cells query anchors) ----
        # No padding mask on anchors (they're all real, learnable).
        cells = self.cross_out(cells, anchors, kv_key_padding_mask=None)
        # Mask padding cells back to zero (cross_out doesn't know about
        # padding on the query side).
        cells = cells * node_mask.unsqueeze(-1).to(cells.dtype)

        # ---- Output head: features + positions ----
        out = self.head(cells)                                         # (B, N, F+2)
        features = out[..., :-2]
        positions = out[..., -2:]

        # Mask + mean-centre positions (same convention as DiT backbone).
        pad_mask = node_mask.unsqueeze(-1).to(features.dtype)
        features = features * pad_mask
        positions = positions * pad_mask
        n_real = node_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1).to(
            positions.dtype
        )
        positions = positions - positions.sum(dim=1, keepdim=True) / n_real
        positions = positions * pad_mask

        return DataHolder(
            node_features=features,
            positions=positions,
            diffusion_time=data.diffusion_time,
            cell_class=data.cell_class,
            cell_ID=data.cell_ID,
            t_int=data.t_int,
            t=data.t,
            node_mask=node_mask,
        )

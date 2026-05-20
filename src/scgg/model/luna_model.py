"""
LUNA-style transformer that predicts x_0 (clean target positions) from a
noisy z_t and cell-expression conditioning.

Faithful port of LUNA's ``models/model.py`` + ``models/transformer.py`` +
``models/self_attention.py`` adapted to the scGG interface. The
peculiar bits that make LUNA's architecture work for spatial-coord
prediction (and that our earlier cross-attention velocity net was
missing) are:

  * **Three separate input streams** (node features, diffusion time,
    positions) processed independently by input MLPs, then updated in
    parallel by each transformer layer. The position information stays
    explicit through the stack instead of being absorbed into a
    single hidden vector at the first linear projection.

  * **Position-direction conditioning** in the attention: at each
    layer, the unit-vector ``pos / |pos|`` is mapped through a small
    MLP to a delta-embedding that is concatenated with the node
    features and broadcast time embedding BEFORE attention. The
    attention sees where each cell currently is (direction-wise)
    relative to the slice centroid.

  * **Position output via norm modulation**: the output position is
    ``pos * f(features, pos, |pos|) / |pos|`` — direction inherited
    from the running position state, magnitude predicted by a
    feature-conditioned MLP. This explicitly factors the output into
    a learned radial structure.

  * **Translation equivariance** via mean subtraction at the output —
    the model can't waste capacity learning the absolute centroid.

  * **Predict x_0 directly**, not velocity. The pairwise-distance loss
    in ``flow_matching.py`` then compares the predicted x_0 to the GT.

We use ``nn.MultiheadAttention`` instead of LUNA's
``LinearAttentionTransformer`` (their dependency on the
``linear_attention_transformer`` package). For cortex slices of
< ~8k cells per forward, standard MHA fits in memory easily.

The forward signature matches our other velocity nets
(``CrossAttentionVelocityNetwork``, ``VelocityNetwork``):
``forward(z_t, t, cell_embed, section_embed, k_target) -> (n_cells, spatial_dim)``.
``section_embed`` and ``k_target`` are accepted but ignored — LUNA's
self-attention provides global context and doesn't condition on
section-level summaries or a k value.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (same design as velocity_net_attention.py)
# ---------------------------------------------------------------------------


class _SinusoidalTimeEmbedding(nn.Module):
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
# Position-equivariant MLP and norm (LUNA's models/layers.py)
# ---------------------------------------------------------------------------


class PositionsMLP(nn.Module):
    """Norm-only MLP: scales position magnitude by an MLP(|pos|),
    preserving direction. Translation-equivariant by construction
    when followed by mean subtraction.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: (n, spatial_dim)
        norm = torch.norm(pos, dim=-1, keepdim=True)             # (n, 1)
        new_norm = self.mlp(norm)                                 # (n, 1)
        new_pos = pos * new_norm / (norm + self.eps)
        # Translation equivariance: subtract the slice centroid.
        new_pos = new_pos - new_pos.mean(dim=0, keepdim=True)
        return new_pos


# ---------------------------------------------------------------------------
# Self-attention with position+time conditioning
# ---------------------------------------------------------------------------


class _LunaSelfAttention(nn.Module):
    """LUNA-style multi-head self-attention that:
       1. transforms positions → direction-MLP → delta-embedding,
       2. concatenates [node, delta, time_broadcast] → linear → hidden,
       3. runs multi-head self-attention over all cells in the slice,
       4. derives a fresh position via Linear(features) → spatial_dim.

    Returns (node_features_new, time_new, position_new). The time
    stream is updated by a separate Linear (the "y_y" projection in
    LUNA's code).
    """

    def __init__(
        self,
        node_dim: int,
        time_dim: int,
        delta_dim: int,
        spatial_dim: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert node_dim % n_heads == 0, (
            f"node_dim ({node_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.node_dim = node_dim
        self.time_dim = time_dim
        self.delta_dim = delta_dim
        self.spatial_dim = spatial_dim
        self.n_heads = n_heads

        # Direction-of-position → delta-embedding (LUNA's
        # `transform_positions_for_attn_mlp`). Operating on the unit
        # vector (not the raw coords) lets the attention condition on
        # WHICH WAY a cell sits relative to the centroid, separate
        # from how far away it is.
        self.pos_dir_mlp = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.ReLU(),
            nn.Linear(64, delta_dim),
        )

        # Identity-ish projection on node features before concat.
        self.lin_node = nn.Linear(node_dim, node_dim)

        # Project [node || delta || time_broadcast] back to node_dim.
        self.concat_proj = nn.Linear(node_dim + delta_dim + time_dim, node_dim)

        # Self-attention over the n_cells "sequence".
        self.attn = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Fresh position from attended features.
        self.feat_to_pos = nn.Linear(node_dim, spatial_dim)

        # Time-stream update (LUNA's `y_y`).
        self.time_update = nn.Linear(time_dim, time_dim)

    def forward(
        self,
        node_features: torch.Tensor,   # (n, node_dim)
        time_features: torch.Tensor,   # (n, time_dim) -- already broadcast per cell
        positions: torch.Tensor,       # (n, spatial_dim)
    ):
        # 1. Direction of position → delta
        norm = torch.norm(positions, dim=-1, keepdim=True).clamp_min(1e-7)
        direction = positions / norm                       # (n, spatial_dim)
        delta = self.pos_dir_mlp(direction)                # (n, delta_dim)

        # 2. Concat + project
        node_lin = self.lin_node(node_features)            # (n, node_dim)
        concat = torch.cat([node_lin, delta, time_features], dim=-1)
        h = self.concat_proj(concat)                       # (n, node_dim)

        # 3. Self-attention over cells. Treat n_cells as the sequence
        # length, with a leading batch dim of 1.
        h = h.unsqueeze(0)                                  # (1, n, node_dim)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        attn_out = attn_out.squeeze(0)                     # (n, node_dim)

        # 4. Fresh position from attended features.
        new_pos = self.feat_to_pos(attn_out)               # (n, spatial_dim)

        # Time update.
        new_time = self.time_update(time_features)         # (n, time_dim)

        return attn_out, new_time, new_pos


# ---------------------------------------------------------------------------
# Transformer block (LUNA's TransformerLayer)
# ---------------------------------------------------------------------------


class _LunaTransformerLayer(nn.Module):
    """One transformer block: self-attention + residual + FFN for each
    of (node, time) streams, plus a position update (no residual on
    position — LUNA recomputes it from features each layer).
    """

    def __init__(
        self,
        node_dim: int,
        time_dim: int,
        delta_dim: int,
        spatial_dim: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.attn = _LunaSelfAttention(
            node_dim=node_dim,
            time_dim=time_dim,
            delta_dim=delta_dim,
            spatial_dim=spatial_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Node feature stream
        self.norm_node_1 = nn.LayerNorm(node_dim, eps=eps)
        self.norm_node_2 = nn.LayerNorm(node_dim, eps=eps)
        self.ff_node = nn.Sequential(
            nn.Linear(node_dim, node_dim * ff_mult),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * ff_mult, node_dim),
            nn.Dropout(dropout),
        )

        # Time feature stream
        self.norm_time_1 = nn.LayerNorm(time_dim, eps=eps)
        self.norm_time_2 = nn.LayerNorm(time_dim, eps=eps)
        self.ff_time = nn.Sequential(
            nn.Linear(time_dim, time_dim * ff_mult),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(time_dim * ff_mult, time_dim),
            nn.Dropout(dropout),
        )

    def forward(self, node_features, time_features, positions):
        # Attention with position+time conditioning.
        attn_node, attn_time, new_pos = self.attn(
            node_features, time_features, positions
        )

        # Node stream: residual + LN + FFN + residual + LN
        node = self.norm_node_1(node_features + attn_node)
        node = self.norm_node_2(node + self.ff_node(node))

        # Time stream: residual + LN + FFN + residual + LN
        time = self.norm_time_1(time_features + attn_time)
        time = self.norm_time_2(time + self.ff_time(time))

        # Position: LUNA recomputes from features each layer (no
        # residual on position). Center for translation equivariance.
        new_pos = new_pos - new_pos.mean(dim=0, keepdim=True)

        return node, time, new_pos


# ---------------------------------------------------------------------------
# Top-level LunaTransformerNet — drop-in replacement for the velocity net
# ---------------------------------------------------------------------------


class LunaTransformerNet(nn.Module):
    """LUNA-style transformer; predicts x_0 (clean target positions).

    Forward signature matches CrossAttentionVelocityNetwork so it
    plugs into the same dispatch in ScGG.

    Args:
        spatial_dim: 2 for (x, y).
        cell_embed_dim: dim of per-cell gene-expression embedding.
        time_embed_dim: dim of sinusoidal time embedding (before MLP).
        node_dim: width of the node feature stream (LUNA's `dx`).
        time_dim_hidden: width of the time stream (LUNA's `dy`).
        delta_dim: width of the position-direction embedding
            (LUNA's `dd`).
        n_layers: number of transformer blocks.
        n_heads: heads in each MHA (must divide `node_dim`).
        ff_mult: FFN expansion factor.
        dropout: attention + FFN dropout.
        k_max: unused, accepted for API parity.
        k_embed_dim: unused, accepted for API parity.
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        cell_embed_dim: int = 128,
        time_embed_dim: int = 64,
        node_dim: int = 256,
        time_dim_hidden: int = 128,
        delta_dim: int = 64,
        n_layers: int = 8,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        k_max: int = 50,           # accepted, ignored
        k_embed_dim: int = 16,     # accepted, ignored
    ):
        super().__init__()
        self.spatial_dim = spatial_dim

        # Input MLPs (LUNA's `mlp_in_*`).
        self.mlp_in_node = nn.Sequential(
            nn.Linear(cell_embed_dim, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
        )

        self.time_embed = _SinusoidalTimeEmbedding(time_embed_dim)
        self.mlp_in_time = nn.Sequential(
            nn.Linear(time_embed_dim, time_dim_hidden * 2),
            nn.ReLU(),
            nn.Linear(time_dim_hidden * 2, time_dim_hidden),
            nn.ReLU(),
        )

        self.mlp_in_pos = PositionsMLP(hidden_dim=node_dim)

        # Stack of transformer layers.
        self.layers = nn.ModuleList([
            _LunaTransformerLayer(
                node_dim=node_dim,
                time_dim=time_dim_hidden,
                delta_dim=delta_dim,
                spatial_dim=spatial_dim,
                n_heads=n_heads,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output: node features → small projection.
        self.mlp_out_node = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim),
        )

        # Norm-modulation MLP: takes [node || pos || |pos|] → scalar
        # new-norm. The output position is `pos * new_norm / |pos|`,
        # so direction comes from the running position and magnitude
        # comes from this MLP. Zero-init the LAST layer's weight + bias
        # so at init the model output is roughly zero — gradient signal
        # starts from a clean slate rather than random garbage.
        out_norm_in = node_dim + spatial_dim + 1
        self.mlp_out_pos_norm = nn.Sequential(
            nn.Linear(out_norm_in, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, 1),
        )
        nn.init.zeros_(self.mlp_out_pos_norm[-1].weight)
        nn.init.zeros_(self.mlp_out_pos_norm[-1].bias)

        self.eps = 1e-9

    def forward(
        self,
        z_t: torch.Tensor,                         # (n, spatial_dim)
        t: torch.Tensor,                           # (n,) flow time in [0, 1]
        cell_embed: torch.Tensor,                  # (n, cell_embed_dim)
        section_embed: Optional[torch.Tensor] = None,  # ignored
        k_target: Optional[torch.Tensor] = None,       # ignored
    ) -> torch.Tensor:
        n = z_t.shape[0]

        # Input projections.
        node = self.mlp_in_node(cell_embed)                          # (n, node_dim)
        time = self.mlp_in_time(self.time_embed(t))                  # (n, time_dim)
        pos = self.mlp_in_pos(z_t)                                   # (n, spatial_dim)

        # Transformer stack.
        for layer in self.layers:
            node, time, pos = layer(node, time, pos)

        # Output: norm-modulated position.
        node_out = self.mlp_out_node(node)                           # (n, node_dim)
        norm = torch.norm(pos, dim=-1, keepdim=True)                 # (n, 1)
        new_norm = self.mlp_out_pos_norm(
            torch.cat([node_out, pos, norm], dim=-1)
        )                                                              # (n, 1)
        new_pos = pos * new_norm / (norm + self.eps)
        # Final translation equivariance.
        new_pos = new_pos - new_pos.mean(dim=0, keepdim=True)

        return new_pos

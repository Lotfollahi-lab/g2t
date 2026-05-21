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

We use LUNA's exact attention implementation — the
``LinearAttentionTransformer`` from the
``linear_attention_transformer`` package (depth=1 = one full
transformer block with LN + linear attention + residual + LN + FFN
+ residual). Linear-time attention (O(n·d²) instead of standard
MHA's O(n²·d)) is essential for 1000-step DDPM reverse sampling
on multi-thousand-cell slices — without it inference would take
hours per slice on cortex data.

Install: ``pip install linear-attention-transformer``

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
import torch.nn.functional as F


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


# ---------------------------------------------------------------------------
# Linear-time attention (Katharopoulos et al. 2020, "Transformers are RNNs")
# Inline replacement for `linear_attention_transformer.LinearAttentionTransformer`
# (the library LUNA imports). Same math: φ(Q)·(φ(K)^T·V) with
# φ(x) = elu(x)+1, O(n·d²) instead of O(n²·d).
# ---------------------------------------------------------------------------


class _LinearAttention(nn.Module):
    """Single-head-merged linear attention with elu+1 feature map.

    Computes ``φ(Q) · ((φ(K)^T · V)) / (φ(Q) · (φ(K)^T · 1))``
    where ``φ(x) = elu(x) + 1`` ensures positivity (matches what
    lucidrains' ``linear_attention_transformer`` uses by default).
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0, (
            f"dim ({dim}) must be divisible by heads ({heads})"
        )
        self.heads = heads
        self.head_dim = dim // heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: (B, N, D),  mask: (B, N) bool — True for valid cells
        B, N, D = x.shape
        H, Hd = self.heads, self.head_dim

        qkv = self.to_qkv(x).reshape(B, N, 3, H, Hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # (B, H, N, Hd)

        # Positive feature map.
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Mask padding cells out of the K^T·V and K^T·1 reductions so
        # they don't contribute to any query's output. Q at padding
        # positions still produces an output; we mask that downstream
        # at the caller. The result is that padding cells in batch
        # element b never affect real cells in batch element b' (or
        # any other real cell).
        if mask is not None:
            m = mask.to(k.dtype).unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
            k = k * m
            v = v * m

        # K^T · V  →  (B, H, Hd, Hd). The "key-value summary" reused
        # across all queries — this is where linear attention's
        # O(n·d²) cost comes from (no n×n matrix).
        kv = torch.einsum("bhnd,bhne->bhde", k, v)

        # Numerator: Q · (K^T · V)  →  (B, H, N, Hd).
        num = torch.einsum("bhnd,bhde->bhne", q, kv)

        # Denominator: Q · (K^T · 1)  →  (B, H, N).
        k_sum = k.sum(dim=2)                                  # (B, H, Hd)
        denom = torch.einsum("bhnd,bhd->bhn", q, k_sum)       # (B, H, N)
        denom = denom.unsqueeze(-1).clamp_min(1e-6)           # (B, H, N, 1)

        out = num / denom                                     # (B, H, N, Hd)
        out = out.transpose(1, 2).reshape(B, N, D)            # (B, N, D)
        return self.to_out(out)


class _LinearAttentionBlock(nn.Module):
    """One pre-norm transformer block: LN + linear-attn + residual +
    LN + FFN + residual. Equivalent to
    ``LinearAttentionTransformer(dim, heads, depth=1)`` in lucidrains'
    library, which LUNA imports and uses.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _LinearAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class PositionsMLP(nn.Module):
    """Norm-only MLP: scales position magnitude by an MLP(|pos|),
    preserving direction. Translation-equivariant by construction
    when followed by mean subtraction.

    Direct port of LUNA's ``models/layers.PositionsMLP``.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, pos: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # pos: (B, N, spatial_dim), mask: (B, N) bool or None
        norm = torch.norm(pos, dim=-1, keepdim=True)             # (B, N, 1)
        new_norm = self.mlp(norm)                                 # (B, N, 1)
        new_pos = pos * new_norm / (norm + self.eps)
        # Translation equivariance: subtract the slice centroid,
        # masked so padding cells don't bias the mean.
        if mask is not None:
            m = mask.to(new_pos.dtype).unsqueeze(-1)              # (B, N, 1)
            new_pos = new_pos * m
            valid_count = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_pos = new_pos.sum(dim=1, keepdim=True) / valid_count
            new_pos = (new_pos - mean_pos) * m                    # re-mask
        else:
            new_pos = new_pos - new_pos.mean(dim=1, keepdim=True)
        return new_pos


class PositionNorm(nn.Module):
    """Scale positions by the mean-norm across cells in the slice,
    multiplied by a learned scalar weight. Direct port of LUNA's
    ``models/layers.PositionNorm``.

    Keeps the position magnitudes anchored as they propagate through
    the transformer stack — without this, the running positions can
    drift to scales the next layer wasn't expecting.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(
        self, pos: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # pos: (B, N, spatial_dim), mask: (B, N) bool or None
        norm = torch.norm(pos, dim=-1, keepdim=True)             # (B, N, 1)
        if mask is not None:
            m = mask.to(norm.dtype).unsqueeze(-1)
            valid_count = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_norm = (norm * m).sum(dim=1, keepdim=True) / valid_count
        else:
            mean_norm = norm.mean(dim=1, keepdim=True)            # (B, 1, 1)
        return self.weight * pos / (mean_norm + self.eps)


# ---------------------------------------------------------------------------
# Self-attention with position+time conditioning
# ---------------------------------------------------------------------------


class _LunaSelfAttention(nn.Module):
    """LUNA-style attention. Direct port of LUNA's
    ``models/self_attention.SelfAttention``:

      1. positions → direction MLP → delta embedding,
      2. concatenate [node, delta, time_broadcast] → Linear → hidden,
      3. **LinearAttentionTransformer(depth=1)** over all cells in the
         slice (linear-time attention + FFN, all internal),
      4. derive a fresh position via Linear(features) → spatial_dim.

    Returns (node_features_new, time_new, position_new). The time
    stream is updated by a separate Linear (LUNA's ``y_y`` projection).

    Args:
        max_seq_len: cap on cells per slice for LinearAttentionTransformer's
            positional bias buffer. LUNA uses 70_000 — large enough for
            any single MERFISH slice.
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
        # `transform_positions_for_attn_mlp`).
        self.pos_dir_mlp = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.ReLU(),
            nn.Linear(64, delta_dim),
        )

        # Identity-ish projection on node features before concat.
        self.lin_node = nn.Linear(node_dim, node_dim)

        # Project [node || delta || time_broadcast] back to node_dim.
        self.concat_proj = nn.Linear(node_dim + delta_dim + time_dim, node_dim)

        # Linear-attention transformer (LUNA's `self.attention`).
        # One full pre-norm transformer block: LN + linear-attn +
        # residual + LN + FFN + residual. Inlined (no external dep)
        # via `_LinearAttentionBlock` above. Mathematically equivalent
        # to `LinearAttentionTransformer(dim, heads, depth=1)` that
        # LUNA imports in `models/self_attention.py`.
        self.attention = _LinearAttentionBlock(
            dim=node_dim,
            heads=n_heads,
            ff_mult=4,
            dropout=dropout,
        )

        # Fresh position from attended features (LUNA's
        # `head_features_to_position`).
        self.feat_to_pos = nn.Linear(node_dim, spatial_dim)

        # Time-stream update (LUNA's `y_y`).
        self.time_update = nn.Linear(time_dim, time_dim)

    def forward(
        self,
        node_features: torch.Tensor,   # (B, N, node_dim)
        time_features: torch.Tensor,   # (B, N, time_dim) -- broadcast per cell
        positions: torch.Tensor,       # (B, N, spatial_dim)
        mask: Optional[torch.Tensor] = None,  # (B, N) bool
    ):
        # 1. Direction of position → delta
        norm = torch.norm(positions, dim=-1, keepdim=True).clamp_min(1e-7)
        direction = positions / norm                        # (B, N, spatial_dim)
        delta = self.pos_dir_mlp(direction)                 # (B, N, delta_dim)

        # 2. Concat + project (mirrors LUNA's `transform_node_features`).
        node_lin = self.lin_node(node_features)             # (B, N, node_dim)
        concat = torch.cat([node_lin, delta, time_features], dim=-1)
        h = self.concat_proj(concat)                        # (B, N, node_dim)

        # 3. Linear-attention transformer block. Mask isolates each
        # batch element so padding cells don't leak into real cells'
        # attention (and across-section attention doesn't happen).
        attn_out = self.attention(h, mask=mask)             # (B, N, node_dim)

        # 4. Fresh position from attended features.
        new_pos = self.feat_to_pos(attn_out)                # (B, N, spatial_dim)
        if mask is not None:
            # Zero out padding cells' positions so they don't propagate
            # into downstream PositionNorm / mean computations.
            new_pos = new_pos * mask.to(new_pos.dtype).unsqueeze(-1)

        # Time update.
        new_time = self.time_update(time_features)          # (B, N, time_dim)

        return attn_out, new_time, new_pos


# ---------------------------------------------------------------------------
# Transformer block (LUNA's TransformerLayer)
# ---------------------------------------------------------------------------


class _LunaTransformerLayer(nn.Module):
    """One transformer block. Direct port of LUNA's
    ``models/transformer.TransformerLayer``: attention with
    position+time conditioning, residual+LN+FFN+residual+LN on each of
    (node, time) streams, ``PositionNorm`` on positions (no residual on
    positions — LUNA recomputes them from features each layer).
    """

    def __init__(
        self,
        node_dim: int,
        time_dim: int,
        delta_dim: int,
        spatial_dim: int,
        n_heads: int,
        dim_ff_node: int,    # LUNA's `dim_ffX`
        dim_ff_time: int,    # LUNA's `dim_ffy`
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
            nn.Linear(node_dim, dim_ff_node),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff_node, node_dim),
            nn.Dropout(dropout),
        )

        # Time feature stream
        self.norm_time_1 = nn.LayerNorm(time_dim, eps=eps)
        self.norm_time_2 = nn.LayerNorm(time_dim, eps=eps)
        self.ff_time = nn.Sequential(
            nn.Linear(time_dim, dim_ff_time),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff_time, time_dim),
            nn.Dropout(dropout),
        )

        # PositionNorm on the layer's output position (LUNA's
        # `norm_positions_1`). Keeps position magnitudes anchored as
        # they flow through the stack.
        self.pos_norm = PositionNorm(eps=1e-8)

    def forward(self, node_features, time_features, positions, mask=None):
        # All tensors are (B, N, *). mask is (B, N) bool or None.
        # Attention with position+time conditioning.
        attn_node, attn_time, new_pos = self.attn(
            node_features, time_features, positions, mask=mask
        )

        # Position: LUNA replaces the old position with the
        # attention-derived one, then applies PositionNorm. No residual
        # connection on the position stream.
        new_pos = self.pos_norm(new_pos, mask=mask)
        # Final mean subtraction for translation equivariance (masked).
        if mask is not None:
            m = mask.to(new_pos.dtype).unsqueeze(-1)
            valid_count = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_pos = (new_pos * m).sum(dim=1, keepdim=True) / valid_count
            new_pos = (new_pos - mean_pos) * m
        else:
            new_pos = new_pos - new_pos.mean(dim=1, keepdim=True)

        # Node stream: residual + LN + FFN + residual + LN
        node = self.norm_node_1(node_features + attn_node)
        node = self.norm_node_2(node + self.ff_node(node))

        # Time stream: residual + LN + FFN + residual + LN
        time = self.norm_time_1(time_features + attn_time)
        time = self.norm_time_2(time + self.ff_time(time))

        return node, time, new_pos


# ---------------------------------------------------------------------------
# Top-level LunaTransformerNet — drop-in replacement for the velocity net
# ---------------------------------------------------------------------------


class LunaTransformerNet(nn.Module):
    """LUNA-style transformer; predicts x_0 (clean target positions).

    Direct port of LUNA's ``models/model.Model``. Hyperparameter names
    map to LUNA's config keys:

        hidden_mlp_dims.X   <-> hidden_mlp_x       (input node MLP hidden)
        hidden_mlp_dims.y   <-> hidden_mlp_y       (input time MLP hidden)
        hidden_mlp_dims.pos <-> hidden_mlp_pos     (PositionsMLP hidden)
        hidden_dims.dx      <-> node_dim
        hidden_dims.dy      <-> time_dim
        hidden_dims.dd      <-> delta_dim
        hidden_dims.dim_ffX <-> dim_ff_node
        hidden_dims.dim_ffy <-> dim_ff_time
        hidden_dims.num_heads <-> n_heads
        hidden_dims.output_features_to_pos_dims <-> output_features_dim

    LUNA's defaults (we mirror them exactly):
        n_layers=8, n_heads=16, node_dim=256, time_dim=1, delta_dim=64,
        dim_ff_node=256, dim_ff_time=256,
        hidden_mlp_x=256, hidden_mlp_y=256, hidden_mlp_pos=64,
        output_features_dim=4.

    Note `time_dim=1` is intentional in LUNA — the time stream stays a
    single scalar per cell through all blocks. Larger doesn't seem to
    help; we leave it at 1 by default.
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        cell_embed_dim: int = 128,
        time_embed_dim: int = 64,
        # LUNA's `hidden_dims`
        node_dim: int = 256,
        time_dim: int = 1,
        delta_dim: int = 64,
        dim_ff_node: int = 256,
        dim_ff_time: int = 256,
        output_features_dim: int = 4,
        # LUNA's `hidden_mlp_dims`
        hidden_mlp_x: int = 256,
        hidden_mlp_y: int = 256,
        hidden_mlp_pos: int = 64,
        # Stack + heads
        n_layers: int = 8,
        n_heads: int = 16,
        dropout: float = 0.1,
        # Accepted, ignored (API parity)
        k_max: int = 50,
        k_embed_dim: int = 16,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim

        if node_dim % n_heads != 0:
            raise ValueError(
                f"node_dim ({node_dim}) must be divisible by n_heads ({n_heads})"
            )

        # Input MLPs (LUNA's `mlp_in_*`).
        # Node: cell_embed_dim → hidden_mlp_x → node_dim
        self.mlp_in_node = nn.Sequential(
            nn.Linear(cell_embed_dim, hidden_mlp_x),
            nn.ReLU(),
            nn.Linear(hidden_mlp_x, node_dim),
            nn.ReLU(),
        )

        # Time: 1 → hidden_mlp_y → time_dim  (LUNA's input dim is 1,
        # a scalar). The sinusoidal embedding is here to give the
        # downstream MLP a richer view of t. Output stays at time_dim
        # which defaults to 1.
        self.time_embed = _SinusoidalTimeEmbedding(time_embed_dim)
        self.mlp_in_time = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_mlp_y),
            nn.ReLU(),
            nn.Linear(hidden_mlp_y, time_dim),
            nn.ReLU(),
        )

        # Position: PositionsMLP with LUNA's hidden_mlp_pos = 64.
        self.mlp_in_pos = PositionsMLP(hidden_dim=hidden_mlp_pos)

        # Stack of transformer layers.
        self.layers = nn.ModuleList([
            _LunaTransformerLayer(
                node_dim=node_dim,
                time_dim=time_dim,
                delta_dim=delta_dim,
                spatial_dim=spatial_dim,
                n_heads=n_heads,
                dim_ff_node=dim_ff_node,
                dim_ff_time=dim_ff_time,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output node MLP: bottleneck to `output_features_dim` (LUNA's
        # `output_features_to_pos_dims`, default 4). LUNA's idea: the
        # output norm MLP gets only a small feature summary alongside
        # the current position + |pos|. Going from 256 down to 4 forces
        # the network to compress what it knows about each cell.
        self.mlp_out_node = nn.Sequential(
            nn.Linear(node_dim, hidden_mlp_x),
            nn.ReLU(),
            nn.Linear(hidden_mlp_x, output_features_dim),
        )

        # Norm-modulation MLP: takes [node_summary(4) || pos(2) || |pos|(1)]
        # = 7 dims → hidden_mlp_x (256) → 1 (scalar new norm).
        # Direction is preserved from the running position; magnitude
        # comes from this MLP.
        #
        # Use the default (Kaiming / Xavier) initialisation: do NOT
        # zero-init the last layer. Zero-init produces x_hat_0 = 0 at
        # step 0, which makes `torch.cdist(zeros, zeros)` return all
        # zeros with NaN gradients (gradient of √(sum (xi - xj)²) at
        # coincident points is 0/0). LUNA itself uses default random
        # init here; we mirror that.
        out_norm_in = output_features_dim + spatial_dim + 1
        self.mlp_out_pos_norm = nn.Sequential(
            nn.Linear(out_norm_in, hidden_mlp_x),
            nn.ReLU(),
            nn.Linear(hidden_mlp_x, 1),
        )

        # NOTE: LUNA's `models/model.Model.__init__` also creates
        # `self.mlp_out_pos = PositionsMLP(hidden_mlp_dims["pos"])`
        # but the `forward()` method NEVER CALLS IT — it's dead code.
        # We intentionally do NOT create or call it here; applying an
        # extra PositionsMLP at the output adds a random norm-
        # modulation that corrupts the prediction.

        self.eps = 1e-9

    def forward(
        self,
        z_t: torch.Tensor,                         # (N, D) OR (B, N, D)
        t: torch.Tensor,                           # (N,) or (B, N)
        cell_embed: torch.Tensor,                  # (N, C) or (B, N, C)
        section_embed: Optional[torch.Tensor] = None,  # ignored
        k_target: Optional[torch.Tensor] = None,       # ignored
        mask: Optional[torch.Tensor] = None,           # (B, N) bool, only with batched input
    ) -> torch.Tensor:
        """Predict x_0 from noisy z_t.

        Accepts both single-section input (``z_t.dim() == 2``) and
        parallel-batched input (``z_t.dim() == 3``). In the batched
        path, ``mask`` must be provided to isolate sections from each
        other (LUNA's `to_dense_batch` + `node_mask` equivalent).
        """
        # Normalize input shape to (B, N, *).
        was_unbatched = z_t.dim() == 2
        if was_unbatched:
            z_t = z_t.unsqueeze(0)                                    # (1, N, D)
            cell_embed = cell_embed.unsqueeze(0)                      # (1, N, C)
            t = t.unsqueeze(0)                                        # (1, N)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, N = z_t.shape[:2]

        # Input projections.
        node = self.mlp_in_node(cell_embed)                           # (B, N, node_dim)
        # Time embedding: t shape (B, N) -> sinusoidal (B, N, time_embed_dim)
        # The sinusoidal embed works on any tensor shape (leading dims preserved).
        time = self.mlp_in_time(self.time_embed(t))                   # (B, N, time_dim)
        pos = self.mlp_in_pos(z_t, mask=mask)                         # (B, N, spatial_dim)

        # Transformer stack.
        for layer in self.layers:
            node, time, pos = layer(node, time, pos, mask=mask)

        # Output stage: norm-modulated position. LUNA's exact recipe.
        node_out = self.mlp_out_node(node)                            # (B, N, output_features_dim)
        norm = torch.norm(pos, dim=-1, keepdim=True)                  # (B, N, 1)
        new_norm = self.mlp_out_pos_norm(
            torch.cat([node_out, pos, norm], dim=-1)
        )                                                              # (B, N, 1)
        new_pos = pos * new_norm / (norm + self.eps)
        # Final translation equivariance — matches LUNA's exact final
        # step in `models/model.Model.forward`. Masked when batched.
        if mask is not None:
            m = mask.to(new_pos.dtype).unsqueeze(-1)
            valid_count = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_pos = (new_pos * m).sum(dim=1, keepdim=True) / valid_count
            new_pos = (new_pos - mean_pos) * m
        else:
            new_pos = new_pos - new_pos.mean(dim=1, keepdim=True)

        if was_unbatched:
            new_pos = new_pos.squeeze(0)
        return new_pos

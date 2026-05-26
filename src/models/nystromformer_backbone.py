"""Nyströmformer backbone — sub-quadratic global self-attention.

Implements Xiong et al. 2021, "Nyströmformer: A Nyström-based
Algorithm for Approximating Self-Attention" (arXiv:2102.03902).
Replaces the dense softmax attention matrix
``A = softmax(Q K^T / sqrt(d)) ∈ R^{N×N}`` with the Nyström matrix
approximation

    A ≈ F · B^+ · G                                                (1)

where, after partitioning the N tokens into M segments and computing
landmark queries / keys as segment means ``Q̃, K̃``:

    F = softmax( Q   · K̃^T / sqrt(d) )      ∈ R^{N×M}
    B = softmax( Q̃ · K̃^T / sqrt(d) )       ∈ R^{M×M}
    G = softmax( Q̃ · K^T / sqrt(d) )        ∈ R^{M×N}

and B^+ is the Moore-Penrose pseudo-inverse of B.

Output = F · B^+ · G · V, with cost O(N · M · d) per attention
call (vs O(N² · d) for dense softmax). As M → N the approximation
becomes exact — that's the formal guarantee that distinguishes
Nyström from heuristic linear-attention kernels (Performer / FAVOR+).

Why this matches the cell-cell setting
--------------------------------------
The user's biological argument applies cleanly here:
* Landmarks are computed by **segment-mean pooling** over the
  ordered cell array (B, N, d). They are NOT clusters of similar
  cells — they're pooled summaries of contiguous blocks. So the
  approximation never assumes that "transcriptionally similar
  cells should attend more strongly to each other".
* Every cell attends to every other cell THROUGH the landmarks.
  The information path is cell_i → all M landmarks → cell_j for any
  i, j. So reachability is preserved exactly the way full attention
  preserves it.
* M is a knob — large M makes the approximation tighter at the cost
  of compute.

Padding handling
----------------
Padding cells are excluded from:
  * Landmark computation (their contributions are zeroed before
    segment mean, and the segment count is corrected so the mean
    is over real cells only).
  * Output: the F·B^+·G·V product is masked back to zero at PAD
    positions on the OUTPUT side (consistent with how the other
    backbones handle padding).
  * Softmax of G's rows: keys at PAD positions get -inf bias so they
    contribute zero weight. F doesn't need a key mask since its
    keys are the landmarks, which are computed only from real cells.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data.dataholder import DataHolder

# Reuse the DiT primitives so time conditioning is consistent across
# the new backbones.
from models.dit_backbone import (
    TimestepEmbedder,
    _modulate,
)
from models.sdpa_attention import SDPAMultiheadAttention


# ---------------------------------------------------------------------------
# Iterative Moore-Penrose pseudo-inverse
# ---------------------------------------------------------------------------


def _moore_penrose_iter(A: torch.Tensor, n_iter: int = 6) -> torch.Tensor:
    """Iterative Moore-Penrose pseudo-inverse via the Schulz / Razavi
    cubic iteration (Razavi et al. 2014, used in the Nyströmformer
    paper). For a square positive matrix ``A`` of shape ``(..., M, M)``
    returns ``A^+`` of the same shape.

    Why not ``torch.linalg.pinv``: pinv is SVD-based and unstable for
    near-singular matrices that we routinely encounter at init (when
    softmax outputs are nearly uniform). Schulz iteration converges
    cubically when the initial estimate is close enough, and is
    stable on near-singular inputs.

    Initial estimate: ``Z_0 = A^T / (||A||_∞ · ||A||_1)`` (the
    standard Ben-Israel/Greville initialisation; bounds the spectrum
    of ``A·Z_0`` into (0, 2) so the iteration converges).
    """
    # ||A||_∞ = max row sum of absolute values; ||A||_1 = max col sum.
    abs_A = A.abs()
    col_sum = abs_A.sum(dim=-1)  # (..., M)  — sum over columns per row
    row_sum = abs_A.sum(dim=-2)  # (..., M)  — sum over rows per col
    # max over rows / cols
    inf_norm = col_sum.amax(dim=-1, keepdim=True)  # (..., 1) — ||A||_∞
    one_norm = row_sum.amax(dim=-1, keepdim=True)  # (..., 1) — ||A||_1
    # Initial estimate; the * .unsqueeze(-1) broadcasts the scalar
    # norm across the M-axis of A.T.
    scale = (inf_norm * one_norm).unsqueeze(-1).clamp_min(1e-12)
    Z = A.transpose(-2, -1) / scale  # (..., M, M)
    I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    # Broadcast I across any leading batch dims.
    I = I.expand(A.shape)

    # Cubic Schulz iteration. Each step:
    #   Z ← 1/4 · Z · ( 13·I − A·Z · (15·I − A·Z · (7·I − A·Z)) )
    for _ in range(n_iter):
        AZ = A @ Z
        Z = 0.25 * Z @ (13.0 * I - AZ @ (15.0 * I - AZ @ (7.0 * I - AZ)))
    return Z


# ---------------------------------------------------------------------------
# Segment-mean pooling for landmark computation
# ---------------------------------------------------------------------------


def _segment_mean_pool(
    x: torch.Tensor,
    mask: torch.Tensor,
    M: int,
) -> torch.Tensor:
    """Partition the N positions into M (approximately equal-size)
    contiguous segments and compute the mean of each, masking padding
    cells out before the average.

    Args:
        x:    (B, H, N, D)  — per-token features (already split into heads).
        mask: (B, N)        — bool, True at real cells.
        M:    int           — number of landmarks (segments).

    Returns:
        (B, H, M, D)  — landmark features.

    Notes:
        * Segment IDs are computed as ``seg_id[n] = (n * M) // N``;
          this distributes the N positions into M roughly-equal-size
          buckets, with the last bucket possibly slightly larger.
        * If a segment ends up with ZERO real cells (all PAD), we set
          its denominator to 1 to avoid division by zero — the
          resulting "landmark" is the zero vector, which contributes
          nothing to softmax weights downstream (it's the worst
          possible key for similarity to any real query). This is
          fine for the approximation; happens only on slices with
          extreme padding.
    """
    B, H, N, D = x.shape
    device = x.device
    dtype = x.dtype

    # Segment IDs: (N,) → maps each cell-index to a landmark bucket.
    n_idx = torch.arange(N, device=device)
    seg_id = (n_idx * M) // N                                  # (N,)
    seg_id = seg_id.clamp(max=M - 1)  # safety against off-by-one at n=N-1

    # Mask out padding: real cells contribute their features, PAD
    # contributes zero.
    m_float = mask.to(dtype).unsqueeze(1).unsqueeze(-1)        # (B, 1, N, 1)
    x_masked = x * m_float                                      # (B, H, N, D)

    # Sum per segment via scatter_add. We need an index tensor of shape
    # (B, H, N, D) where every entry says which OUTPUT segment that
    # input cell goes into.
    seg_id_expanded = seg_id.view(1, 1, N, 1).expand(B, H, N, D)
    sums = torch.zeros(B, H, M, D, device=device, dtype=dtype)
    sums.scatter_add_(2, seg_id_expanded, x_masked)            # (B, H, M, D)

    # Count of real cells per segment. (B, N) mask + (N,) seg_id
    # → (B, M) counts.
    cnt = torch.zeros(B, M, device=device, dtype=dtype)
    cnt_idx = seg_id.view(1, N).expand(B, N)
    cnt.scatter_add_(1, cnt_idx, mask.to(dtype))               # (B, M)
    # Avoid div-by-zero for empty segments (see docstring).
    cnt = cnt.clamp_min(1.0).view(B, 1, M, 1)                  # (B, 1, M, 1)

    landmarks = sums / cnt                                     # (B, H, M, D)
    return landmarks


# ---------------------------------------------------------------------------
# Nyström attention layer
# ---------------------------------------------------------------------------


class NystromAttention(nn.Module):
    """Multi-head Nyström-approximated self-attention.

    Args:
        embed_dim: per-token feature width.
        num_heads: number of attention heads. ``embed_dim % num_heads``
            must be 0.
        n_landmarks: M — number of landmark tokens. Default 64.
            Larger M tightens the approximation; M = N gives exact
            attention (modulo the iterative pseudo-inverse error).
        moore_penrose_iters: number of Schulz iterations for B^+.
            6 is the Nyströmformer paper's default and is enough
            to drive the error well below the natural noise floor
            on softmax matrices.
        exact_until_n: if N ≤ this threshold, fall through to exact
            ``F.scaled_dot_product_attention`` on this forward call
            instead of computing the Nyström approximation. Both
            paths use the SAME projection weights so model behaviour
            stays correct across the switch. Useful for datasets
            that mix small and large slices.
        dropout: applied to the final output projection.
        bias: include bias on q/k/v/out linears.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        n_landmarks: int = 64,
        moore_penrose_iters: int = 6,
        exact_until_n: int = 0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads "
                f"{num_heads} (per-head dim must be integer)."
            )
        self.head_dim = self.embed_dim // self.num_heads
        self.n_landmarks = int(n_landmarks)
        self.moore_penrose_iters = int(moore_penrose_iters)
        self.exact_until_n = int(exact_until_n)
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

    def _exact_attention(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Exact softmax via SDPA — same routing as
        ``SDPAMultiheadAttention``. Used when N is small enough that
        N² attention is cheaper than the Nyström sequence.
        """
        B, H, N, D = q.shape
        sdpa_mask: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            mask_dtype = q.dtype
            neg_inf = torch.finfo(mask_dtype).min
            pad_bias = torch.where(
                key_padding_mask,
                torch.tensor(neg_inf, dtype=mask_dtype, device=q.device),
                torch.tensor(0.0,     dtype=mask_dtype, device=q.device),
            )
            sdpa_mask = pad_bias.view(B, 1, 1, N)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim).
            key_padding_mask: (B, N) bool, True at PAD positions
                (PyTorch's MultiheadAttention convention).

        Returns:
            (B, N, embed_dim).
        """
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        M = self.n_landmarks

        # Project and split heads.
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)    # (B, H, N, D)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)    # (B, H, N, D)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)    # (B, H, N, D)

        # Hybrid switch — fall through to exact SDPA when the slice is
        # small enough that O(N²) is the right trade.
        if N <= self.exact_until_n or N <= M:
            out = self._exact_attention(q, k, v, key_padding_mask)
            out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
            return self.out_proj(out)

        # Build a real-cell mask if not given (all real → no-op).
        if key_padding_mask is None:
            real_mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        else:
            real_mask = ~key_padding_mask                       # True at REAL
        m_float = real_mask.to(q.dtype)                         # (B, N)

        # Compute landmark queries and keys via segment-mean pooling.
        # IMPORTANT: pool real cells only — masked PAD cells contribute
        # zeros and the per-segment count is corrected.
        q_tilde = _segment_mean_pool(q, real_mask, M)           # (B, H, M, D)
        k_tilde = _segment_mean_pool(k, real_mask, M)           # (B, H, M, D)

        scale = 1.0 / math.sqrt(D)

        # F = softmax( Q · K̃^T / sqrt(d) )      ∈ (B, H, N, M)
        F_logits = q @ k_tilde.transpose(-2, -1) * scale
        # No key mask on F — its keys are landmarks (M of them), all valid.
        F_mat = F_logits.softmax(dim=-1)

        # B = softmax( Q̃ · K̃^T / sqrt(d) )    ∈ (B, H, M, M)
        B_logits = q_tilde @ k_tilde.transpose(-2, -1) * scale
        B_mat = B_logits.softmax(dim=-1)

        # G = softmax( Q̃ · K^T / sqrt(d) )      ∈ (B, H, M, N)
        # Keys here ARE the original cells, so we need to mask PAD
        # positions with -inf before the softmax over the last (N)
        # axis. Use finfo.min to avoid NaN in fp16 paths if a query
        # row attends only to PAD (shouldn't happen, but defensive).
        G_logits = q_tilde @ k.transpose(-2, -1) * scale       # (B, H, M, N)
        if key_padding_mask is not None:
            neg_inf = torch.finfo(G_logits.dtype).min
            key_bias = torch.where(
                key_padding_mask,                                # (B, N)
                torch.tensor(neg_inf, dtype=G_logits.dtype, device=q.device),
                torch.tensor(0.0,     dtype=G_logits.dtype, device=q.device),
            ).view(B, 1, 1, N)
            G_logits = G_logits + key_bias
        G_mat = G_logits.softmax(dim=-1)

        # B^+ via iterative Schulz / Razavi (cubic convergence).
        B_pinv = _moore_penrose_iter(B_mat, n_iter=self.moore_penrose_iters)

        # Final approximation: F · B^+ · G · V.
        # Shapes: F (B,H,N,M) · B_pinv (B,H,M,M) · G (B,H,M,N) · V (B,H,N,D)
        # Multiply right-to-left for memory efficiency:
        #   GV = G · V          → (B, H, M, D)
        #   BpGV = B_pinv · GV  → (B, H, M, D)
        #   out = F · BpGV      → (B, H, N, D)
        # This avoids ever materialising an N×N intermediate.
        GV = G_mat @ v
        BpGV = B_pinv @ GV
        out = F_mat @ BpGV                                      # (B, H, N, D)

        # NOTE: we don't zero PAD output rows here. ``out_proj`` has
        # bias, so a zero input would still produce a nonzero output
        # — the bias term would leak. The correctness invariant the
        # layer must satisfy is "outputs at REAL positions don't
        # depend on PAD-position values", which is enforced via the
        # padding-aware landmark pooling (segment_mean_pool excludes
        # PAD) + the key_padding_mask on G. PAD-row zeroing is the
        # backbone's job (DiT / Perceiver follow the same convention)
        # so the final head can mask features and positions in one
        # place.

        # Merge heads.
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        out = self.out_proj(out)
        return out


# ---------------------------------------------------------------------------
# Nyströmformer block (Nyström attention + FFN, adaLN-Zero conditioned)
# ---------------------------------------------------------------------------


class NystromBlock(nn.Module):
    """One Nyströmformer block. Mirrors ``DiTBlock`` 1-for-1 except
    the attention call is ``NystromAttention`` instead of dense SDPA.
    Same adaLN-Zero recipe so each block starts as identity.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_landmarks: int,
        moore_penrose_iters: int,
        exact_until_n: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)

        self.norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = NystromAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            n_landmarks=n_landmarks,
            moore_penrose_iters=moore_penrose_iters,
            exact_until_n=exact_until_n,
        )

        self.norm2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        ff_hidden = int(self.hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, self.hidden_dim),
        )

        # adaLN-Zero modulation (6 vectors: shift/scale/gate × MSA/MLP).
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 6 * self.hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # MSA sub-layer (Nyström attention).
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(h, key_padding_mask=key_padding_mask)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # FFN sub-layer.
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# Final layer — shared with DiT shape, but kept local for clarity.
# ---------------------------------------------------------------------------


class _NystromFinalLayer(nn.Module):
    """adaLN-modulated final norm + linear projection to
    (features, positions). Same structure as DiT's final layer; we
    don't import it to keep this module independent of dit_backbone's
    internals (the dit one might evolve separately).
    """

    def __init__(self, hidden_dim: int, out_features_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.linear = nn.Linear(hidden_dim, out_features_dim + 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Nyströmformer backbone — public interface (DataHolder in / DataHolder out)
# ---------------------------------------------------------------------------


class NystromformerBackbone(nn.Module):
    """Drop-in replacement for ``models.model.Model`` using Nyström-
    approximated global self-attention.

    Architecture mirrors ``DiTBackbone`` (gene + position → token,
    L blocks of Nyström MSA + FFN with adaLN-Zero time conditioning,
    final adaLN → projection). The only difference is the attention
    implementation inside each block.

    Cost vs DiT:
        DiT:           O(N² · d) compute, O(N²) memory (or O(N) with SDPA)
        Nyströmformer: O(N · M · d) compute, O(N · M) memory

    Quality vs LUNA's kernel-based linear attention:
        Nyström has a formal approximation guarantee (the Nyström
        decomposition is a truncated low-rank factorisation of the
        kernel matrix) and approaches exact softmax as M → N. The
        kernel-feature-map approximation used by
        LinearAttentionTransformer (Φ(Q) (Φ(K)^T V)) has no analogous
        convergence property — its quality depends on the heuristic
        choice of Φ.

    Output interface: DataHolder in → DataHolder out, with
    ``pred.node_features`` of width ``output_features_to_pos_dims``
    and ``pred.positions`` of width 2. Compatible with EDM /
    coarse-to-fine / hierarchical wrappers identically to DiT.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        n_layers: int,
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        output_dims: Dict[str, int],
        nystromformer_cfg=None,
    ):
        super().__init__()
        self.out_features_dim = int(hidden_dims["output_features_to_pos_dims"])

        def _g(k, default):
            if nystromformer_cfg is None:
                return default
            return (
                nystromformer_cfg.get(k, default)
                if hasattr(nystromformer_cfg, "get")
                else getattr(nystromformer_cfg, k, default)
            )

        self.hidden_dim = int(_g("hidden_dim", 256))
        self.n_heads = int(_g("n_heads", 8))
        self.mlp_ratio = float(_g("mlp_ratio", 4))
        # Default n_layers comes from cfg.model.n_layers; the
        # backbone-specific override takes precedence.
        self.n_layers = int(_g("n_layers", n_layers))
        self.time_embed_dim = int(_g("time_embed_dim", 256))
        self.n_landmarks = int(_g("n_landmarks", 64))
        self.moore_penrose_iters = int(_g("moore_penrose_iters", 6))
        # Exact-fallthrough threshold: when a slice has N ≤
        # ``exact_until_n`` cells, the layer uses exact SDPA softmax
        # instead of Nyström. 0 = always use Nyström. A sensible
        # default for mixed datasets is ~16_000 (slices smaller than
        # that pay less compute under exact softmax than under
        # Nyström's M-landmark machinery).
        self.exact_until_n = int(_g("exact_until_n", 0))

        gene_in = int(input_dims["node_features_dimensions"])

        # Input projections (gene + position → token), matching DiT.
        self.gene_embed = nn.Sequential(
            nn.Linear(gene_in, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Time embedding.
        self.t_embed = TimestepEmbedder(self.hidden_dim, self.time_embed_dim)

        # Stack of Nyström blocks.
        self.blocks = nn.ModuleList([
            NystromBlock(
                hidden_dim=self.hidden_dim,
                n_heads=self.n_heads,
                n_landmarks=self.n_landmarks,
                moore_penrose_iters=self.moore_penrose_iters,
                exact_until_n=self.exact_until_n,
                mlp_ratio=self.mlp_ratio,
            )
            for _ in range(self.n_layers)
        ])

        # Output head.
        self.final = _NystromFinalLayer(self.hidden_dim, self.out_features_dim)

    def forward(self, data: DataHolder, **_unused_kwargs) -> DataHolder:
        node_mask = data.node_mask                            # (B, N) bool

        # Build per-cell tokens.
        tok = self.gene_embed(data.node_features)             # (B, N, D)
        tok = tok + self.pos_embed(data.positions)
        tok = tok * node_mask.unsqueeze(-1).to(tok.dtype)

        # Time conditioning.
        t = data.diffusion_time
        if t.dim() > 1:
            t = t.view(-1)
        c = self.t_embed(t)                                    # (B, D)

        # NystromAttention's key_padding_mask convention matches MHA's:
        # True at PAD.
        key_padding_mask = ~node_mask                          # (B, N) bool

        for block in self.blocks:
            tok = block(tok, c, key_padding_mask=key_padding_mask)

        # Final projection: features + positions.
        out = self.final(tok, c)                               # (B, N, F+2)
        features = out[..., :-2]
        positions = out[..., -2:]

        # Mask + mean-centre positions (same convention as DiT).
        pad_mask_features = node_mask.unsqueeze(-1).to(features.dtype)
        features = features * pad_mask_features
        positions = positions * pad_mask_features
        n_real = node_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1).to(
            positions.dtype
        )
        positions = positions - positions.sum(dim=1, keepdim=True) / n_real
        positions = positions * pad_mask_features

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

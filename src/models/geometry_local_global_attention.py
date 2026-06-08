"""Geometry Local-Global Attention (geomattn_localglobal) — sub-quadratic
geometry-aware attention via exact local + landmark global.

Idea
----
Dense geometry-coupled attention (``geomattn``) is O(N²): it scores every
cell against every other cell. But under the current coordinate estimate
x_t, the cells that matter most to cell i are its spatial NEIGHBOURS;
the far field only needs a coarse summary. So:

* **Local (exact):** cell i attends with full softmax to its top-K
  nearest cells (kNN under x_t), with the usual learnable, SE(2)-invariant
  geometric bias on the K neighbour distances. Cost O(N·K).
* **Global (approximate):** the far field is summarised by M segment-mean
  **landmarks** (Nyström-style), each with a centroid position so the
  geometric bias applies at the landmark level. Cost O(N·M).

The kNN depends only on positions, which are fixed within one forward, so
it is computed ONCE per forward (chunked cdist+topk: O(N²) compute but
O(N) memory, shared across all layers and heads) — not per-layer-per-head
like dense attention. Every layer's local branch is then a cheap O(N·K)
gather.

Two combination schemes (config ``combine``)
--------------------------------------------
* ``"gated"``: two independent, individually-correct softmaxes —
  local-exact and Nyström-global (F·B⁺·G) — blended by a learned per-head
  sigmoid gate: ``out = g·local + (1-g)·global``. No double-counting; each
  branch is a valid normalised attention output. With g→1 and K=N it
  equals dense softmax.
* ``"unified"``: ONE softmax whose key set is {K nearest cells (exact)}
  ∪ {M landmarks}. Each landmark l contributes with weight equal to its
  *effective* cell count for query i — the number of real cells it
  summarises MINUS how many of i's local neighbours already fall in it.
  This exclusion means at K=N every landmark's effective count is 0, so
  the far-field term vanishes and the output is EXACTLY dense
  geometry-biased softmax. A genuine sparse + coarse softmax
  approximation, exact in the K→N limit.

Both are O(N·(K+M)) — sub-quadratic. SE(2)-invariant (all biases are
functions of distances only). Zero-init geometry gate ⇒ the geometric
bias is 0 at init.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from utils.data.dataholder import DataHolder
from models.dit_backbone import TimestepEmbedder, DiTFinalLayer, _modulate
from models.geometry_coupled_attention import RadialBasis
from models.nystromformer_backbone import _segment_mean_pool, _moore_penrose_iter


# ---------------------------------------------------------------------------
# kNN under the current coordinate estimate (chunked, once per forward)
# ---------------------------------------------------------------------------


def knn_chunked(
    pos: torch.Tensor,            # (B, N, 2)
    real_mask: torch.Tensor,      # (B, N) bool, True at REAL cells
    K: int,
    chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-K nearest neighbours (smallest distance, self included) under
    the Euclidean metric on ``pos``. Padding cells are excluded as
    CANDIDATES (their distance is set to +inf before the top-k).

    Chunked over the query axis so peak memory is O(chunk·N), not O(N²).
    Compute is O(N²) but paid ONCE per forward (positions are fixed
    within a forward) and shared across all layers / heads.

    Returns:
        idx:   (B, N, Keff) long — neighbour cell indices.
        dist:  (B, N, Keff) float — neighbour distances (0 where invalid).
        valid: (B, N, Keff) bool — False where the "neighbour" is a
               padding slot (only happens when a slice has < K real
               cells); masked out of the local softmax downstream.
    where Keff = min(K, N).
    """
    B, N, _ = pos.shape
    Keff = min(int(K), N)
    idx_out = torch.zeros(B, N, Keff, dtype=torch.long, device=pos.device)
    dist_out = torch.zeros(B, N, Keff, dtype=pos.dtype, device=pos.device)

    inf = torch.tensor(float("inf"), dtype=pos.dtype, device=pos.device)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        d = torch.cdist(pos[:, s:e], pos)                      # (B, c, N)
        # Exclude padding candidates from the top-k.
        d = torch.where(real_mask[:, None, :], d, inf)
        dk, ik = torch.topk(d, Keff, dim=-1, largest=False)    # (B, c, Keff)
        idx_out[:, s:e] = ik
        dist_out[:, s:e] = dk

    valid = torch.isfinite(dist_out)
    dist_out = torch.where(valid, dist_out, torch.zeros_like(dist_out))
    return idx_out, dist_out, valid


def _gather_neighbors(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather neighbour features without materialising an N×N tensor.

    Args:
        x:   (B, H, N, D) per-head features.
        idx: (B, N, K) neighbour indices.
    Returns:
        (B, H, N, K, D).
    """
    B, H, N, D = x.shape
    K = idx.shape[-1]
    # (B, N, K) → (B, H, N*K) index along the key (dim=2) axis.
    idx_e = idx.unsqueeze(1).expand(B, H, N, K).reshape(B, H, N * K)
    idx_e = idx_e.unsqueeze(-1).expand(B, H, N * K, D)         # (B,H,N*K,D)
    out = torch.gather(x, 2, idx_e)                            # (B,H,N*K,D)
    return out.view(B, H, N, K, D)


# ---------------------------------------------------------------------------
# Per-head radial bias (shared helper)
# ---------------------------------------------------------------------------


def radial_bias(
    dist: torch.Tensor,           # (B, R, S)
    centers: torch.Tensor,        # (Kr,)
    widths: torch.Tensor,         # (Kr,)
    W: torch.Tensor,              # (H, Kr)
    gamma: torch.Tensor,          # (B, H)
) -> torch.Tensor:
    """γ_h · Σ_k W_{h,k} φ_k(dist). Accumulates over the Kr RBFs on the
    fly (no (B,R,S,Kr) tensor). Returns (B, H, R, S)."""
    B, R, S = dist.shape
    H, Kr = W.shape
    out = dist.new_zeros(B, H, R, S)
    for k in range(Kr):
        phi = torch.exp(-0.5 * ((dist - centers[k]) / widths[k]) ** 2)
        out = out + W[:, k].view(1, H, 1, 1) * phi.unsqueeze(1)
    return gamma.view(B, H, 1, 1) * out


# ---------------------------------------------------------------------------
# Local-Global attention
# ---------------------------------------------------------------------------


class LocalGlobalGeometryAttention(nn.Module):
    """Exact local (kNN) + landmark global geometry-aware attention.

    Shared q/k/v projections feed both branches (the branches differ in
    WHICH keys they attend to, not in the projection). ``combine`` picks
    the gated or unified scheme.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_rbf: int,
        n_local: int = 32,
        n_landmarks: int = 64,
        moore_penrose_iters: int = 6,
        combine: str = "gated",
        bias: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )
        if combine not in ("gated", "unified"):
            raise ValueError(f"combine must be 'gated' or 'unified'; got {combine!r}")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.num_rbf = int(num_rbf)
        self.n_local = int(n_local)
        self.n_landmarks = int(n_landmarks)
        self.moore_penrose_iters = int(moore_penrose_iters)
        self.combine = combine

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

        # Per-head radial kernel + zero-init geometry gate (shared by
        # both branches — "the" geometric kernel for that head).
        self.kernel = nn.Linear(self.num_rbf, self.num_heads, bias=False)
        self.geo_gate = nn.Linear(self.embed_dim, self.num_heads)
        nn.init.zeros_(self.geo_gate.weight)
        nn.init.zeros_(self.geo_gate.bias)

        # Per-head local/global blend gate (gated mode only). Zero-init →
        # sigmoid(0)=0.5 → equal blend at start.
        if self.combine == "gated":
            self.blend_gate = nn.Linear(self.embed_dim, self.num_heads)
            nn.init.zeros_(self.blend_gate.weight)
            nn.init.zeros_(self.blend_gate.bias)

    # ------------------------------------------------------------------
    def _bias(self, dist, radial, gamma):
        widths = torch.exp(radial.log_widths) + 1e-6
        return radial_bias(dist, radial.centers, widths, self.kernel.weight, gamma)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                  # (B, N, embed_dim)
        c: torch.Tensor,                  # (B, embed_dim) time embedding
        radial: RadialBasis,
        # Precomputed-once geometry (passed from the backbone):
        nbr_idx: torch.Tensor,            # (B, N, K)
        nbr_dist: torch.Tensor,           # (B, N, K)
        nbr_valid: torch.Tensor,          # (B, N, K) bool
        positions: torch.Tensor,          # (B, N, 2)  (detached upstream)
        centroids: torch.Tensor,          # (B, M, 2)
        seg_id: torch.Tensor,             # (N,) landmark id per cell index
        base_count: torch.Tensor,         # (B, M) real cells per landmark
        real_mask: torch.Tensor,          # (B, N) bool
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        M = self.n_landmarks
        K = nbr_idx.shape[-1]
        scale = 1.0 / math.sqrt(D)
        neg_inf = torch.finfo(x.dtype).min

        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)       # (B,H,N,D)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)
        gamma = self.geo_gate(c)                                  # (B,H)

        # ---- LOCAL exact branch (logits over K neighbours) ----
        k_nbr = _gather_neighbors(k, nbr_idx)                     # (B,H,N,K,D)
        v_nbr = _gather_neighbors(v, nbr_idx)                     # (B,H,N,K,D)
        # scores: q_i · k_neighbour
        s_local = torch.einsum("bhnd,bhnkd->bhnk", q, k_nbr) * scale
        s_local = s_local + self._bias(nbr_dist, radial, gamma)   # (B,H,N,K)
        # Mask invalid neighbours (padding slots when a slice has < K
        # real cells). nbr_valid is (B,N,K) → broadcast over heads.
        s_local = s_local.masked_fill(~nbr_valid.unsqueeze(1), neg_inf)

        if self.combine == "gated":
            a_local = s_local.softmax(dim=-1)                     # (B,H,N,K)
            out_local = torch.einsum("bhnk,bhnkd->bhnd", a_local, v_nbr)
            out_global = self._nystrom_global(
                q, k, v, positions, centroids, radial, gamma,
                real_mask, key_padding_mask,
            )                                                     # (B,H,N,D)
            g = torch.sigmoid(self.blend_gate(c)).view(B, H, 1, 1)
            out = g * out_local + (1.0 - g) * out_global
        else:  # unified
            out = self._unified(
                q, k, v, v_nbr, s_local, positions, centroids, seg_id,
                base_count, nbr_idx, nbr_valid, radial, gamma,
                key_padding_mask, scale, neg_inf,
            )

        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    def _nystrom_global(
        self, q, k, v, positions, centroids, radial, gamma,
        real_mask, key_padding_mask,
    ) -> torch.Tensor:
        """Far field via Nyström F·B⁺·G with landmark-centroid geometry
        bias. (Same math as geomattn_nystrom; used by the gated scheme.)"""
        B, H, N, D = q.shape
        M = self.n_landmarks
        scale = 1.0 / math.sqrt(D)

        q_tilde = _segment_mean_pool(q, real_mask, M)             # (B,H,M,D)
        k_tilde = _segment_mean_pool(k, real_mask, M)

        d_NM = torch.cdist(positions, centroids)                  # (B,N,M)
        d_MM = torch.cdist(centroids, centroids)                  # (B,M,M)
        d_MN = d_NM.transpose(1, 2)                               # (B,M,N)

        F_logits = (q @ k_tilde.transpose(-2, -1)) * scale
        F_logits = F_logits + self._bias(d_NM, radial, gamma)
        F_mat = F_logits.softmax(dim=-1)

        B_logits = (q_tilde @ k_tilde.transpose(-2, -1)) * scale
        B_logits = B_logits + self._bias(d_MM, radial, gamma)
        B_mat = B_logits.softmax(dim=-1)

        G_logits = (q_tilde @ k.transpose(-2, -1)) * scale
        G_logits = G_logits + self._bias(d_MN, radial, gamma)
        if key_padding_mask is not None:
            neg_inf = torch.finfo(G_logits.dtype).min
            kb = torch.where(
                key_padding_mask,
                torch.tensor(neg_inf, dtype=G_logits.dtype, device=q.device),
                torch.tensor(0.0, dtype=G_logits.dtype, device=q.device),
            ).view(B, 1, 1, N)
            G_logits = G_logits + kb
        G_mat = G_logits.softmax(dim=-1)

        B_pinv = _moore_penrose_iter(B_mat, n_iter=self.moore_penrose_iters)
        return F_mat @ (B_pinv @ (G_mat @ v))                     # (B,H,N,D)

    # ------------------------------------------------------------------
    def _unified(
        self, q, k, v, v_nbr, s_local, positions, centroids, seg_id,
        base_count, nbr_idx, nbr_valid, radial, gamma,
        key_padding_mask, scale, neg_inf,
    ) -> torch.Tensor:
        """One softmax over {K local cells} ∪ {M landmarks}, landmark l
        weighted by its per-query effective count (cells it summarises
        minus i's local neighbours already in it). Count-exclusion makes
        the far field vanish at K=N ⇒ exact dense softmax in that limit.
        """
        B, H, N, D = q.shape
        M = self.n_landmarks
        K = nbr_idx.shape[-1]

        # Landmark key/value summaries (segment-mean pools).
        k_tilde = _segment_mean_pool(k, real_mask, M)             # (B,H,M,D)
        v_tilde = _segment_mean_pool(v, real_mask, M)             # (B,H,M,D)

        # Landmark logits: q_i · k̃_l + geometry bias on cell↔centroid.
        d_NM = torch.cdist(positions, centroids)                  # (B,N,M)
        s_land = torch.einsum("bhnd,bhmd->bhnm", q, k_tilde) * scale
        s_land = s_land + self._bias(d_NM, radial, gamma)         # (B,H,N,M)

        # Per-query effective landmark counts. local_count[b,i,l] = #
        # of i's valid neighbours whose cell falls in landmark l.
        nbr_seg = seg_id[nbr_idx]                                 # (B,N,K) landmark of each neighbour
        local_count = torch.zeros(B, N, M, dtype=q.dtype, device=q.device)
        # scatter_add valid neighbours into their landmark bin.
        local_count.scatter_add_(
            2, nbr_seg, nbr_valid.to(q.dtype),
        )                                                         # (B,N,M)
        n_eff = (base_count.unsqueeze(1) - local_count).clamp_min(0.0)  # (B,N,M)
        # Zero-out landmarks that are entirely padding (base_count==0).
        # (clamp already ≥0; n_eff is 0 there.)

        # log-sum-exp combine over [local K] + [landmark M weighted n_eff].
        m_local = s_local.max(dim=-1, keepdim=True).values        # (B,H,N,1)
        m_land = s_land.max(dim=-1, keepdim=True).values
        m = torch.maximum(m_local, m_land)                        # (B,H,N,1)

        e_local = torch.exp(s_local - m)                          # (B,H,N,K); invalid→0
        # weight landmark exponentials by the (non-negative) effective count
        e_land = torch.exp(s_land - m) * n_eff.unsqueeze(1)       # (B,H,N,M)

        num = (
            torch.einsum("bhnk,bhnkd->bhnd", e_local, v_nbr)
            + torch.einsum("bhnm,bhmd->bhnd", e_land, v_tilde)
        )
        denom = e_local.sum(-1, keepdim=True) + e_land.sum(-1, keepdim=True)
        return num / denom.clamp_min(1e-20)


# ---------------------------------------------------------------------------
# Block + backbone
# ---------------------------------------------------------------------------


class LocalGlobalBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_rbf, n_local, n_landmarks,
                 moore_penrose_iters, combine, mlp_ratio=4.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = LocalGlobalGeometryAttention(
            embed_dim=self.hidden_dim, num_heads=n_heads, num_rbf=num_rbf,
            n_local=n_local, n_landmarks=n_landmarks,
            moore_penrose_iters=moore_penrose_iters, combine=combine,
        )
        self.norm2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        ff_hidden = int(self.hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, ff_hidden), nn.GELU(),
            nn.Linear(ff_hidden, self.hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_dim, 6 * self.hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, radial, geom, key_padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(
            h, c, radial,
            nbr_idx=geom["nbr_idx"], nbr_dist=geom["nbr_dist"],
            nbr_valid=geom["nbr_valid"], positions=geom["positions"],
            centroids=geom["centroids"], seg_id=geom["seg_id"],
            base_count=geom["base_count"], real_mask=geom["real_mask"],
            key_padding_mask=key_padding_mask,
        )
        x = x + gate_msa.unsqueeze(1) * attn_out
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class LocalGlobalBackbone(nn.Module):
    """DiT-shaped backbone with geometry local-global attention.
    Sub-quadratic O(N·(K+M)); same DataHolder contract as DiT."""

    def __init__(self, input_dims, n_layers, hidden_mlp_dims, hidden_dims,
                 output_dims, geomattn_localglobal_cfg=None):
        super().__init__()
        self.out_features_dim = int(hidden_dims["output_features_to_pos_dims"])
        cfg = geomattn_localglobal_cfg

        def _g(key, default):
            if cfg is None:
                return default
            return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)

        self.hidden_dim = int(_g("hidden_dim", 256))
        self.n_heads = int(_g("n_heads", 8))
        self.mlp_ratio = float(_g("mlp_ratio", 4))
        self.n_layers = int(_g("n_layers", n_layers))
        self.time_embed_dim = int(_g("time_embed_dim", 256))
        self.num_rbf = int(_g("num_rbf", 16))
        self.init_max_dist = float(_g("init_max_dist", 3.0))
        self.detach_geometry = bool(_g("detach_geometry", True))
        self.n_local = int(_g("n_local", 32))
        self.n_landmarks = int(_g("n_landmarks", 64))
        self.moore_penrose_iters = int(_g("moore_penrose_iters", 6))
        self.knn_chunk = int(_g("knn_chunk", 1024))
        self.combine = str(_g("combine", "gated"))
        self.grad_checkpoint = bool(_g("grad_checkpoint", True))

        gene_in = int(input_dims["node_features_dimensions"])
        self.gene_embed = nn.Sequential(
            nn.Linear(gene_in, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(2, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.t_embed = TimestepEmbedder(self.hidden_dim, self.time_embed_dim)
        self.radial = RadialBasis(self.num_rbf, self.init_max_dist)
        self.blocks = nn.ModuleList([
            LocalGlobalBlock(
                self.hidden_dim, self.n_heads, self.num_rbf, self.n_local,
                self.n_landmarks, self.moore_penrose_iters, self.combine,
                self.mlp_ratio,
            ) for _ in range(self.n_layers)
        ])
        self.final = DiTFinalLayer(self.hidden_dim, self.out_features_dim)

    def _precompute_geometry(self, positions, node_mask):
        """kNN + landmark centroids + segment ids + counts — all derived
        from the (fixed-per-forward) coordinate estimate, computed once
        and shared by every block."""
        B, N, _ = positions.shape
        M = self.n_landmarks
        real_mask = node_mask                                     # (B,N) bool, True at REAL

        nbr_idx, nbr_dist, nbr_valid = knn_chunked(
            positions, real_mask, self.n_local, self.knn_chunk,
        )

        # Landmark centroids (segment-mean of positions over real cells).
        centroids = _segment_mean_pool(
            positions.unsqueeze(1), real_mask, M,
        ).squeeze(1)                                              # (B,M,2)

        # Segment ids (N,) and per-landmark real-cell counts (B,M).
        n_idx = torch.arange(N, device=positions.device)
        seg_id = ((n_idx * M) // N).clamp(max=M - 1)              # (N,)
        base_count = torch.zeros(B, M, dtype=positions.dtype, device=positions.device)
        base_count.scatter_add_(
            1, seg_id.view(1, N).expand(B, N), real_mask.to(positions.dtype),
        )
        return {
            "nbr_idx": nbr_idx, "nbr_dist": nbr_dist, "nbr_valid": nbr_valid,
            "positions": positions, "centroids": centroids, "seg_id": seg_id,
            "base_count": base_count, "real_mask": real_mask,
        }

    def forward(self, data: DataHolder, **_unused_kwargs) -> DataHolder:
        node_mask = data.node_mask

        tok = self.gene_embed(data.node_features)
        tok = tok + self.pos_embed(data.positions)
        tok = tok * node_mask.unsqueeze(-1).to(tok.dtype)

        t = data.diffusion_time
        if t.dim() > 1:
            t = t.view(-1)
        c = self.t_embed(t)

        pos = data.positions
        if self.detach_geometry:
            pos = pos.detach()
        pos = pos.float()
        geom = self._precompute_geometry(pos, node_mask)

        key_padding_mask = ~node_mask
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                tok = checkpoint(
                    block, tok, c, self.radial, geom, key_padding_mask,
                    use_reentrant=False,
                )
            else:
                tok = block(tok, c, self.radial, geom, key_padding_mask=key_padding_mask)

        out = self.final(tok, c)
        features = out[..., :-2]
        positions = out[..., -2:]
        pad = node_mask.unsqueeze(-1).to(features.dtype)
        features = features * pad
        positions = positions * pad
        n_real = node_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1).to(positions.dtype)
        positions = positions - positions.sum(dim=1, keepdim=True) / n_real
        positions = positions * pad

        return DataHolder(
            node_features=features, positions=positions,
            diffusion_time=data.diffusion_time, cell_class=data.cell_class,
            cell_ID=data.cell_ID, t_int=data.t_int, t=data.t,
            node_mask=node_mask,
        )

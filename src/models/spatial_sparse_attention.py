"""Spatial sparse geometry-aware attention — a zoo of efficient-attention
patterns adapted from sequence-index sparsity to 2-D spatial proximity.

Each backbone takes a famous sub-quadratic attention mechanism (whose
sparsity is normally decided by *sequence index* or content) and decides
it instead by **2-D spatial position under the current coordinate
estimate x_t**. All patterns:
  * carry the same learnable, SE(2)-invariant radial geometry bias
    (γ_h(t)·Σ_k W_{h,k} φ_k(distance)), zero-init gate ⇒ no bias at start;
  * are sub-quadratic in N;
  * share the DiT adaLN-Zero scaffolding + EDM-compatible I/O contract.

Patterns (config ``backbone``):
  geomattn_dilated   — Sparse-Transformer / dilated-Longformer: each cell
                       attends to a DILATED set of its nearest neighbours
                       (every-r-th of its top-P), spanning near→far at a
                       fixed O(N·K) budget.
  geomattn_bigbird   — BigBird: kNN window + R RANDOM distant cells +
                       landmark global, in one softmax (count-excluded
                       landmarks ⇒ no double-count).
  geomattn_axial     — Axial: attend within x-coordinate bands, then within
                       y-coordinate bands (two-pass), averaged. O(N·√N).
  geomattn_swin      — Swin: partition the plane into a grid; attend within
                       each bucket (window); SHIFT the grid by half a cell
                       on alternating layers for cross-window flow.
  geomattn_routing   — Routing Transformer: k-means on positions → attend
                       within each spatial cluster.

The group-based patterns (axial / swin / routing) all reduce to the one
shared, tested ``grouped_dense_attention`` primitive (padded per-group
dense attention + geometry bias), so their correctness rests on a single
validated kernel whose single-group limit is exactly dense softmax.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from utils.data.dataholder import DataHolder
from models.dit_backbone import TimestepEmbedder, DiTFinalLayer, _modulate
from models.geometry_coupled_attention import RadialBasis
from models.geometry_local_global_attention import knn_chunked, _gather_neighbors
from models.nystromformer_backbone import _segment_mean_pool


# ---------------------------------------------------------------------------
# Geometry bias over arbitrary trailing shape
# ---------------------------------------------------------------------------


def radial_bias_nd(dist, centers, widths, W, gamma):
    """γ_h · Σ_k W_{h,k} φ_k(dist) for dist of shape (B, *S). Returns
    (B, H, *S). Accumulates over the K RBFs (no (B,*S,K) tensor)."""
    B = dist.shape[0]
    H, K = W.shape
    extra = tuple(dist.shape[1:])
    out = dist.new_zeros((B, H) + extra)
    wshape = (1, H) + (1,) * len(extra)
    for k in range(K):
        phi = torch.exp(-0.5 * ((dist - centers[k]) / widths[k]) ** 2)   # (B,*S)
        out = out + W[:, k].view(*wshape) * phi.unsqueeze(1)
    gshape = (B, H) + (1,) * len(extra)
    return gamma.view(*gshape) * out


# ---------------------------------------------------------------------------
# Shared grouped dense attention (powers axial / swin / routing)
# ---------------------------------------------------------------------------


def grouped_dense_attention(
    q, k, v,                       # (B, H, N, D)
    positions,                     # (B, N, 2)
    group_id,                      # (B, N) long in [0, n_groups)
    real_mask,                     # (B, N) bool
    n_groups: int,
    centers, widths, kernel_W,     # radial kernel params
    gamma,                         # (B, H)
    max_group: Optional[int] = None,
):
    """Exact softmax attention computed independently WITHIN each group,
    with the per-head SE(2)-invariant geometry bias. Cells in different
    groups don't interact. Cost O(B·H·n_groups·S²·D) with S = max group
    size — sub-quadratic when groups are balanced (S ≈ N/n_groups).

    Implementation: scatter cells into a padded (n_groups, S) layout via
    a per-group running slot index, run batched dense attention with a
    key-occupancy mask, gather the results back. Single-group limit
    (n_groups=1, S=N) is exactly dense softmax over all cells — the
    correctness anchor.
    """
    B, H, N, D = q.shape
    G = n_groups
    device = q.device
    dtype = q.dtype

    rm = real_mask.to(dtype)                                          # (B,N)
    # Per-group running slot index (count of earlier real cells in the
    # same group, original order). onehot+cumsum is O(N·G).
    onehot = torch.zeros(B, N, G, device=device, dtype=dtype)
    onehot.scatter_(2, group_id.unsqueeze(-1), 1.0)
    onehot = onehot * rm.unsqueeze(-1)
    cum = onehot.cumsum(dim=1)                                        # (B,N,G)
    slot = (cum.gather(2, group_id.unsqueeze(-1)).squeeze(-1) - 1.0)  # (B,N)
    slot = slot.clamp_min(0).long()
    counts = onehot.sum(dim=1)                                        # (B,G)
    S = int(counts.max().item()) if max_group is None else int(max_group)
    S = max(S, 1)

    valid_cell = real_mask & (slot < S)                              # (B,N) bool
    flat = (group_id * S + slot).clamp(0, G * S - 1)                 # (B,N)

    def scatter_feat(x):  # (B,H,N,D) -> (B,H,G*S,D)
        out = x.new_zeros(B, H, G * S, D)
        idx = flat.view(B, 1, N, 1).expand(B, H, N, D)
        out.scatter_(2, idx, x * valid_cell.view(B, 1, N, 1).to(x.dtype))
        return out

    qg = scatter_feat(q).view(B, H, G, S, D)
    kg = scatter_feat(k).view(B, H, G, S, D)
    vg = scatter_feat(v).view(B, H, G, S, D)

    pos_out = positions.new_zeros(B, G * S, 2)
    pos_out.scatter_(
        1, flat.view(B, N, 1).expand(B, N, 2),
        positions * valid_cell.view(B, N, 1).to(positions.dtype),
    )
    pg = pos_out.view(B, G, S, 2)

    occ = torch.zeros(B, G * S, device=device, dtype=dtype)
    occ.scatter_(1, flat, valid_cell.to(dtype))
    occ = occ.view(B, G, S)                                          # 1 at occupied slot

    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum("bhgsd,bhgtd->bhgst", qg, kg) * scale      # (B,H,G,S,S)
    dist = torch.cdist(pg, pg)                                       # (B,G,S,S)
    scores = scores + radial_bias_nd(dist, centers, widths, kernel_W, gamma)
    # Mask invalid KEY slots (unoccupied) — finfo.min (not -inf) so an
    # all-empty group's softmax is uniform, not NaN.
    neg_inf = torch.finfo(scores.dtype).min
    key_ok = (occ > 0).view(B, 1, G, 1, S)
    scores = scores.masked_fill(~key_ok, neg_inf)
    attn = scores.softmax(dim=-1)
    out_g = torch.einsum("bhgst,bhgtd->bhgsd", attn, vg)             # (B,H,G,S,D)

    out_flat = out_g.reshape(B, H, G * S, D)
    idx = flat.view(B, 1, N, 1).expand(B, H, N, D)
    out = torch.gather(out_flat, 2, idx)                            # (B,H,N,D)
    return out * valid_cell.view(B, 1, N, 1).to(out.dtype)


# ---------------------------------------------------------------------------
# Base: shared projections, geometry bias, helpers
# ---------------------------------------------------------------------------


class _SparseAttnBase(nn.Module):
    """q/k/v/out projections + per-head radial kernel + zero-init geo gate.
    Subclasses implement ``_attend`` using whatever spatial pattern."""

    def __init__(self, embed_dim, num_heads, num_rbf, bias=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} % num_heads {num_heads} != 0")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.num_rbf = int(num_rbf)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.kernel = nn.Linear(self.num_rbf, self.num_heads, bias=False)
        self.geo_gate = nn.Linear(self.embed_dim, self.num_heads)
        nn.init.zeros_(self.geo_gate.weight)
        nn.init.zeros_(self.geo_gate.bias)

    def _qkv(self, x):
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)
        return q, k, v

    def _widths(self, radial):
        return torch.exp(radial.log_widths) + 1e-6

    def _merge(self, out):
        B, H, N, D = out.shape
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, N, H * D))


# ---------------------------------------------------------------------------
# A. Dilated kNN attention (Sparse Transformer / dilated Longformer)
# ---------------------------------------------------------------------------


class DilatedKNNAttention(_SparseAttnBase):
    def __init__(self, embed_dim, num_heads, num_rbf, n_local, dilation):
        super().__init__(embed_dim, num_heads, num_rbf)
        self.n_local = int(n_local)
        self.dilation = max(int(dilation), 1)

    def forward(self, x, c, radial, geom, key_padding_mask=None):
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q, k, v = self._qkv(x)
        gamma = self.geo_gate(c)
        scale = 1.0 / math.sqrt(D)
        widths = self._widths(radial)

        nbr_idx = geom["nbr_idx"]        # (B,N,P) P = n_local*dilation nearest
        nbr_dist = geom["nbr_dist"]
        nbr_valid = geom["nbr_valid"]
        # Dilated subsample: columns 0, dilation, 2*dilation, ... up to n_local.
        P = nbr_idx.shape[-1]
        sel = torch.arange(0, P, self.dilation, device=x.device)[: self.n_local]
        idx = nbr_idx[:, :, sel]                                     # (B,N,K)
        dist = nbr_dist[:, :, sel]
        valid = nbr_valid[:, :, sel]
        K = idx.shape[-1]

        k_nbr = _gather_neighbors(k, idx)                           # (B,H,N,K,D)
        v_nbr = _gather_neighbors(v, idx)
        s = torch.einsum("bhnd,bhnkd->bhnk", q, k_nbr) * scale
        s = s + radial_bias_nd(dist, radial.centers, widths, self.kernel.weight, gamma)
        neg_inf = torch.finfo(s.dtype).min
        s = s.masked_fill(~valid.unsqueeze(1), neg_inf)
        a = s.softmax(dim=-1)
        out = torch.einsum("bhnk,bhnkd->bhnd", a, v_nbr)
        return self._merge(out)


# ---------------------------------------------------------------------------
# B. BigBird spatial: local kNN + random + landmark global (one softmax)
# ---------------------------------------------------------------------------


class BigBirdSpatialAttention(_SparseAttnBase):
    def __init__(self, embed_dim, num_heads, num_rbf, n_local, n_random,
                 n_landmarks):
        super().__init__(embed_dim, num_heads, num_rbf)
        self.n_local = int(n_local)
        self.n_random = int(n_random)
        self.n_landmarks = int(n_landmarks)

    def forward(self, x, c, radial, geom, key_padding_mask=None):
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        M = self.n_landmarks
        q, k, v = self._qkv(x)
        gamma = self.geo_gate(c)
        scale = 1.0 / math.sqrt(D)
        widths = self._widths(radial)
        positions = geom["positions"]
        real_mask = geom["real_mask"]

        # ---- local neighbours ----
        nbr_idx = geom["nbr_idx"][:, :, : self.n_local]              # (B,N,Kl)
        nbr_dist = geom["nbr_dist"][:, :, : self.n_local]
        nbr_valid = geom["nbr_valid"][:, :, : self.n_local]

        # ---- random distant cells (resampled per forward, real only) ----
        rand_idx = geom["rand_idx"]                                  # (B,N,R)
        rand_valid = geom["rand_valid"]                              # (B,N,R)
        # distance of each random cell from the query (gather positions).
        rand_pos = torch.gather(
            positions, 1, rand_idx.reshape(B, -1, 1).expand(B, N * self.n_random, 2),
        ).view(B, N, self.n_random, 2)
        rand_dist = (positions.unsqueeze(2) - rand_pos).norm(dim=-1)  # (B,N,R)

        # Concatenate local + random into one exact key set.
        idx_cat = torch.cat([nbr_idx, rand_idx], dim=-1)            # (B,N,Kl+R)
        dist_cat = torch.cat([nbr_dist, rand_dist], dim=-1)
        valid_cat = torch.cat([nbr_valid, rand_valid], dim=-1)
        k_cat = _gather_neighbors(k, idx_cat)                      # (B,H,N,Kc,D)
        v_cat = _gather_neighbors(v, idx_cat)
        s_loc = torch.einsum("bhnd,bhnkd->bhnk", q, k_cat) * scale
        s_loc = s_loc + radial_bias_nd(dist_cat, radial.centers, widths, self.kernel.weight, gamma)
        neg_inf = torch.finfo(s_loc.dtype).min
        s_loc = s_loc.masked_fill(~valid_cat.unsqueeze(1), neg_inf)

        # ---- landmark global (count-excluded over local+random) ----
        k_tilde = _segment_mean_pool(k, real_mask, M)              # (B,H,M,D)
        v_tilde = _segment_mean_pool(v, real_mask, M)
        centroids = geom["centroids"]
        d_NM = torch.cdist(positions, centroids)                  # (B,N,M)
        s_land = torch.einsum("bhnd,bhmd->bhnm", q, k_tilde) * scale
        s_land = s_land + radial_bias_nd(d_NM, radial.centers, widths, self.kernel.weight, gamma)

        seg_id = geom["seg_id"]
        base_count = geom["base_count"]
        cat_seg = seg_id[idx_cat]                                  # (B,N,Kc)
        local_count = torch.zeros(B, N, M, dtype=q.dtype, device=x.device)
        local_count.scatter_add_(2, cat_seg, valid_cat.to(q.dtype))
        n_eff = (base_count.unsqueeze(1) - local_count).clamp_min(0.0)  # (B,N,M)

        m = torch.maximum(
            s_loc.max(dim=-1, keepdim=True).values,
            s_land.max(dim=-1, keepdim=True).values,
        )
        e_loc = torch.exp(s_loc - m)
        e_land = torch.exp(s_land - m) * n_eff.unsqueeze(1)
        num = (
            torch.einsum("bhnk,bhnkd->bhnd", e_loc, v_cat)
            + torch.einsum("bhnm,bhmd->bhnd", e_land, v_tilde)
        )
        den = e_loc.sum(-1, keepdim=True) + e_land.sum(-1, keepdim=True)
        out = num / den.clamp_min(1e-20)
        return self._merge(out)


# ---------------------------------------------------------------------------
# C. Axial spatial attention (x-bands then y-bands, averaged)
# ---------------------------------------------------------------------------


class AxialSpatialAttention(_SparseAttnBase):
    def __init__(self, embed_dim, num_heads, num_rbf, band_size):
        super().__init__(embed_dim, num_heads, num_rbf)
        self.band_size = int(band_size)

    def _bands(self, coord, real_mask, N):
        """Assign each cell to a band by rank along ``coord``. Padding
        cells get rank pushed to the end (won't share bands with real
        cells of interest). Returns (group_id (B,N), n_groups)."""
        B = coord.shape[0]
        # Push padding to +inf so it sorts last.
        c = torch.where(real_mask, coord, torch.full_like(coord, float("inf")))
        rank = c.argsort(dim=1).argsort(dim=1)                     # (B,N) 0..N-1
        gid = (rank // self.band_size).long()
        n_groups = (N + self.band_size - 1) // self.band_size
        return gid.clamp(max=n_groups - 1), n_groups

    def forward(self, x, c, radial, geom, key_padding_mask=None):
        B, N, _ = x.shape
        q, k, v = self._qkv(x)
        gamma = self.geo_gate(c)
        widths = self._widths(radial)
        positions = geom["positions"]
        real_mask = geom["real_mask"]
        cen, W = radial.centers, self.kernel.weight

        gid_x, Gx = self._bands(positions[..., 0], real_mask, N)
        gid_y, Gy = self._bands(positions[..., 1], real_mask, N)
        out_x = grouped_dense_attention(
            q, k, v, positions, gid_x, real_mask, Gx, cen, widths, W, gamma,
        )
        out_y = grouped_dense_attention(
            q, k, v, positions, gid_y, real_mask, Gy, cen, widths, W, gamma,
        )
        return self._merge(0.5 * (out_x + out_y))


# ---------------------------------------------------------------------------
# D. Swin spatial attention (grid windows, shifted on alternating layers)
# ---------------------------------------------------------------------------


class SwinSpatialAttention(_SparseAttnBase):
    def __init__(self, embed_dim, num_heads, num_rbf, grid, layer_idx):
        super().__init__(embed_dim, num_heads, num_rbf)
        self.grid = int(grid)                  # grid × grid buckets
        self.layer_idx = int(layer_idx)        # odd layers shift by half a cell

    def _buckets(self, positions, real_mask):
        B, N, _ = positions.shape
        Gd = self.grid
        # Per-slice bbox over real cells.
        big = torch.finfo(positions.dtype).max
        masked = torch.where(real_mask.unsqueeze(-1), positions,
                             torch.full_like(positions, big))
        mn = masked.amin(dim=1, keepdim=True)                      # (B,1,2)
        masked2 = torch.where(real_mask.unsqueeze(-1), positions,
                              torch.full_like(positions, -big))
        mx = masked2.amax(dim=1, keepdim=True)
        span = (mx - mn).clamp_min(1e-6)
        cell = span / Gd
        shift = 0.5 if (self.layer_idx % 2 == 1) else 0.0
        # bucket coord in each axis, shifted; clamp into [0, Gd-1].
        bxy = torch.floor((positions - mn) / cell - shift).clamp(0, Gd - 1).long()
        gid = bxy[..., 0] * Gd + bxy[..., 1]                       # (B,N) in [0, Gd²)
        return gid, Gd * Gd

    def forward(self, x, c, radial, geom, key_padding_mask=None):
        B, N, _ = x.shape
        q, k, v = self._qkv(x)
        gamma = self.geo_gate(c)
        widths = self._widths(radial)
        positions = geom["positions"]
        real_mask = geom["real_mask"]
        gid, G = self._buckets(positions, real_mask)
        out = grouped_dense_attention(
            q, k, v, positions, gid, real_mask, G,
            radial.centers, widths, self.kernel.weight, gamma,
        )
        return self._merge(out)


# ---------------------------------------------------------------------------
# E. Routing spatial attention (k-means clusters of positions)
# ---------------------------------------------------------------------------


def _kmeans_positions(pos, real_mask, C, iters=5, seed=0):
    """Lightweight Lloyd k-means on 2-D positions. Returns cluster id
    (B,N) in [0,C). Deterministic init (evenly-spaced real cells)."""
    B, N, _ = pos.shape
    device = pos.device
    big = torch.finfo(pos.dtype).max
    # init centroids: C evenly-spaced REAL cells per slice.
    gid = torch.zeros(B, N, dtype=torch.long, device=device)
    cent = torch.zeros(B, C, 2, dtype=pos.dtype, device=device)
    for b in range(B):
        real_idx = torch.nonzero(real_mask[b], as_tuple=False).view(-1)
        if real_idx.numel() == 0:
            continue
        pick = real_idx[torch.linspace(0, real_idx.numel() - 1, C, device=device).long()]
        cent[b] = pos[b, pick]
    for _ in range(iters):
        # assign
        d = torch.cdist(pos, cent)                                 # (B,N,C)
        d = torch.where(real_mask.unsqueeze(-1), d, torch.full_like(d, big))
        gid = d.argmin(dim=-1)                                     # (B,N)
        # update
        oh = torch.zeros(B, N, C, dtype=pos.dtype, device=device)
        oh.scatter_(2, gid.unsqueeze(-1), 1.0)
        oh = oh * real_mask.unsqueeze(-1).to(pos.dtype)
        cnt = oh.sum(dim=1).clamp_min(1.0)                         # (B,C)
        cent = torch.einsum("bnc,bnd->bcd", oh, pos) / cnt.unsqueeze(-1)
    return gid


class RoutingSpatialAttention(_SparseAttnBase):
    def __init__(self, embed_dim, num_heads, num_rbf, n_clusters, kmeans_iters):
        super().__init__(embed_dim, num_heads, num_rbf)
        self.n_clusters = int(n_clusters)
        self.kmeans_iters = int(kmeans_iters)

    def forward(self, x, c, radial, geom, key_padding_mask=None):
        q, k, v = self._qkv(x)
        gamma = self.geo_gate(c)
        widths = self._widths(radial)
        positions = geom["positions"]
        real_mask = geom["real_mask"]
        gid = geom["route_gid"]                                    # precomputed once
        out = grouped_dense_attention(
            q, k, v, positions, gid, real_mask, self.n_clusters,
            radial.centers, widths, self.kernel.weight, gamma,
        )
        return self._merge(out)


# ---------------------------------------------------------------------------
# adaLN-Zero block wrapping a pluggable sparse attention
# ---------------------------------------------------------------------------


class _SparseBlock(nn.Module):
    def __init__(self, hidden_dim, attn, mlp_ratio=4.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = attn
        self.norm2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        ff = int(self.hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, ff), nn.GELU(), nn.Linear(ff, self.hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_dim, 6 * self.hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, radial, geom, key_padding_mask=None):
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = _modulate(self.norm1(x), s_msa, sc_msa)
        x = x + g_msa.unsqueeze(1) * self.attn(h, c, radial, geom, key_padding_mask)
        h = _modulate(self.norm2(x), s_mlp, sc_mlp)
        x = x + g_mlp.unsqueeze(1) * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# Backbone — one class, pattern-dispatched
# ---------------------------------------------------------------------------


_PATTERNS = ("dilated", "bigbird", "axial", "swin", "routing")


class SpatialSparseBackbone(nn.Module):
    """DiT-shaped backbone whose attention is one of the spatial sparse
    patterns. ``pattern`` selects which. Same DataHolder I/O contract as
    DiT, so EDM / c2f / gene_recon wrappers compose unchanged."""

    def __init__(self, pattern, input_dims, n_layers, hidden_mlp_dims,
                 hidden_dims, output_dims, cfg=None):
        super().__init__()
        if pattern not in _PATTERNS:
            raise ValueError(f"pattern must be one of {_PATTERNS}; got {pattern!r}")
        self.pattern = pattern
        self.out_features_dim = int(hidden_dims["output_features_to_pos_dims"])

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
        self.grad_checkpoint = bool(_g("grad_checkpoint", True))
        # pattern-specific
        self.n_local = int(_g("n_local", 32))
        self.dilation = int(_g("dilation", 3))
        self.n_random = int(_g("n_random", 8))
        self.n_landmarks = int(_g("n_landmarks", 64))
        self.band_size = int(_g("band_size", 128))
        self.grid = int(_g("grid", 8))
        self.n_clusters = int(_g("n_clusters", 32))
        self.kmeans_iters = int(_g("kmeans_iters", 5))
        self.knn_chunk = int(_g("knn_chunk", 1024))
        # neighbour pool size needed by knn (dilated needs n_local*dilation)
        if pattern == "dilated":
            self._knn_K = self.n_local * self.dilation
        elif pattern == "bigbird":
            self._knn_K = self.n_local
        else:
            self._knn_K = 0   # group patterns don't need kNN

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

        self.blocks = nn.ModuleList(
            [self._make_block(layer_idx=i) for i in range(self.n_layers)]
        )
        self.final = DiTFinalLayer(self.hidden_dim, self.out_features_dim)

    def _make_block(self, layer_idx):
        hd, nh, nr = self.hidden_dim, self.n_heads, self.num_rbf
        if self.pattern == "dilated":
            attn = DilatedKNNAttention(hd, nh, nr, self.n_local, self.dilation)
        elif self.pattern == "bigbird":
            attn = BigBirdSpatialAttention(hd, nh, nr, self.n_local, self.n_random, self.n_landmarks)
        elif self.pattern == "axial":
            attn = AxialSpatialAttention(hd, nh, nr, self.band_size)
        elif self.pattern == "swin":
            attn = SwinSpatialAttention(hd, nh, nr, self.grid, layer_idx)
        else:  # routing
            attn = RoutingSpatialAttention(hd, nh, nr, self.n_clusters, self.kmeans_iters)
        return _SparseBlock(hd, attn, self.mlp_ratio)

    def _precompute_geometry(self, pos, node_mask):
        B, N, _ = pos.shape
        real_mask = node_mask
        geom = {"positions": pos, "real_mask": real_mask}

        if self._knn_K > 0:
            K = min(self._knn_K, N)
            nbr_idx, nbr_dist, nbr_valid = knn_chunked(pos, real_mask, K, self.knn_chunk)
            geom.update(nbr_idx=nbr_idx, nbr_dist=nbr_dist, nbr_valid=nbr_valid)

        if self.pattern == "bigbird":
            # landmark global + random distant cells.
            M = self.n_landmarks
            centroids = _segment_mean_pool(pos.unsqueeze(1), real_mask, M).squeeze(1)
            n_idx = torch.arange(N, device=pos.device)
            seg_id = ((n_idx * M) // N).clamp(max=M - 1)
            base_count = torch.zeros(B, M, dtype=pos.dtype, device=pos.device)
            base_count.scatter_add_(1, seg_id.view(1, N).expand(B, N), real_mask.to(pos.dtype))
            # random real cells per query (resampled each forward).
            R = self.n_random
            rand_idx = torch.randint(0, N, (B, N, R), device=pos.device)
            rand_valid = torch.gather(
                real_mask, 1, rand_idx.reshape(B, -1),
            ).view(B, N, R)
            geom.update(centroids=centroids, seg_id=seg_id, base_count=base_count,
                        rand_idx=rand_idx, rand_valid=rand_valid)

        if self.pattern == "routing":
            geom["route_gid"] = _kmeans_positions(
                pos, real_mask, self.n_clusters, self.kmeans_iters,
            )
        return geom

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

        kpm = ~node_mask
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                tok = checkpoint(block, tok, c, self.radial, geom, kpm, use_reentrant=False)
            else:
                tok = block(tok, c, self.radial, geom, key_padding_mask=kpm)

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
            cell_ID=data.cell_ID, t_int=data.t_int, t=data.t, node_mask=node_mask,
        )

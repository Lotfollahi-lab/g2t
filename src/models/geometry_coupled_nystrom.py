"""Geometry-Coupled Nyström Attention (geomattn_nystrom) — scalable GCA.

Combines two ideas already in scgg:

* **Geometry-Coupled Attention** (``geometry_coupled_attention.py``):
  bias attention toward cells that are spatially near under the current
  coordinate estimate, via a learnable, time-gated, SE(2)-invariant
  radial kernel over pairwise distances. Problem: the dense per-head
  bias is O(B·H·N²) — it forces an N×N tensor and OOMs on large slices.

* **Nyströmformer** (``nystromformer_backbone.py``): approximate the
  N×N softmax attention by ``A ≈ F · B⁺ · G`` using M segment-mean
  *landmarks*, costing O(N·M) instead of O(N²).

The combination resolves GCA's scalability wall. The key observation:
Nyström's landmarks are **segment-mean pools of cells**, so each
landmark ``j`` has a natural *centroid position*
``p̃_j = mean_{i in segment j} x_i``. That gives every entry of the
three Nyström softmax blocks a well-defined distance, so the geometric
bias injects at the landmark level and stays sub-quadratic::

    F = softmax( Q·K̃ᵀ/√d + γ_h(t)·K_h(‖x_i  − p̃_j‖) )   ∈ (B,H,N,M)
    B = softmax( Q̃·K̃ᵀ/√d + γ_h(t)·K_h(‖p̃_i − p̃_j‖) )   ∈ (B,H,M,M)
    G = softmax( Q̃·Kᵀ/√d + γ_h(t)·K_h(‖p̃_i − x_j ‖) )   ∈ (B,H,M,N)

Cost: O(N·M) memory + compute for BOTH the attention and the bias
(the largest bias tensor is (B,H,N,M), ~135 MB at B=6,N=11k,M=64,H=8 —
vs ~23 GB for dense GCA's (B,H,N,N)).

Properties preserved from dense GCA:
* **SE(2)-invariance**: centroids transform rigidly with positions
  (``mean(Rx+t) = R·mean(x)+t``), so all cell↔centroid and
  centroid↔centroid distances — hence the biases — are invariant to
  any rigid motion of the coordinate estimate.
* **Zero-init time-gate** ⇒ all three biases are 0 at init ⇒ the layer
  is identical to plain Nyströmformer at the start of training; the
  coarse-to-fine schedule is learned.
* **M → N exactness**: when ``n_landmarks ≥ N`` the layer falls through
  to *dense* geometry-coupled softmax attention (the exact GCA bias on
  the full N×N scores), so the small-slice / large-M limit recovers
  dense GCA exactly — the same convergence guarantee plain Nyström has.

Interpretation for the paper: landmark centroids define a coarse
spatial partition of the tissue; the F/G biases make the low-rank
attention respect spatial locality at that coarse scale, while the
cell→landmark soft assignment carries the fine structure. Geometry
coupling at O(N·M).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from utils.data.dataholder import DataHolder
from models.dit_backbone import TimestepEmbedder, DiTFinalLayer, _modulate
from models.geometry_coupled_attention import RadialBasis
from models.nystromformer_backbone import (
    _segment_mean_pool,
    _moore_penrose_iter,
)


class GeometryCoupledNystromAttention(nn.Module):
    """Nyström attention with a landmark-level geometric bias.

    Holds its own per-head radial kernel (``kernel``: K→H) and
    time-gate (``gate``: hidden_dim→H, zero-init). The Gaussian RBF
    basis (``radial``) is shared across layers and passed into forward.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_rbf: int,
        n_landmarks: int = 64,
        moore_penrose_iters: int = 6,
        exact_until_n: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads "
                f"{num_heads}."
            )
        self.head_dim = self.embed_dim // self.num_heads
        self.num_rbf = int(num_rbf)
        self.n_landmarks = int(n_landmarks)
        self.moore_penrose_iters = int(moore_penrose_iters)
        self.exact_until_n = int(exact_until_n)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

        # Per-head radial kernel (shared across the F/B/G blocks — it is
        # "the" geometric kernel for that head). No bias: a constant
        # added to all logits is a softmax no-op.
        self.kernel = nn.Linear(self.num_rbf, self.num_heads, bias=False)
        # Per-head time gate, zero-init ⇒ zero bias at start ⇒ identical
        # to plain Nyström at init.
        self.gate = nn.Linear(self.embed_dim, self.num_heads)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    # ------------------------------------------------------------------
    # Geometric bias helper
    # ------------------------------------------------------------------

    def _radial_bias(
        self,
        dist: torch.Tensor,           # (B, R, S)
        radial: RadialBasis,
        gamma: torch.Tensor,          # (B, H)  — precomputed gate
    ) -> torch.Tensor:
        """Per-head additive bias γ_h · Σ_k W_{h,k} φ_k(dist).

        Accumulates over the K radial-basis functions on the fly (no
        (B,R,S,K) tensor). Returns (B, H, R, S).
        """
        B, R, S = dist.shape
        H = self.num_heads
        centers = radial.centers                                  # (K,)
        widths = torch.exp(radial.log_widths) + 1e-6              # (K,)
        W = self.kernel.weight                                    # (H, K)
        out = dist.new_zeros(B, H, R, S)
        for k in range(self.num_rbf):
            phi_k = torch.exp(-0.5 * ((dist - centers[k]) / widths[k]) ** 2)
            out = out + W[:, k].view(1, H, 1, 1) * phi_k.unsqueeze(1)
        return gamma.view(B, H, 1, 1) * out

    # ------------------------------------------------------------------
    # Exact fallthrough (dense GCA) for small slices / large-M limit
    # ------------------------------------------------------------------

    def _exact_geometry_attention(
        self,
        q, k, v,                       # (B,H,N,D)
        positions,                     # (B,N,2)
        radial, gamma,
        key_padding_mask,
    ) -> torch.Tensor:
        """Dense geometry-coupled softmax via SDPA — recovers dense GCA.
        Used when N ≤ n_landmarks (or ≤ exact_until_n), and as the
        M → N exactness limit."""
        B, H, N, D = q.shape
        dist = torch.cdist(positions, positions)                  # (B,N,N)
        bias = self._radial_bias(dist, radial, gamma)             # (B,H,N,N)
        if key_padding_mask is not None:
            neg_inf = torch.finfo(q.dtype).min
            pad = torch.where(
                key_padding_mask,
                torch.tensor(neg_inf, dtype=q.dtype, device=q.device),
                torch.tensor(0.0, dtype=q.dtype, device=q.device),
            ).view(B, 1, 1, N)
            bias = bias + pad
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False,
        )

    def forward(
        self,
        x: torch.Tensor,               # (B, N, embed_dim)
        positions: torch.Tensor,       # (B, N, 2)  — current x_t (detached upstream)
        radial: RadialBasis,
        c: torch.Tensor,               # (B, embed_dim) time embedding
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        M = self.n_landmarks

        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)       # (B,H,N,D)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)

        gamma = self.gate(c)                                      # (B, H)

        # Hybrid / exactness switch: dense GCA when the slice is small
        # or M ≥ N (the M → N exactness limit).
        if N <= self.exact_until_n or N <= M:
            out = self._exact_geometry_attention(
                q, k, v, positions, radial, gamma, key_padding_mask,
            )
            out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
            return self.out_proj(out)

        if key_padding_mask is None:
            real_mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        else:
            real_mask = ~key_padding_mask

        # Landmark FEATURES (segment-mean pool of q, k) — same as plain
        # Nyström.
        q_tilde = _segment_mean_pool(q, real_mask, M)             # (B,H,M,D)
        k_tilde = _segment_mean_pool(k, real_mask, M)

        # Landmark POSITIONS (segment-mean pool of the coordinate
        # estimate over the SAME segments). Treat positions as a
        # 1-head (B,1,N,2) tensor so we can reuse _segment_mean_pool,
        # then squeeze the head axis → (B, M, 2).
        centroids = _segment_mean_pool(
            positions.unsqueeze(1), real_mask, M,
        ).squeeze(1)                                              # (B, M, 2)

        # Pairwise distances for the bias. cdist is cheap at these
        # shapes: (B,N,M) and (B,M,M).
        d_NM = torch.cdist(positions, centroids)                  # (B, N, M)
        d_MM = torch.cdist(centroids, centroids)                  # (B, M, M)
        # d_MN == d_NM transposed (distance is symmetric).
        d_MN = d_NM.transpose(1, 2)                               # (B, M, N)

        scale = 1.0 / math.sqrt(D)

        # F = softmax( Q·K̃ᵀ·scale + bias(d_NM) )   (B,H,N,M)
        F_logits = (q @ k_tilde.transpose(-2, -1)) * scale
        F_logits = F_logits + self._radial_bias(d_NM, radial, gamma)
        F_mat = F_logits.softmax(dim=-1)

        # B = softmax( Q̃·K̃ᵀ·scale + bias(d_MM) )   (B,H,M,M)
        B_logits = (q_tilde @ k_tilde.transpose(-2, -1)) * scale
        B_logits = B_logits + self._radial_bias(d_MM, radial, gamma)
        B_mat = B_logits.softmax(dim=-1)

        # G = softmax( Q̃·Kᵀ·scale + bias(d_MN) + key_pad )   (B,H,M,N)
        G_logits = (q_tilde @ k.transpose(-2, -1)) * scale
        G_logits = G_logits + self._radial_bias(d_MN, radial, gamma)
        if key_padding_mask is not None:
            neg_inf = torch.finfo(G_logits.dtype).min
            key_bias = torch.where(
                key_padding_mask,
                torch.tensor(neg_inf, dtype=G_logits.dtype, device=q.device),
                torch.tensor(0.0, dtype=G_logits.dtype, device=q.device),
            ).view(B, 1, 1, N)
            G_logits = G_logits + key_bias
        G_mat = G_logits.softmax(dim=-1)

        B_pinv = _moore_penrose_iter(B_mat, n_iter=self.moore_penrose_iters)

        # F · B⁺ · G · V, right-to-left (never materialises N×N).
        GV = G_mat @ v                                            # (B,H,M,D)
        BpGV = B_pinv @ GV                                        # (B,H,M,D)
        out = F_mat @ BpGV                                        # (B,H,N,D)

        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(out)


class GeometryCoupledNystromBlock(nn.Module):
    """Nyström block + landmark-level geometric bias, adaLN-Zero
    conditioned (mirrors NystromBlock / DiTBlock)."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        num_rbf: int,
        n_landmarks: int,
        moore_penrose_iters: int,
        exact_until_n: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = GeometryCoupledNystromAttention(
            embed_dim=self.hidden_dim,
            num_heads=n_heads,
            num_rbf=num_rbf,
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
        positions: torch.Tensor,
        radial: RadialBasis,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(h, positions, radial, c, key_padding_mask=key_padding_mask)
        x = x + gate_msa.unsqueeze(1) * attn_out
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class GeometryCoupledNystromBackbone(nn.Module):
    """DiT-shaped backbone with geometry-coupled Nyström attention.

    Same DataHolder-in / DataHolder-out contract as DiT / Nyströmformer.
    O(N·M) memory + compute for both attention and the geometric bias —
    the scalable counterpart to ``GeometryCoupledBackbone`` (geomattn).
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        n_layers: int,
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        output_dims: Dict[str, int],
        geomattn_nystrom_cfg=None,
    ):
        super().__init__()
        self.out_features_dim = int(hidden_dims["output_features_to_pos_dims"])

        def _g(k, default):
            cfg = geomattn_nystrom_cfg
            if cfg is None:
                return default
            return cfg.get(k, default) if hasattr(cfg, "get") else getattr(cfg, k, default)

        self.hidden_dim = int(_g("hidden_dim", 256))
        self.n_heads = int(_g("n_heads", 8))
        self.mlp_ratio = float(_g("mlp_ratio", 4))
        self.n_layers = int(_g("n_layers", n_layers))
        self.time_embed_dim = int(_g("time_embed_dim", 256))
        self.num_rbf = int(_g("num_rbf", 16))
        self.init_max_dist = float(_g("init_max_dist", 3.0))
        self.detach_geometry = bool(_g("detach_geometry", True))
        self.n_landmarks = int(_g("n_landmarks", 64))
        self.moore_penrose_iters = int(_g("moore_penrose_iters", 6))
        self.exact_until_n = int(_g("exact_until_n", 0))
        # Nyström is already O(N·M) so checkpointing is rarely needed;
        # exposed for parity / very deep stacks. Default off.
        self.grad_checkpoint = bool(_g("grad_checkpoint", False))

        gene_in = int(input_dims["node_features_dimensions"])
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
        self.t_embed = TimestepEmbedder(self.hidden_dim, self.time_embed_dim)
        self.radial = RadialBasis(self.num_rbf, self.init_max_dist)

        self.blocks = nn.ModuleList([
            GeometryCoupledNystromBlock(
                hidden_dim=self.hidden_dim,
                n_heads=self.n_heads,
                num_rbf=self.num_rbf,
                n_landmarks=self.n_landmarks,
                moore_penrose_iters=self.moore_penrose_iters,
                exact_until_n=self.exact_until_n,
                mlp_ratio=self.mlp_ratio,
            )
            for _ in range(self.n_layers)
        ])
        self.final = DiTFinalLayer(self.hidden_dim, self.out_features_dim)

    def forward(self, data: DataHolder, **_unused_kwargs) -> DataHolder:
        node_mask = data.node_mask                                # (B, N) bool

        tok = self.gene_embed(data.node_features)
        tok = tok + self.pos_embed(data.positions)
        tok = tok * node_mask.unsqueeze(-1).to(tok.dtype)

        t = data.diffusion_time
        if t.dim() > 1:
            t = t.view(-1)
        c = self.t_embed(t)                                       # (B, D)

        # Coordinate estimate driving the geometric bias (detached by
        # default — self-conditioning semantics).
        pos = data.positions
        if self.detach_geometry:
            pos = pos.detach()
        pos = pos.float()

        key_padding_mask = ~node_mask
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                tok = checkpoint(
                    block, tok, c, pos, self.radial, key_padding_mask,
                    use_reentrant=False,
                )
            else:
                tok = block(tok, c, pos, self.radial, key_padding_mask=key_padding_mask)

        out = self.final(tok, c)
        features = out[..., :-2]
        positions = out[..., -2:]

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

"""Geometry-Coupled Attention (GCA) backbone — methodological extension.

Motivation
----------
In spatial reconstruction the "ideal" attention pattern is the spatial
neighbourhood graph: a cell should aggregate information preferentially
from cells that are physically close to it in the tissue. But the
positions are exactly what we are trying to recover — we do not know
the neighbourhood a priori.

Flow matching / diffusion gives a natural escape from this chicken-and-egg
problem: at every ODE/diffusion step ``t`` we hold a *current estimate*
of the cell positions ``x_t``. GCA uses that estimate to bias attention
toward spatially-near cells, then the refined features sharpen the next
position estimate, which sharpens the next attention pattern, and so on.
The attention pattern and the geometry **co-evolve along the sampling
trajectory** (noise → data): early steps lean on content (gene) signal
because the geometry is still noise; late steps lean on a tight spatial
neighbourhood because the geometry is nearly resolved.

The mechanism
-------------
On top of standard scaled-dot-product attention we add a per-head,
learnable, **SE(2)-invariant** geometric bias::

    score_ij^{h} = (q_i · k_j) / sqrt(d)  +  gamma_h(t) · K_h( ||x_t,i - x_t,j|| )

where

* ``||x_t,i - x_t,j||`` is the Euclidean distance under the CURRENT
  coordinate estimate (detached — treated as a conditioning signal, in
  the spirit of self-conditioning in diffusion models). Because the
  bias depends only on pairwise *distances*, it is exactly invariant to
  any rigid motion (rotation / translation / reflection) of ``x_t`` —
  the same symmetry the EDM head exploits at the output side.

* ``K_h(·)`` is a per-head learnable radial kernel, parameterised as a
  small linear map over a shared radial-basis-function (RBF) expansion
  of the distance. This lets each head learn its own spatial
  receptive-field shape (e.g. "attend within ~1 unit", "avoid the
  immediate ring", ...).

* ``gamma_h(t)`` is a per-head, time-conditioned scalar gate produced by
  a tiny MLP on the time embedding. It is **zero-initialised**, so at the
  start of training the geometric bias is exactly zero and GCA reduces
  to a plain DiT block — the "can't hurt at init" property, mirroring
  adaLN-Zero. Training then learns *how much* to trust the geometry at
  each noise level: ``gamma_h(t) -> 0`` at high noise (geometry is
  garbage), large at low noise (geometry is reliable). The coarse-to-fine
  attention schedule is learned, not hand-designed.

Relationship to prior work
---------------------------
This is a continuous, 2-D, *dynamic* relative-position bias: like T5 /
ALiBi relative-position bias, but where the "position" is the model's
own evolving spatial prediction rather than a fixed sequence index, and
the bias kernel is learnable per head. Distinct from linear-attention
(a speed approximation), Nyströmformer (a low-rank softmax
approximation), and Perceiver (an anchor bottleneck): GCA changes *what
the attention attends to*, not how cheaply it is computed.

Cost / memory
-------------
The bias needs the N×N distance matrix and a (B, N, N, K) RBF tensor
(K = ``num_rbf``), so GCA is O(N²) memory like dense DiT (the RBF tensor
is the dominant term: B·N²·K·4 bytes). This is fine for MMC-scale slices
(~5-7k cells); for very large CNS slices prefer ``num_rbf`` small or a
sub-quadratic backbone. The RBF basis is computed ONCE per forward and
shared across layers (each layer only learns its own K→H linear + gate),
so the per-layer marginal cost is one (B, H, N, N) bias tensor.

Output interface matches DiT / LUNA Model: DataHolder in → DataHolder
out with ``pred.node_features`` (B, N, out_dim) and ``pred.positions``
(B, N, 2), so the EDM / c2f / gene-recon wrappers compose unchanged.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from utils.data.dataholder import DataHolder
from models.sdpa_attention import SDPAMultiheadAttention
# Reuse the DiT primitives verbatim so GCA stays a minimal, auditable
# diff against the proven DiT backbone (same time embedding, same
# adaLN-Zero modulation, same final layer).
from models.dit_backbone import (
    TimestepEmbedder,
    DiTFinalLayer,
    _modulate,
)


# ---------------------------------------------------------------------------
# Shared radial-basis-function expansion of the pairwise distances
# ---------------------------------------------------------------------------


class RadialBasis(nn.Module):
    """Expand a scalar distance ``d`` into ``num_rbf`` Gaussian features

        phi_k(d) = exp( -0.5 * ((d - mu_k) / sigma_k)^2 ).

    Centres ``mu_k`` and (log-)widths ``sigma_k`` are learnable so the
    basis can adapt to whatever scale the (normalised) coordinates live
    on. Shared across all attention layers — the expensive N×N×K tensor
    is built once per forward.
    """

    def __init__(self, num_rbf: int = 16, init_max_dist: float = 3.0):
        super().__init__()
        self.num_rbf = int(num_rbf)
        # Centres evenly spaced over [0, init_max_dist]; learnable.
        centers = torch.linspace(0.0, float(init_max_dist), self.num_rbf)
        self.centers = nn.Parameter(centers)
        # Width initialised to the centre spacing (so neighbouring RBFs
        # overlap ~1 sigma). Parameterised in log-space to keep it
        # positive under gradient descent.
        spacing = float(init_max_dist) / max(self.num_rbf - 1, 1)
        self.log_widths = nn.Parameter(
            torch.full((self.num_rbf,), torch.log(torch.tensor(spacing + 1e-6)))
        )

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: (B, N, N) pairwise Euclidean distances.
        Returns:
            (B, N, N, num_rbf) RBF features.
        """
        widths = torch.exp(self.log_widths) + 1e-6                  # (K,)
        # (B, N, N, 1) - (K,) → (B, N, N, K)
        diff = dist.unsqueeze(-1) - self.centers.view(1, 1, 1, -1)
        return torch.exp(-0.5 * (diff / widths.view(1, 1, 1, -1)) ** 2)


# ---------------------------------------------------------------------------
# Geometry-coupled attention block (DiT block + geometric bias)
# ---------------------------------------------------------------------------


class GeometryCoupledBlock(nn.Module):
    """A DiT block whose self-attention carries a learnable, time-gated,
    SE(2)-invariant geometric bias derived from the current coordinate
    estimate.

    Identical to ``DiTBlock`` except the attention call receives an
    additive bias ``gamma_h(t) · (W_h · phi(d_ij))`` via SDPA's float
    ``attn_mask``.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        num_rbf: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} must be divisible by n_heads {n_heads}"
            )
        self.num_rbf = int(num_rbf)

        self.norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = SDPAMultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
        )
        self.norm2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        ff_hidden = int(self.hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, self.hidden_dim),
        )

        # adaLN-Zero modulation (same as DiTBlock).
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 6 * self.hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        # Per-head learnable radial kernel: maps the shared RBF features
        # (K) to one bias scalar per head. No bias term — a constant
        # offset on the attention scores is a no-op after softmax.
        self.kernel = nn.Linear(self.num_rbf, self.n_heads, bias=False)

        # Per-head, time-conditioned gate gamma_h(t). Zero-initialised so
        # the geometric bias is exactly 0 at the start of training and
        # the block is identical to a plain DiT block. Training learns
        # the coarse-to-fine schedule.
        self.gate = nn.Linear(self.hidden_dim, self.n_heads)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def _geometry_bias(
        self,
        dist: torch.Tensor,
        radial: "RadialBasis",
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Build the additive attention bias.

        Memory-efficient formulation: we accumulate the per-head bias
        directly over the RBF index ``k`` WITHOUT ever materialising the
        full (B, N, N, K) RBF tensor — that tensor was the dominant
        memory term (B·N²·K floats) and OOM'd on cortex-scale slices
        with the default batch size. The K-loop holds only one
        (B, N, N) Gaussian at a time plus the (B, H, N, N) accumulator.

        Note: the (B, H, N, N) accumulator itself is irreducible for a
        *dense, per-head* additive attention bias — SDPA must add it to
        the N×N scores, so it cannot ride FlashAttention's O(N)-memory
        path. This term is O(B·H·N²); use ``train.batch_size=1`` for
        ``geomattn`` on cortex/CNS-scale slices (see the backbone
        docstring).

        Args:
            dist:   (B, N, N) pairwise Euclidean distances (detached).
            radial: the shared RadialBasis (centres + log-widths).
            c:      (B, D) time/conditioning embedding.
        Returns:
            (B*H, N, N) float bias for SDPA's attn_mask.
        """
        B, N, _ = dist.shape
        H = self.n_heads
        centers = radial.centers                                    # (K,)
        widths = torch.exp(radial.log_widths) + 1e-6                # (K,)
        W = self.kernel.weight                                      # (H, K)

        bias = dist.new_zeros(B, H, N, N)
        for k in range(self.num_rbf):
            # phi_k(d) = exp(-0.5 ((d - mu_k)/sigma_k)^2)  — (B, N, N)
            phi_k = torch.exp(-0.5 * ((dist - centers[k]) / widths[k]) ** 2)
            # Add W[h,k] * phi_k to every head h.
            bias = bias + W[:, k].view(1, H, 1, 1) * phi_k.unsqueeze(1)

        # Time-conditioned per-head gate: (B, D) → (B, H) → (B, H, 1, 1).
        gamma = self.gate(c).view(B, H, 1, 1)
        bias = gamma * bias                                         # (B, H, N, N)
        # SDPA wants (B*H, N, N) for the rank-3 attn_mask form.
        return bias.reshape(B * H, N, N)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        dist: torch.Tensor,
        radial: "RadialBasis",
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # MSA sub-layer with geometric bias.
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        geo_bias = self._geometry_bias(dist, radial, c)            # (B*H, N, N)
        attn_out, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=geo_bias,
        )
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP sub-layer.
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


# ---------------------------------------------------------------------------
# Geometry-coupled backbone (DataHolder in / DataHolder out)
# ---------------------------------------------------------------------------


class GeometryCoupledBackbone(nn.Module):
    """DiT-style backbone with geometry-coupled attention.

    Same input/output contract as ``DiTBackbone``; the only structural
    difference is that each block's attention is biased by an
    SE(2)-invariant learnable radial kernel over the current coordinate
    estimate ``data.positions`` (= the noisy interpolant ``x_t`` at this
    FM step).
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        n_layers: int,
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        output_dims: Dict[str, int],
        geomattn_cfg=None,
    ):
        super().__init__()
        self.out_features_dim = int(hidden_dims["output_features_to_pos_dims"])

        def _g(k, default):
            if geomattn_cfg is None:
                return default
            return (
                geomattn_cfg.get(k, default)
                if hasattr(geomattn_cfg, "get")
                else getattr(geomattn_cfg, k, default)
            )

        self.hidden_dim = int(_g("hidden_dim", 256))
        self.n_heads = int(_g("n_heads", 8))
        self.mlp_ratio = float(_g("mlp_ratio", 4))
        self.n_layers = int(_g("n_layers", n_layers))
        self.time_embed_dim = int(_g("time_embed_dim", 256))
        self.num_rbf = int(_g("num_rbf", 16))
        self.init_max_dist = float(_g("init_max_dist", 3.0))
        # Detach the positions used to build the geometric bias so the
        # coordinate losses do not backprop *through* the bias into the
        # (input) interpolant — the geometry is a conditioning signal,
        # not a parameter (self-conditioning semantics). Exposed as a
        # knob for ablation.
        self.detach_geometry = bool(_g("detach_geometry", True))
        # Gradient checkpointing: recompute each GCA block in the
        # backward pass instead of storing its attention scores +
        # geometric bias. GCA's dense per-head bias is O(B·H·N²) and
        # the SDPA-with-float-mask path stores the N×N attention per
        # layer for backward, so a deep stack on cortex-scale slices
        # (N ~ 10k) otherwise needs tens of GB × n_layers. Checkpointing
        # trades ~30% extra compute for a large memory cut and is ON by
        # default for geomattn (unlike DiT, which gets FlashAttention's
        # O(N) memory for free and doesn't need it). Only active in
        # training (no-op at inference / under no_grad).
        self.grad_checkpoint = bool(_g("grad_checkpoint", True))

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

        # Shared RBF basis (one per backbone, reused by every block).
        self.radial = RadialBasis(self.num_rbf, self.init_max_dist)

        self.blocks = nn.ModuleList([
            GeometryCoupledBlock(
                self.hidden_dim, self.n_heads, self.num_rbf, self.mlp_ratio,
            )
            for _ in range(self.n_layers)
        ])
        self.final = DiTFinalLayer(self.hidden_dim, self.out_features_dim)

    def forward(self, data: DataHolder, **_unused_kwargs) -> DataHolder:
        node_mask = data.node_mask                                 # (B, N) bool

        # Per-cell tokens: gene + position embedding (same as DiT).
        tok = self.gene_embed(data.node_features)
        tok = tok + self.pos_embed(data.positions)
        tok = tok * node_mask.unsqueeze(-1).to(tok.dtype)

        # Time conditioning.
        t = data.diffusion_time
        if t.dim() > 1:
            t = t.view(-1)
        c = self.t_embed(t)                                        # (B, D)

        # --- Geometric bias basis (computed ONCE, shared by all blocks).
        # Distances under the CURRENT coordinate estimate x_t. Detached
        # by default so the bias is a pure conditioning signal.
        # We pass the (B, N, N) DISTANCE matrix (not the (B, N, N, K)
        # RBF expansion) to each block: the block accumulates its
        # per-head bias over the K basis functions on the fly, which
        # avoids ever materialising the B·N²·K RBF tensor (the dominant
        # memory term that OOM'd on cortex-scale slices).
        pos = data.positions
        if self.detach_geometry:
            pos = pos.detach()
        pos = pos.float()
        # cdist is O(N²) but exact and numerically clean. Padded cells
        # produce finite garbage rows that the key_padding_mask (-inf)
        # kills downstream.
        dist = torch.cdist(pos, pos)                               # (B, N, N)

        key_padding_mask = ~node_mask                              # (B, N) bool, True at PAD

        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                # use_reentrant=False is the modern checkpoint API: it
                # handles non-tensor args (the shared ``radial`` module,
                # the bool key_padding_mask) cleanly and supports the
                # autograd graph correctly. Recomputes the block's
                # attention + geometric bias in backward.
                tok = checkpoint(
                    block, tok, c, dist, self.radial, key_padding_mask,
                    use_reentrant=False,
                )
            else:
                tok = block(
                    tok, c, dist, self.radial, key_padding_mask=key_padding_mask,
                )

        out = self.final(tok, c)                                   # (B, N, F+2)
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

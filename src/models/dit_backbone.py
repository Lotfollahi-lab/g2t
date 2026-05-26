"""DiT (Diffusion Transformer) backbone — Architectural extension #2.

Implements Peebles & Xie 2022, "Scalable Diffusion Models with
Transformers" (arXiv:2212.09748), the architecture that became state
of the art for image diffusion. Adapted to spatial-transcriptomics
shape:

- Each cell is one token.
- The token = ``gene_embed(gene_features) + pos_embed(positions)``,
  with a learnable token dropout / dropout-style mask absorbed by
  padding.
- Time enters via **adaLN-Zero**: per-block, a small MLP on the time
  embedding produces six modulation vectors (scale_msa, shift_msa,
  gate_msa, scale_mlp, shift_mlp, gate_mlp). LayerNorm × (1 + scale)
  + shift, with attention/FFN output passed through a learnable
  gate. Gates are initialised to zero so each DiT block starts as
  identity — critical for training stability with deep stacks.

Output interface matches ``models.model.Model`` and ``models.egnn.EGNNModel``:
DataHolder in → DataHolder out, with ``pred.node_features`` of width
``output_features_to_pos_dims`` (default 32 under the new defaults)
and ``pred.positions`` of width 2.

Attention path
--------------
Each block uses ``SDPAMultiheadAttention`` (see
``models/sdpa_attention.py``) which routes explicitly through
``F.scaled_dot_product_attention`` — FlashAttention-2 on Ampere+
GPUs, memory-efficient kernel on older GPUs, math fallback on CPU.
O(N²) compute but **O(N) memory** for the attention matrix.

Why on top of EDM-FM
--------------------
The LUNA transformer's 3-stream attention has the gene / time /
position streams interact via hand-coded projections. DiT replaces
that with standard scaled-dot-product attention + adaLN-Zero time
conditioning, which is the proven recipe for FM and diffusion
trajectories. Pairs naturally with the EDM head (the head consumes
``pred.node_features`` independent of which backbone produced it).
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data.dataholder import DataHolder
from models.sdpa_attention import SDPAMultiheadAttention


# ---------------------------------------------------------------------------
# Time embedding (sinusoidal, projected through an MLP)
# ---------------------------------------------------------------------------


def _sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding of a scalar time t.

    Args:
        t: (B,) or (B, 1) tensor of scalar times in [0, 1] (FM-style).
        dim: embedding dimensionality (must be even — half sin, half cos).

    Returns:
        (B, dim) embedding.
    """
    if t.dim() > 1:
        t = t.view(-1)
    half = dim // 2
    # Frequencies on a log scale (the "Attention is All You Need" recipe).
    freqs = torch.exp(
        -math.log(10_000.0)
        * torch.arange(0, half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)        # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # (B, dim)
    if dim % 2 == 1:
        # Pad an extra zero to make the embedding the requested odd dim.
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding → 2-layer MLP. Output width = ``hidden_dim``."""

    def __init__(self, hidden_dim: int, sinusoidal_dim: int = 256):
        super().__init__()
        self.sinusoidal_dim = int(sinusoidal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.sinusoidal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(_sinusoidal_time_embed(t, self.sinusoidal_dim))


# ---------------------------------------------------------------------------
# adaLN-Zero modulation helper
# ---------------------------------------------------------------------------


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: ``x * (1 + scale) + shift``.

    Args:
        x:     (B, N, D) per-token features.
        shift: (B, D)    per-batch modulation shift (broadcast across N).
        scale: (B, D)    per-batch modulation scale (broadcast across N).

    Returns:
        (B, N, D) modulated features.
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# DiT block: MSA + MLP with adaLN-Zero conditioning
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    """One DiT block: pre-LayerNorm → adaLN modulation → multi-head
    self-attention → residual + gate. Then the same pattern for the
    FFN sublayer. Six modulation vectors per block (shift/scale/gate
    × MSA/MLP) come from a tiny MLP on the conditioning ``c``.

    Gates are initialised to zero so each block starts as identity —
    the standard adaLN-Zero trick that lets deep DiT stacks train
    stably from scratch.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} must be divisible by n_heads {n_heads}"
            )

        # ``SDPAMultiheadAttention`` routes EXPLICITLY through
        # ``F.scaled_dot_product_attention`` (FlashAttention-2 on
        # Ampere+, memory-efficient kernel on older GPUs, math
        # fallback on CPU). Same forward signature as
        # nn.MultiheadAttention(batch_first=True) but guaranteed not
        # to silently fall back to the O(N²)-memory native path the
        # way stock MHA can when its fast-path conditions aren't met.
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

        # adaLN-Zero modulation. 6 vectors of size hidden_dim per block.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 6 * self.hidden_dim),
        )
        # Zero-init the final linear in adaLN_modulation so each block
        # starts as exactly identity (shift=0, scale=0, gate=0 → block
        # output = block input). Standard adaLN-Zero recipe.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) per-token features.
            c: (B, D)    conditioning vector (time embedding).
            key_padding_mask: (B, N) bool, ``True`` at PAD positions
                (PyTorch's MultiheadAttention convention — preserved
                here so call sites work unchanged after the swap to
                ``SDPAMultiheadAttention``). Padding positions don't
                contribute to attention weights but their token slots
                are kept in the output (we zero them via the input
                mask outside the block).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # MSA sub-layer.
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        # SDPAMultiheadAttention takes (B, N, D) with batch_first
        # baked in; key_padding_mask is (B, N) bool, True at PAD
        # (same convention as nn.MultiheadAttention).
        attn_out, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP sub-layer.
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


# ---------------------------------------------------------------------------
# Final layer (token → output features + positions)
# ---------------------------------------------------------------------------


class DiTFinalLayer(nn.Module):
    """Final adaLN modulation followed by a Linear projection to the
    two output heads (per-cell node_features of width
    ``out_features_dim`` and per-cell position of width 2).
    """

    def __init__(self, hidden_dim: int, out_features_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        # 2 vectors (shift + scale) for the final modulation, no gate
        # — the projection itself is the gate effectively.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        # Project to (features, positions) jointly so the two heads
        # share the same context. We DON'T zero-init this — earlier
        # revision zero'd both weight and bias to start positions
        # near zero, but that also zero'd the features path, which
        # collapsed cdist gradients at training start (cdist's grad
        # at x=y is degenerate). Default Kaiming-uniform init gives
        # small random outputs whose mean-centred positions are still
        # near zero on average but not pathologically so.
        self.linear = nn.Linear(hidden_dim, out_features_dim + 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# DiT backbone — the public interface (DataHolder in / DataHolder out)
# ---------------------------------------------------------------------------


class DiTBackbone(nn.Module):
    """Drop-in replacement for ``models.model.Model`` using a DiT-style
    architecture. Matches the LUNA Model's input/output contract:

    Inputs (via ``data``):
        ``data.node_features``    (B, N, gene_dim)
        ``data.positions``        (B, N, 2)
        ``data.diffusion_time``   (B, 1)   — scalar time per slice
        ``data.node_mask``        (B, N)   — bool, True for real cells

    Outputs (via returned DataHolder):
        ``pred.node_features``    (B, N, output_features_to_pos_dims)
        ``pred.positions``        (B, N, 2)

    The position-stream is fused into the token via additive embedding
    (vs LUNA's separate position stream). This matches DiT-2D's patch-
    position convention. Padding cells get zeroed inputs and zeroed
    outputs (PyTorch's MultiheadAttention key_padding_mask zeroes
    attention contributions from padding; we additionally mask the
    output to be safe).
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        n_layers: int,
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        output_dims: Dict[str, int],
        dit_cfg=None,
    ):
        super().__init__()
        # ``hidden_dims["output_features_to_pos_dims"]`` is the per-cell
        # feature width on output — same convention as LUNA Model so
        # downstream wrappers (EDM, c2f, gene_recon) don't need to know
        # what backbone produced them.
        self.out_features_dim = int(hidden_dims["output_features_to_pos_dims"])

        def _g(k, default):
            if dit_cfg is None:
                return default
            return (
                dit_cfg.get(k, default)
                if hasattr(dit_cfg, "get")
                else getattr(dit_cfg, k, default)
            )

        self.hidden_dim = int(_g("hidden_dim", 256))
        self.n_heads = int(_g("n_heads", 8))
        self.mlp_ratio = float(_g("mlp_ratio", 4))
        # Default n_layers comes from cfg.model.n_layers (passed in via
        # the constructor's positional arg); the dit-specific override
        # in cfg.model.dit.n_layers takes precedence if set.
        self.n_layers = int(_g("n_layers", n_layers))
        self.time_embed_dim = int(_g("time_embed_dim", 256))

        gene_in = int(input_dims["node_features_dimensions"])

        # Input projections: gene → token, position → token.
        # The two are summed to give the per-cell token. We use 2-layer
        # MLPs so the input projections can absorb scale differences
        # between gene-expression counts and normalised positions.
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

        # DiT blocks.
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_dim, self.n_heads, self.mlp_ratio)
            for _ in range(self.n_layers)
        ])

        # Output head.
        self.final = DiTFinalLayer(self.hidden_dim, self.out_features_dim)

    def forward(self, data: DataHolder, **_unused_kwargs) -> DataHolder:
        # Ignore unused kwargs (e.g. true_positions from c2f wrapper);
        # DiTBackbone doesn't use teacher forcing.
        node_mask = data.node_mask                            # (B, N) bool

        # Build per-cell tokens: gene + position embedding.
        tok = self.gene_embed(data.node_features)             # (B, N, D)
        tok = tok + self.pos_embed(data.positions)            # (B, N, D)
        # Zero out padding tokens (the attention mask handles attention,
        # but downstream the residual stream would still pass through
        # non-zero values at PAD; explicit zero keeps PAD tokens at 0).
        tok = tok * node_mask.unsqueeze(-1).to(tok.dtype)

        # Time conditioning.
        # data.diffusion_time may be (B, 1) or (B,) depending on caller.
        t = data.diffusion_time
        if t.dim() > 1:
            t = t.view(-1)
        c = self.t_embed(t)                                    # (B, D)

        # MultiheadAttention's key_padding_mask wants True at PAD.
        key_padding_mask = ~node_mask                          # (B, N) bool

        # DiT stack.
        for block in self.blocks:
            tok = block(tok, c, key_padding_mask=key_padding_mask)

        # Final projection: features + positions jointly, then split.
        out = self.final(tok, c)                               # (B, N, F+2)
        features = out[..., :-2]                               # (B, N, F)
        positions = out[..., -2:]                              # (B, N, 2)

        # Mask + mean-center positions to match LUNA Model's contract.
        # ``mlp_out_pos_norm`` in LUNA does normalisation per cell; for
        # DiT we lean on adaLN modulation in the final layer plus the
        # zero-init of the position output rows to keep positions
        # well-scaled. We DO mean-center here because downstream
        # pairwise-distance / EDM losses operate on the relative
        # positions only.
        pad_mask_features = node_mask.unsqueeze(-1).to(features.dtype)
        features = features * pad_mask_features
        positions = positions * pad_mask_features
        # Mean-centre over the masked cells, per slice.
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

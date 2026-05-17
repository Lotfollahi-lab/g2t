"""
Metric embedding heads for ScGG.

Two architectures, both producing a d-dim metric embedding per cell that
is used directly for kNN graph construction at inference:

  * `MetricHead` — per-cell MLP. Each cell's metric embedding is a function
    of (its own cell_embed, the single section_embed). No information flows
    between cells in the slice. Cheap, but fundamentally limited: two cells
    with the same gene expression in the same slice get the same embedding,
    so the model cannot break ties between same-type cells.

  * `CrossCellMetricHead` — self-attention transformer over all cells in
    the slice (LUNA-style). Each cell's representation depends on every
    other cell's cell_embed. Substantially more capacity at the cost of
    O(B^2) attention. The recommended choice for the cortex benchmark and
    the one that closes the structural gap vs. LUNA.

Both are L2-normalized so FAISS Euclidean kNN matches cosine ranking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MetricHead(nn.Module):
    """Projects (cell, section) embeddings into a metric space for kNN retrieval.

    The output dimension is decoupled from physical space (default d=32),
    giving the model substantially more capacity than the 2-D coordinate
    bottleneck used by LUNA / coordinate-based methods. FAISS kNN over
    these embeddings is what produces the final spatial graph at inference.

    Args:
        cell_embed_dim: Input cell embedding dimension.
        section_embed_dim: Input section embedding dimension.
        hidden_dims: Hidden layer sizes for the projection MLP.
        embed_dim: Output metric dimension.
        normalize: If True, L2-normalize the output so Euclidean kNN matches
            cosine ranking. FAISS IndexFlatL2 on normalized vectors then
            produces the same ranking as cosine similarity.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        cell_embed_dim: int = 128,
        section_embed_dim: int = 64,
        hidden_dims: List[int] = [256, 128],
        embed_dim: int = 32,
        normalize: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.normalize = normalize

        in_dim = cell_embed_dim + section_embed_dim
        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, embed_dim)

    def forward(
        self,
        cell_embed: torch.Tensor,
        section_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cell_embed: (batch_size, cell_embed_dim).
            section_embed: (section_embed_dim,) or (batch_size, section_embed_dim).

        Returns:
            Metric embedding, shape (batch_size, embed_dim). L2-normalized
            if self.normalize.
        """
        if section_embed.dim() == 1:
            section_embed = section_embed.unsqueeze(0).expand(cell_embed.shape[0], -1)
        h = torch.cat([cell_embed, section_embed], dim=-1)
        h = self.mlp(h)
        z = self.out(h)
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return z


class CrossCellMetricHead(nn.Module):
    """Cross-cell-attention metric head (LUNA-style).

    Treats the cells in one slice as a single sequence and runs a stack of
    standard Transformer encoder layers (multi-head self-attention + MLP)
    over them. The output is then projected to a d-dim metric space and
    L2-normalized.

    Critically, each cell's output depends on *every other cell* in the
    slice via attention, so two cells with identical gene expression but
    different spatial roles in the tissue can still receive different
    metric embeddings. This is the inductive bias that LUNA's diffusion
    process gets from attention, and the bottleneck in the per-cell MLP
    variant.

    Section context can optionally be injected as an additional learnable
    bias added to every cell token (defaults to off — the section context
    is already implicit in the attention over slice cells).

    Args:
        cell_embed_dim: Input cell embedding dimension (per-cell encoder
            output). This dimension is preserved through the attention block;
            the final projection maps cell_embed_dim -> embed_dim.
        section_embed_dim: If using section_embed as conditioning, its
            dimensionality.
        embed_dim: Output metric dimension.
        n_layers: Number of transformer encoder layers (default 2).
        n_heads: Attention heads per layer (default 4).
        ff_mult: Feedforward expansion multiplier inside each layer (default 4).
        dropout: Dropout in attention + feedforward.
        normalize: L2-normalize the output (default True).
        use_section_embed: If True, add a per-cell bias derived from
            section_embed before the attention stack. Default False —
            attention itself integrates slice context.

    Notes on memory and speed:
        Full-attention is O(B^2 d) memory + compute. For B=8192 cells and
        d=128, each attention matrix is 64M floats per head per layer. With
        n_heads=4 and n_layers=2 this is 512M floats (~2 GB fp32) before
        considering activations. For the LUNA cortex benchmark (all slices
        < 7,500 cells) this fits comfortably; for million-cell atlases you
        would need linear / windowed attention instead.
    """

    def __init__(
        self,
        cell_embed_dim: int = 128,
        section_embed_dim: int = 64,
        embed_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        normalize: bool = True,
        use_section_embed: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.normalize = normalize
        self.use_section_embed = use_section_embed

        # Optional section-bias projector (used only when use_section_embed=True).
        if use_section_embed:
            self.section_proj = nn.Linear(section_embed_dim, cell_embed_dim)
        else:
            self.section_proj = None

        # Standard transformer encoder layers.
        layer = nn.TransformerEncoderLayer(
            d_model=cell_embed_dim,
            nhead=n_heads,
            dim_feedforward=cell_embed_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm: more stable for our scale
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Output projection to metric space.
        self.out = nn.Linear(cell_embed_dim, embed_dim)

    def forward(
        self,
        cell_embed: torch.Tensor,
        section_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            cell_embed: (B, cell_embed_dim) — all cells in the current slice
                seen by attention together.
            section_embed: (section_embed_dim,) or (B, section_embed_dim).
                Only consumed when self.use_section_embed=True.

        Returns:
            Metric embedding, shape (B, embed_dim). L2-normalized if
            self.normalize.
        """
        x = cell_embed  # (B, d)

        if self.use_section_embed and section_embed is not None:
            if section_embed.dim() == 1:
                section_embed = section_embed.unsqueeze(0).expand(x.shape[0], -1)
            x = x + self.section_proj(section_embed)

        # (B, d) -> (1, B, d) for the transformer (batch_first means
        # first dim is "batch of sequences", second is the sequence).
        x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.squeeze(0)  # (B, d)

        z = self.out(x)
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return z

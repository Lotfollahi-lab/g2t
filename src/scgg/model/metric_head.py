"""
Metric embedding head for ScGG.

Projects (cell embedding, section embedding) -> d-dim metric embedding.
The metric embedding is the object used directly for kNN graph construction
at inference: build a kNN index over the d-dim embeddings and return the
top-k neighbors per cell.

In the "contrastive" training objective (the default and primary mode),
this head is trained with a supervised InfoNCE / listwise rank loss against
the ground-truth spatial kNN graph (PinSage-style retrieval). The flow
matching machinery is retained as an ablation and ignores this head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


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

"""
Gene expression encoder and section-level encoder.

The gene expression encoder maps per-cell expression vectors to compact
embeddings. The section encoder aggregates cell embeddings into a single
section-level context vector via attention pooling.

Design rationale:
- Per-cell encoder with NO cell-cell message passing, to avoid creating
  dependency on expression graph structure (which shifts during domain transfer).
- Section-level encoder provides global tissue context (composition, tissue
  identity) without per-cell coupling.
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional


class GeneExpressionEncoder(nn.Module):
    """Encodes per-cell gene expression vectors into compact embeddings.

    Architecture: MLP with residual connections, layer normalization, and
    GELU activations. Operates independently per cell (no cell-cell interaction).

    Args:
        n_genes: Number of input genes.
        hidden_dims: List of hidden layer dimensions.
        embed_dim: Output embedding dimension.
        dropout: Dropout rate.
        norm: Normalization type ('layernorm', 'batchnorm', 'none').
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dims: List[int] = [512, 256],
        embed_dim: int = 128,
        dropout: float = 0.1,
        norm: str = "layernorm",
    ):
        super().__init__()
        self.n_genes = n_genes
        self.embed_dim = embed_dim

        layers = []
        in_dim = n_genes
        for h_dim in hidden_dims:
            layers.append(self._make_block(in_dim, h_dim, dropout, norm))
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # Final projection to embedding space
        self.proj = nn.Linear(in_dim, embed_dim)

    def _make_block(
        self, in_dim: int, out_dim: int, dropout: float, norm: str
    ) -> nn.Module:
        layers = [nn.Linear(in_dim, out_dim)]
        if norm == "layernorm":
            layers.append(nn.LayerNorm(out_dim))
        elif norm == "batchnorm":
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Gene expression matrix, shape (n_cells, n_genes).

        Returns:
            Cell embeddings, shape (n_cells, embed_dim).
        """
        h = self.backbone(x)
        return self.proj(h)


class SectionEncoder(nn.Module):
    """Computes a section-level context embedding via attention pooling.

    Uses a DeepSets-style architecture: per-cell embeddings are aggregated
    into a single section-level vector using multi-head attention pooling
    (with a learned query). This captures global tissue context (cell type
    composition, overall expression patterns) without depending on cell order
    or pairwise relationships.

    For scalability with large sections (>100k cells), supports subsampling
    a fixed number of cells for the pooling computation.

    Args:
        cell_embed_dim: Dimension of input cell embeddings.
        hidden_dim: Hidden dimension for attention.
        embed_dim: Output section embedding dimension.
        n_attention_heads: Number of attention heads.
        subsample_size: Maximum cells to use for pooling (None = use all).
    """

    def __init__(
        self,
        cell_embed_dim: int = 128,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        n_attention_heads: int = 4,
        subsample_size: Optional[int] = 4096,
    ):
        super().__init__()
        self.subsample_size = subsample_size
        self.embed_dim = embed_dim

        # Learned query for attention pooling
        self.query = nn.Parameter(torch.randn(1, n_attention_heads, hidden_dim // n_attention_heads))
        nn.init.xavier_uniform_(self.query)

        # Project cell embeddings to keys and values
        self.key_proj = nn.Linear(cell_embed_dim, hidden_dim)
        self.value_proj = nn.Linear(cell_embed_dim, hidden_dim)

        self.n_heads = n_attention_heads
        self.head_dim = hidden_dim // n_attention_heads
        self.scale = math.sqrt(self.head_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, cell_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cell_embeddings: Shape (n_cells, cell_embed_dim).

        Returns:
            Section embedding, shape (embed_dim,).
        """
        n_cells = cell_embeddings.shape[0]

        # Subsample for scalability — ONLY during training. At eval we use the
        # full set so the section embedding is deterministic across runs and
        # the headline benchmark numbers don't wobble between repeated
        # inference passes on the same checkpoint.
        if (
            self.training
            and self.subsample_size is not None
            and n_cells > self.subsample_size
        ):
            idx = torch.randperm(n_cells, device=cell_embeddings.device)[: self.subsample_size]
            cell_embeddings = cell_embeddings[idx]

        # Compute keys and values: (n_cells, n_heads, head_dim)
        keys = self.key_proj(cell_embeddings).view(-1, self.n_heads, self.head_dim)
        values = self.value_proj(cell_embeddings).view(-1, self.n_heads, self.head_dim)

        # Attention: query (1, n_heads, head_dim) @ keys^T (n_heads, head_dim, n_cells)
        # -> (1, n_heads, n_cells)
        query = self.query.expand(1, -1, -1)  # (1, n_heads, head_dim)
        attn_weights = torch.einsum("qhd,nhd->qhn", query, keys) / self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)  # (1, n_heads, n_cells)

        # Weighted sum: (1, n_heads, n_cells) @ (n_cells, n_heads, head_dim) -> (1, n_heads, head_dim)
        pooled = torch.einsum("qhn,nhd->qhd", attn_weights, values)
        pooled = pooled.reshape(1, -1).squeeze(0)  # (hidden_dim,)

        return self.output_proj(pooled)

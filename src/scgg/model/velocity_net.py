"""
Velocity network for conditional flow matching.

Predicts the velocity field v(z_t, t | c) where:
- z_t is the current spatial embedding at flow time t
- t is the flow time (0 = noise, 1 = spatial coordinates)
- c is the conditioning: cell expression embedding + section embedding + k target

The network uses sinusoidal time embeddings and learned k-embeddings for
controlling the target graph sparsity.
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the flow time step."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values, shape (batch_size,) or scalar.

        Returns:
            Time embeddings, shape (batch_size, dim).
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class VelocityNetwork(nn.Module):
    """Predicts flow velocity conditioned on cell/section context and target k.

    Architecture: MLP with conditioning via concatenation. The network takes
    the current spatial state z_t, time embedding, cell embedding, section
    embedding, and k embedding as input, and predicts the velocity dz/dt.

    We use a simple MLP rather than a transformer because:
    1. Each cell's velocity is predicted independently (no cell-cell attention),
       which is key for scalability and domain transfer robustness.
    2. Cell-cell context is captured through the section-level embedding.
    3. The flow matching framework provides the generative power; the velocity
       net just needs to be a good function approximator.

    Args:
        spatial_dim: Dimension of spatial embeddings (2 for 2D coordinates).
        cell_embed_dim: Dimension of cell expression embeddings.
        section_embed_dim: Dimension of section-level embeddings.
        hidden_dims: Hidden layer dimensions.
        time_embed_dim: Dimension of sinusoidal time embeddings.
        k_embed_dim: Dimension of learned k (graph sparsity) embeddings.
        k_max: Maximum k value for the embedding table.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        cell_embed_dim: int = 128,
        section_embed_dim: int = 64,
        hidden_dims: List[int] = [512, 512, 256],
        time_embed_dim: int = 64,
        k_embed_dim: int = 16,
        k_max: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim

        # Embedding layers
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
        )

        # Learned embedding for target k (graph control parameter)
        # k=0 reserved for "no control" / marginal mode
        self.k_embed = nn.Embedding(k_max + 1, k_embed_dim)

        # Input: z_t + time_embed + cell_embed + section_embed + k_embed
        input_dim = spatial_dim + time_embed_dim + cell_embed_dim + section_embed_dim + k_embed_dim

        # Build MLP with residual connections where dimensions match
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(ResidualBlock(in_dim, h_dim, dropout))
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)

        # Output projection to velocity
        self.out_proj = nn.Linear(in_dim, spatial_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize output layer to near-zero for stable training start."""
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_embed: torch.Tensor,
        section_embed: torch.Tensor,
        k_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t: Current spatial state, shape (batch_size, spatial_dim).
            t: Flow time, shape (batch_size,) with values in [0, 1].
            cell_embed: Cell expression embeddings, shape (batch_size, cell_embed_dim).
            section_embed: Section embedding, shape (section_embed_dim,) or
                (batch_size, section_embed_dim). Broadcast to all cells.
            k_target: Target k for graph construction, shape (batch_size,) of ints.

        Returns:
            Predicted velocity, shape (batch_size, spatial_dim).
        """
        batch_size = z_t.shape[0]

        # Time embedding
        t_emb = self.time_mlp(self.time_embed(t))

        # k embedding
        k_emb = self.k_embed(k_target)

        # Broadcast section embedding if needed
        if section_embed.dim() == 1:
            section_embed = section_embed.unsqueeze(0).expand(batch_size, -1)

        # Concatenate all inputs
        x = torch.cat([z_t, t_emb, cell_embed, section_embed, k_emb], dim=-1)

        # MLP forward
        h = self.mlp(x)
        return self.out_proj(h)


class ResidualBlock(nn.Module):
    """MLP block with optional residual connection (when dims match)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.use_residual = in_dim == out_dim
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.act(self.norm(self.fc(x))))
        if self.use_residual:
            return h + x
        return h

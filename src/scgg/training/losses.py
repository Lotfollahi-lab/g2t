"""
Loss functions for ScGG training.

Primary loss: Conditional flow matching loss (MSE on velocity prediction).
Auxiliary loss: Contrastive spatial loss that uses ground truth spatial
graph edges to define positive/negative pairs, encouraging the encoder
to learn spatially informative representations.

The contrastive loss is a key differentiator from LUNA, which only uses
coordinate-level losses. By directly optimizing for graph structure, our
encoder learns what matters for spatial connectivity rather than exact
coordinate placement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from typing import Optional, Tuple, Dict


class ScGGLoss(nn.Module):
    """Combined loss for ScGG training.

    Args:
        lambda_contrastive: Weight for the contrastive spatial loss.
        temperature: Temperature for InfoNCE contrastive loss.
        n_negatives: Number of negative samples per positive pair.
    """

    def __init__(
        self,
        lambda_contrastive: float = 0.1,
        temperature: float = 0.1,
        n_negatives: int = 64,
    ):
        super().__init__()
        self.lambda_contrastive = lambda_contrastive
        self.temperature = temperature
        self.n_negatives = n_negatives

    def contrastive_spatial_loss(
        self,
        cell_embeddings: torch.Tensor,
        spatial_adj: sparse.csr_matrix,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Contrastive loss based on ground truth spatial graph.

        For each cell, its spatial neighbors (from the ground truth graph)
        are positive pairs, and randomly sampled non-neighbors are negatives.
        This trains the encoder to produce embeddings where spatial neighbors
        are closer than non-neighbors.

        Note: This loss operates on ENCODER outputs (gene expression embeddings),
        not on the flow-generated spatial embeddings. This is intentional:
        it directly shapes the conditioning signal so the flow has better
        information to work with.

        Args:
            cell_embeddings: Encoder outputs, shape (batch_size, embed_dim).
            spatial_adj: Ground truth spatial adjacency (sparse, full section).
            batch_indices: Indices of batch cells within the full section.
                If None, assumes batch = full section.

        Returns:
            Scalar contrastive loss.
        """
        batch_size = cell_embeddings.shape[0]
        device = cell_embeddings.device

        if batch_size < 4:
            return torch.tensor(0.0, device=device)

        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(cell_embeddings, dim=-1)

        # Get adjacency for batch cells
        if batch_indices is not None:
            idx = batch_indices.cpu().numpy()
            adj_batch = spatial_adj[idx][:, idx]
        else:
            adj_batch = spatial_adj

        # Convert to COO for efficient iteration
        adj_coo = adj_batch.tocoo()

        if adj_coo.nnz == 0:
            return torch.tensor(0.0, device=device)

        # Sample a subset of positive pairs for efficiency
        n_pos = min(adj_coo.nnz, batch_size * 4)
        if adj_coo.nnz > n_pos:
            perm = torch.randperm(adj_coo.nnz)[:n_pos]
            pos_i = torch.tensor(adj_coo.row, dtype=torch.long)[perm].to(device)
            pos_j = torch.tensor(adj_coo.col, dtype=torch.long)[perm].to(device)
        else:
            pos_i = torch.tensor(adj_coo.row, dtype=torch.long, device=device)
            pos_j = torch.tensor(adj_coo.col, dtype=torch.long, device=device)

        # Positive similarities
        pos_sim = torch.sum(
            embeddings_norm[pos_i] * embeddings_norm[pos_j], dim=-1
        )  # (n_pos,)

        # Sample negative indices for each positive pair
        n_neg = min(self.n_negatives, batch_size - 1)
        neg_indices = torch.randint(0, batch_size, (len(pos_i), n_neg), device=device)

        # Negative similarities: (n_pos, n_neg)
        anchor_emb = embeddings_norm[pos_i].unsqueeze(1)  # (n_pos, 1, dim)
        neg_emb = embeddings_norm[neg_indices]  # (n_pos, n_neg, dim)
        neg_sim = torch.sum(anchor_emb * neg_emb, dim=-1)  # (n_pos, n_neg)

        # InfoNCE loss
        # log(exp(pos/tau) / (exp(pos/tau) + sum(exp(neg/tau))))
        logits = torch.cat(
            [pos_sim.unsqueeze(-1) / self.temperature,
             neg_sim / self.temperature],
            dim=-1,
        )  # (n_pos, 1 + n_neg)

        labels = torch.zeros(len(pos_i), dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(
        self,
        fm_loss: torch.Tensor,
        cell_embeddings: Optional[torch.Tensor] = None,
        spatial_adj: Optional[sparse.csr_matrix] = None,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            fm_loss: Flow matching loss (already computed by flow module).
            cell_embeddings: Encoder outputs for contrastive loss.
            spatial_adj: Ground truth spatial graph.
            batch_indices: Indices of batch cells in full section.

        Returns:
            total_loss: Combined scalar loss.
            metrics: Dict of individual loss components.
        """
        total_loss = fm_loss
        metrics = {"fm_loss": fm_loss.item()}

        if (
            self.lambda_contrastive > 0
            and cell_embeddings is not None
            and spatial_adj is not None
        ):
            c_loss = self.contrastive_spatial_loss(
                cell_embeddings, spatial_adj, batch_indices
            )
            total_loss = total_loss + self.lambda_contrastive * c_loss
            metrics["contrastive_loss"] = c_loss.item()

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics

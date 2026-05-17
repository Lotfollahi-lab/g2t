"""
Loss functions for ScGG training.

ScGG supports two training objectives, selected via config["training"]["objective"]:

1. "contrastive" (default, primary): PinSage-style supervised contrastive /
   listwise ranking on the metric-head output. For each anchor cell, the
   positives are its ground-truth spatial kNN neighbors and the negatives
   are sampled non-neighbors (in-batch by default). This directly optimizes
   what we care about: that the kNN of the learned embedding matches the
   true spatial kNN.

2. "flow_matching" (ablation): retains the original conditional flow
   matching objective on 2-D coordinates. This is kept for ablation
   experiments and is not the default. Optionally augmented by an
   auxiliary contrastive loss on encoder outputs (legacy behavior).

The implementation is intentionally split so the trainer can pick one path
and never instantiate the other.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Primary objective: supervised contrastive on the metric embedding
# ---------------------------------------------------------------------------


class ContrastiveRankingLoss(nn.Module):
    """Primary ScGG loss: supervised InfoNCE on the metric embedding.

    For each anchor cell i, we sample one positive p_i from its ground-truth
    spatial kNN neighbors and treat all other cells in the batch as negatives.
    The loss is the standard InfoNCE cross-entropy:

        L_i = -log( exp(sim(z_i, z_{p_i})/tau)
                    / sum_{j in batch} exp(sim(z_i, z_j)/tau) )

    Cells with no neighbors inside the current batch (e.g., the batch missed
    them via subsampling) are skipped.

    Args:
        temperature: InfoNCE temperature; smaller -> sharper, larger -> softer.
        positives_per_anchor: Number of positives sampled per anchor (averaged).
            Set >1 to better cover the k=10 neighborhood with one batch pass.
        exclude_self: If True, exclude the anchor itself from the candidate set.
        normalize_inputs: If True, L2-normalize embeddings before similarity.
            Pass False if the MetricHead already normalizes (avoids double work).
    """

    def __init__(
        self,
        temperature: float = 0.1,
        positives_per_anchor: int = 1,
        exclude_self: bool = True,
        normalize_inputs: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.positives_per_anchor = positives_per_anchor
        self.exclude_self = exclude_self
        self.normalize_inputs = normalize_inputs

    def forward(
        self,
        embeddings: torch.Tensor,
        spatial_adj: sparse.csr_matrix,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: Metric embeddings for the batch, shape (B, d).
            spatial_adj: Sparse ground-truth spatial adjacency for the *full*
                section, shape (N_section, N_section).
            batch_indices: Indices of the batch cells within the full section,
                shape (B,). If None, assumes batch == full section.

        Returns:
            loss: Scalar InfoNCE loss.
            metrics: Dict with diagnostic metrics.
        """
        device = embeddings.device
        B = embeddings.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=device), {
                "contrastive_loss": 0.0,
                "n_anchors_used": 0,
                "n_anchors_skipped": B,
            }

        # Slice the section-level adjacency down to the batch
        if batch_indices is not None:
            idx_np = batch_indices.detach().cpu().numpy()
            adj_batch = spatial_adj[idx_np][:, idx_np]
        else:
            adj_batch = spatial_adj
        adj_csr = adj_batch.tocsr()

        if self.normalize_inputs:
            embeddings = F.normalize(embeddings, dim=-1)

        # Pairwise cosine/Euclidean-on-normalized similarity over the batch
        # If embeddings are L2-normalized, dot product == cosine similarity.
        sim = embeddings @ embeddings.t()  # (B, B)
        if self.exclude_self:
            diag_mask = torch.eye(B, dtype=torch.bool, device=device)
            sim = sim.masked_fill(diag_mask, float("-inf"))
        sim = sim / self.temperature

        # For each anchor, collect indices of in-batch positives
        # adj_csr.indptr[i]:indptr[i+1] gives neighbors of anchor i.
        indptr = adj_csr.indptr
        indices = adj_csr.indices

        losses = []
        n_skipped = 0

        # We sample positives per anchor on CPU for clarity; the heavy work
        # (the softmax) stays on GPU.
        for i in range(B):
            start, end = indptr[i], indptr[i + 1]
            n_pos = end - start
            if n_pos == 0:
                n_skipped += 1
                continue

            # Sample positives
            if n_pos <= self.positives_per_anchor:
                pos_idx = torch.from_numpy(indices[start:end].copy()).to(
                    device=device, dtype=torch.long
                )
            else:
                perm = torch.randperm(n_pos, device="cpu")[: self.positives_per_anchor]
                pos_idx = torch.from_numpy(indices[start:end][perm.numpy()].copy()).to(
                    device=device, dtype=torch.long
                )

            # InfoNCE: log-sum-exp over all candidates, minus pos logit
            # (averaging over multiple positives if requested).
            denom = torch.logsumexp(sim[i], dim=0)
            pos_logits = sim[i, pos_idx]
            # log mean exp of positives is approximated by mean of positives'
            # log-probs in cross-entropy fashion; we use mean cross-entropy.
            losses.append(-(pos_logits - denom).mean())

        if not losses:
            return torch.tensor(0.0, device=device), {
                "contrastive_loss": 0.0,
                "n_anchors_used": 0,
                "n_anchors_skipped": int(n_skipped),
            }

        loss = torch.stack(losses).mean()
        metrics = {
            "contrastive_loss": loss.item(),
            "n_anchors_used": int(B - n_skipped),
            "n_anchors_skipped": int(n_skipped),
        }
        return loss, metrics


# ---------------------------------------------------------------------------
# Ablation objective: conditional flow matching (+ optional aux contrastive)
# ---------------------------------------------------------------------------


class FlowMatchingLoss(nn.Module):
    """Flow matching loss with an optional auxiliary contrastive term.

    This is the legacy training path: the velocity network is trained with
    MSE on conditional velocities (computed by ConditionalFlowMatching.compute_loss),
    optionally augmented with a contrastive loss on encoder cell embeddings.

    Args:
        lambda_contrastive: Weight for the auxiliary contrastive term (0 disables).
        temperature: InfoNCE temperature for the auxiliary term.
        n_negatives: Negatives per positive pair for the auxiliary term.
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

    @staticmethod
    def _auxiliary_contrastive(
        cell_embeddings: torch.Tensor,
        spatial_adj: sparse.csr_matrix,
        batch_indices: Optional[torch.Tensor],
        temperature: float,
        n_negatives: int,
    ) -> torch.Tensor:
        """In-batch InfoNCE on encoder outputs (legacy auxiliary loss).

        Identical behavior to the previous ScGGLoss.contrastive_spatial_loss
        so flow-matching ablations match the prior runs.
        """
        B = cell_embeddings.shape[0]
        device = cell_embeddings.device
        if B < 4:
            return torch.tensor(0.0, device=device)

        emb_n = F.normalize(cell_embeddings, dim=-1)

        if batch_indices is not None:
            idx = batch_indices.detach().cpu().numpy()
            adj_batch = spatial_adj[idx][:, idx]
        else:
            adj_batch = spatial_adj
        adj_coo = adj_batch.tocoo()
        if adj_coo.nnz == 0:
            return torch.tensor(0.0, device=device)

        n_pos = min(adj_coo.nnz, B * 4)
        if adj_coo.nnz > n_pos:
            perm = torch.randperm(adj_coo.nnz)[:n_pos]
            pos_i = torch.tensor(adj_coo.row, dtype=torch.long)[perm].to(device)
            pos_j = torch.tensor(adj_coo.col, dtype=torch.long)[perm].to(device)
        else:
            pos_i = torch.tensor(adj_coo.row, dtype=torch.long, device=device)
            pos_j = torch.tensor(adj_coo.col, dtype=torch.long, device=device)

        pos_sim = torch.sum(emb_n[pos_i] * emb_n[pos_j], dim=-1)
        n_neg = min(n_negatives, B - 1)
        neg_indices = torch.randint(0, B, (len(pos_i), n_neg), device=device)
        anchor = emb_n[pos_i].unsqueeze(1)
        neg = emb_n[neg_indices]
        neg_sim = torch.sum(anchor * neg, dim=-1)

        logits = torch.cat(
            [pos_sim.unsqueeze(-1) / temperature, neg_sim / temperature],
            dim=-1,
        )
        labels = torch.zeros(len(pos_i), dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        fm_loss: torch.Tensor,
        cell_embeddings: Optional[torch.Tensor] = None,
        spatial_adj: Optional[sparse.csr_matrix] = None,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            fm_loss: Scalar flow matching loss (already computed by
                ConditionalFlowMatching.compute_loss).
            cell_embeddings: Encoder outputs (for the auxiliary contrastive).
            spatial_adj: Section-level GT spatial adjacency.
            batch_indices: Indices of batch cells within the section.

        Returns:
            total_loss, metrics dict.
        """
        total = fm_loss
        metrics = {"fm_loss": fm_loss.item()}

        if (
            self.lambda_contrastive > 0
            and cell_embeddings is not None
            and spatial_adj is not None
        ):
            c = self._auxiliary_contrastive(
                cell_embeddings,
                spatial_adj,
                batch_indices,
                self.temperature,
                self.n_negatives,
            )
            total = total + self.lambda_contrastive * c
            metrics["aux_contrastive_loss"] = c.item()

        metrics["total_loss"] = total.item()
        return total, metrics


# ---------------------------------------------------------------------------
# Backwards-compat alias
# ---------------------------------------------------------------------------

# Old name kept for any code that imports ScGGLoss directly. Resolves to the
# flow-matching variant since that was the previous default behavior.
ScGGLoss = FlowMatchingLoss

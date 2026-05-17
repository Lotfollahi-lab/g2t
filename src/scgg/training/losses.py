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

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary objective: supervised contrastive on the metric embedding
# ---------------------------------------------------------------------------


class ContrastiveRankingLoss(nn.Module):
    """Primary ScGG loss: supervised InfoNCE on the metric embedding.

    For each anchor cell *i*, one positive p_i is sampled from its in-batch
    ground-truth spatial neighbors and all other cells in the batch are
    treated as negatives. The loss is the standard InfoNCE cross-entropy:

        L_i = -log( exp(sim(z_i, z_{p_i})/tau)
                    / sum_{j in batch, j != i} exp(sim(z_i, z_j)/tau) )

    Implementation notes:

    * Fully vectorized — no Python loop over anchors. Builds a dense (B, B)
      positive mask once per mini-batch, samples a single positive per anchor
      via masked-argmax, and gathers similarities with `torch.gather`. This is
      ~50–100× faster than a per-cell Python loop on GPU.
    * Defensive against self-loops in the adjacency. We force the diagonal of
      the positive mask to False so `pos_idx` can never equal `i`, even if the
      GT graph accidentally contains a self-loop (which can happen with
      duplicate coordinates + FAISS).
    * Defensive against non-finite per-anchor losses. Any anchor that produces
      `inf` or `nan` is dropped with a warning instead of poisoning the batch
      loss. The number dropped is exposed as `n_anchors_clamped`.

    Args:
        temperature: InfoNCE temperature; smaller -> sharper, larger -> softer.
        positives_per_anchor: Currently the vectorized path samples one positive
            per anchor per forward. Higher values are accepted but are
            silently treated as 1 for now (the loop-based multi-positive
            variant was the source of the slowdown).
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
        if positives_per_anchor != 1:
            logger.warning(
                "positives_per_anchor=%d is currently treated as 1 in the "
                "vectorized path. The previous loop-based multi-positive "
                "implementation was a major training-time bottleneck.",
                positives_per_anchor,
            )

    def _build_pos_mask(
        self,
        spatial_adj: sparse.csr_matrix,
        batch_indices: Optional[torch.Tensor],
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a (B, B) boolean mask: pos_mask[i, j] iff (i, j) is a GT edge.

        Always forces the diagonal to False as a defense against
        self-loops in the upstream GT graph.
        """
        if batch_indices is not None:
            idx_np = batch_indices.detach().cpu().numpy()
            adj_batch = spatial_adj[idx_np][:, idx_np]
        else:
            adj_batch = spatial_adj
        adj_coo = adj_batch.tocoo()

        mask = torch.zeros(B, B, dtype=torch.bool, device=device)
        if adj_coo.nnz > 0:
            row = torch.as_tensor(adj_coo.row, dtype=torch.long, device=device)
            col = torch.as_tensor(adj_coo.col, dtype=torch.long, device=device)
            # Filter out any self-loops (defensive).
            keep = row != col
            mask[row[keep], col[keep]] = True
        # Belt-and-braces: zero the diagonal even if no edges loaded.
        mask.fill_diagonal_(False)
        return mask

    def forward(
        self,
        embeddings: torch.Tensor,
        spatial_adj: sparse.csr_matrix,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: (B, d) metric embeddings for the batch.
            spatial_adj: sparse (N_section, N_section) GT adjacency for the
                full section.
            batch_indices: (B,) indices of batch cells within the section.
                If None, batch == full section.

        Returns:
            loss: scalar InfoNCE loss (zero tensor with grad_fn if no usable
                anchors, so the trainer's backward call stays well-defined).
            metrics: dict with diagnostic counters.
        """
        device = embeddings.device
        B = embeddings.shape[0]

        # Trivial cases.
        if B < 4:
            return embeddings.sum() * 0.0, {
                "contrastive_loss": 0.0,
                "n_anchors_used": 0,
                "n_anchors_skipped": B,
                "n_anchors_clamped": 0,
            }

        pos_mask = self._build_pos_mask(spatial_adj, batch_indices, B, device)
        has_pos = pos_mask.any(dim=1)  # (B,) — True if anchor has any positive
        if not has_pos.any():
            return embeddings.sum() * 0.0, {
                "contrastive_loss": 0.0,
                "n_anchors_used": 0,
                "n_anchors_skipped": int(B),
                "n_anchors_clamped": 0,
            }

        if self.normalize_inputs:
            embeddings = F.normalize(embeddings, dim=-1)

        # Scaled similarity matrix (B, B). If embeddings are L2-normalized,
        # this is cosine sim / temperature.
        sim = (embeddings @ embeddings.t()) / self.temperature

        # Sample one positive per anchor: pick the argmax of (uniform noise
        # masked to -inf on non-positives). Anchors with no positives get a
        # garbage value that we drop via `has_pos`.
        rand = torch.rand(B, B, device=device, dtype=embeddings.dtype)
        rand = rand.masked_fill(~pos_mask, float("-inf"))
        pos_idx = rand.argmax(dim=1)  # (B,)

        # Gather positive logits in a vectorized way.
        pos_logits = torch.gather(sim, 1, pos_idx.unsqueeze(1)).squeeze(1)  # (B,)

        # Denominator: log-sum-exp over all candidates excluding self.
        if self.exclude_self:
            eye = torch.eye(B, dtype=torch.bool, device=device)
            sim_denom = sim.masked_fill(eye, float("-inf"))
        else:
            sim_denom = sim
        denom = torch.logsumexp(sim_denom, dim=1)  # (B,)

        # Per-anchor InfoNCE cross-entropy.
        loss_per_anchor = denom - pos_logits  # (B,)
        loss_per_anchor = loss_per_anchor[has_pos]

        # Drop any anchors that came out non-finite (defensive; should not
        # happen now that we force the pos-mask diagonal to False, but cheap
        # to check).
        finite = torch.isfinite(loss_per_anchor)
        n_clamped = int((~finite).sum().item())
        if n_clamped > 0:
            logger.warning(
                "ContrastiveRankingLoss: dropped %d / %d anchors with "
                "non-finite per-anchor loss; check for degenerate embeddings "
                "or adjacency self-loops.",
                n_clamped, int(has_pos.sum().item()),
            )
            loss_per_anchor = loss_per_anchor[finite]

        if loss_per_anchor.numel() == 0:
            return embeddings.sum() * 0.0, {
                "contrastive_loss": 0.0,
                "n_anchors_used": 0,
                "n_anchors_skipped": int(B - has_pos.sum().item()),
                "n_anchors_clamped": n_clamped,
            }

        loss = loss_per_anchor.mean()

        n_used = int(has_pos.sum().item()) - n_clamped
        return loss, {
            "contrastive_loss": loss.item(),
            "n_anchors_used": n_used,
            "n_anchors_skipped": int(B - has_pos.sum().item()),
            "n_anchors_clamped": n_clamped,
        }


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

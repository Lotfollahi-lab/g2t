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
    """Primary ScGG loss: supervised contrastive (SupCon, Khosla 2020) on the
    metric embedding.

    For each anchor cell *i*, ALL of its in-batch ground-truth spatial
    neighbors are positives; all other cells in the batch (excluding self)
    are negatives. The loss is averaged log-softmax over the positive set:

        L_i = -1/|P_i| * sum_{p in P_i} log( exp(sim(z_i, z_p)/tau)
                                              / sum_{j != i} exp(sim(z_i, z_j)/tau) )

    Compared to the single-positive InfoNCE used previously, SupCon gives
    ~k× stronger gradient signal per step (where k is the GT graph degree,
    typically 10) at essentially zero extra cost. Khosla et al. (2020)
    showed this generalizes better than single-positive InfoNCE in
    classification settings; the spatial-graph case is analogous.

    Implementation notes:

    * Fully vectorized: one matmul for similarity, one logsumexp for the
      denominator, one masked sum for the numerator. No Python loop.
    * Defensive against self-loops in the adjacency: the positive mask
      diagonal is forced to False.
    * Defensive against non-finite per-anchor losses: `n_anchors_clamped`
      counts how many anchors had to be dropped (should always be 0).

    Args:
        temperature: SupCon temperature. Smaller = sharper, harder positives.
            Default 0.07 (SimCLR / SupCon convention).
        max_positives_per_anchor: Cap the number of positives used per anchor
            for memory reasons. None or <=0 means "use all in-batch positives"
            (recommended). Set to a small int (e.g. 4) if memory pressure
            requires it.
        exclude_self: If True, exclude the anchor itself from the candidate set.
        normalize_inputs: If True, L2-normalize embeddings before similarity.
            Pass False if the MetricHead already normalizes (avoids double work).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        max_positives_per_anchor: Optional[int] = None,
        exclude_self: bool = True,
        normalize_inputs: bool = False,
        # Backwards-compat alias (older configs used `positives_per_anchor`).
        positives_per_anchor: Optional[int] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.exclude_self = exclude_self
        self.normalize_inputs = normalize_inputs

        if positives_per_anchor is not None and max_positives_per_anchor is None:
            # Treat legacy positives_per_anchor=1 as "no cap" — i.e., default
            # to all-positives SupCon. We never want the old 1-positive mode
            # because it strictly under-uses the supervision signal.
            if positives_per_anchor != 1:
                max_positives_per_anchor = positives_per_anchor
        self.max_positives_per_anchor = max_positives_per_anchor

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
        """Supervised contrastive (SupCon) loss with all in-batch positives.

        Args:
            embeddings: (B, d) metric embeddings for the batch.
            spatial_adj: sparse (N_section, N_section) GT adjacency for the
                full section.
            batch_indices: (B,) indices of batch cells within the section.
                If None, batch == full section.

        Returns:
            loss: scalar loss (zero tensor with grad_fn if no usable anchors).
            metrics: dict with diagnostic counters.
        """
        device = embeddings.device
        B = embeddings.shape[0]

        if B < 4:
            return embeddings.sum() * 0.0, {
                "contrastive_loss": 0.0,
                "n_anchors_used": 0,
                "n_anchors_skipped": B,
                "n_anchors_clamped": 0,
                "mean_positives_per_anchor": 0.0,
            }

        pos_mask = self._build_pos_mask(spatial_adj, batch_indices, B, device)

        # Optional cap on positives per anchor (memory; usually leave uncapped).
        cap = self.max_positives_per_anchor
        if cap is not None and cap > 0:
            # For each row, keep at most `cap` positives chosen at random.
            rand = torch.rand(B, B, device=device, dtype=embeddings.dtype)
            rand = rand.masked_fill(~pos_mask, float("-inf"))
            # Get the top-`cap` indices per row; build a new mask.
            topk = min(cap, B)
            _, kept_idx = rand.topk(topk, dim=1)
            capped = torch.zeros_like(pos_mask)
            capped.scatter_(1, kept_idx, True)
            pos_mask = pos_mask & capped

        n_pos_per_anchor = pos_mask.sum(dim=1)  # (B,)
        has_pos = n_pos_per_anchor > 0
        if not has_pos.any():
            return embeddings.sum() * 0.0, {
                "contrastive_loss": 0.0,
                "n_anchors_used": 0,
                "n_anchors_skipped": int(B),
                "n_anchors_clamped": 0,
                "mean_positives_per_anchor": 0.0,
            }

        if self.normalize_inputs:
            embeddings = F.normalize(embeddings, dim=-1)

        # Scaled similarity matrix (B, B). Cosine if L2-normalized.
        sim = (embeddings @ embeddings.t()) / self.temperature

        # Denominator: log-sum-exp over all candidates excluding self.
        if self.exclude_self:
            eye = torch.eye(B, dtype=torch.bool, device=device)
            sim_denom = sim.masked_fill(eye, float("-inf"))
        else:
            sim_denom = sim
        log_denom = torch.logsumexp(sim_denom, dim=1, keepdim=True)  # (B, 1)

        # Log-softmax over candidates: P(j | i) = exp(sim_ij/tau) / sum_l exp(sim_il/tau)
        log_p = sim - log_denom  # (B, B)

        # For each anchor i, average log P(p | i) over its positives p.
        pos_mask_f = pos_mask.float()
        # sum_{p in P_i} log P(p | i)
        sum_log_p_pos = (log_p * pos_mask_f).sum(dim=1)  # (B,)
        loss_per_anchor = -sum_log_p_pos / n_pos_per_anchor.clamp(min=1).float()

        loss_per_anchor = loss_per_anchor[has_pos]

        # Defensive: drop non-finite anchors with a warning.
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
                "n_anchors_skipped": int(B - int(has_pos.sum().item())),
                "n_anchors_clamped": n_clamped,
                "mean_positives_per_anchor": 0.0,
            }

        loss = loss_per_anchor.mean()
        n_used = int(has_pos.sum().item()) - n_clamped
        mean_pos = float(n_pos_per_anchor[has_pos].float().mean().item())

        return loss, {
            "contrastive_loss": loss.item(),
            "n_anchors_used": n_used,
            "n_anchors_skipped": int(B - int(has_pos.sum().item())),
            "n_anchors_clamped": n_clamped,
            "mean_positives_per_anchor": mean_pos,
        }


# ---------------------------------------------------------------------------
# Pairwise distance regression (LUNA-style; optimizes the Spearman target)
# ---------------------------------------------------------------------------


class DistanceRegressionLoss(nn.Module):
    """LUNA-style pairwise distance preservation, via Pearson correlation.

    LUNA's coordinate generator is trained with a pairwise-distance MSE
    objective (eq. 11 of the paper):

        L = (1/m^2) * sum_{i,j} (||r_pred_i - r_pred_j||^2 - ||r_true_i - r_true_j||^2)^2

    That objective is the structural reason LUNA scores high on the Spearman
    metric — the metric is per-cell Spearman of pairwise distance rows, so a
    model that preserves all pairwise distances directly optimizes it.

    Our SupCon contrastive loss only cares about the top-k neighborhood; two
    cells at GT distance 3 and 30 are both "negatives" and get pushed away
    identically, even though their relative ordering is the whole point of
    Spearman. This module adds the structural-distance term back in.

    Implementation:
      * Pairwise squared Euclidean distance between cells in BOTH the metric
        embedding space (B, d) and the GT coordinate space (B, 2).
      * Take the upper triangle (i < j) to avoid self-pairs and double-counting.
      * Standardize both vectors (z-score) and compute Pearson correlation =
        mean of the element-wise product.
      * Loss = 1 - Pearson, so minimizing it maximizes correlation.

    Why squared distance rather than Euclidean?
      For L2-normalized embeddings, ||z_i - z_j||^2 = 2 - 2*cos(z_i, z_j),
      so this is monotonic in cosine similarity and has slightly smoother
      gradients than raw Euclidean.

    Why Pearson rather than MSE?
      Pearson is scale-invariant in both arguments. The L2-normalized
      embedding's distances are bounded in [0, 2] (so squared in [0, 4]),
      while GT coords have arbitrary scale; MSE would couple the loss to
      that scale mismatch, Pearson sidesteps it entirely.

    Args:
        use_squared: Use squared distances if True (default), Euclidean if False.
        eps: Numerical safety for the standardization step.
    """

    def __init__(self, use_squared: bool = True, eps: float = 1e-8):
        super().__init__()
        self.use_squared = use_squared
        self.eps = eps

    def forward(
        self,
        embeddings: torch.Tensor,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: (B, d) metric embeddings for the batch.
            coords: (B, 2) ground-truth 2-D coordinates for the same B cells.

        Returns:
            loss in [0, 2], metrics dict.
        """
        B = embeddings.shape[0]
        if B < 4 or coords.shape[0] != B:
            return embeddings.sum() * 0.0, {
                "distance_pearson": 0.0,
                "distance_loss": 0.0,
            }

        pred_d = torch.cdist(embeddings, embeddings)
        true_d = torch.cdist(coords, coords)
        if self.use_squared:
            pred_d = pred_d ** 2
            true_d = true_d ** 2

        # Upper triangle (i < j); avoids self-pairs and counts each pair once.
        triu = torch.triu_indices(B, B, offset=1, device=embeddings.device)
        pred_v = pred_d[triu[0], triu[1]]
        true_v = true_d[triu[0], triu[1]]

        # Standardize both vectors. Mean of pairwise products of z-scored
        # vectors equals Pearson correlation.
        pred_v = (pred_v - pred_v.mean()) / (pred_v.std() + self.eps)
        true_v = (true_v - true_v.mean()) / (true_v.std() + self.eps)

        pearson = (pred_v * true_v).mean()
        loss = 1.0 - pearson  # in [0, 2]

        return loss, {
            "distance_pearson": float(pearson.item()),
            "distance_loss": float(loss.item()),
        }


# ---------------------------------------------------------------------------
# Optional cell-class auxiliary classification head
# ---------------------------------------------------------------------------


class CellClassAuxLoss(nn.Module):
    """Optional auxiliary cell-class classifier on the encoder embedding.

    Adds a small head on top of the cell embedding and trains it with cross
    entropy against GT cell-class labels. Same-class cells often co-localize
    spatially in tissue (cortical layers, glomerular structures, etc.), so a
    class-aware encoder is a useful inductive bias for spatial-graph
    prediction.

    Default `enabled: false` in config; when enabled, weighted into the
    primary loss via `loss.cell_class_aux.weight`.
    """

    def __init__(
        self,
        cell_embed_dim: int,
        n_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list = []
        if hidden_dim > 0:
            layers += [nn.Linear(cell_embed_dim, hidden_dim), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        else:
            in_dim = cell_embed_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        cell_embed: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = self.head(cell_embed)
        loss = F.cross_entropy(logits, class_labels)
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == class_labels).float().mean().item()
        return loss, {
            "cellclass_aux_loss": loss.item(),
            "cellclass_aux_acc": acc,
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

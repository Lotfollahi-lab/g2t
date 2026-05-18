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

    Class-stratified variant (`class_stratified=True`)
    --------------------------------------------------
    The all-pairs Pearson above is dominated by between-class pair distances
    in laminar tissues (different cortical layers are far apart; within-layer
    distances are small). A model that perfectly clusters by cell type
    already scores high Pearson without learning any within-class spatial
    geometry — and that turns out to be exactly the failure mode the
    inference UMAP diagnostic surfaces.

    With `class_stratified=True`, the loss is computed *per cell class*:
    for each class with >= `min_cells_per_class` cells in the batch, we
    compute Pearson on within-class pairs, then average across classes.
    Between-class pairs no longer contribute, so the gradient is pushed
    entirely onto within-class pair ordering — which is the signal the
    SupCon term is *not* providing.

    `cell_class` must be passed to `forward()` when stratification is on.
    Cells with `class == -1` (= unknown vocabulary, see ABC / cortex loaders)
    are excluded. The unstratified metric is still reported alongside as
    `distance_pearson_global` for direct comparison to the previous regime.

    Args:
        use_squared: Use squared distances if True (default), Euclidean if False.
        eps: Numerical safety for the standardization step.
        class_stratified: If True, compute Pearson per cell class and average.
        min_cells_per_class: A class needs at least this many cells in the
            batch (so at least min_cells_per_class*(min_cells_per_class-1)/2
            pairs) to be included in the per-class average. Default 6
            (= 15 pairs minimum).
    """

    def __init__(
        self,
        use_squared: bool = True,
        eps: float = 1e-8,
        class_stratified: bool = False,
        min_cells_per_class: int = 6,
    ):
        super().__init__()
        self.use_squared = use_squared
        self.eps = eps
        self.class_stratified = class_stratified
        self.min_cells_per_class = int(min_cells_per_class)

    @staticmethod
    def _pair_pearson(
        pred_d: torch.Tensor,
        true_d: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Pearson correlation between selected pairwise-distance entries."""
        pred_v = pred_d[rows, cols]
        true_v = true_d[rows, cols]
        pred_v = (pred_v - pred_v.mean()) / (pred_v.std() + eps)
        true_v = (true_v - true_v.mean()) / (true_v.std() + eps)
        return (pred_v * true_v).mean()

    def forward(
        self,
        embeddings: torch.Tensor,
        coords: torch.Tensor,
        cell_class: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: (B, d) metric embeddings for the batch.
            coords: (B, 2) ground-truth 2-D coordinates for the same B cells.
            cell_class: (B,) long tensor of integer class ids, or None.
                Required when `class_stratified=True`. Cells with class < 0
                are excluded.

        Returns:
            loss in [0, 2], metrics dict.
        """
        B = embeddings.shape[0]
        zero_out = (embeddings.sum() * 0.0, {
            "distance_pearson": 0.0,
            "distance_loss": 0.0,
            "distance_pearson_global": 0.0,
            "n_classes_used": 0,
        })
        if B < 4 or coords.shape[0] != B:
            return zero_out

        pred_d = torch.cdist(embeddings, embeddings)
        true_d = torch.cdist(coords, coords)
        if self.use_squared:
            pred_d = pred_d ** 2
            true_d = true_d ** 2

        # Always compute the global (all-pairs) Pearson as a diagnostic,
        # even when stratifying — useful for direct comparison to the
        # previous training regime.
        triu = torch.triu_indices(B, B, offset=1, device=embeddings.device)
        pearson_global = self._pair_pearson(
            pred_d, true_d, triu[0], triu[1], self.eps,
        )

        if not self.class_stratified:
            loss = 1.0 - pearson_global
            return loss, {
                "distance_pearson": float(pearson_global.item()),
                "distance_loss": float(loss.item()),
                "distance_pearson_global": float(pearson_global.item()),
                "n_classes_used": 0,
            }

        # Class-stratified path: average Pearson within each class with
        # enough cells. Cells with class < 0 (unknown) are excluded.
        if cell_class is None:
            # Fall back to global rather than silently changing behavior.
            loss = 1.0 - pearson_global
            return loss, {
                "distance_pearson": float(pearson_global.item()),
                "distance_loss": float(loss.item()),
                "distance_pearson_global": float(pearson_global.item()),
                "n_classes_used": 0,
            }

        cell_class = cell_class.to(embeddings.device)
        valid_classes = torch.unique(cell_class[cell_class >= 0])
        per_class_pearsons: list = []
        for c in valid_classes.tolist():
            mask = cell_class == c
            n_c = int(mask.sum())
            if n_c < self.min_cells_per_class:
                continue
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            sub_triu = torch.triu_indices(
                n_c, n_c, offset=1, device=embeddings.device,
            )
            rows = idx[sub_triu[0]]
            cols = idx[sub_triu[1]]
            per_class_pearsons.append(
                self._pair_pearson(pred_d, true_d, rows, cols, self.eps)
            )

        if not per_class_pearsons:
            # Nothing usable in this batch — fall back to global.
            loss = 1.0 - pearson_global
            return loss, {
                "distance_pearson": float(pearson_global.item()),
                "distance_loss": float(loss.item()),
                "distance_pearson_global": float(pearson_global.item()),
                "n_classes_used": 0,
            }

        pearson_strat = torch.stack(per_class_pearsons).mean()
        loss = 1.0 - pearson_strat
        return loss, {
            "distance_pearson": float(pearson_strat.item()),
            "distance_loss": float(loss.item()),
            "distance_pearson_global": float(pearson_global.item()),
            "n_classes_used": len(per_class_pearsons),
        }


# ---------------------------------------------------------------------------
# Direct 2-D coordinate regression with Procrustes-aligned MSE
# ---------------------------------------------------------------------------


class CoordRegressionLoss(nn.Module):
    """Direct (x, y) regression head with Procrustes-aligned MSE.

    Adds a learned Linear(metric_dim, 2) head on top of the metric
    embedding. Each batch:
      1. Project embedding -> 2-D predicted coords.
      2. Solve for the best similarity transform (scale + rotation +
         optional reflection + translation) that maps the prediction
         onto GT coords (Kabsch-Umeyama). The transform parameters are
         computed under torch.no_grad — gradients only flow through the
         predicted coords, not through the alignment.
      3. Apply the alignment and compute MSE between the aligned
         prediction and GT, normalized by GT coord variance so the loss
         is comparable across slices with different physical scales.

    Why a separate, head-based loss when we already have pairwise-distance
    regression?

      The pairwise-distance Pearson loss is dominated by whichever spatial
      axis carries the most variance — in laminar tissue, that's the depth
      axis. The model can score Pearson ~0.8 while leaving the tangential
      axis essentially unconstrained, and *any* 2-D readout of the
      resulting embedding ends up scrambled in the tangential direction.

      This loss directly punishes that failure mode: there is no rotation
      or projection of the prediction that lets it ignore one of the two
      output dimensions. Both must be encoded as linear directions of the
      metric embedding for the loss to be small.

    The Kabsch alignment makes the loss invariant to rotation / scale /
    reflection of the embedding frame, so the model isn't penalized for
    learning coords in an arbitrary orientation. Stop-gradding R means
    the model can't "trick" the loss by exploiting how the alignment
    would compensate for its own errors — it has to produce intrinsically
    well-shaped coords.

    Args:
        embed_dim: dimensionality of the metric embedding fed to the head.
            Should match `model.metric_head.embed_dim`.
        allow_reflection: if True, the Kabsch transform may flip chirality;
            useful when the embedding can come out mirrored relative to GT
            without that being a real model error. Default False (rotations
            only) — cortex has a definite handedness.
        normalize_by_var: divide the MSE by mean GT coord variance so the
            loss sits in roughly [0, 1] like the other losses. Default True.
    """

    def __init__(
        self,
        embed_dim: int,
        allow_reflection: bool = False,
        normalize_by_var: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.head = nn.Linear(embed_dim, 2)
        self.allow_reflection = bool(allow_reflection)
        self.normalize_by_var = bool(normalize_by_var)
        self.eps = float(eps)

    def forward(
        self,
        embeddings: torch.Tensor,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: (B, embed_dim) metric embeddings.
            coords: (B, 2) GT 2-D coordinates.

        Returns:
            loss (scalar) + metrics dict.
        """
        B = embeddings.shape[0]
        zero_out = (
            embeddings.sum() * 0.0,
            {
                "coord_loss": 0.0,
                "coord_mse": 0.0,
                "coord_normalized_mse": 0.0,
                "coord_scale": 0.0,
            },
        )
        if B < 3 or coords.shape[0] != B:
            return zero_out

        pred = self.head(embeddings)  # (B, 2)

        # Procrustes / Kabsch-Umeyama similarity transform. Computed under
        # no_grad so gradients flow through pred only — the model can't
        # exploit the alignment to lower the loss without producing
        # intrinsically well-shaped predictions.
        with torch.no_grad():
            mu_p = pred.mean(dim=0)
            mu_c = coords.mean(dim=0)
            pc = pred - mu_p
            cc = coords - mu_c
            var_p = (pc ** 2).sum() / B  # mean squared norm of centered pred
            if float(var_p.item()) < self.eps:
                # Predictions collapsed to a point; fall back to plain MSE
                # so gradients can still escape this regime.
                s = torch.ones((), device=pred.device, dtype=pred.dtype)
                R = torch.eye(2, device=pred.device, dtype=pred.dtype)
                t = mu_c - mu_p
                degenerate = True
            else:
                cov = (cc.T @ pc) / B  # (2, 2) target.T @ source
                U, S_vals, Vt = torch.linalg.svd(cov)
                if self.allow_reflection:
                    d_diag = torch.ones(2, device=pred.device, dtype=pred.dtype)
                else:
                    det_uv = torch.linalg.det(U @ Vt)
                    d_diag = torch.stack(
                        [torch.ones_like(det_uv), det_uv.sign()]
                    ).to(pred.dtype)
                R = U @ torch.diag(d_diag) @ Vt
                s = (S_vals * d_diag).sum() / (var_p + self.eps)
                t = mu_c - s * (R @ mu_p)
                degenerate = False

        # Apply alignment WITH grad on pred.
        aligned = s * (pred @ R.T) + t
        mse = ((aligned - coords) ** 2).mean()

        if self.normalize_by_var:
            with torch.no_grad():
                coord_var = ((coords - coords.mean(dim=0)) ** 2).mean()
                coord_var = coord_var + self.eps
            normalized = mse / coord_var
            loss = normalized
            normalized_val = float(normalized.item())
        else:
            loss = mse
            normalized_val = float("nan")

        return loss, {
            "coord_loss": float(loss.item()),
            "coord_mse": float(mse.item()),
            "coord_normalized_mse": normalized_val,
            "coord_scale": float(s.item()),
            "coord_degenerate": float(degenerate),
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

"""k-NN graph generation output head — scGG fundamental method #4.

Motivation
----------
LUNA's output is coordinates X ∈ R^{N×2}. But for most downstream
spatial-transcriptomics analyses (niche enrichment, cell-cell
communication, lineage inference), the actual quantity of interest is
the spatial neighbourhood GRAPH — "which cells are next to which" —
not the absolute coordinates.

This head reframes the problem: predict, for each cell, its k spatial
nearest neighbours directly. The output is a graph (or equivalently a
ranking over candidate edges), not a coordinate tensor. Coordinates
are derived post-hoc by Laplacian eigenmaps (spectral graph embedding)
so downstream code that reads ``pred.positions`` keeps working.

Properties
----------
- Graph output is invariant to rigid motion AND global scale by
  construction.
- The learning target is RELATIVE ("rank these 20 candidate neighbours
  for cell i by spatial proximity") rather than ABSOLUTE ("place cell i
  at (x, y)"). Flatter loss landscape on small datasets — fewer
  symmetry-related minima to confuse the optimiser.
- Directly aligned with downstream biology.

Loss
----
Contrastive BCE with negative sampling: for each cell i, the k true
spatial nearest neighbours are positives; ``n_negatives`` random
non-neighbours are negatives. The edge score s_ij = -‖h_i − h_j‖²
(more-negative-distance → higher score → likelier-edge). The model
has to push positives' scores up and negatives' down.

Inference
---------
At each forward (or once at the end of the sampling chain), take the
top-k positive-scoring cells for each cell to form the predicted
graph. For coordinate recovery, run Laplacian eigenmaps on the
symmetrised adjacency: eigvecs 1 and 2 of (D − A) give the 2D
embedding. Procrustes-align to the input x_t frame for FM/DDPM
trajectory consistency (same trick as EDM).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from utils.data.dataholder import DataHolder


# ---------------------------------------------------------------------------
# Spectral graph embedding (Laplacian eigenmaps) for inference-time
# coordinate recovery
# ---------------------------------------------------------------------------


def _laplacian_eigenmaps_2d(A: torch.Tensor) -> torch.Tensor:
    """Compute 2D Laplacian-eigenmap embedding from a symmetric
    adjacency matrix A (n × n, nonneg).

    The unnormalised graph Laplacian L = D − A has eigenvalue 0 with
    eigenvector ∝ 1 (constant); the next two smallest eigenvectors
    give the 2D embedding (Belkin & Niyogi 2003).
    """
    n = A.shape[0]
    device = A.device
    dtype = A.dtype
    # Symmetrise (in case A came in non-symmetric — e.g. mutual-kNN
    # asymmetry).
    A_sym = 0.5 * (A + A.T)
    # Row sums = node degrees.
    deg = A_sym.sum(dim=1)
    L = torch.diag(deg) - A_sym
    # Symmetrise L (numerical safety).
    L = 0.5 * (L + L.T)
    # Eigendecomposition. Smallest 3 eigenvalues: 0 (constant), then 2
    # informative ones. Take eigenvectors 1 and 2 (0-indexed) skipping
    # the constant.
    evals, evecs = torch.linalg.eigh(L)
    # evals ascending. Skip evec[:, 0] (constant), take 1 and 2.
    if n < 3:
        return torch.zeros(n, 2, device=device, dtype=dtype)
    return evecs[:, 1:3]                              # (n, 2)


def _procrustes_align(x_src: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Orthogonal Procrustes (rotation+reflection). See edm_head.py for
    the same routine — duplicated here to keep this module self-
    contained (knn_graph and edm shouldn't depend on each other).

    R = U @ Vh from SVD(x_src.T @ x_ref) — Schönemann 1966.
    """
    M = x_src.T @ x_ref
    U, _S, Vh = torch.linalg.svd(M)
    R = U @ Vh
    return x_src @ R


# ---------------------------------------------------------------------------
# Output wrapper
# ---------------------------------------------------------------------------


class KNNGraphOutputWrapper(nn.Module):
    """Wraps an inner backbone with a k-NN graph output head.

    Per forward:
      1. Inner backbone runs as usual; ``pred.node_features`` and
         ``pred.positions`` are produced.
      2. Projector maps per-cell features to embeddings h ∈ R^k.
      3. Edge logits L_ij = -‖h_i - h_j‖² (closer-in-embedding → likelier
         edge). Stashed on ``pred.knn_logits`` for the loss.
      4. At inference (or any time `spectral_layout=True`): build a
         soft adjacency from the logits, run Laplacian eigenmaps to
         get a 2D layout, Procrustes-align to the input frame, replace
         ``pred.positions``.

    The contrastive loss (negative sampling) is built into
    LossFunction._compute_knn_graph_loss, which reads ``pred.knn_logits``
    and the true positions to derive positive/negative edges.
    """

    def __init__(
        self,
        inner_model: nn.Module,
        inner_out_dim: int,
        embed_dim: int = 16,
        spectral_layout: bool = True,
        k_for_layout: int = 10,
        temperature: float = 0.1,
        spectral_layout_gradient: bool = False,
    ) -> None:
        super().__init__()
        self.inner_model = inner_model
        self.embed_dim = int(embed_dim)
        self.spectral_layout = bool(spectral_layout)
        # k for the inference-time top-k → adjacency step. The loss
        # has its own k (in LossFunction config). Default tied to
        # ``cfg.model.knn_graph.k``.
        self.k_for_layout = int(k_for_layout)
        # Contrastive temperature. See cfg.model.knn_graph.temperature
        # for the rationale and the NaN-divergence story.
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError(
                f"knn_graph.temperature must be > 0; got {temperature}"
            )
        # Mirror of ``model.edm.mds_align_gradient`` — same trade-off,
        # same fix. When False (legacy default), the spectral-layout
        # ``pred.positions`` is detached and any position-space
        # auxiliary loss (shape_matching, sinkhorn, knn_rank,
        # persistent_homology, pairwise_distance_mse-on-positions)
        # has ZERO gradient through this path. When True, gradient
        # flows AND a torch.nan_to_num backward hook contains the
        # eigh-degenerate NaN risk to the affected slice. See
        # configs/model/default.yaml::knn_graph.spectral_layout_gradient
        # for the user-facing explanation.
        self.spectral_layout_gradient = bool(spectral_layout_gradient)

        in_dim = int(inner_out_dim) + 2
        hidden = max(32, 2 * self.embed_dim)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
        )

        # Propagate the c2f teacher-forcing marker through the wrapper
        # chain — see the matching comment in edm_head.py for why.
        if (
            hasattr(inner_model, "_c2f_uses_true_positions")
            or "CoarseToFineWrapper" in type(inner_model).__name__
        ):
            self._c2f_uses_true_positions = True

    def forward(self, data: DataHolder, **kwargs) -> DataHolder:
        pred = self.inner_model(data, **kwargs)

        feat_in = torch.cat([pred.node_features, pred.positions], dim=-1)
        h = self.projector(feat_in)                    # (B, N, k)
        mask = data.node_mask.to(h.dtype).unsqueeze(-1)
        # L2-normalise embeddings BEFORE masking, so unit-norm holds for
        # real cells. Padding cells then get zeroed by `* mask` and
        # contribute zero to pairwise distances regardless of where
        # their pre-mask gradients point. clamp_min(eps) prevents
        # division-by-zero for the (rare) degenerate case of a zero
        # projector output.
        h_norm = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        h_norm = h_norm * mask
        # Backward-compat: expose the (unnormalised) projector output too.
        pred.knn_h = h_norm

        # Edge logits = temperature-scaled NEGATIVE squared distance in
        # the L2-normalised embedding space. For unit-norm h:
        #     ‖h_i − h_j‖² = 2 − 2·cos(θ_ij) ∈ [0, 4]
        # so logits = -‖h_i − h_j‖² / T ∈ [-4/T, 0]. BCE-with-logits on
        # these is well-bounded; gradient on h is bounded; training is
        # stable. See the NaN-divergence comment in
        # configs/model/default.yaml::knn_graph.temperature.
        diff = h_norm.unsqueeze(2) - h_norm.unsqueeze(1)   # (B, N, N, k)
        d_sq = (diff * diff).sum(dim=-1)                    # (B, N, N)
        logits = -d_sq / self.temperature                   # (B, N, N)

        # Mask padding rows/cols. We set padding entries to a large
        # negative number so the top-k step never picks them.
        m1 = data.node_mask                            # (B, N)
        m2 = m1.unsqueeze(2) * m1.unsqueeze(1)         # (B, N, N) bool
        logits = logits.masked_fill(~m2, float("-inf"))

        pred.knn_logits = logits

        if self.spectral_layout:
            if self.spectral_layout_gradient:
                # GRADIENT-CARRYING spectral-layout path. Same trade-off
                # as edm_head.py::mds_align_gradient: the Laplacian
                # eigenmaps eigh has 1/(λ_i − λ_j) backward terms
                # that diverge at near-degenerate eigenvalues. We
                # install a torch.nan_to_num backward hook on the
                # spectral-layout output below, so a degenerate slice
                # at most loses its own auxiliary-loss gradient
                # contribution rather than poisoning the entire
                # backward graph. The common case (well-separated
                # eigvals) flows through cleanly. Enable this when
                # combining knn_graph with shape_matching / sinkhorn
                # / knn_rank / persistent_homology / pairwise_distance_mse;
                # otherwise those losses contribute zero gradient.
                new_pos = self._spectral_layout(
                    logits, pred.positions, data.node_mask,
                )
                new_pos = new_pos * data.node_mask.unsqueeze(-1).to(new_pos.dtype)
                if new_pos.requires_grad:
                    def _nan_to_zero(grad):
                        return torch.nan_to_num(
                            grad, nan=0.0, posinf=0.0, neginf=0.0,
                        )
                    new_pos.register_hook(_nan_to_zero)
            else:
                # Legacy detached path. The k-NN loss reads
                # ``pred.knn_logits`` directly (not pred.positions),
                # so this path is fine in isolation. But under any
                # auxiliary loss that reads pred.positions, the
                # gradient through this path is IDENTICALLY ZERO —
                # those losses still show non-zero values on wandb
                # (positions improve as logits improve), but the
                # model is not trained by them. The fail-loud guard
                # in ``LossFunction.__init__`` catches this combo
                # at config-load time so it can't silently happen.
                with torch.no_grad():
                    new_pos = self._spectral_layout(
                        logits.detach(), pred.positions.detach(), data.node_mask,
                    )
                new_pos = new_pos * data.node_mask.unsqueeze(-1).to(new_pos.dtype)
            pred.positions = new_pos

        return pred

    # ------------------------------------------------------------------
    # Spectral graph embedding per slice
    # ------------------------------------------------------------------
    def _spectral_layout(
        self,
        logits: torch.Tensor,
        x_ref: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build a sparse adjacency from top-k logits per cell, run
        Laplacian eigenmaps, Procrustes-align. Per-slice loop because
        n_valid varies across the batch.
        """
        B, N, _ = x_ref.shape
        aligned_list = []
        for b in range(B):
            m = node_mask[b]
            valid_idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
            n_valid = int(valid_idx.numel())
            if n_valid < 4:
                aligned_list.append(x_ref[b])
                continue
            L_v = logits[b].index_select(0, valid_idx).index_select(1, valid_idx)
            # Top-k per row, but skip self (diagonal = 0 since h_i−h_i=0).
            # Set diagonal to -inf so it's never picked.
            diag_mask = torch.eye(n_valid, device=L_v.device, dtype=torch.bool)
            L_v_topk = L_v.masked_fill(diag_mask, float("-inf"))
            k_eff = min(self.k_for_layout, n_valid - 1)
            # Soft adjacency: weight = softmax over top-k logits per row.
            # We use SOFTMAX rather than hard top-k so the layout step
            # is smooth in the logits (matters if we ever want to
            # backprop through it — currently only used at inference).
            top_vals, top_idx = torch.topk(L_v_topk, k=k_eff, dim=1)
            top_weights = torch.softmax(top_vals, dim=1)
            # Scatter into (n_valid, n_valid) adjacency.
            A = torch.zeros(n_valid, n_valid, device=L_v.device, dtype=L_v.dtype)
            A = A.scatter(1, top_idx, top_weights)
            # Symmetrise (mutual graph).
            try:
                x_spec = _laplacian_eigenmaps_2d(A)        # (n_valid, 2)
            except Exception:
                aligned_list.append(x_ref[b])
                continue
            # Rescale eigenmaps to match x_ref's scale (eigenmaps are
            # unit-norm-ish; x_ref is in normalised coord scale).
            ref_scale = x_ref[b].index_select(0, valid_idx).abs().mean().clamp_min(1e-6)
            spec_scale = x_spec.abs().mean().clamp_min(1e-6)
            x_spec = x_spec * (ref_scale / spec_scale)
            x_ref_v = x_ref[b].index_select(0, valid_idx)
            x_ref_c = x_ref_v - x_ref_v.mean(dim=0, keepdim=True)
            x_spec_c = x_spec - x_spec.mean(dim=0, keepdim=True)
            # framework=regression zeroes positions; Procrustes on
            # zero ref is degenerate. Skip alignment in that case —
            # the raw spectral frame is arbitrary but consistent, and
            # the downstream metrics (Spearman / pairwise MSE) are
            # frame-invariant anyway. See the matching guard in
            # edm_head.py for the full rationale.
            ref_norm = x_ref_c.abs().mean().item()
            if ref_norm < 1e-8:
                x_aligned = x_spec_c
            else:
                try:
                    x_aligned = _procrustes_align(x_spec_c, x_ref_c)
                except Exception:
                    aligned_list.append(x_ref[b])
                    continue
            padded = torch.zeros(N, 2, device=x_ref.device, dtype=x_ref.dtype)
            padded = padded.index_copy(0, valid_idx, x_aligned)
            aligned_list.append(padded)
        return torch.stack(aligned_list, dim=0)

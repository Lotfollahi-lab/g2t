"""Canonical-frame mapping for point clouds — the shared primitive for
the gauge-fixed flow-matching variants (Option 1: canonicalized
coordinate flow; Option 2: gauge-fixed relational / h-space flow).

Motivation
----------
A cell cloud's absolute coordinates are defined only up to a rigid
motion + reflection (SE(2) / O(2)): the per-cell Spearman metric and the
EDM distance loss are both invariant to it. Coordinate flow matching,
however, regresses toward ONE specific rotated/reflected copy of the
target, so the model must waste capacity being equivariant — which is
why the pipeline carries train-time rotation/reflection AUGMENTATION
(stochastic frame averaging) plus Procrustes alignment.

``canonicalize_cloud`` replaces that stochastic averaging with a
DETERMINISTIC frame fix: each slice is mapped to a single canonical
representative (mean-centred, PCA-rotated so the principal axis is x,
sign-fixed by skewness so chirality/reflection is resolved). The flow
then targets a unique cloud instead of an orbit.

Applied to GROUND-TRUTH targets only, so it runs under ``no_grad`` —
there is no eigh-backward stability concern (unlike the MDS path in the
EDM head). Degenerate slices (n < 3, isotropic covariance, near-zero
skewness) fall back gracefully and never produce NaN; the result is
always deterministic for fixed input, so the canonical frame of a slice
is identical across epochs (no caching needed).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def canonicalize_cloud(
    positions: torch.Tensor,
    node_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Map each slice's 2D point cloud to its O(2)/translation canonical
    frame.

    Args:
        positions: (B, N, D) coordinates (D typically 2).
        node_mask: (B, N) bool, True at real cells.
        eps: numerical floor (unused in the linear-algebra but kept for
            signature symmetry / future scale-normalisation).

    Returns:
        (B, N, D) canonicalised coordinates. Padding rows are zero.

    The transform per slice is an isometry of the real cells, so it does
    NOT change any pairwise distance — only the global frame. It is
    therefore a no-op for distance-based losses (edm_distance_mse,
    sparse_local_distance) and only affects the flow's interpolant frame.
    """
    B, N, D = positions.shape
    out = torch.zeros_like(positions)
    for b in range(B):
        m = node_mask[b]
        idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
        n = int(idx.numel())
        if n == 0:
            continue
        x = positions[b].index_select(0, idx)              # (n, D)
        x = x - x.mean(dim=0, keepdim=True)                # translation gauge
        if n < 3:
            # Too few points to define a stable principal axis; centring
            # is the most we can canonicalise without ambiguity.
            out[b].index_copy_(0, idx, x.to(out.dtype))
            continue
        # PCA via the (D, D) covariance eigendecomposition. D is tiny
        # (2), so this is trivially cheap; eigh gives ASCENDING
        # eigenvalues with orthonormal eigenvectors as columns.
        cov = (x.t() @ x) / float(n)                       # (D, D)
        evals, evecs = torch.linalg.eigh(cov)              # (D,), (D, D)
        # Order axes by DESCENDING variance (principal axis first).
        order = torch.argsort(evals, descending=True)
        R = evecs.index_select(1, order)                   # (D, D)
        x_rot = x @ R                                      # (n, D) rotate
        # Resolve the remaining sign/reflection gauge deterministically:
        # flip each axis so its third moment (skewness) is >= 0. Two
        # mirror-image clouds map to the SAME canonical frame, matching
        # the reflection-invariance the pipeline already assumes. For a
        # (near-)symmetric axis skewness ~ 0 and the sign is arbitrary
        # but DETERMINISTIC for fixed data (same slice -> same frame
        # every epoch).
        skew = (x_rot ** 3).sum(dim=0)                     # (D,)
        signs = torch.where(
            skew < 0.0,
            torch.full_like(skew, -1.0),
            torch.ones_like(skew),
        )
        x_rot = x_rot * signs.unsqueeze(0)
        out[b].index_copy_(0, idx, x_rot.to(out.dtype))
    return out

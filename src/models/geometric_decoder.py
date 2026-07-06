"""Differentiable geometric decoders for the EDM head — refine the 2D
layout beyond classical (spectral) MDS.

Classical MDS takes the top-2 eigenvectors of the double-centred distance
matrix; that is exact ONLY when the predicted distances are perfectly
2D-Euclidean-embeddable. Predicted distances rarely are, so the top-2
truncation silently discards the residual (the ``SCGG_LOG_MDS_VAR``
diagnostic in edm_head measures exactly this — ``top2_frac`` < 1).

SMACOF ("Scaling by MAjorizing a COmplicated Function") instead directly
minimises the raw stress

    sigma(X) = sum_{i<j} ( ||x_i - x_j|| - delta_ij )^2

by iterated Guttman transforms, so it yields a lower-stress 2D layout
than classical MDS whenever the distances are not cleanly 2D-embeddable.
Each Guttman step is plain differentiable linear algebra (NO
eigendecomposition), so this is both (a) a drop-in better decoder at
inference and (b) safe to backprop through for END-TO-END geometry
training (v2) — unlike the ``eigh`` in classical MDS, whose backward has
1/(lambda_i - lambda_j) blow-ups.

v2 (gradient-carrying) design:
  * The classical-MDS init is passed in DETACHED (a warm-start only), so
    no eigh/lobpcg backward is ever exercised.
  * The trainable signal flows through the target distances ``delta`` used
    inside the Guttman steps -> back to the predicted distance matrix ->
    the embedding -> the backbone.
  * ``grad_safe=True`` uses a floored pairwise distance sqrt(.^2 + eps^2)
    (differentiable everywhere) instead of ``torch.cdist`` (whose backward
    NaNs on the zero diagonal).
  * ``n_grad_iter`` runs only the LAST k iterations with gradient (the
    earlier ones are a no_grad warm-up), bounding backward memory to k
    Guttman steps instead of the full iteration count.
"""
from __future__ import annotations

import torch


def _pairwise_dist(x: torch.Tensor, eps: float, grad_safe: bool) -> torch.Tensor:
    """(n, 2) -> (n, n) Euclidean distances. ``grad_safe`` floors the sqrt
    argument so the backward is finite everywhere (cdist NaNs at d=0)."""
    if grad_safe:
        diff = x.unsqueeze(-2) - x.unsqueeze(-3)          # (n, n, 2)
        return torch.sqrt((diff * diff).sum(-1) + eps * eps)
    return torch.cdist(x, x)


def _guttman_step(delta: torch.Tensor, x: torch.Tensor, eps: float,
                  grad_safe: bool) -> torch.Tensor:
    """One SMACOF Guttman transform (unit weights):
      B_ij = -delta_ij / d_ij (off-diag, 0 where d~0); B_ii = row-sum;
      x <- (1/n) B x, recentred.
    """
    n = x.shape[0]
    d = _pairwise_dist(x, eps, grad_safe)
    inv = torch.where(d > eps, 1.0 / d, torch.zeros_like(d))
    ratio = delta * inv
    ratio = ratio - torch.diag_embed(torch.diagonal(ratio))   # zero diagonal
    B = -ratio + torch.diag_embed(ratio.sum(dim=1))
    x = (B @ x) / float(n)
    return x - x.mean(dim=0, keepdim=True)


def smacof_refine(
    delta: torch.Tensor,
    x_init: torch.Tensor,
    n_iter: int = 30,
    n_grad_iter: int = 0,
    eps: float = 1e-8,
    grad_safe: bool = False,
) -> torch.Tensor:
    """Single-slice SMACOF (unit weights) stress minimisation.

    Args:
        delta:  (n, n) target distances (NOT squared).
        x_init: (n, 2) initial config (e.g. detached classical-MDS output).
        n_iter: total Guttman iterations.
        n_grad_iter: how many of the LAST iterations carry gradient
            (0 = fully detached refinement — inference / v1). The earlier
            ``n_iter - n_grad_iter`` iterations run under no_grad, so
            backward memory is bounded to ``n_grad_iter`` steps.
        eps: floor for the 1/d division (and the grad-safe sqrt).
        grad_safe: use the floored pairwise distance (see _pairwise_dist);
            set True whenever the grad iterations are active.

    Returns (n, 2). The true target config is a fixed point (unit test),
    and stress is non-increasing across iterations (majorization).
    """
    n = int(x_init.shape[0])
    if n < 3 or int(n_iter) <= 0:
        return x_init
    delta = delta.clamp_min(0.0)
    n_grad = max(0, min(int(n_iter), int(n_grad_iter)))
    n_detach = int(n_iter) - n_grad

    x = x_init
    if n_detach > 0:
        with torch.no_grad():
            for _ in range(n_detach):
                x = _guttman_step(delta, x, eps, grad_safe=False)
        x = x.detach()
    for _ in range(n_grad):
        x = _guttman_step(delta, x, eps, grad_safe=grad_safe)
    return x


def stress(delta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Raw MDS stress sum_{i<j} (||x_i - x_j|| - delta_ij)^2 (logging/tests)."""
    d = torch.cdist(x, x)
    n = x.shape[0]
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
    return ((d - delta)[triu] ** 2).sum()

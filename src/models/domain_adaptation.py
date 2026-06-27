"""Domain-adaptation primitives for cross-platform G2T training.

Replaces the external Harmony integration step with an END-TO-END,
in-model domain-invariance objective: instead of pre-integrating the
joint MERFISH+scRNA expression matrix with Harmony and feeding the model
a coarse, both-sign 600-D latent, we train on RAW genes and make the
per-cell gene embedding platform-invariant *jointly* with the geometry
objective. Invariance is then shaped to preserve the spatial-predictive
signal, not generic integration.

Setting (transductive UDA): SOURCE = MERFISH (labelled — has true
coords, drives the geometry/FM losses), TARGET = scRNA-seq (unlabelled —
coords masked, contributes only the domain objective). This is the same
data assumption Harmony already makes (it sees the target at integration
time); we just move it end-to-end.

Two complementary alignment objectives (compose; either can be off):
  * ADVERSARIAL (DANN): a gradient-reversal layer + domain discriminator.
    The discriminator learns to tell source from target; the reversed
    gradient pushes the encoder toward features it CANNOT classify ->
    platform-invariant. ``lambd`` (reversal strength) is usually warmed
    up from 0 to avoid early instability.
  * CORAL / MMD: directly align the source/target feature *distributions*
    (2nd-order CORAL, or kernel MMD). Stable, no adversary; good on its
    own and a strong stabiliser alongside DANN.

The known DANN failure mode is that GLOBAL invariance also erases the
biology the geometry head needs ("aligns away cell type"). The
``class_conditional_coral`` helper aligns platforms WITHIN cell-class so
cross-type biology is preserved — the recommended novelty for this task.

This module is intentionally self-contained (only torch) and operates on
generic ``(B, N, D)`` embeddings + a ``node_mask`` + per-cell domain
labels, so it can attach to whatever per-cell gene embedding the backbone
exposes (e.g. ``edm_h``). The numeric core (CORAL/MMD) is mirrored by a
pure-numpy test for off-cluster validation.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (Ganin & Lempitsky, 2015)
# ---------------------------------------------------------------------------
class _GradReverse(torch.autograd.Function):
    """Identity in the forward pass; negates + scales the gradient in the
    backward pass. Placed between the encoder and the domain discriminator
    so minimising the discriminator's loss MAXIMISES domain confusion in
    the encoder."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # reverse + scale; second output (for ``lambd``) is None
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


def grl_lambda(step: int, warmup_steps: int, max_lambda: float = 1.0) -> float:
    """DANN's schedule: ramp the reversal strength 0 -> max_lambda over
    ``warmup_steps`` via the standard 2/(1+exp(-10p))-1 curve (p in [0,1]).
    warmup_steps<=0 -> constant max_lambda."""
    if warmup_steps <= 0:
        return float(max_lambda)
    import math
    p = min(1.0, max(0.0, float(step) / float(warmup_steps)))
    return float(max_lambda) * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ---------------------------------------------------------------------------
# Domain discriminator
# ---------------------------------------------------------------------------
class DomainDiscriminator(nn.Module):
    """Per-cell MLP: embedding -> domain logit(s). Binary (source/target)
    by default (single logit + BCE)."""

    def __init__(self, in_dim: int, hidden: int = 128, n_domains: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.n_domains = int(n_domains)
        out = 1 if self.n_domains == 2 else self.n_domains
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def domain_adversarial_loss(
    embed: torch.Tensor,
    domain_label: torch.Tensor,
    node_mask: torch.Tensor,
    discriminator: DomainDiscriminator,
    lambd: float = 1.0,
) -> torch.Tensor:
    """Masked domain-classification loss through a gradient-reversal layer.

    Args:
        embed: (B, N, D) per-cell embeddings (the rep to make invariant).
        domain_label: (B,) or (B, N) in {0,1} (0=source, 1=target).
        node_mask: (B, N) bool — real cells.
        discriminator: maps (..., D) -> (..., 1) logit (binary).
        lambd: gradient-reversal strength (the encoder sees -lambd * grad).

    Returns scalar mean BCE over real cells. Minimising it trains the
    discriminator; the reversed gradient makes the encoder invariant.
    """
    z = grad_reverse(embed, lambd)
    logits = discriminator(z).squeeze(-1)             # (B, N)
    tgt = domain_label
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(1).expand(logits.shape)
    tgt = tgt.to(logits.dtype)
    m = node_mask.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    return (bce * m).sum() / (m.sum() + 1e-8)


# ---------------------------------------------------------------------------
# Distribution-alignment losses (CORAL / MMD) on flattened, masked embeds
# ---------------------------------------------------------------------------
def _covariance(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Sample covariance of (M, D) -> (D, D) (unbiased, mean-centred)."""
    m = z.shape[0]
    z = z - z.mean(dim=0, keepdim=True)
    denom = max(1, m - 1)
    cov = (z.t() @ z) / denom
    return cov + eps * torch.eye(z.shape[1], device=z.device, dtype=z.dtype)


def coral_loss(zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """Deep CORAL: squared Frobenius distance between the source/target
    feature covariances, normalised by 4·D² (Sun & Saenko, 2016).
    ``zs`` (Ns, D), ``zt`` (Nt, D) are masked-flattened embeddings."""
    if zs.shape[0] < 2 or zt.shape[0] < 2:
        return zs.new_zeros(())
    d = zs.shape[1]
    cs = _covariance(zs)
    ct = _covariance(zt)
    return ((cs - ct) ** 2).sum() / (4.0 * d * d)


def mmd_loss(
    zs: torch.Tensor, zt: torch.Tensor,
    sigmas: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0),
) -> torch.Tensor:
    """Multi-bandwidth RBF MMD² between source/target embeddings:
    E[k(s,s')] + E[k(t,t')] - 2 E[k(s,t)], k = Σ_σ exp(-‖·‖²/(2σ²))."""
    if zs.shape[0] < 2 or zt.shape[0] < 2:
        return zs.new_zeros(())

    def _k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(a, b) ** 2
        out = torch.zeros_like(d2)
        for s in sigmas:
            out = out + torch.exp(-d2 / (2.0 * float(s) ** 2))
        return out / float(len(sigmas))

    return _k(zs, zs).mean() + _k(zt, zt).mean() - 2.0 * _k(zs, zt).mean()


def _flatten_by_domain(
    embed: torch.Tensor, domain_label: torch.Tensor, node_mask: torch.Tensor,
):
    """(B,N,D) + domain + mask -> (source (Ns,D), target (Nt,D)) flattened
    over real cells only."""
    if domain_label.dim() == 1:
        domain_label = domain_label.unsqueeze(1).expand(node_mask.shape)
    m = node_mask.bool()
    flat = embed[m]                                   # (M, D)
    dom = domain_label[m]                             # (M,)
    return flat[dom == 0], flat[dom == 1]


def alignment_loss(
    embed: torch.Tensor,
    domain_label: torch.Tensor,
    node_mask: torch.Tensor,
    mode: str = "coral",
) -> torch.Tensor:
    """Distribution-alignment loss (``coral`` | ``mmd``) between the
    source and target embeddings in a batch."""
    zs, zt = _flatten_by_domain(embed, domain_label, node_mask)
    if mode == "coral":
        return coral_loss(zs, zt)
    if mode == "mmd":
        return mmd_loss(zs, zt)
    raise ValueError(f"alignment mode must be 'coral' or 'mmd'; got {mode!r}")


def class_conditional_coral(
    embed: torch.Tensor,
    domain_label: torch.Tensor,
    class_label: torch.Tensor,
    node_mask: torch.Tensor,
    min_per_class: int = 2,
) -> torch.Tensor:
    """Align source/target covariances WITHIN each cell-class, averaged
    over classes. Preserves cross-class (biological) variation while
    removing the platform shift — the recommended anti-"aligns-away-
    biology" variant. Classes lacking ≥``min_per_class`` cells on BOTH
    sides are skipped."""
    if domain_label.dim() == 1:
        domain_label = domain_label.unsqueeze(1).expand(node_mask.shape)
    if class_label.dim() == 1:
        class_label = class_label.unsqueeze(1).expand(node_mask.shape)
    m = node_mask.bool()
    flat = embed[m]
    dom = domain_label[m]
    cls = class_label[m]
    terms = []
    for c in torch.unique(cls):
        sc = flat[(cls == c) & (dom == 0)]
        tc = flat[(cls == c) & (dom == 1)]
        if sc.shape[0] >= min_per_class and tc.shape[0] >= min_per_class:
            terms.append(coral_loss(sc, tc))
    if not terms:
        return embed.new_zeros(())
    return torch.stack(terms).mean()

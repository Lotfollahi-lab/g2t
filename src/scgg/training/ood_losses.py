"""
Optional cross-modality OOD loss components.

These are NOT enabled by default. They are scaffolded here so the trainer can
flip them on once the OOD gap between ST and scRNA has been quantified with
the bare contrastive objective.

We deliberately do NOT rely on PASTE-style cross-section alignment: the user
has matched adjacent ST + scRNA sections, but cells are not paired (sequencing
is destructive). The losses below assume only section-level correspondence
(section s in ST <-> section s' in scRNA), never cell-level pairing.

Components:

* CrossModalityContrastiveLoss
    InfoNCE over cells from paired-section ST and scRNA mini-batches. For an
    ST anchor in section s, positives are scRNA cells in the matched section
    s' (modeling: "similar tissue context"), negatives are scRNA cells from
    other sections. Pulls per-cell embeddings closer across modalities at
    the section-pair level without claiming exact cell correspondences.

* SectionEmbeddingConsistencyLoss
    L2 (or cosine) penalty between section embeddings of paired adjacent
    ST/scRNA sections. Directly aligns the DeepSets context vector.

* DomainAdversarialLoss
    Gradient-reversal classifier that tries to predict modality from the
    cell embedding; the encoder is penalized for being classifiable. Cheap
    domain-invariance regularizer.

All three are inert unless the trainer explicitly invokes them with the
necessary tensors. When disabled in config, the trainer never even
instantiates them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# Cross-modality contrastive
# ---------------------------------------------------------------------------


class CrossModalityContrastiveLoss(nn.Module):
    """Pull ST cells and scRNA cells from matched-adjacent sections together.

    Cells are NOT paired across modalities — only sections are. For each ST
    anchor cell with section id s, positives are scRNA cells with the matched
    section id s' (membership-level positives), and negatives are scRNA cells
    from any *non*-matched section. We use a soft section-membership target,
    not a hard cell pairing.

    This is intentionally weak supervision and is meant to act as a
    distributional regularizer, not a precise cell-cell teacher.

    Args:
        temperature: InfoNCE temperature.
        normalize_inputs: L2-normalize before similarity if True.
        symmetric: If True, also compute the reverse direction
            (scRNA anchors -> ST positives) and average. Standard for
            cross-modality contrastive (e.g., CLIP).
    """

    def __init__(
        self,
        temperature: float = 0.1,
        normalize_inputs: bool = False,
        symmetric: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize_inputs = normalize_inputs
        self.symmetric = symmetric

    def _directional(
        self,
        anchors: torch.Tensor,            # (A, d)
        candidates: torch.Tensor,         # (C, d)
        anchor_section_ids: torch.Tensor, # (A,)
        candidate_section_match: torch.Tensor,  # (C,) -- the *paired* section id
    ) -> torch.Tensor:
        """One-directional cross-modality InfoNCE.

        For each anchor with paired-section id s, a candidate c is a positive
        iff `candidate_section_match[c] == s` (i.e., that candidate sits in
        the section that is paired with the anchor's section).
        """
        if self.normalize_inputs:
            anchors = F.normalize(anchors, dim=-1)
            candidates = F.normalize(candidates, dim=-1)

        sim = (anchors @ candidates.t()) / self.temperature  # (A, C)
        # positives mask
        pos_mask = anchor_section_ids.view(-1, 1) == candidate_section_match.view(1, -1)
        # log-sum-exp denominator over all candidates
        denom = torch.logsumexp(sim, dim=1)

        # Sum of pos logits per anchor (mean-of-pos cross-entropy form)
        losses = []
        for i in range(anchors.shape[0]):
            pos_i = pos_mask[i]
            if not pos_i.any():
                continue
            losses.append(-(sim[i][pos_i].mean() - denom[i]))
        if not losses:
            return anchors.new_zeros(())
        return torch.stack(losses).mean()

    def forward(
        self,
        st_embeddings: torch.Tensor,
        st_section_ids: torch.Tensor,
        sc_embeddings: torch.Tensor,
        sc_section_ids: torch.Tensor,
        section_pairing: Dict[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            st_embeddings: (N_st, d) ST cell embeddings.
            st_section_ids: (N_st,) section ids for ST cells.
            sc_embeddings: (N_sc, d) scRNA cell embeddings.
            sc_section_ids: (N_sc,) section ids for scRNA cells.
            section_pairing: dict mapping ST section id -> matched scRNA
                section id (and ideally the reverse too; we'll build the
                reverse on the fly).

        Returns:
            loss, metrics.
        """
        device = st_embeddings.device

        if st_embeddings.numel() == 0 or sc_embeddings.numel() == 0:
            return torch.tensor(0.0, device=device), {"cm_contrastive_loss": 0.0}

        # Map each scRNA cell's section to its paired ST section (so the mask
        # can be written as: anchor.st_section == candidate.paired_st_section).
        reverse = {v: k for k, v in section_pairing.items()}
        sc_paired_st = torch.tensor(
            [reverse.get(int(s.item()), -1) for s in sc_section_ids],
            device=device,
            dtype=st_section_ids.dtype,
        )

        l_st = self._directional(
            st_embeddings, sc_embeddings, st_section_ids, sc_paired_st
        )

        if self.symmetric:
            # Reverse: scRNA anchors, ST candidates. Need the inverse mapping:
            # for each ST cell, what scRNA section is *it* paired with.
            st_paired_sc = torch.tensor(
                [section_pairing.get(int(s.item()), -1) for s in st_section_ids],
                device=device,
                dtype=sc_section_ids.dtype,
            )
            l_sc = self._directional(
                sc_embeddings, st_embeddings, sc_section_ids, st_paired_sc
            )
            loss = 0.5 * (l_st + l_sc)
        else:
            loss = l_st

        return loss, {"cm_contrastive_loss": loss.item()}


# ---------------------------------------------------------------------------
# Section embedding consistency
# ---------------------------------------------------------------------------


class SectionEmbeddingConsistencyLoss(nn.Module):
    """Match section embeddings of paired adjacent ST and scRNA sections.

    Given two stacked tensors (ST section embeddings, scRNA section embeddings)
    that are aligned along dim 0 (row i in ST <-> row i in scRNA via the
    user-provided pairing), pulls them together. Cheap and very directly
    encodes the inductive bias that adjacent sections share tissue context.
    """

    def __init__(self, distance: str = "l2", normalize: bool = False):
        super().__init__()
        assert distance in ("l2", "cosine"), distance
        self.distance = distance
        self.normalize = normalize

    def forward(
        self,
        st_section_embeds: torch.Tensor,  # (P, d)
        sc_section_embeds: torch.Tensor,  # (P, d)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if st_section_embeds.shape != sc_section_embeds.shape:
            raise ValueError(
                f"Section-embedding consistency requires aligned tensors; got "
                f"{st_section_embeds.shape} vs {sc_section_embeds.shape}"
            )
        if st_section_embeds.numel() == 0:
            return torch.tensor(0.0, device=st_section_embeds.device), {
                "section_consistency_loss": 0.0
            }

        if self.normalize:
            st_section_embeds = F.normalize(st_section_embeds, dim=-1)
            sc_section_embeds = F.normalize(sc_section_embeds, dim=-1)

        if self.distance == "l2":
            loss = F.mse_loss(st_section_embeds, sc_section_embeds)
        else:  # cosine
            sim = (st_section_embeds * sc_section_embeds).sum(dim=-1)
            loss = (1.0 - sim).mean()

        return loss, {"section_consistency_loss": loss.item()}


# ---------------------------------------------------------------------------
# Domain-adversarial (gradient reversal + classifier)
# ---------------------------------------------------------------------------


class _GradientReversalFn(torch.autograd.Function):
    """Reverses (and scales) the gradient on the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def gradient_reversal(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lambd)


class DomainAdversarialLoss(nn.Module):
    """DANN-style domain classifier with gradient reversal.

    The classifier head sits on top of (cell or section) embeddings; the
    gradient reversal layer flips the sign of gradients flowing back to the
    encoder, so the encoder is incentivized to produce embeddings the
    classifier cannot tell apart. Domain labels are integers (0=ST, 1=scRNA),
    extendable to more than two domains.

    Args:
        in_dim: Dimension of input embeddings.
        n_domains: Number of domain classes.
        hidden_dim: Hidden width of the classifier.
        lambd: Strength of the reversed gradient (often ramped during training).
    """

    def __init__(
        self,
        in_dim: int,
        n_domains: int = 2,
        hidden_dim: int = 128,
        lambd: float = 1.0,
    ):
        super().__init__()
        self.lambd = lambd
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: (N, in_dim) — typically cell embeddings.
            domain_labels: (N,) integer domain ids.

        Returns:
            loss, metrics.
        """
        if embeddings.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device), {
                "domain_adv_loss": 0.0,
                "domain_adv_acc": 0.0,
            }
        reversed_ = gradient_reversal(embeddings, self.lambd)
        logits = self.classifier(reversed_)
        loss = F.cross_entropy(logits, domain_labels)
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == domain_labels).float().mean().item()
        return loss, {"domain_adv_loss": loss.item(), "domain_adv_acc": acc}

"""Latent-VAE building blocks for the LDM framework.

Three modules:

  * ``LatentVAEEncoder`` — small transformer that maps (gene, position)
    pairs to per-cell ``(mu, logvar)`` of the latent posterior
    ``q(z | x)``. Cross-cell mixing happens inside the transformer so
    each ``z_i`` is tissue-aware, not just a per-cell function of
    ``(gene_i, position_i)``.

  * ``LatentVAEDecoder`` — small transformer that maps ``(gene, z)``
    pairs back to per-cell positions. Symmetric to the encoder.

  * Helper functions: ``reparameterize`` (Gaussian reparam trick) and
    ``kl_normal_standard`` (KL of ``q(z|x) = N(mu, sigma²)`` against
    the standard Gaussian prior ``N(0, I)``).

These are deliberately small (2 layers each, configurable) because
joint training with the denoiser is already a 3-loss optimisation
and we don't want the VAE to dominate. The denoiser is where the
heavy lifting happens (DiT-style backbone).

Used by:
  * ``models.latent_diffusion_wrapper.LatentDiffusionWrapper`` — wires
    encoder + denoiser + decoder into one ``self.model`` for the
    LightningModule.
  * ``utils.diffusion_model.diffusion.latent_diffusion_model.LatentDiffusionModel``
    — the noise model that operates in latent (z) space.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Small transformer block — pre-LN, self-attention, FFN with residuals.
# Standard "ViT-style" block. No time conditioning (the VAE doesn't see
# diffusion time — only the denoiser does).
# ---------------------------------------------------------------------------


class _VAEBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"VAE block hidden_dim {hidden_dim} must be divisible by "
                f"n_heads {n_heads}"
            )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        ff_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN self-attention, padding-aware.
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Encoder: (gene, position) → (mu, logvar)
# ---------------------------------------------------------------------------


class LatentVAEEncoder(nn.Module):
    """Per-cell ``q(z | gene, position)`` parameterised as a Gaussian.

    Input contract:
        gene_features  (B, N, gene_dim)
        positions      (B, N, 2)
        node_mask      (B, N)   bool, True for real cells

    Output:
        mu     (B, N, latent_dim)
        logvar (B, N, latent_dim)   ← log-variance per dim; KL math uses this
                                       directly without a softplus.
    """

    def __init__(
        self,
        gene_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        # Project gene features and positions to per-cell tokens, then sum
        # (matches the additive-token convention used by the DiT backbone).
        self.gene_embed = nn.Linear(int(gene_dim), self.hidden_dim)
        self.pos_embed = nn.Linear(2, self.hidden_dim)
        self.blocks = nn.ModuleList([
            _VAEBlock(self.hidden_dim, n_heads, mlp_ratio)
            for _ in range(int(n_layers))
        ])
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        # Project to (mu, logvar). 2*latent_dim output dim; split downstream.
        self.proj_out = nn.Linear(self.hidden_dim, 2 * self.latent_dim)
        # Initialise logvar to roughly 0 (so sigma ≈ 1) and mu to roughly 0
        # (matches the N(0,I) prior at init). Bias only — the random-init
        # weights add a small perturbation that the KL loss will damp.
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        gene_features: torch.Tensor,
        positions: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.gene_embed(gene_features) + self.pos_embed(positions)
        # Zero out padding tokens.
        tok = tok * node_mask.unsqueeze(-1).to(tok.dtype)
        # PyTorch's MultiheadAttention key_padding_mask is True at PAD.
        kpm = ~node_mask
        for block in self.blocks:
            tok = block(tok, key_padding_mask=kpm)
        tok = self.final_norm(tok)
        out = self.proj_out(tok)                              # (B, N, 2k)
        mu, logvar = out.chunk(2, dim=-1)                     # each (B, N, k)
        # Mask padding cells (mu, logvar stay zero there).
        m = node_mask.unsqueeze(-1).to(mu.dtype)
        mu = mu * m
        logvar = logvar * m
        return mu, logvar


# ---------------------------------------------------------------------------
# Decoder: (gene, z) → positions
# ---------------------------------------------------------------------------


class LatentVAEDecoder(nn.Module):
    """Per-cell ``p(position | gene, z)`` — symmetric to the encoder.

    Input contract:
        gene_features  (B, N, gene_dim)
        z              (B, N, latent_dim)
        node_mask      (B, N)

    Output:
        positions      (B, N, 2)
    """

    def __init__(
        self,
        gene_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gene_embed = nn.Linear(int(gene_dim), self.hidden_dim)
        self.z_embed = nn.Linear(int(latent_dim), self.hidden_dim)
        self.blocks = nn.ModuleList([
            _VAEBlock(self.hidden_dim, n_heads, mlp_ratio)
            for _ in range(int(n_layers))
        ])
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        # Project to 2D position.
        self.proj_out = nn.Linear(self.hidden_dim, 2)

    def forward(
        self,
        gene_features: torch.Tensor,
        z: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        tok = self.gene_embed(gene_features) + self.z_embed(z)
        tok = tok * node_mask.unsqueeze(-1).to(tok.dtype)
        kpm = ~node_mask
        for block in self.blocks:
            tok = block(tok, key_padding_mask=kpm)
        tok = self.final_norm(tok)
        pos = self.proj_out(tok)                              # (B, N, 2)
        # Mask padding cells to zero and mean-centre over real cells.
        # Matches the convention used by Model / DiTBackbone / etc.
        m = node_mask.unsqueeze(-1).to(pos.dtype)
        pos = pos * m
        n_real = node_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1).to(
            pos.dtype
        )
        pos = pos - pos.sum(dim=1, keepdim=True) / n_real
        pos = pos * m
        return pos


# ---------------------------------------------------------------------------
# Reparameterization + KL helpers
# ---------------------------------------------------------------------------


def reparameterize(
    mu: torch.Tensor, logvar: torch.Tensor,
) -> torch.Tensor:
    """Standard Gaussian reparameterization: z = mu + sigma * eps.

    Clamps logvar to a sane range so the optimiser can't push the
    encoder into a degenerate near-deterministic regime (which would
    make the KL loss numerically pathological).
    """
    # exp(20) ≈ 5e8 — generous upper bound, beyond which sigma is dominated
    # by encoder pathology rather than meaningful variance. Lower bound -20
    # → sigma ~ 4.5e-5 — small enough that effectively z = mu.
    logvar = logvar.clamp(min=-20.0, max=20.0)
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


def kl_normal_standard(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) per real cell, averaged.

    For a diagonal Gaussian q = N(mu, diag(sigma²)) vs N(0, I):

        KL = 0.5 * sum_d ( mu_d² + sigma_d² − 1 − log(sigma_d²) )
           = 0.5 * sum_d ( mu_d² + exp(logvar_d) − 1 − logvar_d )

    The sum is over the latent dimensions for each cell; we then
    average across REAL cells (padding cells excluded so the loss
    scales the same way regardless of how much padding the batch has).
    """
    logvar = logvar.clamp(min=-20.0, max=20.0)
    per_cell = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)
    # per_cell: (B, N). Mask + mean over real cells.
    mask = node_mask.to(per_cell.dtype)
    per_cell = per_cell * mask
    n_real = mask.sum().clamp_min(1.0)
    return per_cell.sum() / n_real

"""
LUNA-aligned DDPM training + sampling.

Direct port of LUNA's
    utils/diffusion_model/diffusion/noise_model.py  (cosine β schedule + apply_noise + reverse process)
    utils/diffusion_model/train/train.py            (training step: apply noise → predict x_0 → cdist MSE)
    utils/diffusion_model/sample/sample.py          (iterative stochastic reverse)
    metrics/loss_function.py                        (MSE on pairwise distance matrix)

Used when ``model.diffusion.type='ddpm'`` (the default for the LUNA-aligned
config). The alternative is the OT-CFM path in ``flow_matching.py``,
which we keep for ablation but which doesn't reproduce LUNA's results.

Key differences from OT-CFM:

  * **Discrete cosine-β diffusion schedule** (1000 timesteps by default),
    not continuous OT-CFM linear interpolation. Forward kernel is
    ``z_t = α̅_t · x_0 + σ̅_t · ε``.

  * **Time convention is LUNA's**: ``t_int`` ∈ [1, T], ``t_float = t_int / T``
    ∈ (0, 1]. ``t_float`` near 0 ⇔ clean target. ``t_float`` near 1 ⇔ noise.
    (This is the OPPOSITE of our OT-CFM convention where t=0 is noise.)

  * **Stochastic reverse sampling**: at each step from t=T down to t=1,
    predict x_0 given z_t, then sample z_{t-1} ~ N(μ, σ² I) with
    fresh noise. The noise injection at each step regularises error
    accumulation — this is precisely the inductive bias OT-CFM's
    deterministic Euler integration was missing.

  * **Per-slice t**: one t per slice, broadcast to all cells. Already
    matched in the OT-CFM path.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Cosine β schedule (LUNA's `utils/diffusion_model/diffusion/diffusion_utils.py`)
# ---------------------------------------------------------------------------


def _cosine_beta_schedule_discrete(
    timesteps: int, nu: float = 2.0, s: float = 0.008
) -> np.ndarray:
    """LUNA's cosine_beta_schedule_discrete, specialised to a single
    component (positions). Returns (timesteps+1,) array of β values."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    # alphas_cumprod = cos(0.5 π ((x/steps)^ν + s) / (1+s))^2
    alphas_cumprod = (
        np.cos(0.5 * np.pi * (((x / steps) ** nu) + s) / (1.0 + s)) ** 2
    )
    alphas_cumprod_norm = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod_norm[1:] / alphas_cumprod_norm[:-1]
    betas = 1.0 - alphas
    return betas.astype(np.float64)


# ---------------------------------------------------------------------------
# DDPM noise model
# ---------------------------------------------------------------------------


class DDPMNoiseModel(nn.Module):
    """Cosine-β VP-DDPM noise model for 2-D position diffusion.

    Direct port of LUNA's ``NoiseModel`` restricted to the 'p' (positions)
    component. All schedule tensors are precomputed at construction and
    registered as buffers (so they move with the module to GPU).

    Conventions:
      * ``t_int`` is an integer in [1, T]. ``t=0`` corresponds to the
        clean target, ``t=T`` to the noise prior.
      * ``α̅_t``, ``σ̅_t`` satisfy ``α̅_t² + σ̅_t² = 1`` (VP).
      * ``apply_noise(x_0)`` samples a single ``t`` (per-slice) and
        returns ``z_t``, the broadcast ``t_float``, and ``t_int``.

    Args:
        timesteps: number of diffusion steps T (LUNA default: 1000).
        nu: cosine-schedule exponent for positions (LUNA default: 2.0).
    """

    def __init__(self, timesteps: int = 1000, nu: float = 2.0):
        super().__init__()
        self.timesteps = int(timesteps)
        self.max_diffusion_steps = int(timesteps)

        # Precompute β, α, log α̅, α̅, σ̅²= -expm1(2·log α̅), σ̅.
        betas_np = _cosine_beta_schedule_discrete(self.timesteps, nu=nu)
        betas = torch.from_numpy(betas_np).float()                     # (T+1,)
        alphas = 1.0 - betas.clamp(min=0.0, max=0.9999)                # (T+1,)
        log_alpha = torch.log(alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)                 # (T+1,)
        alphas_bar = torch.exp(log_alpha_bar)
        sigma2_bar = -torch.expm1(2.0 * log_alpha_bar).clamp(min=0.0)
        sigma_bar = torch.sqrt(sigma2_bar)

        self.register_buffer("_betas", betas)
        self.register_buffer("_alphas", alphas)
        self.register_buffer("_log_alpha_bar", log_alpha_bar)
        self.register_buffer("_alphas_bar", alphas_bar)
        self.register_buffer("_sigma2_bar", sigma2_bar)
        self.register_buffer("_sigma_bar", sigma_bar)

    # ---- Lookups (mirror LUNA's get_* helpers) -------------------------

    def get_alpha_bar(self, t_int: torch.Tensor) -> torch.Tensor:
        return self._alphas_bar[t_int.long()]

    def get_sigma_bar(self, t_int: torch.Tensor) -> torch.Tensor:
        return self._sigma_bar[t_int.long()]

    def get_log_alpha_bar(self, t_int: torch.Tensor) -> torch.Tensor:
        return self._log_alpha_bar[t_int.long()]

    # ---- Forward (apply noise) -----------------------------------------

    def apply_noise(
        self, x_0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a per-slice t and return the noised z_t.

        Args:
            x_0: ``(n_cells, spatial_dim)`` clean target positions
                (already mean-centred is recommended).

        Returns:
            z_t: ``(n_cells, spatial_dim)`` noised positions.
            t_per_cell: ``(n_cells,)`` broadcast t_float in (0, 1].
            t_int: scalar tensor, the sampled integer timestep.
        """
        device = x_0.device
        n_cells = x_0.shape[0]

        # Per-SLICE t: one integer in [1, T] for all cells in this slice.
        t_int = torch.randint(
            1, self.max_diffusion_steps + 1, size=(1,), device=device
        )

        # α̅, σ̅ at this t. Each is a (1,) scalar.
        a = self.get_alpha_bar(t_int)
        s = self.get_sigma_bar(t_int)

        # Noise with mean removed (preserves translation equivariance).
        noise = torch.randn_like(x_0)
        noise = noise - noise.mean(dim=0, keepdim=True)

        # z_t = α̅ · x_0 + σ̅ · ε  (broadcasts (1,) × (n, 2) → (n, 2))
        z_t = a * x_0 + s * noise

        t_float = t_int.float() / float(self.max_diffusion_steps)      # (1,)
        t_per_cell = t_float.expand(n_cells)                            # (n,)
        return z_t, t_per_cell, t_int

    # ---- Initial noise (used at sampling start) -----------------------

    def sample_limit_dist(
        self, n_cells: int, spatial_dim: int, device: torch.device
    ) -> torch.Tensor:
        """Sample z_T ~ N(0, I), mean-removed. (LUNA's `sample_limit_dist`.)"""
        z_T = torch.randn(n_cells, spatial_dim, device=device)
        z_T = z_T - z_T.mean(dim=0, keepdim=True)
        return z_T

    # ---- Reverse (one denoising step) ---------------------------------

    @torch.no_grad()
    def sample_zs_from_zt_and_pred(
        self,
        z_t: torch.Tensor,                # (n, spatial_dim)
        x_0_pred: torch.Tensor,           # (n, spatial_dim)
        t_int: torch.Tensor,              # scalar
        s_int: torch.Tensor,              # scalar
    ) -> torch.Tensor:
        """Sample z_s ~ p(z_s | z_t, x_0_pred). Direct port of LUNA's
        ``sample_zs_from_zt_and_pred``, with mean-removal on the noise.
        """
        device = z_t.device

        # log α̅ at t and s.
        log_a_t = self.get_log_alpha_bar(t_int)
        log_a_s = self.get_log_alpha_bar(s_int)

        # α̅_t / α̅_s, and its square (LUNA's `get_alpha_pos_ts(_sq)`).
        alpha_ts = torch.exp(log_a_t - log_a_s)
        alpha_ts_sq = torch.exp(2.0 * log_a_t - 2.0 * log_a_s)

        # σ²_s / σ²_t = exp(log σ²_s - log σ²_t).
        # σ²_t := -expm1(2 log α̅_t).
        s2_s = -torch.expm1(2.0 * log_a_s).clamp(min=1e-30)
        s2_t = -torch.expm1(2.0 * log_a_t).clamp(min=1e-30)
        sigma_sq_ratio = torch.exp(torch.log(s2_s) - torch.log(s2_t))

        # μ = z_t_prefactor · z_t + positions_prefactor · x_0_pred.
        z_t_prefactor = alpha_ts * sigma_sq_ratio
        a_s = self.get_alpha_bar(s_int)
        positions_prefactor = a_s * (1.0 - alpha_ts_sq * sigma_sq_ratio)
        mu = z_t_prefactor * z_t + positions_prefactor * x_0_pred

        # Noise term: scale = √((σ̅_t - σ̅_s · α̅²_{t,s}) · σ²_s/σ²_t).
        # LUNA's exact arithmetic; see noise_model.py:411-417.
        sigma_bar_t = self.get_sigma_bar(t_int)
        sigma_bar_s = self.get_sigma_bar(s_int)
        sigma2_t_s = (sigma_bar_t - sigma_bar_s * alpha_ts_sq).clamp(min=0.0)
        noise_prefactor_sq = sigma2_t_s * sigma_sq_ratio
        noise_prefactor = torch.sqrt(noise_prefactor_sq.clamp(min=0.0))

        # Fresh noise, mean-removed.
        noise = torch.randn_like(z_t)
        noise = noise - noise.mean(dim=0, keepdim=True)

        z_s = mu + noise_prefactor * noise
        return z_s


# ---------------------------------------------------------------------------
# Top-level wrapper: training + sampling
# ---------------------------------------------------------------------------


class DiffusionDDPM(nn.Module):
    """LUNA-aligned DDPM wrapper.

    Combines:
      * ``DDPMNoiseModel``: forward noising + reverse sampling.
      * A network (``velocity_net`` — typically ``LunaTransformerNet``)
        that predicts ``x_0`` from noised ``z_t``.

    The interface matches ``ConditionalFlowMatching`` so this is a
    drop-in replacement.

    Training (``compute_loss``):
      1. Centre target positions.
      2. ``z_t = α̅_t · x_0 + σ̅_t · ε`` via the noise model.
      3. Predict ``x_0_pred = model(z_t, t_per_cell, cell_embed)``.
      4. Loss = ``MSE(cdist(x_0_pred), cdist(x_0))`` — LUNA's exact loss.

    Sampling (``sample``):
      * Iterate T=1000 stochastic DDPM reverse steps. At each step,
        predict x_0 from the current z_t and sample z_{t-1}.
    """

    def __init__(
        self,
        velocity_net: nn.Module,
        timesteps: int = 1000,
        nu: float = 2.0,
        translation_equivariant: bool = True,
        pairwise_dist_max_cells: int = 8192,
    ):
        super().__init__()
        self.velocity_net = velocity_net  # network predicting x_0
        self.spatial_dim = velocity_net.spatial_dim
        self.noise_model = DDPMNoiseModel(timesteps=timesteps, nu=nu)
        self.translation_equivariant = bool(translation_equivariant)
        self.pairwise_dist_max_cells = int(pairwise_dist_max_cells)

    @staticmethod
    def _center(x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=0, keepdim=True)

    # ---- Training step ------------------------------------------------

    def compute_loss(
        self,
        z_1: torch.Tensor,
        cell_embed: torch.Tensor,
        section_embed: Optional[torch.Tensor] = None,
        k_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """One training step of LUNA's DDPM."""
        device = z_1.device
        n_cells = z_1.shape[0]

        # Centre targets (matches LUNA's `center_positions` inside `.mask()`).
        if self.translation_equivariant:
            z_1 = self._center(z_1)

        # Forward noising: z_t = α̅ · z_1 + σ̅ · ε.
        z_t, t_per_cell, t_int = self.noise_model.apply_noise(z_1)
        if self.translation_equivariant:
            z_t = self._center(z_t)

        # Predict x_0 from z_t.
        x_0_pred = self.velocity_net(
            z_t, t_per_cell, cell_embed, section_embed, k_target
        )
        if self.translation_equivariant:
            x_0_pred = self._center(x_0_pred)

        # Subsample for memory if needed (LUNA itself doesn't subsample;
        # cortex slices < 8k fit easily without subsampling).
        if (
            self.pairwise_dist_max_cells > 0
            and n_cells > self.pairwise_dist_max_cells
        ):
            idx = torch.randperm(n_cells, device=device)[
                : self.pairwise_dist_max_cells
            ]
            x_pred_sub = x_0_pred[idx]
            z_1_sub = z_1[idx]
        else:
            x_pred_sub = x_0_pred
            z_1_sub = z_1

        d_pred = torch.cdist(x_pred_sub, x_pred_sub, p=2.0)
        d_true = torch.cdist(z_1_sub, z_1_sub, p=2.0)
        loss = torch.mean((d_pred - d_true) ** 2)

        metrics = {
            "fm_loss": loss.item(),
            "fm_pairwise_dist_mse": loss.item(),
            "fm_velocity_mse": 0.0,
            "v_norm": 0.0,
            "u_norm": (
                z_1.std(dim=0).mean().item() * math.sqrt(self.spatial_dim)
            ),
            "x_hat_0_norm": x_0_pred.norm(dim=-1).mean().item(),
            "t_int": int(t_int.item()),
        }
        return loss, metrics

    # ---- Sampling ----------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        cell_embed: torch.Tensor,
        section_embed: Optional[torch.Tensor] = None,
        k_target: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,   # ignored — uses noise_model.T
        solver: Optional[str] = None,    # ignored — DDPM reverse is the path
    ) -> torch.Tensor:
        """LUNA-style stochastic reverse process. Iterate T=1000 steps
        from t=T down to t=1, each step predicting x_0 and sampling
        z_{t-1} from N(μ, σ² I). Final z_0 is the prediction.
        """
        device = cell_embed.device
        n_cells = cell_embed.shape[0]
        T = self.noise_model.max_diffusion_steps

        # Start from z_T ~ N(0, I), mean-removed.
        z_t = self.noise_model.sample_limit_dist(n_cells, self.spatial_dim, device)

        for t in range(T, 0, -1):
            s = t - 1
            t_int = torch.tensor(t, dtype=torch.long, device=device)
            s_int = torch.tensor(s, dtype=torch.long, device=device)

            # Predict x_0 from the current z_t.
            t_per_cell = (t_int.float() / float(T)).expand(n_cells)
            x_0_pred = self.velocity_net(
                z_t, t_per_cell, cell_embed, section_embed, k_target
            )
            if self.translation_equivariant:
                x_0_pred = self._center(x_0_pred)

            if s > 0:
                # Sample z_{t-1} ~ N(μ, σ² I).
                z_t = self.noise_model.sample_zs_from_zt_and_pred(
                    z_t, x_0_pred, t_int, s_int
                )
            else:
                # Last step (t=1, s=0): the standard convention is that
                # at s=0 the prediction IS the final output (no further
                # noise is added). LUNA's loop also exits with z at the
                # final step.
                z_t = x_0_pred

            if self.translation_equivariant:
                z_t = self._center(z_t)

        return z_t

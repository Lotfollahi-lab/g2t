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
        self,
        x_0: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a per-section t and return the noised z_t.

        Args:
            x_0: ``(n_cells, spatial_dim)`` for single-section, or
                ``(B, n_cells, spatial_dim)`` for parallel-batched.
            mask: ``(B, n_cells)`` bool, True for valid cells. Only
                used when ``x_0`` is batched.

        Returns:
            z_t: same shape as ``x_0`` (noised positions).
            t_per_cell: ``(n_cells,)`` or ``(B, n_cells)`` broadcast
                t_float in (0, 1].
            t_int: ``(1,)`` or ``(B,)`` integer timesteps actually
                sampled (one per section).
        """
        was_unbatched = x_0.dim() == 2
        if was_unbatched:
            x_0 = x_0.unsqueeze(0)                                    # (1, N, D)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, N, D = x_0.shape
        device = x_0.device

        # Per-SECTION t (one integer in [1, T] per batch element).
        # LUNA: `t_int = torch.randint(1, T+1, size=(bs, 1), ...)`
        t_int = torch.randint(
            1, self.max_diffusion_steps + 1, size=(B,), device=device
        )

        # α̅, σ̅ per section, shaped for broadcast with (B, N, D).
        a = self.get_alpha_bar(t_int).view(B, 1, 1)
        s = self.get_sigma_bar(t_int).view(B, 1, 1)

        # Noise per section, mean-removed within each section (masked).
        noise = torch.randn_like(x_0)
        if mask is not None:
            m = mask.to(noise.dtype).unsqueeze(-1)                    # (B, N, 1)
            noise = noise * m
            valid = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            noise_mean = noise.sum(dim=1, keepdim=True) / valid
            noise = (noise - noise_mean) * m
        else:
            noise = noise - noise.mean(dim=1, keepdim=True)

        # z_t = α̅ · x_0 + σ̅ · ε
        z_t = a * x_0 + s * noise

        t_float = t_int.float() / float(self.max_diffusion_steps)     # (B,)
        t_per_cell = t_float.unsqueeze(-1).expand(B, N)               # (B, N)

        if was_unbatched:
            z_t = z_t.squeeze(0)
            t_per_cell = t_per_cell.squeeze(0)
            # t_int kept as (1,) for backward compat
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
        z_t: torch.Tensor,                # (N, D) OR (B, N, D)
        x_0_pred: torch.Tensor,           # same shape as z_t
        t_int: torch.Tensor,              # scalar OR (B,)
        s_int: torch.Tensor,              # scalar OR (B,) — same shape as t_int
        mask: Optional[torch.Tensor] = None,  # (B, N) bool — batched only
    ) -> torch.Tensor:
        """Sample z_s ~ p(z_s | z_t, x_0_pred). Direct port of LUNA's
        ``sample_zs_from_zt_and_pred``, with mean-removal on the noise.
        Supports both single-section (legacy) and parallel-batched
        (LUNA-equivalent) inputs.
        """
        was_unbatched = z_t.dim() == 2
        if was_unbatched:
            z_t = z_t.unsqueeze(0)
            x_0_pred = x_0_pred.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            # t_int / s_int might be scalar or (1,); make them (1,)
            if t_int.dim() == 0:
                t_int = t_int.unsqueeze(0)
            if s_int.dim() == 0:
                s_int = s_int.unsqueeze(0)

        B, N, D = z_t.shape

        # log α̅ at t and s — shape (B,) → (B, 1, 1) for broadcast.
        log_a_t = self.get_log_alpha_bar(t_int).view(B, 1, 1)
        log_a_s = self.get_log_alpha_bar(s_int).view(B, 1, 1)

        # α̅_t / α̅_s, and its square (LUNA's `get_alpha_pos_ts(_sq)`).
        alpha_ts = torch.exp(log_a_t - log_a_s)
        alpha_ts_sq = torch.exp(2.0 * log_a_t - 2.0 * log_a_s)

        # σ²_s / σ²_t = exp(log σ²_s - log σ²_t).
        s2_s = -torch.expm1(2.0 * log_a_s).clamp(min=1e-30)
        s2_t = -torch.expm1(2.0 * log_a_t).clamp(min=1e-30)
        sigma_sq_ratio = torch.exp(torch.log(s2_s) - torch.log(s2_t))

        # μ = z_t_prefactor · z_t + positions_prefactor · x_0_pred.
        z_t_prefactor = alpha_ts * sigma_sq_ratio
        a_s = self.get_alpha_bar(s_int).view(B, 1, 1)
        positions_prefactor = a_s * (1.0 - alpha_ts_sq * sigma_sq_ratio)
        mu = z_t_prefactor * z_t + positions_prefactor * x_0_pred

        # Noise term scale per section.
        sigma_bar_t = self.get_sigma_bar(t_int).view(B, 1, 1)
        sigma_bar_s = self.get_sigma_bar(s_int).view(B, 1, 1)
        sigma2_t_s = (sigma_bar_t - sigma_bar_s * alpha_ts_sq).clamp(min=0.0)
        noise_prefactor_sq = sigma2_t_s * sigma_sq_ratio
        noise_prefactor = torch.sqrt(noise_prefactor_sq.clamp(min=0.0))

        # Fresh noise per section, mean-removed (masked).
        noise = torch.randn_like(z_t)
        if mask is not None:
            m = mask.to(noise.dtype).unsqueeze(-1)
            noise = noise * m
            valid = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            noise_mean = noise.sum(dim=1, keepdim=True) / valid
            noise = (noise - noise_mean) * m
        else:
            noise = noise - noise.mean(dim=1, keepdim=True)

        z_s = mu + noise_prefactor * noise
        if mask is not None:
            # zero padding (cosmetic — caller centers anyway)
            z_s = z_s * mask.to(z_s.dtype).unsqueeze(-1)

        if was_unbatched:
            z_s = z_s.squeeze(0)
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
    def _center(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Subtract the per-section mean. Works on (N, D) or (B, N, D).
        When `mask` is given (only with (B, N, D)), only counts valid
        cells in the mean and zeros out padding cells in the output.
        """
        if x.dim() == 2:
            return x - x.mean(dim=0, keepdim=True)
        # (B, N, D)
        if mask is None:
            return x - x.mean(dim=1, keepdim=True)
        m = mask.to(x.dtype).unsqueeze(-1)
        x = x * m
        valid = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = x.sum(dim=1, keepdim=True) / valid
        return (x - mean) * m

    # ---- Training step ------------------------------------------------

    def compute_loss(
        self,
        z_1: torch.Tensor,
        cell_embed: torch.Tensor,
        section_embed: Optional[torch.Tensor] = None,
        k_target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """One training step of LUNA's DDPM.

        Supports both single-section ``z_1: (N, D)`` and
        parallel-batched ``z_1: (B, N, D)`` inputs. In the batched
        path, ``mask: (B, N)`` is required and isolates each section
        from the others — equivalent to LUNA's ``to_dense_batch +
        node_mask`` setup. Loss is the mean over sections of each
        section's masked pairwise-distance MSE — matches LUNA's
        ``metrics.loss_function.LossFunction.compute_loss``.
        """
        was_unbatched = z_1.dim() == 2
        if was_unbatched:
            z_1 = z_1.unsqueeze(0)
            cell_embed = cell_embed.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, N, D = z_1.shape
        device = z_1.device

        # Centre targets per section (masked).
        if self.translation_equivariant:
            z_1 = self._center(z_1, mask=mask)

        # Forward noising: z_t = α̅ · z_1 + σ̅ · ε, one t per section.
        z_t, t_per_cell, t_int = self.noise_model.apply_noise(z_1, mask=mask)
        if self.translation_equivariant:
            z_t = self._center(z_t, mask=mask)

        # Predict x_0. velocity_net handles batched input + mask.
        x_0_pred = self.velocity_net(
            z_t, t_per_cell, cell_embed, section_embed, k_target, mask=mask,
        )
        if self.translation_equivariant:
            x_0_pred = self._center(x_0_pred, mask=mask)

        # Per-section pairwise-distance MSE, then mean across sections.
        # IMPORTANT: must compute cdist per-section on ONLY the valid
        # cells, not on the full padded (B, N, N) tensor. Padding cells
        # all share position (0, 0), and `torch.cdist`'s backward at
        # coincident points is `(x_i - x_j) / 0 = NaN`. PyTorch's
        # autograd propagates NaN through the sum even when we
        # multiply by mask_2d = 0 — the NaN poisons the whole
        # gradient. Per-section cdist on valid cells avoids this and
        # exactly mirrors LUNA's `LossFunction.compute_loss`.
        loss_per_section_list = []
        for b in range(B):
            if mask is not None:
                valid = mask[b]
                x_pred_b = x_0_pred[b][valid]                         # (n_b, D)
                z_1_b = z_1[b][valid]                                 # (n_b, D)
            else:
                x_pred_b = x_0_pred[b]
                z_1_b = z_1[b]
            d_pred_b = torch.cdist(x_pred_b, x_pred_b, p=2.0)
            d_true_b = torch.cdist(z_1_b, z_1_b, p=2.0)
            loss_per_section_list.append(((d_pred_b - d_true_b) ** 2).mean())
        loss_per_section = torch.stack(loss_per_section_list)         # (B,)

        # LUNA: `mse_loss = torch.mean(stacked_losses)` — mean across sections.
        loss = loss_per_section.mean()

        # Diagnostics — average across sections.
        if mask is not None:
            m1d = mask.to(z_1.dtype).unsqueeze(-1)
            z_1_norm_per_cell = z_1.norm(dim=-1) * mask.to(z_1.dtype)
            x_norm_per_cell = x_0_pred.norm(dim=-1) * mask.to(x_0_pred.dtype)
            valid = mask.to(z_1.dtype).sum().clamp_min(1.0)
            u_norm = (z_1_norm_per_cell.sum() / valid).item()
            x_hat_0_norm = (x_norm_per_cell.sum() / valid).item()
        else:
            u_norm = z_1.norm(dim=-1).mean().item()
            x_hat_0_norm = x_0_pred.norm(dim=-1).mean().item()

        metrics = {
            "fm_loss": loss.item(),
            "fm_pairwise_dist_mse": loss.item(),
            "fm_velocity_mse": 0.0,
            "v_norm": 0.0,
            "u_norm": u_norm,
            "x_hat_0_norm": x_hat_0_norm,
            "t_int": float(t_int.float().mean().item()),
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
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """LUNA-style stochastic reverse process. Iterate T=1000 steps
        from t=T down to t=1, each step predicting x_0 and sampling
        z_{t-1} from N(μ, σ² I). Final z_0 is the prediction.

        Supports both single-section ``cell_embed: (N, d)`` and
        parallel-batched ``cell_embed: (B, N, d)`` inputs. In the
        batched path, ``mask: (B, N)`` isolates each section from the
        others — equivalent to LUNA's ``to_dense_batch + node_mask``.
        """
        device = cell_embed.device
        T = self.noise_model.max_diffusion_steps

        was_unbatched = cell_embed.dim() == 2
        if was_unbatched:
            n_cells = cell_embed.shape[0]
            z_t = self.noise_model.sample_limit_dist(n_cells, self.spatial_dim, device)
        else:
            B, N, _ = cell_embed.shape
            # Per-section initial noise z_T ~ N(0, I), mean-removed (masked).
            z_t = torch.randn(B, N, self.spatial_dim, device=device)
            if mask is not None:
                m = mask.to(z_t.dtype).unsqueeze(-1)
                z_t = z_t * m
                valid = m.sum(dim=1, keepdim=True).clamp_min(1.0)
                z_t = (z_t - z_t.sum(dim=1, keepdim=True) / valid) * m
            else:
                z_t = z_t - z_t.mean(dim=1, keepdim=True)

        for t in range(T, 0, -1):
            s = t - 1
            if was_unbatched:
                t_int = torch.tensor(t, dtype=torch.long, device=device)
                s_int = torch.tensor(s, dtype=torch.long, device=device)
                t_per_cell = (t_int.float() / float(T)).expand(z_t.shape[0])
            else:
                t_int = torch.full((B,), t, dtype=torch.long, device=device)
                s_int = torch.full((B,), s, dtype=torch.long, device=device)
                t_per_cell = (t_int.float() / float(T)).unsqueeze(-1).expand(B, N)

            # Predict x_0 from the current z_t.
            x_0_pred = self.velocity_net(
                z_t, t_per_cell, cell_embed, section_embed, k_target,
                mask=mask if not was_unbatched else None,
            )
            if self.translation_equivariant:
                x_0_pred = self._center(x_0_pred, mask=None if was_unbatched else mask)

            # Sample z_{t-1} ~ N(μ, σ² I). At s=0, the math collapses to
            # z_{0} = x_0_pred (the position prefactor goes to 1, noise
            # scale goes to 0) — see the σ_sq_ratio computation, which
            # produces ~0 at s=0 because σ²_s = -expm1(0) = 0.
            z_t = self.noise_model.sample_zs_from_zt_and_pred(
                z_t, x_0_pred, t_int, s_int,
                mask=mask if not was_unbatched else None,
            )

            if self.translation_equivariant:
                z_t = self._center(z_t, mask=None if was_unbatched else mask)

        return z_t

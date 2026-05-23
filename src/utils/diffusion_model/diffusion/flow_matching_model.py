"""Flow-matching alternative to LUNA's DDPM ``NoiseModel``.

This is a CONFIG SWITCH for scgg — enabled via
``--override model.framework=flow_matching``. LUNA's DDPM path is
untouched and remains the default.

Why
---
DDPM in LUNA does 1000 sequential denoising steps at inference. Each
step is a network forward, so the per-slice sampling cost is
dominated by that. Rectified flow / x_0-parameterised flow matching
replaces the iterative denoising with a smooth ODE solve in ~20-50
steps, giving 20-50× faster inference at equivalent quality. Per-step
training cost is unchanged; the loss landscape is often smoother so
convergence can be 1.5-3× faster in epochs.

Interface contract
------------------
This class exposes EXACTLY the methods that scgg's training loop
(``utils/diffusion_model/train/train.py``) and sampling loop
(``utils/diffusion_model/sample/sample.py``) call on
``self.noise_model``::

    apply_noise(data, train_flag=True)            -> DataHolder
    sample_limit_dist(node_features, node_mask,
                      cell_ID, cell_class)        -> DataHolder
    sample_zs_from_zt_and_pred(z_t, pred, s_int)  -> DataHolder

Same input/return types as ``NoiseModel``. The Lightning module
(``diffusion_model.py``) picks which class fills ``self.noise_model``
based on ``cfg.model.framework`` — neither the training step nor the
sampling loop needs to know which is in use.

Math
----
Forward (training) — sample a perturbed point on the straight line
between data and noise::

    t ∼ 𝒰(eps_t, 1)        # one t per slice
    x_1 ∼ 𝒩(0, I)           # mean-centered noise per slice
    x_t = (1 - t)·x_0 + t·x_1

The network is trained to predict ``x_0`` from ``x_t``. Loss is
unchanged from the diffusion path — ``LossFunction`` already operates
on (pred_positions, true_positions). All opt-in loss components
(persistent homology, kNN-rank) transfer without modification.

Sampling — Euler ODE on the velocity field, going t = 1 → 0::

    v_t(x_t)  = (x_t - x_0_pred(x_t, t)) / t       # for the linear schedule,
                                                    # v is constant along the trajectory
    x_s       = x_t + (s - t)·v_t
              = (s/t)·x_t  +  ((t-s)/t)·x_0_pred

At ``s = 0`` this returns ``x_0_pred`` (the network's final clean
estimate). The 1/t factor is the reason we cap ``t ≥ eps_t`` at
training time — gradients through 1/t at very small t are noisy.

Why x_0-parameterisation and not velocity-pred
----------------------------------------------
LUNA's backbones (both the transformer and the EGNN) output
"predicted clean positions" directly, and ``LossFunction`` consumes
those positions in every loss component. Switching to v-pred would
force a rewrite of every loss term. x_0-pred is mathematically
equivalent (just a re-parameterisation of the same vector field) and
keeps the rest of the pipeline 100% framework-agnostic.
"""

from __future__ import annotations

from typing import Optional

import torch
import wandb

from utils.data.dataholder import DataHolder
from utils.data.load import remove_mean_with_mask


class FlowMatchingModel:
    """x_0-parameterised rectified flow with linear interpolation
    schedule. API-compatible with ``NoiseModel`` (DDPM)."""

    def __init__(self, cfg):
        # Read FM-specific knobs with safe defaults. The cfg block
        # is declared in ``configs/model/default.yaml`` so this lookup
        # never KeyErrors under composition; ``getattr`` with fallback
        # is belt-and-braces for older saved configs.
        fm_cfg = getattr(cfg.model, "flow_matching", None)
        if fm_cfg is None:
            self.n_sampling_steps = 50
            self.eps_t = 1.0e-3
        else:
            self.n_sampling_steps = int(getattr(fm_cfg, "n_sampling_steps", 50))
            self.eps_t = float(getattr(fm_cfg, "eps_t", 1.0e-3))

        # ``max_diffusion_steps`` is the attribute the sample loop in
        # utils/diffusion_model/sample/sample.py uses to size the
        # outer ``reversed(range(0, max_diffusion_steps))`` iteration.
        # For DDPM that's 1000; for FM we want the ODE step count
        # (e.g. 50). The Lightning module reads its own copy of this
        # from ``cfg.model.diffusion_steps`` directly, so we ALSO
        # rewrite it on the module from the FM branch of
        # FullDenoisingDiffusion.__init__ — see diffusion_model.py.
        self.max_diffusion_steps = self.n_sampling_steps

        # Carried for parity with NoiseModel; not used in FM math.
        self.noise_schedule = "rectified_flow_linear"

    # ------------------------------------------------------------------
    # Training-time perturbation
    # ------------------------------------------------------------------
    def apply_noise(self, data: DataHolder, train_flag: bool = True) -> DataHolder:
        """Sample one ``t`` per slice, draw Gaussian ``x_1``, return the
        linear-interpolant point ``x_t = (1-t)·x_0 + t·x_1``.

        Mirrors ``NoiseModel.apply_noise`` in signature and return
        type so training_step_func is framework-agnostic.
        """
        B = data.node_features.size(0)
        device = data.node_features.device

        # t ∼ 𝒰(eps_t, 1), one per slice. We cap below at eps_t so the
        # 1/t factor in the sampling-time Euler step (and equivalently
        # the velocity = (x_t - x_0)/t target) stays bounded — the
        # standard rectified-flow workaround for the singular limit.
        u = torch.rand(B, 1, device=device)
        t_float = self.eps_t + (1.0 - self.eps_t) * u                   # (B, 1)

        # Integer time bookkeeping — the DataHolder schema carries a
        # ``t_int`` slot. The training loop doesn't actually use it
        # in FM (the loss is on positions only) but downstream code
        # may peek, so we keep the field populated and consistent.
        t_int = torch.round(t_float * self.max_diffusion_steps).long()  # (B, 1)

        if wandb.run is not None and train_flag:
            wandb.log({"fm_t/histogram": wandb.Histogram(t_float[0].cpu().numpy())})

        # x_1 ∼ 𝒩(0, I), masked and mean-subtracted per slice (same
        # convention LUNA's NoiseModel uses for its diffusion noise —
        # keeps the centroid at the origin, consistent with the
        # backbone's translation-invariance assumption).
        noise_pos = torch.randn(data.positions.shape, device=device)
        noise_positions_masked = noise_pos * data.node_mask.unsqueeze(-1)
        x_1 = remove_mean_with_mask(
            x=noise_positions_masked, node_mask=data.node_mask
        )

        # Linear interpolation. t_float is (B, 1); broadcast to
        # (B, 1, 1) for per-slice scalar multiplication against
        # positions of shape (B, N, 2).
        t_b = t_float.unsqueeze(-1)                                     # (B, 1, 1)
        pos_t = (1.0 - t_b) * data.positions + t_b * x_1

        z_t = DataHolder(
            node_features=data.node_features,
            positions=pos_t,
            cell_class=data.cell_class,
            node_mask=data.node_mask,
            t_int=t_int,
            t=t_float,
            diffusion_time=t_float,
        ).mask()
        return z_t

    # ------------------------------------------------------------------
    # Inference start point: pure noise at t = 1
    # ------------------------------------------------------------------
    def sample_limit_dist(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        cell_ID: torch.Tensor,
        cell_class: torch.Tensor,
    ) -> DataHolder:
        """Initialise the ODE at ``t = 1``: positions ∼ 𝒩(0, I),
        masked + mean-subtracted. Matches NoiseModel.sample_limit_dist
        signature so ``sample.sample_noise`` works framework-agnostically.
        """
        B, N = node_mask.shape
        device = node_mask.device

        positions = torch.randn(B, N, 2, device=device)
        positions = positions * node_mask.unsqueeze(-1)
        positions = remove_mean_with_mask(positions, node_mask)

        t_array = positions.new_ones((B, 1))                            # t = 1
        t_int_array = (self.max_diffusion_steps * t_array).long()       # = n_sampling_steps

        result = DataHolder(
            node_features=node_features,
            positions=positions,
            node_mask=node_mask,
            cell_class=cell_class,
            cell_ID=cell_ID,
            t_int=t_int_array,
            t=t_array,
            diffusion_time=t_array,
        )
        return result.mask()

    # ------------------------------------------------------------------
    # ODE step: z_t (current) + x_0_pred  ->  z_s (next, with s < t)
    # ------------------------------------------------------------------
    def sample_zs_from_zt_and_pred(
        self,
        z_t: DataHolder,
        pred: DataHolder,
        s_int: torch.Tensor,
    ) -> DataHolder:
        """One Euler step of the rectified-flow ODE.

        ``z_t.t`` is the current (source) time in ``(0, 1]``; ``s_int``
        is the integer index of the target step (in
        ``[0, n_sampling_steps)``). We compute the target time
        ``s = s_int / n_sampling_steps`` and Euler-step::

            x_s = (s/t)·z_t + ((t-s)/t)·x_0_pred

        which equals ``x_0_pred`` at ``s = 0`` (final clean estimate).

        For the linear interpolation schedule the velocity field is
        constant along each trajectory, so one Euler step is exact
        between any two times — the only source of step error is the
        network's ``x_0_pred`` changing slightly as we move along the
        ODE. ~50 steps is canonical; fewer (≥20) usually fine.
        """
        node_mask = z_t.node_mask

        # Source/target times. We read t from z_t.t (the continuous
        # value the previous step wrote) rather than re-deriving it
        # from z_t.t_int — avoids round-off drift over many steps.
        t = z_t.t                                                       # (B, 1), in (0, 1]
        s_float = s_int.float() / float(self.max_diffusion_steps)       # scalar / broadcast

        # Guard against t = 0 (shouldn't happen if the sampling loop
        # is set up correctly, but defensive — the 1/t in the Euler
        # step is the singular point).
        t_safe = t.clamp_min(self.eps_t)

        # Reshape time scalars to (B, 1, 1) for broadcasting over
        # positions of shape (B, N, 2).
        t_b = t_safe.unsqueeze(-1)
        s_b = s_float.unsqueeze(-1) if s_float.dim() == 2 else s_float.view(-1, 1, 1)

        # Euler step. No noise added — flow matching is fully
        # deterministic at inference (unlike DDPM, which injects
        # fresh Gaussian noise at every step).
        ratio_t = s_b / t_b                                             # (B, 1, 1)
        ratio_pred = (t_b - s_b) / t_b                                  # (B, 1, 1)
        positions = ratio_t * z_t.positions + ratio_pred * pred.positions

        # Re-mask + re-center to keep padding cells at the origin and
        # the real-cell centroid at zero (matches NoiseModel's
        # convention; the loss is translation-invariant but the
        # backbone expects centered inputs).
        positions = positions * node_mask.unsqueeze(-1)
        positions = remove_mean_with_mask(positions, node_mask)

        # Write the new (continuous) source time and its integer
        # surrogate into the returned DataHolder so the next loop
        # iteration reads the right t.
        if s_float.dim() == 2:
            new_t = s_float
        else:
            # s_int came in as a global scalar; broadcast to (B, 1).
            new_t = s_float.view(1, 1).expand(node_mask.size(0), 1).contiguous()

        z_s = DataHolder(
            node_features=z_t.node_features,
            positions=positions,
            node_mask=node_mask,
            t_int=s_int if s_int.dim() == 2 else s_int.view(1, 1).expand_as(new_t),
            t=new_t,
            diffusion_time=new_t,
        ).mask()
        return z_s

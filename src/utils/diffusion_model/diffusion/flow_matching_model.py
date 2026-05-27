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

import pickle
from pathlib import Path
from typing import Dict, Optional

import torch
import wandb

from utils.data.dataholder import DataHolder
from utils.data.load import remove_mean_with_mask


# ---------------------------------------------------------------------------
# Empirical-GMM prior helpers (rotation-averaged, per-cell-type)
# ---------------------------------------------------------------------------


def _load_prior_pool(path: str) -> dict:
    """Load the prior pool pickle produced by
    ``scripts/precompute_prior_pool.py``. Pre-computes per-class
    Cholesky factors of the covariance matrices so sampling is just
    a matmul + standard-normal draw.

    Returns a dict with three keys:
      * ``gmms``: ``{cell_class_int → {means_t, chols_t, weights_t, K}}``
        with `means_t`, `chols_t`, `weights_t` as CPU torch tensors
        (moved to device on first sample call).
      * ``default_scale`` (float): the median per-slice bbox-diag
        scale from the training set. Used at inference when we don't
        have x_0 to compute a slice-specific scale.
      * ``meta`` (dict): provenance for debugging.
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)
    out_gmms: Dict[int, dict] = {}
    for cls_int, gmm in raw["gmms"].items():
        means = torch.from_numpy(gmm["means"]).float()              # (K, 2)
        covs  = torch.from_numpy(gmm["covariances"]).float()        # (K, 2, 2)
        weights = torch.from_numpy(gmm["weights"]).float()          # (K,)
        # Precompute Cholesky once. Add a tiny diagonal jitter so the
        # cholesky never fails on a numerically-singular covariance.
        eye = torch.eye(2).unsqueeze(0).expand_as(covs)
        chols = torch.linalg.cholesky(covs + 1.0e-6 * eye)          # (K, 2, 2)
        out_gmms[int(cls_int)] = dict(
            means_t=means, chols_t=chols, weights_t=weights, K=gmm["K"],
        )
    return dict(
        gmms=out_gmms,
        default_scale=float(raw["default_scale"]),
        meta=raw.get("meta", {}),
    )


def _sample_x1_from_gmm_pool(
    cell_class: torch.Tensor,
    node_mask: torch.Tensor,
    pool: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample one position per cell from the per-cell-type GMM pool.

    Args:
        cell_class: ``(B, N)`` or ``(B, N, 1)`` int tensor of cell-class
            indices. Values matching keys in ``pool['gmms']`` get sampled
            from their per-type GMM; unmatched values fall back to
            ``N(0, I)``.
        node_mask: ``(B, N)`` bool, True for real cells. Padding cells
            get exactly ``(0, 0)`` (matches the existing ``noise * mask``
            convention).
        pool: as returned by ``_load_prior_pool``.
        device, dtype: target tensor properties.

    Returns:
        ``(B, N, 2)`` float tensor of sampled positions in NORMALIZED
        scale (so each component sits in roughly ``[-1, 1]²``). The
        caller is responsible for rescaling to slice-specific or
        default scale.
    """
    if cell_class.dim() == 3:
        cell_class = cell_class.squeeze(-1)
    B, N = cell_class.shape
    out = torch.zeros(B, N, 2, device=device, dtype=dtype)

    # Iterate unique classes in this batch — at most a few per batch,
    # so the loop is cheap. Each iteration vectorizes over all cells of
    # that class.
    unique_cls = cell_class.unique().tolist()
    for cls_int in unique_cls:
        mask = (cell_class == cls_int) & node_mask                   # (B, N)
        n_cells = int(mask.sum().item())
        if n_cells == 0:
            continue
        gmm = pool["gmms"].get(int(cls_int))
        if gmm is None:
            # Cell class not in the pool (rare — e.g., a class present
            # in test but not train). Fall back to N(0, I) for these
            # cells. Logged once-per-call would be noisy; instead we
            # rely on the precompute script's verbosity to surface
            # missing classes.
            samples = torch.randn(n_cells, 2, device=device, dtype=dtype)
        else:
            means = gmm["means_t"].to(device=device, dtype=dtype)    # (K, 2)
            chols = gmm["chols_t"].to(device=device, dtype=dtype)    # (K, 2, 2)
            weights = gmm["weights_t"].to(device=device, dtype=dtype)  # (K,)
            # Sample mixture-component indices.
            comp_idx = torch.multinomial(
                weights, n_cells, replacement=True,
            )                                                         # (n_cells,)
            mean_per_cell = means.index_select(0, comp_idx)          # (n_cells, 2)
            chol_per_cell = chols.index_select(0, comp_idx)          # (n_cells, 2, 2)
            z = torch.randn(n_cells, 2, device=device, dtype=dtype)
            # mean + chol @ z, batched.
            samples = mean_per_cell + torch.einsum(
                "ncd,nd->nc", chol_per_cell, z,
            )
        out[mask] = samples
    return out


def _per_slice_scale_from_x0(
    positions: torch.Tensor, node_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute half-bbox-diagonal per slice from x_0 (training-time).

    Returns ``(B, 1, 1)`` so it broadcasts against ``(B, N, 2)`` x_1.
    Padding cells (mask=False) are excluded from the bbox calculation.
    Slices with <2 valid cells fall back to scale = 1.0.
    """
    B = positions.shape[0]
    scales = positions.new_ones((B, 1, 1))
    for b in range(B):
        m = node_mask[b]
        pos = positions[b][m]
        if pos.shape[0] < 2:
            continue
        bbox = pos.max(dim=0).values - pos.min(dim=0).values
        scales[b, 0, 0] = 0.5 * bbox.norm().clamp_min(1.0e-8)
    return scales


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
            self.sampler = "euler"
            self.prediction = "x0"
        else:
            self.n_sampling_steps = int(getattr(fm_cfg, "n_sampling_steps", 50))
            self.eps_t = float(getattr(fm_cfg, "eps_t", 1.0e-3))
            self.sampler = str(getattr(fm_cfg, "sampler", "euler")).lower()
            self.prediction = str(getattr(fm_cfg, "prediction", "x0")).lower()
        if self.sampler not in ("euler", "heun"):
            raise ValueError(
                f"Unknown model.flow_matching.sampler={self.sampler!r}. "
                f"Expected 'euler' or 'heun'."
            )
        if self.prediction not in ("x0", "v"):
            raise ValueError(
                f"Unknown model.flow_matching.prediction={self.prediction!r}. "
                f"Expected 'x0' or 'v'."
            )

        # ---- Optional empirical-GMM prior ----------------------------
        # When ``prior_mode == "empirical_gmm"``, ``x_1`` (the FM noise
        # endpoint) is sampled from a per-cell-type Gaussian Mixture
        # fit OFFLINE on the training set by
        # ``scripts/precompute_prior_pool.py``. The GMM is fit on
        # per-slice-normalized but NOT rotated positions, so the
        # prior is rotation-invariant in distribution.
        #
        # Default ``"gaussian"`` keeps the byte-identical legacy
        # behavior (``x_1 ~ N(0, I)``).
        self.prior_mode = "gaussian"
        self.prior_pool: Optional[dict] = None
        if fm_cfg is not None:
            mode = str(getattr(fm_cfg, "prior_mode", "gaussian")).lower()
            if mode not in ("gaussian", "empirical_gmm"):
                raise ValueError(
                    f"Unknown model.flow_matching.prior_mode={mode!r}. "
                    f"Expected 'gaussian' or 'empirical_gmm'."
                )
            self.prior_mode = mode
            if mode == "empirical_gmm":
                pool_path = getattr(fm_cfg, "prior_pool_path", None)
                if not pool_path:
                    raise ValueError(
                        "model.flow_matching.prior_mode='empirical_gmm' "
                        "requires model.flow_matching.prior_pool_path to "
                        "point at the .pkl produced by "
                        "scripts/precompute_prior_pool.py."
                    )
                if not Path(pool_path).is_file():
                    raise FileNotFoundError(
                        f"prior_pool_path={pool_path!r} does not exist. "
                        f"Run scripts/precompute_prior_pool.py to "
                        f"generate it."
                    )
                self.prior_pool = _load_prior_pool(pool_path)
                print(
                    f"[FlowMatchingModel] empirical_gmm prior loaded: "
                    f"{len(self.prior_pool['gmms'])} cell-class GMMs, "
                    f"default_scale={self.prior_pool['default_scale']:.3g}"
                )

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

        # Sample x_1 from the configured prior. Default "gaussian" is
        # byte-identical to the legacy behavior (N(0, I), masked,
        # mean-subtracted per slice — same as LUNA's NoiseModel).
        # "empirical_gmm" samples from the per-cell-type GMM pool
        # produced by precompute_prior_pool.py, then rescales to each
        # slice's bounding-box scale (computed from x_0 at training).
        if self.prior_mode == "empirical_gmm" and self.prior_pool is not None:
            # Sample in normalized scale (roughly [-1, 1]² per cell).
            x_1_norm = _sample_x1_from_gmm_pool(
                cell_class=data.cell_class,
                node_mask=data.node_mask,
                pool=self.prior_pool,
                device=device,
                dtype=data.positions.dtype,
            )
            # Scale per slice using x_0's bounding-box diagonal — at
            # training we know the slice's natural scale from x_0.
            slice_scale = _per_slice_scale_from_x0(
                positions=data.positions, node_mask=data.node_mask,
            )                                                       # (B, 1, 1)
            noise_positions_masked = (
                x_1_norm * slice_scale
            ) * data.node_mask.unsqueeze(-1)
        else:
            # Legacy Gaussian prior.
            noise_pos = torch.randn(data.positions.shape, device=device)
            noise_positions_masked = noise_pos * data.node_mask.unsqueeze(-1)

        # Mean-subtract per slice — required regardless of prior so that
        # the FM endpoint's centroid sits at the origin (matches x_0,
        # which is also mean-centered by the data pipeline). With the
        # empirical-GMM prior this also normalises away any global mean
        # the GMM happens to put on a cell-type-uneven sample.
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

        # Same branching as apply_noise. At inference we don't have x_0
        # so we use the train-median bbox-diag scale (``default_scale``)
        # from the prior pool — a single global rescaling factor.
        if self.prior_mode == "empirical_gmm" and self.prior_pool is not None:
            positions = _sample_x1_from_gmm_pool(
                cell_class=cell_class,
                node_mask=node_mask,
                pool=self.prior_pool,
                device=device,
                dtype=torch.float32,
            )
            positions = positions * float(self.prior_pool["default_scale"])
        else:
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
            # Preserve cell-class / cell-ID across the sampling chain.
            # Backbones can condition on cell_class; without these
            # passed through, every step after the first one passes
            # None and the conditioning silently disappears.
            # EDM-FM and LDM step functions already forward these;
            # the FM path was inconsistent and missed them.
            cell_class=getattr(z_t, "cell_class", None),
            cell_ID=getattr(z_t, "cell_ID", None),
            t_int=s_int if s_int.dim() == 2 else s_int.view(1, 1).expand_as(new_t),
            t=new_t,
            diffusion_time=new_t,
        ).mask()
        return z_s

    # ------------------------------------------------------------------
    # Heun (2nd-order / "improved Euler") correction
    # ------------------------------------------------------------------
    def heun_correction(
        self,
        z_t: DataHolder,
        pred1: DataHolder,
        z_s_euler: DataHolder,
        pred2: DataHolder,
        s_int: torch.Tensor,
    ) -> DataHolder:
        """Apply the trapezoidal Heun correction on top of an Euler
        prestep.

        Called by ``sample.sample_zs_from_zt`` AFTER the standard
        single-forward Euler step has produced ``z_s_euler`` and a
        SECOND network forward has produced ``pred2`` at the Euler
        endpoint. We compute the velocity at both ends of the step
        and re-take the step with the averaged velocity::

            v₁ = (z_t        − pred1) / t                # at (z_t, t)
            v₂ = (z_s_euler  − pred2) / s                # at (z_s_euler, s)
            v̄  = ½·(v₁ + v₂)
            z_s = z_t − (t − s)·v̄

        Discretization error is O(dt²) vs Euler's O(dt). The 1/s
        factor diverges at the final step (s=0), so we fall back to
        the Euler result there — that's the canonical x_0_pred
        anyway, so the fallback is mathematically clean.

        Cost: this method itself is cheap; the dominant per-step
        cost is the SECOND ``self.forward(z_s_euler)`` call in
        ``sample.sample_zs_from_zt`` (2× forwards per step vs Euler's
        1×).
        """
        node_mask = z_t.node_mask

        t = z_t.t                                                       # (B, 1)
        s_float = s_int.float() / float(self.max_diffusion_steps)

        # Normalise s to the same (B, 1) shape as t for broadcasting.
        if s_float.dim() == 2:
            s_per_slice = s_float
        else:
            s_per_slice = s_float.view(1, 1).expand_as(t).contiguous()

        # If we're already at the final step (s ≤ eps), Heun's 1/s
        # factor is undefined — return the Euler result. This is the
        # canonical x_0_pred (Euler's last step IS the clean
        # prediction by construction).
        if (s_per_slice <= self.eps_t).all():
            return z_s_euler

        t_safe = t.clamp_min(self.eps_t).unsqueeze(-1)                  # (B, 1, 1)
        s_safe = s_per_slice.clamp_min(self.eps_t).unsqueeze(-1)        # (B, 1, 1)
        dt_b = (t_safe - s_safe)                                        # (B, 1, 1)

        # Velocity at both endpoints of the step.
        v1 = (z_t.positions - pred1.positions) / t_safe
        v2 = (z_s_euler.positions - pred2.positions) / s_safe
        v_avg = 0.5 * (v1 + v2)

        positions = z_t.positions - dt_b * v_avg
        positions = positions * node_mask.unsqueeze(-1)
        positions = remove_mean_with_mask(positions, node_mask)

        z_s = DataHolder(
            node_features=z_t.node_features,
            positions=positions,
            node_mask=node_mask,
            t_int=z_s_euler.t_int,
            t=z_s_euler.t,
            diffusion_time=z_s_euler.diffusion_time,
        ).mask()
        return z_s

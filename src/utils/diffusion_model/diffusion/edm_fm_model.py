"""True EDM diffusion (FM in h-space) — Architectural extension #4.

Implements the variant pitched in the May ablation menu: the
flow-matching trajectory operates in the k-D embedding space whose
pairwise distances ARE the Euclidean Distance Matrix, rather than
in 2D position space with EDM as the output projection.

Differences from the existing FlowMatchingModel
-----------------------------------------------
Current (FlowMatchingModel + EDMOutputWrapper):
    1. apply_noise: x_t = (1−t)·x_0 + t·noise in **2D**.
    2. forward: inner Model(x_t, gene, t) → pred.positions (2D);
       EDM head projects pred.node_features+pred.positions → h
       and stashes pred.edm_D = ‖h_i − h_j‖² for the loss.
    3. The trajectory variable is 2D position; the embedding h
       is only the output projection.

True EDM diffusion (this model):
    1. apply_noise: h_t = (1−t)·h_0 + t·noise in **k-D embedding
       space**, where h_0 = (true_positions, 0_pad) padded to k
       dimensions. We feed h_t[..., :2] as x_t to the inner Model
       (so the backbone sees noisy 2D positions just like in
       vanilla FM).
    2. forward: same backbone, same EDM head. pred.edm_h is now the
       FM denoising target.
    3. The trajectory variable is h ∈ R^k; the 2D positions are an
       output projection (MDS at sample end) of h.

Why
---
The current EDM-FM stack does FM in 2D + EDM projection at output.
That's FM with EDM-supervised output, not "diffusion on the EDM
manifold". This model is the proper version: the trajectory lives
in the embedding space whose pairwise distances are the EDM, the
denoising target is the embedding itself, and the 2D positions are
recovered exactly once at the end via MDS.

Interface
---------
Same three methods as ``FlowMatchingModel`` /
``RegressionPredictor`` so the training step and sampling loop
work without modification:
    * ``apply_noise(data)`` → DataHolder with positions = h_t[..., :2]
      AND ``h_t`` stashed as a private attribute for the sampling
      loop to read.
    * ``sample_limit_dist(...)`` → initial h_T ~ N(0,I) k-D, project
      to 2D positions via h[..., :2].
    * ``sample_zs_from_zt_and_pred(z_t, pred, s_int)`` → FM Euler
      step in h-space using pred.edm_h, then update z_t.positions to
      reflect the new h_t.

Constraints
-----------
* Requires ``cfg.model.edm.enabled = true`` (the EDM head produces
  the pred.edm_h that the FM Euler step consumes).
* Requires ``cfg.model.framework = "flow_matching"`` (this is
  selected by ``framework=flow_matching`` + the
  ``edm.diffuse_in_embed_space`` flag).
* The embed_dim parameterises the trajectory dim. ``edm.embed_dim=2``
  reduces to standard FM in 2D — useful as a debugging sanity check.
"""

from __future__ import annotations

from typing import Optional

import torch

from utils.data.dataholder import DataHolder


def _remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Inlined copy of ``utils.data.load.remove_mean_with_mask`` to
    avoid pulling scanpy through ``utils.data.load``'s module-level
    import — keeps this file importable from a venv that doesn't
    have scanpy (the lite test suite imports it that way).

    Subtracts the per-batch-element mean over the REAL cells from
    ``x`` (shape ``(B, N, D)``). The PAD-row safety assertions in
    the canonical helper are dropped here — apply_noise already
    masks ``x`` to zero on padding before calling this.
    """
    m = node_mask.unsqueeze(-1)                        # (B, N, 1)
    n = m.sum(dim=1, keepdim=True).clamp_min(1)        # (B, 1, 1)
    mean = (x * m).sum(dim=1, keepdim=True) / n        # (B, 1, D)
    return (x - mean) * m                              # zero at PAD


class EDMFlowMatchingModel:
    """Drop-in replacement for ``FlowMatchingModel`` that operates in
    the k-D embedding space whose pairwise distances are the EDM.

    Selected by setting ``cfg.model.edm.diffuse_in_embed_space=true``
    in addition to ``cfg.model.framework="flow_matching"`` and
    ``cfg.model.edm.enabled=true``. The Lightning module routes the
    framework dispatch through ``EDMFlowMatchingModel`` when all
    three are set.
    """

    def __init__(self, cfg):
        # ----- Embed dimension (= trajectory dim) -----
        edm_cfg = getattr(cfg.model, "edm", None)
        if edm_cfg is None or not bool(getattr(edm_cfg, "enabled", False)):
            raise ValueError(
                "EDMFlowMatchingModel requires cfg.model.edm.enabled=true. "
                "h-space FM presupposes the EDM head produces pred.edm_h "
                "as the denoising target."
            )
        self.embed_dim = int(getattr(edm_cfg, "embed_dim", 8))
        if self.embed_dim < 2:
            raise ValueError(
                f"edm.embed_dim must be >= 2 (got {self.embed_dim}); "
                f"the first 2 dimensions are reserved for the 2D position "
                f"alias fed into the inner backbone."
            )

        # ----- FM hyperparameters (re-using cfg.model.flow_matching) -----
        fm_cfg = getattr(cfg.model, "flow_matching", None)
        self.n_sampling_steps = int(
            getattr(fm_cfg, "n_sampling_steps", 50)
        ) if fm_cfg is not None else 50
        self.eps_t = float(
            getattr(fm_cfg, "eps_t", 1.0e-3)
        ) if fm_cfg is not None else 1.0e-3
        # h-space FM only supports x0 prediction. v-prediction would
        # need conversion math written for h-space; not implemented.
        # (The Lightning module already raises a clear error when
        # edm.enabled=true + flow_matching.prediction=v, so we'd hit
        # that guard before reaching here in practice.)
        self.prediction = "x0"
        # No 2nd-order sampler in this model — we use Euler only.
        # Adding Heun for h-space is straightforward (same math) but
        # we defer for the first revision.
        self.sampler = "euler"
        self.noise_schedule = "linear"   # informational, not used
        self.max_diffusion_steps = self.n_sampling_steps

    # ------------------------------------------------------------------
    # Helper: lift true positions to k-D h_0 by zero-padding.
    # ------------------------------------------------------------------
    def _lift_to_embed_space(self, positions_2d: torch.Tensor) -> torch.Tensor:
        """Embed (B, N, 2) true positions into (B, N, embed_dim) by
        zero-padding the extra dimensions.

        Why zero-padding works for the EDM loss:
            ‖h_0_i − h_0_j‖² = ‖(true_pos_i − true_pos_j, 0..0)‖²
                              = ‖true_pos_i − true_pos_j‖²
        i.e. pairwise distances in the padded h-space equal pairwise
        distances of the true 2D positions. The model has to learn
        to push the padded dimensions toward zero on its predicted
        h_0; the EDM head's pred.edm_h has room (k > 2) but the
        information content is 2D.
        """
        B, N, _ = positions_2d.shape
        k = self.embed_dim
        if k == 2:
            return positions_2d
        padding = torch.zeros(
            B, N, k - 2,
            device=positions_2d.device,
            dtype=positions_2d.dtype,
        )
        return torch.cat([positions_2d, padding], dim=-1)

    # ------------------------------------------------------------------
    # Training-time noising: sample t, build h_t, hand 2D alias to backbone
    # ------------------------------------------------------------------
    def apply_noise(
        self, data: DataHolder, train_flag: bool = True,
    ) -> DataHolder:
        """Sample a random FM time t and perturb h_0 (the lifted true
        positions) to h_t. The 2D alias h_t[..., :2] is what the
        inner backbone consumes as ``data.positions``.

        Two private attributes are stashed on the returned DataHolder
        for the EDM-FM machinery downstream:
            * ``_edm_h_t``   — the full k-D noisy embedding (B, N, k)
            * ``_edm_h_0``   — the lifted ground-truth h_0 (B, N, k);
              used by the loss path for h-space MSE if you ever want
              it. The current default loss path reads pred.edm_D, so
              this attribute is just there for future use.
        """
        B = data.node_features.size(0)
        device = data.node_features.device
        dtype = data.positions.dtype

        # h_0 = lift true positions to k-D.
        h_0 = self._lift_to_embed_space(data.positions)               # (B, N, k)

        # Sample t in [eps_t, 1] uniformly per slice.
        t = torch.rand(B, 1, device=device, dtype=dtype)
        t = self.eps_t + (1.0 - self.eps_t) * t                       # (B, 1)

        # Sample iid Gaussian noise in k-D. Mask to zero on padding so
        # h_t at padding cells stays zero and pairwise distances at
        # those cells remain zero (mirrors the standard FM behaviour).
        noise = torch.randn(B, h_0.shape[1], self.embed_dim,
                            device=device, dtype=dtype)
        mask = data.node_mask.unsqueeze(-1).to(dtype)
        noise = noise * mask
        # Mean-remove the noise per slice over the REAL cells, so the
        # FM trajectory's centroid stays at zero — matches the
        # FlowMatchingModel convention (which removes x_1's mean,
        # i.e. the noise endpoint). Without this the EDM-FM
        # trajectory's centroid drifts O(1/sqrt(N)) per slice and
        # the backbone has to learn to recover a random shift the
        # standard FM model never saw. Apply to all k channels so
        # the higher dims of h_t (used by the EDM head) are also
        # centroid-free.
        noise = _remove_mean_with_mask(noise, data.node_mask)

        # h_t = (1-t)·h_0 + t·noise.
        t_b = t.unsqueeze(-1)                                          # (B, 1, 1)
        h_t = (1.0 - t_b) * h_0 + t_b * noise                          # (B, N, k)
        h_t = h_t * mask

        # 2D alias for the inner backbone: h_t[..., :2]. We don't run
        # MDS here — feeding h_t[..., :2] is the cheap, correct-on-
        # expectation choice (h_0[..., :2] = true positions, so
        # h_t[..., :2] = (1-t)·true + t·noise[..., :2] which is the
        # same trajectory the existing FM gives the backbone).
        x_t = h_t[..., :2]                                             # (B, N, 2)

        # diffusion_time scalars — mirror FlowMatchingModel's choices.
        t_int = (t * 1000.0).long()                                    # (B, 1)

        out = DataHolder(
            node_features=data.node_features,
            positions=x_t,
            diffusion_time=t,
            cell_class=data.cell_class,
            cell_ID=getattr(data, "cell_ID", None),
            t_int=t_int,
            t=t,
            node_mask=data.node_mask,
        )
        # Stash the h-space state so sampling and (potentially) the
        # loss can read it. DataHolder is a plain Python object so
        # arbitrary attributes attach cleanly.
        out._edm_h_t = h_t
        out._edm_h_0 = h_0
        return out.mask()

    # ------------------------------------------------------------------
    # Inference start: h_T ~ N(0, I) in k-D; 2D alias as backbone input
    # ------------------------------------------------------------------
    def sample_limit_dist(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        cell_ID: torch.Tensor,
        cell_class: torch.Tensor,
    ) -> DataHolder:
        """Initial state of the reverse trajectory: h_T ~ N(0, I) in
        k-D, masked to zero on padding. The 2D alias h_T[..., :2] is
        the backbone's input at the first sampling step.
        """
        B, N = node_mask.shape
        device = node_features.device
        dtype = node_features.dtype if node_features.is_floating_point() else torch.float32

        h_T = torch.randn(B, N, self.embed_dim, device=device, dtype=dtype)
        mask = node_mask.unsqueeze(-1).to(dtype)
        h_T = h_T * mask
        # Mean-remove so the reverse trajectory starts on the same
        # centroid manifold the apply_noise endpoint lives on. Same
        # rationale as in apply_noise above.
        h_T = _remove_mean_with_mask(h_T, node_mask)
        x_T = h_T[..., :2]                                             # (B, N, 2)

        # t=1 at the start of the reverse trajectory.
        t = torch.ones(B, 1, device=device, dtype=dtype)
        t_int = (t * 1000.0).long()

        out = DataHolder(
            node_features=node_features,
            positions=x_T,
            diffusion_time=t,
            cell_class=cell_class,
            cell_ID=cell_ID,
            t_int=t_int,
            t=t,
            node_mask=node_mask,
        )
        out._edm_h_t = h_T
        return out.mask()

    # ------------------------------------------------------------------
    # FM Euler step in h-space
    # ------------------------------------------------------------------
    def sample_zs_from_zt_and_pred(
        self,
        z_t: DataHolder,
        pred: DataHolder,
        s_int: torch.Tensor,
    ) -> DataHolder:
        """One reverse Euler step in h-space.

        Reads:
            * z_t._edm_h_t        — current h_t
            * pred.edm_h          — model's predicted h_0 (set by the
              EDMOutputWrapper). REQUIRED — without it we can't do
              the h-space update.

        Writes:
            * new z_s.positions   — 2D alias h_s[..., :2] for the next
              backbone forward pass
            * z_s._edm_h_t        — new h-space state for the step
              after this one
        """
        h_t = getattr(z_t, "_edm_h_t", None)
        if h_t is None:
            raise RuntimeError(
                "EDMFlowMatchingModel.sample_zs_from_zt_and_pred requires "
                "z_t._edm_h_t to be set. Sampling chain not started via "
                "sample_limit_dist? Or somewhere the DataHolder was "
                "rebuilt without preserving the attribute."
            )
        h0_pred = getattr(pred, "edm_h", None)
        if h0_pred is None:
            raise RuntimeError(
                "EDMFlowMatchingModel.sample_zs_from_zt_and_pred requires "
                "pred.edm_h to be set by the EDMOutputWrapper. Check that "
                "cfg.model.edm.enabled=true."
            )

        # Current and next FM time. ``s_int`` is the next discrete
        # step index in [0, n_sampling_steps); the time at step k is
        # k / n_sampling_steps.
        n = self.n_sampling_steps
        t_val = z_t.t                                                  # (B, 1)
        # s_int may arrive as shape (), (1, 1), or (B, 1); normalise.
        if s_int.dim() == 0:
            s_int_b = s_int.view(1, 1).expand(t_val.size(0), 1)
        elif s_int.dim() == 1:
            s_int_b = s_int.view(-1, 1)
        else:
            s_int_b = s_int.view(t_val.size(0), 1)
        s_val = s_int_b.to(t_val.dtype) / float(n)                     # (B, 1)
        t_b = t_val.unsqueeze(-1)                                      # (B, 1, 1)
        s_b = s_val.unsqueeze(-1)                                      # (B, 1, 1)

        # FM x0-pred Euler step (linear schedule). Equivalent to
        #     h_s = h_t + ((s - t) / t) · (h_t − h_0_pred)
        # but written in the same convex-combination form as the
        # existing FlowMatchingModel for byte-equivalence at k=2:
        #     h_s = (s/t) · h_t + ((t - s)/t) · h_0_pred
        # Derivation: from x_t = (1-t)·x_0 + t·x_1 the velocity is
        # v = (x_t - x_0)/t, so backward Euler x_s = x_t + (s-t)·v
        # gives the same expression after one line of algebra. At
        # s=0 this reduces to h_s = h_0_pred (the prediction *is*
        # the answer at the last step), and at s=t it reduces to
        # h_s = h_t (no-op step).
        t_safe = t_b.clamp_min(self.eps_t)
        ratio_t = s_b / t_safe
        ratio_pred = (t_b - s_b) / t_safe
        h_s = ratio_t * h_t + ratio_pred * h0_pred

        # New state. Mask + 2D alias for the next backbone forward.
        mask = z_t.node_mask.unsqueeze(-1).to(h_s.dtype)
        h_s = h_s * mask
        x_s = h_s[..., :2]                                              # (B, N, 2)
        t_int_s = (s_val * 1000.0).long()

        z_s = DataHolder(
            node_features=z_t.node_features,
            positions=x_s,
            diffusion_time=s_val,
            cell_class=z_t.cell_class,
            cell_ID=getattr(z_t, "cell_ID", None),
            t_int=t_int_s,
            t=s_val,
            node_mask=z_t.node_mask,
        )
        z_s._edm_h_t = h_s
        return z_s.mask()

"""Latent Diffusion noise model — Architectural extension #5 (full).

Mirrors the ``FlowMatchingModel`` / ``RegressionPredictor`` interface
so the training step and sampling loop work with minimal modification.
The framework lives in latent space throughout:

  Training (apply_noise):
      mu, logvar = encoder(gene, true_positions)
      z_0 = reparam(mu, logvar)
      z_t = (1 − t) · z_0 + t · noise
      ... and stash z_0, mu, logvar on the returned DataHolder so
      the loss path (latent_fm_mse + latent_kl) can read them.

  Inference start (sample_limit_dist):
      z_T = N(0, I)
      decode(z_T) for a "visualisable" 2D x_T (optional;
      LightningModule reads only the latent state through z_t.t and
      data._ldm_z_t).

  Reverse FM step (sample_zs_from_zt_and_pred):
      Standard linear-FM Euler:
          z_s = (s/t) · z_t + ((t − s)/t) · z_0_pred
      where z_0_pred is the denoiser's prediction (stashed on
      ``pred._ldm_z_0_pred`` by ``LatentDiffusionWrapper``).

This noise model **requires** that ``self.model`` is the
``LatentDiffusionWrapper`` — it relies on the wrapper's encoder
during apply_noise and the wrapper's decoder during sampling. The
LightningModule's framework dispatch guarantees that pairing.

Joint training caveat
---------------------
The encoder is unfrozen during diffusion training. That's the
"single-phase joint training" choice we committed to — simpler than
two-phase training but with a chicken-and-egg: the diffusion model
is learning to denoise a distribution that's also moving as the
encoder trains. In practice this is stable enough on cortex-scale
data with the KL prior pulling the encoder's outputs toward N(0,I).
Two-phase training (freeze encoder after a pretrain warmup) is a
follow-up enhancement; the current design doesn't preclude it.
"""

from __future__ import annotations

from typing import Optional

import torch

from utils.data.dataholder import DataHolder


class LatentDiffusionModel:
    """Latent-space flow-matching noise model. Pairs with
    ``models.latent_diffusion_wrapper.LatentDiffusionWrapper`` as
    ``self.model`` in the LightningModule.
    """

    def __init__(self, cfg):
        # Latent dim — must match what the LatentDiffusionWrapper was
        # constructed with. Both pull from cfg.model.latent_diffusion.
        ldm_cfg = getattr(cfg.model, "latent_diffusion", None)
        if ldm_cfg is None:
            raise ValueError(
                "framework=latent_diffusion requires the "
                "cfg.model.latent_diffusion block to be present in the "
                "model config. (Should be there by default; check that "
                "your overrides haven't deleted it.)"
            )
        self.latent_dim = int(getattr(ldm_cfg, "latent_dim", 128))
        # Borrow the FM hyperparameters — eps_t for time-sampling
        # stability, n_sampling_steps for the reverse trajectory.
        self.n_sampling_steps = int(getattr(ldm_cfg, "n_diffusion_steps", 50))
        self.eps_t = float(getattr(ldm_cfg, "epsilon_t", 1.0e-3))
        # Mirror FlowMatchingModel attributes the training/sampling
        # loop reads.
        self.sampler = "euler"
        self.noise_schedule = "linear"
        self.max_diffusion_steps = self.n_sampling_steps
        self.prediction = "x0"
        # The wrapper is set on the LightningModule after this
        # noise-model is built — we look it up lazily via the
        # ``_ldm_wrapper`` attribute the LightningModule attaches to
        # the noise model after both are constructed.
        self._ldm_wrapper: Optional["torch.nn.Module"] = None

    # ------------------------------------------------------------------
    # Helper: pull the wrapper's encoder / decoder
    # ------------------------------------------------------------------
    def _require_wrapper(self):
        if self._ldm_wrapper is None:
            raise RuntimeError(
                "LatentDiffusionModel.apply_noise / sample_limit_dist "
                "requires _ldm_wrapper to be set on the noise model. "
                "The LightningModule should do this after constructing "
                "both self.model (the wrapper) and self.noise_model "
                "(this object). If you're invoking apply_noise outside "
                "the standard LightningModule chain, set "
                "noise_model._ldm_wrapper = your_wrapper manually."
            )
        return self._ldm_wrapper

    # ------------------------------------------------------------------
    # Training-time noising
    # ------------------------------------------------------------------
    def apply_noise(
        self, data: DataHolder, train_flag: bool = True,
    ) -> DataHolder:
        """Encode the true positions, noise the latent at FM time t.

        Stashes on the returned DataHolder:
            _ldm_z_t       — (B, N, k) noisy latent (FM input to forward)
            _ldm_z_0_target— (B, N, k) clean latent sample (FM target)
            _ldm_mu,
            _ldm_logvar    — (B, N, k) encoder outputs (KL target)

        Also writes a "visualisable" 2D position into data.positions
        — the decoder's output for z_t — so debug logging / plot
        helpers reading data.positions at training time see a sensible
        2D state.
        """
        wrap = self._require_wrapper()

        # Use whatever device/dtype the input is on.
        device = data.node_features.device
        B = data.node_features.size(0)
        dtype = (
            data.positions.dtype
            if data.positions.is_floating_point()
            else torch.float32
        )

        # Encode true positions.
        mu, logvar = wrap.encode(
            data.node_features, data.positions, data.node_mask,
        )
        # Reparameterised z_0 target. The reparam ensures gradient
        # flows back through the encoder into the KL + FM losses.
        from models.latent_vae import reparameterize
        z_0_target = reparameterize(mu, logvar)               # (B, N, k)

        # Sample FM time per slice. Same convention as FlowMatchingModel.
        t = torch.rand(B, 1, device=device, dtype=dtype)
        t = self.eps_t + (1.0 - self.eps_t) * t                # (B, 1)
        t_b = t.unsqueeze(-1)                                  # (B, 1, 1)

        # Noise the latent.
        noise = torch.randn_like(z_0_target)
        # Mask padding cells in the noise so z_t at padding stays zero
        # (matches the encoder's masked output).
        m = data.node_mask.unsqueeze(-1).to(dtype)
        noise = noise * m
        z_t = (1.0 - t_b) * z_0_target + t_b * noise           # (B, N, k)
        z_t = z_t * m

        # Decode z_t for a "visualisable" 2D position. This is purely
        # for human readability of the data.positions field — the
        # actual training loss reads pred.positions (which the wrapper
        # decodes from z_0_PRED inside forward()).
        with torch.no_grad():
            x_t_vis = wrap.decode(data.node_features, z_t, data.node_mask)

        out = DataHolder(
            node_features=data.node_features,
            positions=x_t_vis,
            diffusion_time=t,
            cell_class=data.cell_class,
            cell_ID=getattr(data, "cell_ID", None),
            t_int=(t * 1000.0).long(),
            t=t,
            node_mask=data.node_mask,
        )
        out._ldm_z_t = z_t
        out._ldm_z_0_target = z_0_target
        out._ldm_mu = mu
        out._ldm_logvar = logvar
        return out.mask()

    # ------------------------------------------------------------------
    # Inference start
    # ------------------------------------------------------------------
    def sample_limit_dist(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        cell_ID: torch.Tensor,
        cell_class: torch.Tensor,
    ) -> DataHolder:
        """Initial state of the reverse trajectory: z_T ~ N(0, I)."""
        wrap = self._require_wrapper()
        B, N = node_mask.shape
        device = node_features.device
        dtype = (
            node_features.dtype
            if node_features.is_floating_point() else torch.float32
        )

        z_T = torch.randn(B, N, self.latent_dim, device=device, dtype=dtype)
        m = node_mask.unsqueeze(-1).to(dtype)
        z_T = z_T * m

        with torch.no_grad():
            x_T_vis = wrap.decode(node_features, z_T, node_mask)

        t = torch.ones(B, 1, device=device, dtype=dtype)
        out = DataHolder(
            node_features=node_features,
            positions=x_T_vis,
            diffusion_time=t,
            cell_class=cell_class,
            cell_ID=cell_ID,
            t_int=(t * 1000.0).long(),
            t=t,
            node_mask=node_mask,
        )
        out._ldm_z_t = z_T
        # Encoder outputs are absent at inference; the LDM loss
        # components check for these attributes and no-op if missing,
        # so the inference path is consistent.
        return out.mask()

    # ------------------------------------------------------------------
    # FM Euler reverse step in latent space
    # ------------------------------------------------------------------
    def sample_zs_from_zt_and_pred(
        self,
        z_t: DataHolder,
        pred: DataHolder,
        s_int: torch.Tensor,
    ) -> DataHolder:
        """One reverse Euler step in latent space.

        Reads ``z_t._ldm_z_t`` (current latent) and
        ``pred._ldm_z_0_pred`` (denoiser's prediction of clean z_0).
        Returns a new DataHolder whose ``_ldm_z_t`` is the next-step
        latent and whose ``positions`` is the decoded 2D state for
        visualisation.
        """
        wrap = self._require_wrapper()

        z_t_state = getattr(z_t, "_ldm_z_t", None)
        if z_t_state is None:
            raise RuntimeError(
                "LatentDiffusionModel.sample_zs_from_zt_and_pred "
                "requires z_t._ldm_z_t — sampling chain not started "
                "via sample_limit_dist?"
            )
        z_0_pred = getattr(pred, "_ldm_z_0_pred", None)
        if z_0_pred is None:
            raise RuntimeError(
                "LatentDiffusionModel.sample_zs_from_zt_and_pred "
                "requires pred._ldm_z_0_pred (set by "
                "LatentDiffusionWrapper.forward). Check that "
                "self.model is the LDM wrapper."
            )

        # FM Euler step (linear schedule, x0-pred form):
        #     z_s = (s/t) z_t + ((t-s)/t) z_0_pred
        # Identical to the existing FlowMatchingModel formula, just
        # in latent space.
        n = self.n_sampling_steps
        t_val = z_t.t                                          # (B, 1)
        if s_int.dim() == 0:
            s_int_b = s_int.view(1, 1).expand(t_val.size(0), 1)
        elif s_int.dim() == 1:
            s_int_b = s_int.view(-1, 1)
        else:
            s_int_b = s_int.view(t_val.size(0), 1)
        s_val = s_int_b.to(t_val.dtype) / float(n)             # (B, 1)
        t_b = t_val.unsqueeze(-1)                              # (B, 1, 1)
        s_b = s_val.unsqueeze(-1)
        t_safe = t_b.clamp_min(self.eps_t)
        ratio_t = s_b / t_safe
        ratio_pred = (t_b - s_b) / t_safe
        z_s = ratio_t * z_t_state + ratio_pred * z_0_pred

        # Mask padding cells.
        m = z_t.node_mask.unsqueeze(-1).to(z_s.dtype)
        z_s = z_s * m

        # Decode for the 2D position output (this becomes pred.positions
        # of the next forward — the model itself only re-derives
        # z_0_pred from z_s, so the decoded 2D is purely for
        # visualisation / metric consumption).
        with torch.no_grad():
            x_s = wrap.decode(z_t.node_features, z_s, z_t.node_mask)

        out = DataHolder(
            node_features=z_t.node_features,
            positions=x_s,
            diffusion_time=s_val,
            cell_class=z_t.cell_class,
            cell_ID=getattr(z_t, "cell_ID", None),
            t_int=(s_val * 1000.0).long(),
            t=s_val,
            node_mask=z_t.node_mask,
        )
        out._ldm_z_t = z_s
        return out.mask()

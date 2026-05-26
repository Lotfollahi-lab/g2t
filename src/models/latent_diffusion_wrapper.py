"""LatentDiffusionWrapper — the ``self.model`` slot for the LDM framework.

Wraps encoder + denoiser + decoder into a single forward that the
LightningModule can drive. Encapsulates the LDM-specific machinery so
the surrounding training step / sampling code stays
framework-agnostic.

Forward contract (DataHolder-in / DataHolder-out, same as every other
``self.model``):

  Input ``data`` must carry, in addition to the usual fields:
    * ``data._ldm_z_t``         — (B, N, latent_dim)  noisy latent
                                  (set by ``LatentDiffusionModel.apply_noise``
                                  / ``sample_limit_dist``)
    * ``data._ldm_mu``,
      ``data._ldm_logvar``,
      ``data._ldm_z_0_target``  — set ONLY at training time (when
                                  the encoder ran inside apply_noise).
                                  Absent at inference. Forward passes
                                  them through onto the returned
                                  DataHolder so the loss can read them.

  Output ``pred`` carries:
    * ``pred.positions``       — (B, N, 2) decoded positions
    * ``pred.node_features``   — (B, N, F) inner denoiser features
                                  (provided so downstream wrappers
                                  that read pred.node_features — EDM /
                                  knn_graph — don't crash; their flags
                                  are mutually exclusive with LDM, so
                                  in practice this field is unused).
    * ``pred._ldm_z_0_pred``   — (B, N, latent_dim) the denoiser's
                                  predicted clean latent. Used by the
                                  ``latent_fm_mse`` loss to compute
                                  the denoising objective.
    * ``pred._ldm_mu``,
      ``pred._ldm_logvar``     — passed through from input if present;
                                  used by ``latent_kl``.

Why no time conditioning here separate from the denoiser: the
denoiser is the only component that should "see" the FM timestep,
because the encoder/decoder are time-agnostic compression (they
operate on x_0-equivalent states in our joint-training setup). The
denoiser uses its existing adaLN-Zero time path.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from utils.data.dataholder import DataHolder
from models.latent_vae import (
    LatentVAEEncoder,
    LatentVAEDecoder,
)


class LatentDiffusionWrapper(nn.Module):
    """Joins encoder/denoiser/decoder into one ``self.model`` for the
    LightningModule.

    Construction takes the same ``input_dims`` / ``hidden_dims`` /
    ``output_dims`` triple as the LUNA Model so the LightningModule's
    backbone-dispatch code can pass them uniformly. The DENOISER is
    the existing DiTBackbone (chosen because it has clean adaLN-Zero
    time conditioning); we just give it our latent state as input
    via the position channel.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        n_layers: int,
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        output_dims: Dict[str, int],
        ldm_cfg=None,
    ):
        super().__init__()

        def _g(k, default):
            if ldm_cfg is None:
                return default
            return (
                ldm_cfg.get(k, default)
                if hasattr(ldm_cfg, "get")
                else getattr(ldm_cfg, k, default)
            )

        self.latent_dim = int(_g("latent_dim", 128))
        self.vae_hidden_dim = int(_g("vae_hidden_dim", 256))
        self.vae_n_layers = int(_g("vae_n_layers", 4))

        gene_in = int(input_dims["node_features_dimensions"])

        # ----- Encoder ----------------------------------------------------
        # Takes (gene, true_positions) → (mu, logvar). Used ONLY at
        # training time; absent at inference. Lives inside this wrapper
        # so its parameters move/save with the rest of the model.
        self.encoder = LatentVAEEncoder(
            gene_dim=gene_in,
            latent_dim=self.latent_dim,
            hidden_dim=self.vae_hidden_dim,
            n_layers=self.vae_n_layers,
            n_heads=int(_g("vae_n_heads", 4)),
        )

        # ----- Decoder ----------------------------------------------------
        # Takes (gene, z) → predicted positions. Used at both training
        # (for the reconstruction loss) and inference (final z → pos).
        self.decoder = LatentVAEDecoder(
            gene_dim=gene_in,
            latent_dim=self.latent_dim,
            hidden_dim=self.vae_hidden_dim,
            n_layers=self.vae_n_layers,
            n_heads=int(_g("vae_n_heads", 4)),
        )

        # ----- Denoiser ---------------------------------------------------
        # The "DiT in latent space" — takes (gene, z_t, t) → z_0_pred.
        # We re-purpose the DiTBackbone here by feeding the latent z_t
        # through what is normally the position-stream input. The
        # DiTBackbone's mlp_in_position takes 2-D positions; here we
        # extend it by overriding the position-embed input dim to
        # latent_dim so it eats the full latent. We do this by passing
        # ``input_dims['position_dim']`` via a custom dict — see below.

        # The DiTBackbone module has hard-coded 2-D position embedding
        # (``self.pos_embed = nn.Linear(2, hidden_dim)``). We need
        # latent_dim instead. Build a small replacement DiT here whose
        # `pos_embed` accepts latent_dim. Reuse all DiT primitives
        # (TimestepEmbedder, DiTBlock, _modulate, DiTFinalLayer) via
        # the dit_backbone module.
        from models.dit_backbone import (
            TimestepEmbedder,
            DiTBlock,
        )

        self.den_hidden = int(_g("denoiser_hidden_dim", 256))
        self.den_n_layers = int(_g("denoiser_n_layers", n_layers))
        self.den_n_heads = int(_g("denoiser_n_heads", 8))
        self.den_mlp_ratio = float(_g("denoiser_mlp_ratio", 4))
        self.den_time_dim = int(_g("denoiser_time_embed_dim", 256))

        # Input projections for the denoiser: gene + z_t → tokens.
        self.den_gene_embed = nn.Sequential(
            nn.Linear(gene_in, self.den_hidden),
            nn.SiLU(),
            nn.Linear(self.den_hidden, self.den_hidden),
        )
        self.den_z_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.den_hidden),
            nn.SiLU(),
            nn.Linear(self.den_hidden, self.den_hidden),
        )
        self.den_t_embed = TimestepEmbedder(self.den_hidden, self.den_time_dim)
        self.den_blocks = nn.ModuleList([
            DiTBlock(self.den_hidden, self.den_n_heads, self.den_mlp_ratio)
            for _ in range(self.den_n_layers)
        ])
        self.den_final_norm = nn.LayerNorm(
            self.den_hidden, elementwise_affine=False, eps=1e-6,
        )
        # adaLN modulation for the final layer (shift + scale only, no gate).
        self.den_final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.den_hidden, 2 * self.den_hidden),
        )
        nn.init.zeros_(self.den_final_modulation[-1].weight)
        nn.init.zeros_(self.den_final_modulation[-1].bias)
        # Project denoiser hidden → predicted clean latent z_0.
        self.den_proj_out = nn.Linear(self.den_hidden, self.latent_dim)
        # Also produce a small "node_features" output of size
        # ``output_features_to_pos_dims`` so downstream code that reads
        # pred.node_features (e.g. gene_recon_head) doesn't crash on
        # LDM-mode runs. Width is taken from the LUNA hidden_dims dict
        # for consistency with the other backbones.
        self.node_features_out_dim = int(hidden_dims["output_features_to_pos_dims"])
        self.den_node_features_proj = nn.Linear(
            self.den_hidden, self.node_features_out_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(
        self,
        gene_features: torch.Tensor,
        positions: torch.Tensor,
        node_mask: torch.Tensor,
    ):
        """Wrap the encoder's forward — sugar for the noise model."""
        return self.encoder(gene_features, positions, node_mask)

    def decode(
        self,
        gene_features: torch.Tensor,
        z: torch.Tensor,
        node_mask: torch.Tensor,
    ):
        """Wrap the decoder's forward — sugar for the noise model and
        the sampling chain.
        """
        return self.decoder(gene_features, z, node_mask)

    def _denoise(
        self,
        gene_features: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """DiT-style denoiser: (gene, z_t, t) → z_0_pred."""
        from models.dit_backbone import _modulate

        tok = self.den_gene_embed(gene_features) + self.den_z_embed(z_t)
        tok = tok * node_mask.unsqueeze(-1).to(tok.dtype)

        # Time conditioning (scalar t per slice).
        if t.dim() > 1:
            t_flat = t.view(-1)
        else:
            t_flat = t
        c = self.den_t_embed(t_flat)                          # (B, D)

        kpm = ~node_mask
        for block in self.den_blocks:
            tok = block(tok, c, key_padding_mask=kpm)

        # Final adaLN + projection → predicted z_0 in latent space.
        shift, scale = self.den_final_modulation(c).chunk(2, dim=-1)
        tok = _modulate(self.den_final_norm(tok), shift, scale)
        z0 = self.den_proj_out(tok)                           # (B, N, k)
        # Mask padding cells.
        m = node_mask.unsqueeze(-1).to(z0.dtype)
        z0 = z0 * m
        # Also project to "node_features" for downstream compatibility.
        node_features = self.den_node_features_proj(tok) * m
        return z0, node_features

    # ------------------------------------------------------------------
    # DataHolder-in / DataHolder-out forward
    # ------------------------------------------------------------------
    def forward(self, data: DataHolder, **_unused_kwargs) -> DataHolder:
        """Joint denoise + decode in one forward.

        Reads ``data._ldm_z_t`` (must be set by the noise model's
        ``apply_noise`` or ``sample_limit_dist``). Predicts ``z_0`` via
        the denoiser; decodes to 2D positions via the decoder.
        Propagates training-time stashes (mu, logvar, z_0_target) onto
        the returned DataHolder so the loss can read them.
        """
        z_t = getattr(data, "_ldm_z_t", None)
        if z_t is None:
            raise RuntimeError(
                "LatentDiffusionWrapper.forward requires data._ldm_z_t "
                "(set by LatentDiffusionModel.apply_noise / "
                "sample_limit_dist). If you're invoking this wrapper "
                "outside the LDM framework, check that "
                "cfg.model.framework='latent_diffusion'."
            )

        node_mask = data.node_mask
        t = data.t if data.t is not None else data.diffusion_time

        # Denoise + reach into the decoder to get a position prediction.
        z_0_pred, node_features = self._denoise(
            data.node_features, z_t, t, node_mask,
        )
        positions_pred = self.decoder(data.node_features, z_0_pred, node_mask)

        pred = DataHolder(
            node_features=node_features,
            positions=positions_pred,
            diffusion_time=data.diffusion_time,
            cell_class=data.cell_class,
            cell_ID=data.cell_ID,
            t_int=data.t_int,
            t=data.t,
            node_mask=node_mask,
        )
        # Stash LDM-specific quantities for the loss path.
        pred._ldm_z_0_pred = z_0_pred
        # Propagate any encoder outputs set by apply_noise (absent at
        # inference time — the encoder doesn't run then).
        for attr in ("_ldm_mu", "_ldm_logvar", "_ldm_z_0_target"):
            val = getattr(data, attr, None)
            if val is not None:
                setattr(pred, attr, val)
        return pred

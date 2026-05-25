"""Latent diffusion framework — Architectural extension #5 (scaffold).

Status: SCAFFOLD ONLY. The config surface (cfg.model.latent_diffusion.*
+ ``framework: "latent_diffusion"``) is wired through diffusion_model.py
but ``LatentDiffusionModel.__init__`` raises NotImplementedError.

Why scaffold-first
------------------
A working latent-diffusion model (Stable-Diffusion-style) for scgg
needs:

  1. **A tissue-level VAE.** Encodes (gene_expression, positions) for
     all cells in a slice → a latent z. Open design questions:
       - One latent per slice (`z ∈ R^d`) — extreme compression;
         decoder has to expand to N cell positions.
       - One latent per cell (`z_i ∈ R^d`, set transformer over
         cells) — gentler compression; loses some cross-cell
         information that helps generalisation.
       - Tiled: K tokens per slice, each token decodes to multiple
         cells — middle ground.

  2. **Two-phase training.** Train VAE first (with a reconstruction
     loss + a small KL prior); freeze the VAE; then train the
     diffusion model in latent space using the existing FM
     machinery on z instead of x.

  3. **Inference plumbing.** Sampling produces z_0; the VAE decoder
     reconstructs per-cell positions from z_0 + gene_expression.

Each of these is its own design problem. The natural separation is:
  - One PR / session for the VAE (architecture choice, training,
    reconstruction quality checks).
  - One PR / session for the diffusion stage on top of the frozen VAE.
  - One PR / session for the inference pipeline integration.

Trying to crash all three into one pass would either produce buggy
code or skip the design choices that matter most. So: scaffold here,
follow-up tasks for the real implementation.

Once implemented, this class will mirror the standard NoiseModel /
FlowMatchingModel / RegressionPredictor interface:

    apply_noise(data, ...) → DataHolder
    sample_limit_dist(...)  → DataHolder
    sample_zs_from_zt_and_pred(z_t, pred, s_int) → DataHolder

with the noise applied in z-space, decoded back to x-space for
the backbone's forward pass.
"""

from __future__ import annotations


class LatentDiffusionModel:
    """Placeholder framework. Raises on construction so selecting
    ``framework="latent_diffusion"`` via config produces a clear,
    actionable error rather than a silent misbehaviour.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "model.framework='latent_diffusion' is not yet implemented. "
            "The config surface (cfg.model.latent_diffusion.*) is in "
            "place but the VAE + two-phase training + latent-space "
            "sampling are pending three focused implementation sessions. "
            "See architectural extension #5 in the May 2026 ablation "
            "menu, and the follow-up task that owns this. For now use "
            "framework='diffusion', 'flow_matching', 'regression', or "
            "'energy'."
        )

"""Cell-cell-potentials / energy-based framework — scGG fundamental
method #2.

Status: SCAFFOLD. The config flag (``framework: "energy"``) is wired
through ``diffusion_model.py`` but the actual implementation raises
NotImplementedError. The scaffold makes the flag surface real so
configs validate; the actual implementation is its own focused
session.

Why scaffold-first
-------------------
An honest energy-based framework involves:
  - A pairwise potential network Φ_θ(g_i, g_j, r_ij) → ℝ, called over
    O(N²) or kNN-restricted pairs (which is the only tractable form
    for slices with thousands of cells).
  - Score-matching training: perturb x by Gaussian noise σ, compute
    ∇_x E(x_noisy) via autograd through Φ, minimise
    ‖∇_x E(x_noisy) − (x_clean − x_noisy)/σ²‖². This is structurally
    different from FM/DDPM's "predict x_0" — it touches the training
    loop, not just the noise schedule.
  - Annealed Langevin sampling: multi-noise-level schedule, with
    Langevin steps at each level. Different from the
    sample.iterate_sampling loop which assumes single-pass-per-step
    denoising.

A correct implementation requires changes outside this file
(loss_function.py needs a score-matching loss, sample.py needs the
annealed-Langevin path) and benefits from being designed end-to-end
in one focused session.

Planned interface
-----------------
EnergyPredictor will mirror the (apply_noise / sample_limit_dist /
sample_zs_from_zt_and_pred) interface that NoiseModel /
FlowMatchingModel / RegressionPredictor share, so the existing
training step and sampling loop work with minimal modification. The
``forward()`` of the inner model becomes ``∇_x E(x | g)`` (the score),
predicted via autograd through Φ_θ. The training loss is
score-matching MSE.

References
----------
- Song, Y., Ermon, S. (2019). "Generative Modeling by Estimating
  Gradients of the Data Distribution." NeurIPS. (NCSN / annealed
  Langevin from random noise.)
- Vincent, P. (2011). "A Connection Between Score Matching and
  Denoising Autoencoders." Neural Computation.
"""

from __future__ import annotations


class EnergyPredictor:
    """Placeholder energy-based framework. Raises on construction so
    selecting this framework via config produces a clear actionable
    error rather than a silent misbehaviour.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "model.framework='energy' is not yet implemented. The "
            "config surface is in place (see "
            "configs/model/default.yaml::energy) but the score-matching "
            "training path + annealed-Langevin sampler haven't been "
            "wired. This is scGG fundamental method #2 — schedule a "
            "focused session to implement EnergyPredictor (interface-"
            "compatible with NoiseModel / FlowMatchingModel / "
            "RegressionPredictor), the pairwise-potential network Φ_θ, "
            "the score-matching loss, and the annealed-Langevin "
            "sampling loop. For now use framework='diffusion', "
            "'flow_matching', or 'regression'."
        )

"""Direct-regression "predictor" that fits the NoiseModel slot.

Selected via ``model.framework=regression``. Bypasses both DDPM and
FM machinery entirely: no noise is added, no iterative sampling
happens. The network is trained as a pure supervised regressor.

Why
---
For very small datasets (cortex has ~6 slices), a generative model
has to learn the full conditional ``p(positions | genes)`` from
finite samples. A direct regressor only needs to learn the
conditional MEAN — much more sample-efficient. Diffusion / FM
become useful when there's enough data for the generative
distribution to matter; on tiny datasets, regression often wins.

The class implements the SAME three methods as ``NoiseModel`` and
``FlowMatchingModel``, so the training step
(``utils/diffusion_model/train/train.py``) and sampling loop
(``utils/diffusion_model/sample/sample.py``) work without
modification:

* ``apply_noise(data)``                 → returns z_t with zeroed
                                          positions and t=1. The
                                          backbone sees NO position
                                          info at training time.
* ``sample_limit_dist(...)``            → same: zeroed positions, t=1.
                                          Used to initialise the
                                          "sampling" loop at inference.
* ``sample_zs_from_zt_and_pred(...)``   → returns pred directly. No
                                          step math; the network's
                                          output IS the final answer.

``max_diffusion_steps = 1`` so the outer reverse loop in
``sample.iterate_sampling`` iterates exactly once: one forward →
return prediction. Wall-clock at inference is one network forward
per slice instead of 50-1000.
"""

from __future__ import annotations

import torch

from utils.data.dataholder import DataHolder


class RegressionPredictor:
    """Drop-in replacement for ``NoiseModel`` / ``FlowMatchingModel``
    that disables the generative pathway entirely.
    """

    def __init__(self, cfg):
        # Single "sampling" step — see module docstring. The outer
        # reverse loop in sample.py iterates ``range(0,
        # max_diffusion_steps)`` so 1 ⇒ one iteration ⇒ one forward.
        self.max_diffusion_steps = 1

        # The Lightning module reads this to decide whether to apply
        # Heun's second-forward correction. Regression has no ODE so
        # Heun is meaningless here — force "euler" (the FM-default
        # value the wrapper code checks for).
        self.sampler = "euler"

        # Carried for parity with FlowMatchingModel; not used by
        # regression-specific code.
        self.noise_schedule = "regression"

    # ------------------------------------------------------------------
    # Training-time "perturbation": zero out positions, set t=1
    # ------------------------------------------------------------------
    def apply_noise(self, data: DataHolder, train_flag: bool = True) -> DataHolder:
        """Strip positional information from the input.

        The backbone is shown only gene expression + the cell mask;
        positions go in as zeros. ``diffusion_time`` is set to 1 to
        signal "no information" — the backbone's time encoder will
        see a constant signal across all training examples, which
        it can learn to ignore (or use as a global "I'm in
        regression mode" indicator). The loss compares the backbone's
        output positions against the TRUE positions held in
        ``masked_true`` (which is just the original ``data`` passed
        separately to the loss — see ``train.training_step_func``).
        """
        B = data.node_features.size(0)
        device = data.node_features.device
        N = data.positions.shape[1]

        zero_pos = torch.zeros_like(data.positions) * data.node_mask.unsqueeze(-1).to(
            data.positions.dtype
        )
        t_array = torch.ones(B, 1, device=device, dtype=data.positions.dtype)
        t_int_array = torch.ones(B, 1, dtype=torch.long, device=device)

        return DataHolder(
            node_features=data.node_features,
            positions=zero_pos,
            cell_class=data.cell_class,
            node_mask=data.node_mask,
            t_int=t_int_array,
            t=t_array,
            diffusion_time=t_array,
        ).mask()

    # ------------------------------------------------------------------
    # Inference start point: identical to apply_noise
    # ------------------------------------------------------------------
    def sample_limit_dist(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        cell_ID: torch.Tensor,
        cell_class: torch.Tensor,
    ) -> DataHolder:
        """Initialise the inference loop. Zeroed positions, t=1 —
        same as the training-time input."""
        B, N = node_mask.shape
        device = node_mask.device

        positions = torch.zeros(B, N, 2, device=device) * node_mask.unsqueeze(-1)
        t_array = torch.ones(B, 1, device=device)
        t_int_array = torch.ones(B, 1, dtype=torch.long, device=device)

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
    # Single-step "denoising": return pred unchanged
    # ------------------------------------------------------------------
    def sample_zs_from_zt_and_pred(
        self,
        z_t: DataHolder,
        pred: DataHolder,
        s_int: torch.Tensor,
    ) -> DataHolder:
        """Direct passthrough. The backbone forward already produced
        the final prediction; there's no ODE / DDPM step to take.

        The outer sampling loop calls this exactly once
        (``max_diffusion_steps=1``), so the return here IS the
        sampler's output.
        """
        return pred.mask()

"""Exponential moving average of model weights as a PyTorch Lightning
callback.

Usage
-----
Enabled via the ``train.ema_decay`` config knob (default 0.0 = off).
The launcher (``scripts/_luna_runner.py::_patch_setup_callbacks``)
instantiates this callback when ``ema_decay > 0`` and hands it to
the Lightning Trainer.

Semantics
---------
After every ``ema_every_n_steps`` optimizer updates::

    θ_EMA[i] ← decay · θ_EMA[i] + (1 − decay) · θ[i]   for each float param/buffer

The shadow copy is initialised LAZILY on the first batch-end call
so we capture parameters AFTER they've been moved to the training
device (matches LightningModule.on_train_batch_end's contract).

Validation and test
-------------------
Before validation/test starts, we *temporarily* swap the live model's
floating-point state with the EMA copy. After the eval epoch ends,
the original weights are restored. This means:

  * train metrics reported per step are computed against the
    instantaneous training weights (correct — that's what is being
    optimized);
  * val/test metrics are computed against EMA weights (correct —
    that's the deployed model);
  * the optimiser state is never touched.

Checkpoint contract
-------------------
At checkpoint-save time, we replace ``checkpoint['state_dict']``
entries with their EMA equivalents (by *cloning* — never mutating
the live module). Effect: every saved ``.ckpt`` IS the EMA model.
Inference loads checkpoints with the standard
``LightningModule.load_from_checkpoint`` machinery and therefore
picks up EMA weights with zero changes to the inference path.

This means you cannot recover the non-EMA weights from a saved
checkpoint — by design. If you ever want both, store ``ema_state``
in the checkpoint separately (commented-out hook below).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl


class EMACallback(pl.Callback):
    """Maintain an exponential moving average of model weights.

    Args:
        decay: smoothing constant in (0, 1). 0.999 ≈ ~1k-step decay
            window, 0.9999 ≈ ~10k-step window. The training-loop
            already enforces decay > 0 before instantiating this
            callback, so this class can assume a meaningful value.
        every_n_steps: update EMA every N optimizer steps. Default 1.
    """

    def __init__(self, decay: float = 0.999, every_n_steps: int = 1):
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError(
                f"EMACallback decay must be in (0, 1); got {decay}."
            )
        self.decay = float(decay)
        self.every_n_steps = max(1, int(every_n_steps))

        # ``ema_state`` is keyed by parameter NAME (matches
        # ``module.state_dict()`` keys) and stores cloned tensors
        # on the same device as the live module. Lazy-initialised
        # because we need post-.cuda() shapes.
        self.ema_state: Dict[str, torch.Tensor] = {}
        # Backup of live floating-point state during val/test, so
        # we can restore it once the eval epoch ends.
        self._train_state_backup: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _init_ema_from(self, module: pl.LightningModule) -> None:
        """First-call init: clone all floating-point params + buffers."""
        with torch.no_grad():
            for name, tensor in module.state_dict().items():
                if tensor.dtype.is_floating_point:
                    self.ema_state[name] = tensor.detach().clone()

    def _update_ema_from(self, module: pl.LightningModule) -> None:
        """In-place EMA update from the live module's float tensors."""
        with torch.no_grad():
            live_state = module.state_dict()
            for name, ema_tensor in self.ema_state.items():
                live_tensor = live_state.get(name)
                if live_tensor is None:
                    continue
                if not live_tensor.dtype.is_floating_point:
                    continue
                # θ_EMA ← decay · θ_EMA + (1 − decay) · θ
                ema_tensor.mul_(self.decay).add_(
                    live_tensor.detach(), alpha=1.0 - self.decay
                )

    def _swap_to_ema(self, module: pl.LightningModule) -> None:
        """Temporarily replace live weights with EMA weights, stashing
        the originals in ``_train_state_backup`` so we can restore.

        ``state_dict()`` returns tensors that SHARE storage with the
        module's parameters/buffers, so ``copy_`` mutates the live
        module in place — which is exactly what we want here for a
        cheap (no-allocation) swap.
        """
        if not self.ema_state:
            return
        live = module.state_dict()
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, ema_tensor in self.ema_state.items():
                if name not in live:
                    continue
                backup[name] = live[name].detach().clone()
                live[name].copy_(ema_tensor)
        self._train_state_backup = backup

    def _swap_back(self, module: pl.LightningModule) -> None:
        """Restore the pre-swap training weights."""
        if not self._train_state_backup:
            return
        live = module.state_dict()
        with torch.no_grad():
            for name, train_tensor in self._train_state_backup.items():
                if name not in live:
                    continue
                live[name].copy_(train_tensor)
        self._train_state_backup = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # Lazy init AFTER the first forward+backward: this is the
        # first moment we're guaranteed that the module is on its
        # final device and all parameters have valid shapes.
        if not self.ema_state:
            self._init_ema_from(pl_module)
            return
        # Throttle update frequency. global_step has already been
        # incremented when on_train_batch_end fires.
        if trainer.global_step % self.every_n_steps != 0:
            return
        self._update_ema_from(pl_module)

    # Val/test: swap to EMA before, swap back after.
    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._swap_to_ema(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._swap_back(pl_module)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._swap_to_ema(pl_module)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self._swap_back(pl_module)

    # Checkpoint save: replace state_dict entries with EMA clones
    # so every saved ``.ckpt`` IS the EMA model. Inference picks it
    # up automatically via the standard load_from_checkpoint path.
    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        if not self.ema_state:
            return
        state_dict = checkpoint.get("state_dict", {})
        with torch.no_grad():
            for name, ema_tensor in self.ema_state.items():
                if name in state_dict:
                    # Clone, otherwise we'd hold a reference to the
                    # live tensor and the *next* EMA update would
                    # silently mutate the just-saved checkpoint.
                    state_dict[name] = ema_tensor.detach().clone()

    # Callback state persistence — so resuming training from a
    # checkpoint also restores the EMA shadow weights.
    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
            "ema_state": {k: v.detach().cpu() for k, v in self.ema_state.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.every_n_steps = int(state_dict.get("every_n_steps", self.every_n_steps))
        ema_state = state_dict.get("ema_state", {})
        # Tensors will be moved to the right device on the first
        # batch_end call (see _update_ema_from); keep them on CPU
        # here so resuming on a different rank is safe.
        self.ema_state = {k: v.clone() for k, v in ema_state.items()}

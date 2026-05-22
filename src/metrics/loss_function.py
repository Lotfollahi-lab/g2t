"""scGG training loss.

Composed of multiple weighted components, each toggleable via
``cfg.model.loss.<name>.enabled`` / ``weight`` / extra knobs. The default
config keeps only LUNA's original pairwise-distance MSE on, so an
out-of-the-box scgg training run reproduces the LUNA baseline; each
extra component is opt-in for ablations.

Adding a new component
----------------------
1. Add a ``<name>: {enabled, weight, ...}`` block to
   ``scgg/src/configs/model/default.yaml`` under ``loss:`` (defaulted off).
2. Add a ``_compute_<name>(self, masked_pred, masked_true)`` method here
   that returns a scalar tensor.
3. Register the new component in ``_COMPONENT_REGISTRY`` below — that's
   the only edit ``compute_loss`` needs.
4. Optional: bump ``log_epoch_metrics`` to also report it.

Per-component scalars are exposed in the returned ``log_dict`` under
``train_loss/<name>`` / ``val_loss/<name>``, so ablation runs become
directly comparable from the wandb / metrics CSV side without having
to re-derive contributions.
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf

from utils.data.dataholder import DataHolder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg_get(cfg, *keys, default=None):
    """Safe nested-key access on a DictConfig / dict. Returns ``default``
    when any key in the chain is missing.
    """
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None) if not hasattr(cur, "get") else cur.get(k, None)
    return default if cur is None else cur


# ---------------------------------------------------------------------------
# LossFunction
# ---------------------------------------------------------------------------


class LossFunction(nn.Module):
    """Composable training/validation loss.

    Each component is computed independently and combined as
    ``sum_i (weight_i * loss_i)`` to produce the optimised scalar.
    Components and their weights/parameters are sourced from
    ``cfg.model.loss``; turning ``loss.<name>.enabled = false`` removes
    that component from the total *and* from logging, which is what
    you want for honest ablations.
    """

    def __init__(self, cfg: Optional[DictConfig] = None) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.cfg = cfg

        # Per-step accumulators (mirrors original LUNA pattern of
        # remembering the last `forward()` so log_epoch_metrics can
        # recompute the value without a second forward pass).
        self.true_positions: Optional[List[torch.Tensor]] = None
        self.pred_positions: Optional[List[torch.Tensor]] = None
        self.node_mask: Optional[List[torch.Tensor]] = None

        # Resolve component configs once, with safe defaults that
        # collapse to LUNA's original loss when no `loss:` block is
        # set on the cfg (e.g. older training configs).
        self._pwd_enabled = bool(_cfg_get(cfg, "model", "loss",
                                          "pairwise_distance_mse", "enabled",
                                          default=True))
        self._pwd_weight = float(_cfg_get(cfg, "model", "loss",
                                          "pairwise_distance_mse", "weight",
                                          default=1.0))

        self._knn_enabled = bool(_cfg_get(cfg, "model", "loss",
                                          "knn_rank", "enabled",
                                          default=False))
        self._knn_weight = float(_cfg_get(cfg, "model", "loss",
                                          "knn_rank", "weight", default=1.0))
        self._knn_k = int(_cfg_get(cfg, "model", "loss", "knn_rank", "k",
                                   default=20))

        # The component registry. Each entry is
        # (name, enabled, weight, callable(self, pred, true) -> Tensor).
        # ``compute_loss`` iterates this list — adding a new component
        # is a one-line addition here once the helper method exists.
        self._components: List[Tuple[str, bool, float, Callable]] = [
            ("pairwise_distance_mse", self._pwd_enabled, self._pwd_weight,
             self._compute_pairwise_distance_mse),
            ("knn_rank", self._knn_enabled, self._knn_weight,
             self._compute_knn_rank),
        ]

        # One-time summary so the train.log shows what's active.
        active = [
            f"{name}(w={w:.3g}"
            + (f", k={self._knn_k}" if name == "knn_rank" else "")
            + ")"
            for (name, on, w, _) in self._components if on
        ]
        print(f"[LossFunction] active components: {', '.join(active) or '(none!)'}")

    # ----------------------------- component implementations --------------

    @staticmethod
    def _masked_cdist(pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pairwise Euclidean distance matrix restricted to masked cells.
        Returns shape ``(n_valid, n_valid)``.
        """
        return torch.cdist(pos[mask], pos[mask], p=2)

    def _compute_pairwise_distance_mse(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> torch.Tensor:
        """LUNA's headline loss: per-slice MSE on the pairwise-distance
        matrix, averaged across slices in the batch.
        """
        losses = []
        for true_pos, pred_pos, mask in zip(
            masked_true.positions, masked_pred.positions, masked_true.node_mask,
        ):
            d_true = self._masked_cdist(true_pos, mask)
            d_pred = self._masked_cdist(pred_pos, mask)
            losses.append(self.mse(d_pred, d_true))
        return torch.mean(torch.stack(losses))

    def _compute_knn_rank(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> torch.Tensor:
        """Local kNN-rank preservation surrogate.

        For each masked cell *i*::

            1. Compute true pairwise distances on the slice (no grad).
            2. Pick the k nearest TRUE neighbours of *i* (excluding self).
            3. Gather true and predicted distances on that k-cell subset.
            4. Compute Pearson correlation between the two length-k
               vectors. (Pearson on raw distances is a smooth lower
               bound on Spearman in this range; cheap and differentiable
               unlike a strict soft-rank.)

        Loss returned is ``1 - mean(corr_i)`` averaged across slices,
        so it's 0 when local geometry is perfectly preserved and 1
        when prediction is uncorrelated with truth.

        Direction-matters: we deliberately pick neighbours in TRUE
        space (the geometry we want to preserve), not in predicted
        space — otherwise the loss has a degenerate solution where the
        model predicts a tight cluster and "preserves" the trivial
        ordering.
        """
        slice_losses: List[torch.Tensor] = []
        eps = 1e-8
        k = self._knn_k

        for true_pos, pred_pos, mask in zip(
            masked_true.positions, masked_pred.positions, masked_true.node_mask,
        ):
            true_pos_m = true_pos[mask]  # (n, D)
            pred_pos_m = pred_pos[mask]  # (n, D)
            n = true_pos_m.shape[0]
            # A slice with too few cells can't form a k-NN; skip
            # rather than crash. The other slices in the batch carry
            # the gradient.
            k_eff = min(k, max(n - 1, 1))
            if n < 3 or k_eff < 2:
                continue

            # k-NN selection: gradient-free.
            with torch.no_grad():
                d_true_ref = torch.cdist(true_pos_m, true_pos_m, p=2)
                # Mask self-distance so cell i never picks itself.
                d_true_ref_masked = d_true_ref.clone()
                d_true_ref_masked.fill_diagonal_(float("inf"))
                _, knn_idx = torch.topk(
                    d_true_ref_masked, k=k_eff, largest=False, dim=1,
                )

            # Distances on the same matrix — true side is detached
            # (constant data), pred side carries gradients.
            d_true = d_true_ref.detach()
            d_pred = torch.cdist(pred_pos_m, pred_pos_m, p=2)

            true_knn = torch.gather(d_true, 1, knn_idx)   # (n, k_eff)
            pred_knn = torch.gather(d_pred, 1, knn_idx)   # (n, k_eff)

            # Per-cell Pearson(true_knn[i, :], pred_knn[i, :]).
            true_c = true_knn - true_knn.mean(dim=1, keepdim=True)
            pred_c = pred_knn - pred_knn.mean(dim=1, keepdim=True)
            cov = (true_c * pred_c).sum(dim=1)
            true_norm = torch.sqrt((true_c ** 2).sum(dim=1) + eps)
            pred_norm = torch.sqrt((pred_c ** 2).sum(dim=1) + eps)
            corr = cov / (true_norm * pred_norm)            # (n,)
            slice_losses.append(1.0 - corr.mean())

        if not slice_losses:
            # No slice had enough cells. Return a zero tensor that
            # tracks the model's parameters so backward doesn't crash.
            zero = (masked_pred.positions[0].sum() * 0.0
                    if len(masked_pred.positions) > 0
                    else torch.tensor(0.0))
            return zero

        return torch.mean(torch.stack(slice_losses))

    # ----------------------------- aggregation ----------------------------

    def compute_loss(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compose all enabled components into the optimised scalar.

        Returns:
            total_loss: weighted sum of enabled component values.
            per_component: ``{name: float}`` for logging (raw value
                before weighting, plus the weighted contribution under
                ``"<name>_weighted"``). Disabled components are absent
                from this dict.
        """
        total = None
        per_component: Dict[str, float] = {}
        for name, enabled, weight, fn in self._components:
            if not enabled:
                continue
            val = fn(masked_pred, masked_true)
            per_component[name] = float(val.detach().item())
            weighted = weight * val
            per_component[f"{name}_weighted"] = float(weighted.detach().item())
            total = weighted if total is None else (total + weighted)
        if total is None:
            # All components disabled — define total as 0 attached to
            # the prediction graph so backward is still well-defined.
            total = masked_pred.positions[0].sum() * 0.0
        per_component["total"] = float(total.detach().item())
        return total, per_component

    # ----------------------------- pl-style hooks -------------------------

    def forward(
        self,
        masked_pred: DataHolder,
        masked_true: DataHolder,
        train_stage: bool = True,
        log: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        # Remember the last batch so log_epoch_metrics can re-derive
        # without forcing a second forward (matches LUNA's pattern).
        self.true_positions = masked_true.positions
        self.pred_positions = masked_pred.positions
        self.node_mask = masked_true.node_mask

        loss, per_component = self.compute_loss(masked_pred, masked_true)

        to_log: Optional[Dict[str, float]] = None
        if log:
            prefix = "train_loss" if train_stage else "val_loss"
            to_log = {f"{prefix}/{k}": v for k, v in per_component.items()}
            # Keep the LUNA-shaped key alive too so existing notebooks
            # that read `*_loss/position_mse` keep working.
            if "pairwise_distance_mse" in per_component:
                to_log[f"{prefix}/position_mse"] = per_component[
                    "pairwise_distance_mse"
                ]
            if wandb.run:
                wandb.log(to_log, commit=True)

        return loss, to_log

    def reset(self) -> None:
        """Hook kept for parity with LUNA's pl callbacks. No state to clear."""
        pass

    def log_epoch_metrics(self) -> Dict[str, float]:
        """Recompute the loss from the most recent forward's tensors and
        log it as `train_epoch/*`. Used by LUNA's training-step wrapper.
        """
        if (
            self.true_positions is None
            or self.pred_positions is None
            or self.node_mask is None
        ):
            return {}

        # Wrap the remembered tensors back into the DataHolder-shaped
        # objects compute_loss expects. Only the three attributes we
        # actually look at need to be present.
        class _Bag:
            pass
        masked_pred = _Bag()
        masked_pred.positions = self.pred_positions
        masked_pred.node_mask = self.node_mask
        masked_true = _Bag()
        masked_true.positions = self.true_positions
        masked_true.node_mask = self.node_mask

        _, per_component = self.compute_loss(masked_pred, masked_true)

        to_log = {
            f"train_epoch/{k}": v for k, v in per_component.items()
        }
        # Backward-compat alias.
        if "pairwise_distance_mse" in per_component:
            to_log["train_epoch/position_mse"] = per_component[
                "pairwise_distance_mse"
            ]
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log

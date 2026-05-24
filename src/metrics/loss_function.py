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
                                          "knn_rank", "weight", default=0.05))
        self._knn_k = int(_cfg_get(cfg, "model", "loss", "knn_rank", "k",
                                   default=20))
        self._knn_scope = str(_cfg_get(cfg, "model", "loss",
                                       "knn_rank", "scope",
                                       default="global")).lower()
        if self._knn_scope not in ("global", "local"):
            raise ValueError(
                f"model.loss.knn_rank.scope must be 'global' or 'local'; "
                f"got {self._knn_scope!r}"
            )

        # 0-dim persistent homology loss (MST / Wasserstein-2²).
        self._ph_enabled = bool(_cfg_get(cfg, "model", "loss",
                                         "persistent_homology", "enabled",
                                         default=False))
        self._ph_weight = float(_cfg_get(cfg, "model", "loss",
                                         "persistent_homology", "weight",
                                         default=0.5))
        ph_subsample = _cfg_get(cfg, "model", "loss",
                                "persistent_homology", "subsample", default=None)
        self._ph_subsample = int(ph_subsample) if ph_subsample else None
        self._ph_dim = int(_cfg_get(cfg, "model", "loss",
                                    "persistent_homology", "dim", default=0))
        self._ph_frequency = max(1, int(_cfg_get(
            cfg, "model", "loss", "persistent_homology", "frequency", default=1,
        )))
        self._ph_cache_true = bool(_cfg_get(
            cfg, "model", "loss", "persistent_homology", "cache_true", default=True,
        ))
        # Step counter for `frequency` knob; cache for true MST edges.
        # Cache size is bounded by the dataset's slice count, which is
        # small (LUNA cortex: 33; ABCA Animal-1: 147). No eviction
        # needed.
        self._ph_step_counter = 0
        self._ph_true_cache: dict = {}
        if self._ph_dim != 0:
            raise NotImplementedError(
                f"model.loss.persistent_homology.dim={self._ph_dim} not "
                f"supported. Only 0-dim (MST formulation) is implemented; "
                f"1-dim needs the alpha complex (gudhi) — TODO."
            )

        # Sinkhorn / OT divergence loss component.
        self._sk_enabled = bool(_cfg_get(cfg, "model", "loss",
                                         "sinkhorn", "enabled",
                                         default=False))
        self._sk_weight = float(_cfg_get(cfg, "model", "loss",
                                         "sinkhorn", "weight", default=0.1))
        self._sk_blur = float(_cfg_get(cfg, "model", "loss",
                                       "sinkhorn", "blur", default=0.01))
        self._sk_scaling = float(_cfg_get(cfg, "model", "loss",
                                          "sinkhorn", "scaling", default=0.9))
        self._sk_p = int(_cfg_get(cfg, "model", "loss",
                                  "sinkhorn", "p", default=2))
        sk_sub = _cfg_get(cfg, "model", "loss", "sinkhorn", "subsample",
                          default=None)
        self._sk_subsample = int(sk_sub) if sk_sub else None
        # Lazy-init: only build the SamplesLoss object on first call,
        # so users who never enable the component never pay the
        # geomloss import cost.
        self._sk_loss_fn = None

        # The component registry. Each entry is
        # (name, enabled, weight, callable(self, pred, true) -> Tensor).
        # ``compute_loss`` iterates this list — adding a new component
        # is a one-line addition here once the helper method exists.
        self._components: List[Tuple[str, bool, float, Callable]] = [
            ("pairwise_distance_mse", self._pwd_enabled, self._pwd_weight,
             self._compute_pairwise_distance_mse),
            ("knn_rank", self._knn_enabled, self._knn_weight,
             self._compute_knn_rank),
            ("persistent_homology", self._ph_enabled, self._ph_weight,
             self._compute_persistent_homology),
            ("sinkhorn", self._sk_enabled, self._sk_weight,
             self._compute_sinkhorn),
        ]

        # One-time summary so the train.log shows what's active.
        active = []
        for (name, on, w, _) in self._components:
            if not on:
                continue
            extra = ""
            if name == "knn_rank":
                extra = f", scope={self._knn_scope}"
                if self._knn_scope == "local":
                    extra += f", k={self._knn_k}"
            elif name == "persistent_homology":
                extra = f", dim={self._ph_dim}"
                if self._ph_frequency > 1:
                    extra += f", every {self._ph_frequency} steps"
                if self._ph_cache_true:
                    extra += ", cache_true=on"
                if self._ph_subsample:
                    extra += f", subsample={self._ph_subsample}"
            elif name == "sinkhorn":
                extra = (
                    f", p={self._sk_p}, blur={self._sk_blur:.3g}, "
                    f"scaling={self._sk_scaling:.3g}"
                )
                if self._sk_subsample:
                    extra += f", subsample={self._sk_subsample}"
            active.append(f"{name}(w={w:.3g}{extra})")
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
        """Rank-preservation correlation surrogate.

        For each masked cell *i*, compute Pearson correlation between
        the row of true pairwise distances and the row of predicted
        pairwise distances. Loss is ``1 - mean(corr_i)`` averaged
        across slices.

        Scope:
          - ``global`` (default): the full distance row — directly
            mirrors the eval metric (per-cell Spearman on full row).
            Recommended starting point.
          - ``local``: restricted to the k nearest TRUE neighbours of
            each cell. Targets local visual fidelity but, used as a
            primary loss, admits degenerate solutions (tight local
            clusters that preserve rank within k but break global
            scale). Always use with a small weight (≤ 1e-1) as a
            regulariser when ``scope=local``.

        Why pick neighbours in TRUE space (not PRED) for the local
        scope: picking in pred space rewards the model for putting
        any k cells close together, regardless of correctness.
        Anchoring on true neighbours forces it to recover the right
        local set.
        """
        slice_losses: List[torch.Tensor] = []
        eps = 1e-8
        scope = self._knn_scope
        k = self._knn_k

        for true_pos, pred_pos, mask in zip(
            masked_true.positions, masked_pred.positions, masked_true.node_mask,
        ):
            true_pos_m = true_pos[mask]  # (n, D)
            pred_pos_m = pred_pos[mask]  # (n, D)
            n = true_pos_m.shape[0]
            # Too few cells → can't compute meaningful row correlation;
            # skip but let other slices carry the gradient.
            min_n = 3
            if scope == "local":
                min_n = max(min_n, 3)
            if n < min_n:
                continue

            # True distances: gradient-free reference.
            with torch.no_grad():
                d_true_ref = torch.cdist(true_pos_m, true_pos_m, p=2)
            d_true = d_true_ref.detach()
            # Pred distances: gradient-bearing.
            d_pred = torch.cdist(pred_pos_m, pred_pos_m, p=2)

            if scope == "global":
                # Full row; exclude the self-distance (always 0 on
                # both sides — it would just inflate the correlation).
                # Mask self by gathering all-but-diagonal entries.
                # Easiest: subtract off the diagonal contribution by
                # working on (n, n-1) tensors via gather of the
                # non-self column indices.
                idx_all = torch.arange(n, device=d_true.device)
                # For each row i, the non-self columns are
                # [0..i-1, i+1..n-1]. Build that as a (n, n-1) index.
                col = idx_all.unsqueeze(0).expand(n, n)  # (n, n)
                row = idx_all.unsqueeze(1).expand(n, n)  # (n, n)
                non_self = col != row                    # (n, n) bool
                # Reshape to (n, n-1) by masking out the diagonal.
                non_self_idx = col[non_self].reshape(n, n - 1)
                true_row = torch.gather(d_true, 1, non_self_idx)
                pred_row = torch.gather(d_pred, 1, non_self_idx)
            else:  # scope == "local"
                k_eff = min(k, max(n - 1, 1))
                if k_eff < 2:
                    continue
                with torch.no_grad():
                    d_true_masked = d_true_ref.clone()
                    d_true_masked.fill_diagonal_(float("inf"))
                    _, knn_idx = torch.topk(
                        d_true_masked, k=k_eff, largest=False, dim=1,
                    )
                true_row = torch.gather(d_true, 1, knn_idx)
                pred_row = torch.gather(d_pred, 1, knn_idx)

            # Per-cell Pearson(true_row[i, :], pred_row[i, :]).
            true_c = true_row - true_row.mean(dim=1, keepdim=True)
            pred_c = pred_row - pred_row.mean(dim=1, keepdim=True)
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

    def _compute_persistent_homology(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> torch.Tensor:
        """0-dim persistent homology loss via minimum spanning tree.

        Per slice: PD_0(true) = sorted MST-edge-lengths of true coords;
        PD_0(pred) = sorted MST-edge-lengths of pred coords. The
        Wasserstein-2² distance between two equal-cardinality 0-dim
        PDs reduces to the MSE between the sorted vectors, which is
        what we return. The loss is averaged across slices.

        Differentiability: scipy's MST gives us the (row, col) indices
        of the N−1 spanning edges. We gather distances at those
        indices from the *gradient-bearing* ``pred_dist`` tensor;
        gradients flow into ``pred_pos`` via the cdist chain rule.
        The MST STRUCTURE is treated as constant (standard "fixed
        structure" differentiable-persistence trick — Hofer 2019,
        Bruel-Gabrielsson 2020).

        Speed knobs:
          * ``frequency``: compute PH every N training steps; other
            steps return a detached zero. The PH signal direction is
            identical, just sampled — a 10× speedup at frequency=10.
          * ``cache_true``: skip the (constant) true cdist + true MST
            on cache hits. True positions never change across epochs
            so this is a pure win.
        """
        import numpy as np
        from scipy.sparse.csgraph import minimum_spanning_tree

        # Detached-zero shortcut for "this isn't a PH step". Attached
        # to the prediction graph via *0 so backward stays valid when
        # this term is summed into the total loss.
        def _zero():
            if len(masked_pred.positions) > 0:
                return masked_pred.positions[0].sum() * 0.0
            return torch.tensor(0.0)

        # `frequency` skip — early return BEFORE we do any cdist /
        # CPU transfer / scipy work. This is where the wall-clock
        # win comes from for frequency > 1.
        self._ph_step_counter += 1
        if (self._ph_step_counter - 1) % self._ph_frequency != 0:
            return _zero()

        slice_losses: List[torch.Tensor] = []
        subsample = self._ph_subsample

        for true_pos, pred_pos, mask in zip(
            masked_true.positions, masked_pred.positions, masked_true.node_mask,
        ):
            true_pos_m = true_pos[mask]
            pred_pos_m = pred_pos[mask]
            n = true_pos_m.shape[0]
            if n < 3:
                continue

            # Optional random subsampling for large slices. Stochastic
            # per step — the subsample changes each iteration, which is
            # fine here (the PH signal is robust to it as long as the
            # subsample covers the same density profile). Note: when
            # subsample is on, the true-MST cache is bypassed because
            # the random subsample changes per step.
            using_subsample = subsample is not None and n > subsample
            if using_subsample:
                with torch.no_grad():
                    idx = torch.randperm(n, device=true_pos_m.device)[:subsample]
                true_pos_m = true_pos_m[idx]
                pred_pos_m = pred_pos_m[idx]
                n = subsample

            # ---- True MST: cache hit fast-path ----
            cache_key = None
            true_edges_sorted = None
            if self._ph_cache_true and not using_subsample:
                # Fingerprint by cell count + a few values + sum. This
                # is unique-enough across slices and stable across
                # training steps (true_pos is the same tensor content
                # every step). data_ptr won't work because boolean
                # indexing creates fresh tensors per step.
                with torch.no_grad():
                    cache_key = (
                        n,
                        float(true_pos_m[0, 0].item()),
                        float(true_pos_m[-1, -1].item()),
                        float(true_pos_m.sum().item()),
                    )
                cached = self._ph_true_cache.get(cache_key)
                if cached is not None:
                    true_edges_sorted = cached.to(
                        device=true_pos_m.device, dtype=true_pos_m.dtype,
                    )

            if true_edges_sorted is None:
                with torch.no_grad():
                    d_true = torch.cdist(true_pos_m, true_pos_m, p=2)
                    d_true_np = d_true.detach().cpu().numpy()
                    true_mst = minimum_spanning_tree(d_true_np).tocoo()
                    true_edges_sorted = torch.from_numpy(
                        np.sort(true_mst.data)
                    ).to(true_pos_m.device, dtype=true_pos_m.dtype)
                # Stash on CPU to avoid GPU memory growth across the
                # dataset; we'll move it back per step.
                if cache_key is not None:
                    self._ph_true_cache[cache_key] = true_edges_sorted.detach().cpu()

            # ---- Pred MST: must recompute every step ----
            d_pred = torch.cdist(pred_pos_m, pred_pos_m, p=2)
            with torch.no_grad():
                d_pred_np = d_pred.detach().cpu().numpy()
                pred_mst = minimum_spanning_tree(d_pred_np).tocoo()
                pred_rows = torch.from_numpy(pred_mst.row).long().to(d_pred.device)
                pred_cols = torch.from_numpy(pred_mst.col).long().to(d_pred.device)
            pred_edges = d_pred[pred_rows, pred_cols]    # (n-1,), gradient-bearing
            pred_edges_sorted, _ = torch.sort(pred_edges)

            # Equal-cardinality W2² between PDs collapses to MSE between
            # sorted edge vectors. (Both have n-1 entries by construction.)
            w2_sq = ((pred_edges_sorted - true_edges_sorted) ** 2).mean()
            slice_losses.append(w2_sq)

        if not slice_losses:
            return _zero()
        return torch.mean(torch.stack(slice_losses))

    def _compute_sinkhorn(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> torch.Tensor:
        """Sinkhorn-regularized OT divergence between predicted and true
        cell point clouds.

        Computes the *debiased* Sinkhorn divergence
        ``S_ε(α, β) = L_ε(α, β) − ½(L_ε(α, α) + L_ε(β, β))``
        between the two N-cell clouds per slice, then averages across
        slices. Debiasing makes ``S_ε`` zero when the clouds are equal
        (the entropic-regularizer bias cancels out), so the loss
        landscape behaves like vanilla Wasserstein near optimum.

        Complementary to ``_compute_pairwise_distance_mse``:
          * pairwise MSE matches the *distribution of pair distances*
            but is blind to which cell is where (a permutation of the
            cells with the same pairwise structure gets zero loss);
          * Sinkhorn matches the *cells themselves* via an optimal
            transport plan — penalises individual cell misplacement
            within an otherwise-correct overall structure.

        With weight ~0.1× the pairwise term it acts as a regulariser:
        pairwise pins global structure, Sinkhorn pins placement.

        Implementation: lazy-imports ``geomloss.SamplesLoss``. Each
        slice is processed independently because slices have different
        cell counts (geomloss SamplesLoss supports batched dispatch
        only when all clouds in the batch share an ``N``).

        Gradient: flows through ``pred_pos`` via SamplesLoss's
        differentiable implementation (Feydy et al. 2019). ``true_pos``
        is treated as the target; geomloss handles it as a non-leaf
        constant.
        """
        if self._sk_loss_fn is None:
            try:
                from geomloss import SamplesLoss
            except ImportError as e:
                raise ImportError(
                    "model.loss.sinkhorn.enabled=true requires the "
                    "geomloss package. Install via `pip install geomloss` "
                    "(installs KeOps too — recommended for GPU speed). "
                    f"Underlying error: {e}"
                )
            # Cached for the lifetime of the LossFunction. The object
            # is stateless (it's just a callable wrapper around the
            # Sinkhorn iteration scheme), so reuse across batches is
            # fine.
            self._sk_loss_fn = SamplesLoss(
                loss="sinkhorn",
                p=self._sk_p,
                blur=self._sk_blur,
                scaling=self._sk_scaling,
                backend="auto",   # KeOps if installed, else tensorized PyTorch
                debias=True,      # always — see docstring
            )

        slice_losses: List[torch.Tensor] = []
        for true_pos, pred_pos, mask in zip(
            masked_true.positions, masked_pred.positions, masked_true.node_mask,
        ):
            true_real = true_pos[mask]          # (n, 2), gradient-free target
            pred_real = pred_pos[mask]          # (n, 2), gradient-bearing
            n = true_real.shape[0]
            # Sinkhorn needs at least a couple of points to define a
            # non-degenerate transport plan; skip pathological tiny
            # slices the same way the other components do.
            if n < 4:
                continue

            # Optional subsampling for very large slices. The
            # subsample indices MUST be the same for true and pred,
            # otherwise we'd be comparing different cell subsets and
            # the loss becomes a distribution comparison rather than
            # an alignment comparison.
            if self._sk_subsample is not None and n > self._sk_subsample:
                with torch.no_grad():
                    idx = torch.randperm(n, device=true_real.device)[:self._sk_subsample]
                true_real = true_real[idx]
                pred_real = pred_real[idx]

            # SamplesLoss(x, y) with x, y of shape (N, D) returns a
            # 0-dim tensor — the divergence between the two clouds
            # under uniform weights. Gradient flows back through x.
            slice_losses.append(self._sk_loss_fn(pred_real, true_real))

        if not slice_losses:
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

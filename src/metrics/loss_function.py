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


def _graph_zero(masked_pred) -> torch.Tensor:
    """Return a scalar tensor of value 0 that's safely attachable to
    the loss-sum autograd graph.

    History (task #48 / #103): the obvious choice
    ``masked_pred.positions[0].sum() * 0.0`` looked safe because the
    value is zero, but with ``model.edm.mds_align_gradient=true``
    ``pred.positions`` is the output of MDS, whose ``eigh`` backward
    formula contains ``1/(λ_i − λ_j)`` terms that go to ±Inf at
    near-degenerate eigenvalues. Autograd's chain rule then
    computes ``0 × Inf = NaN`` even though the leaf gradient is
    nominally zero, poisoning the entire backward graph.

    The fix: source the graph-attached zero from a tensor whose
    backward path does NOT include any spectral op. In order of
    preference:
      1. ``masked_pred.edm_D`` — the EDM head's predicted squared-
         distance matrix. Always gradient-bearing under EDM
         (through projector → backbone), never goes through MDS.
      2. ``masked_pred.knn_logits`` — same role under the kNN-graph
         head.
      3. ``masked_pred.positions[0].sum() * 0.0`` — fallback only
         when no head-native gradient-bearing stash exists (e.g.,
         direct regression framework with no EDM/kNN head). In that
         configuration ``pred.positions`` IS the inner backbone's
         direct output (no MDS overwrite), so the zero is safe.
      4. A constant ``torch.tensor(0.0)`` if even ``positions``
         isn't available (defensive; shouldn't happen in practice).
    """
    edm_D = getattr(masked_pred, "edm_D", None)
    if edm_D is not None:
        return edm_D.sum() * 0.0
    knn_logits = getattr(masked_pred, "knn_logits", None)
    if knn_logits is not None:
        return knn_logits.sum() * 0.0
    positions = getattr(masked_pred, "positions", None)
    if positions is not None and len(positions) > 0:
        return positions[0].sum() * 0.0
    return torch.tensor(0.0)


# ---------------------------------------------------------------------------
# Differentiable Procrustes (Kabsch / Schönemann 1966) — for the
# rotation+reflection-invariant Sinkhorn loss.
# ---------------------------------------------------------------------------


def _procrustes_align_2d(
    x_src: torch.Tensor, x_ref: torch.Tensor,
) -> torch.Tensor:
    """Orthogonal Procrustes (rotation + reflection) alignment of
    ``x_src`` onto ``x_ref``.

    Both inputs must be 2-D point clouds of the same shape (N, 2)
    and pre-centered (mean-subtracted). Returns ``x_src @ R`` where
    ``R`` is the optimal orthogonal matrix minimising
    ``||x_src @ R - x_ref||_F``.

    Closed form (Schönemann 1966): ``R = U @ V^T`` where
    ``U·diag(S)·V^T = SVD(x_src^T · x_ref)``.

    Gradient pattern: R is computed UNDER ``torch.no_grad()`` and
    detached before the final ``x_src @ R`` multiply. The gradient
    therefore flows through ``x_src`` as if R were a CONSTANT
    rotation:
        d L / d x_src = (d L / d (x_src @ R)) @ R^T
    This is the "fixed-structure differentiable alignment" pattern
    — same idea as differentiable persistent homology (Hofer 2019),
    where the simplex structure is fixed per step but the values
    flow gradient. The alignment is recomputed FROM SCRATCH every
    step using the current x_src, so it stays fresh; we just don't
    let gradient back-propagate through the SVD that produced R.

    Why we detach R
    ---------------
    SVD's backward formula contains ``1/(σ_i² − σ_j²)`` terms that
    diverge at degenerate (or near-degenerate) singular values. A
    detached-snapshot threshold check (M_max, σ_min/σ_max,
    (σ_max−σ_min)/σ_max) is a NECESSARY-but-not-sufficient guard —
    it catches obvious-init-time degeneracy, but once the model
    has trained to a state where pred ≈ true (up to rotation), the
    Gram matrix M = pred_c^T · true_c becomes approximately a
    scaled identity, σ_max ≈ σ_min, and the SVD backward is
    intrinsically ill-conditioned. The 2026-05-27 observation:
    ``train_loss/sinkhorn`` bounded ~0.015 (FORWARD is fine) but
    backward still NaN's; SVD backward of nearly-degenerate M is
    the structural cause.

    Detaching R sidesteps the SVD-backward path entirely. The
    geometric meaning is preserved: we still pull each cell toward
    its truth-position in the best-aligned frame; we just don't
    learn to choose poses that are "easier to align" — which is a
    rotation-invariant target anyway, so no information is lost.

    Used by ``LossFunction._compute_sinkhorn`` to make the Sinkhorn
    divergence rotation+reflection-invariant. Same closed-form
    Procrustes math as ``models.edm_head._procrustes_align`` and
    ``models.knn_graph_head._procrustes_align`` — but those modules
    let gradient flow through R (controlled by the
    ``mds_align_gradient`` / ``spectral_layout_gradient`` flags),
    because the position head's output IS the post-alignment cloud
    in those paths and gradient must propagate through the
    alignment for the head to train. Here in the loss, the model's
    pred has already been produced; alignment is an internal step
    of THIS LOSS COMPONENT only, so detaching R is structurally
    safe.
    """
    # Compute R entirely outside autograd. The detached-snapshot
    # degeneracy check stays for defense-in-depth — if M is truly
    # zero (very first step before any training), the SVD itself
    # can return NaN in U/V (not just in its backward). The
    # threshold catches that case and returns x_src unchanged.
    with torch.no_grad():
        M_detached = (x_src.detach().T @ x_ref.detach())          # (2, 2)
        M_max = M_detached.abs().max()
        if (not torch.isfinite(M_max).item()) or M_max.item() < 1e-6:
            return x_src
        # SVD's own forward is well-defined for non-zero M even when
        # σ's are close — only the BACKWARD has the degeneracy.
        # Since we're under no_grad, we can compute R freely.
        U, _S, Vh = torch.linalg.svd(M_detached, full_matrices=False)
        R = (U @ Vh).detach()                                     # (2, 2)
    return x_src @ R                                              # (N, 2)


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
        # 1-dim PH only: Vietoris-Rips max edge length.
        self._ph_max_edge_length = float(_cfg_get(
            cfg, "model", "loss", "persistent_homology",
            "max_edge_length", default=0.2,
        ))
        # Step counter for `frequency` knob; cache for true PD info.
        # For dim=0 the cache stores sorted MST edge lengths; for
        # dim=1 it stores the persistence-pair simplex indices so
        # we can re-evaluate (birth, death) values from current
        # true positions cheaply. Cache size is bounded by the
        # dataset's slice count, which is small (LUNA cortex: 33;
        # ABCA Animal-1: 147). No eviction needed.
        self._ph_step_counter = 0
        self._ph_true_cache: dict = {}
        if self._ph_dim not in (0, 1):
            raise NotImplementedError(
                f"model.loss.persistent_homology.dim={self._ph_dim} not "
                f"supported. Only 0-dim (MST formulation, scipy) and "
                f"1-dim (loop structure, gudhi Vietoris-Rips) are "
                f"implemented."
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
        # Backend selection. Default "tensorized" because the
        # KeOps-accelerated backends ("online", "multiscale", and
        # "auto" when N is large) need a working KeOps install which
        # is fragile in practice (LLVM + CUDA + Python ABI all have
        # to line up). See the comment in configs/model/default.yaml
        # for the full menu and how to verify KeOps if you want
        # to switch to "auto".
        self._sk_backend = str(_cfg_get(cfg, "model", "loss",
                                        "sinkhorn", "backend",
                                        default="tensorized"))
        # Differentiable Procrustes alignment before Sinkhorn.
        # Default ON because Sinkhorn-on-unaligned-clouds measures
        # rotation noise rather than per-cell placement error in
        # scgg's setup (EDM Procrustes-to-noise + FM rotation
        # augmentation). See _compute_sinkhorn docstring for the
        # full diagnosis.
        self._sk_procrustes_align = bool(_cfg_get(
            cfg, "model", "loss", "sinkhorn", "procrustes_align",
            default=True,
        ))
        # Scale-invariant mode: divide both clouds by truth's RMS
        # before Sinkhorn so ``blur`` is in units of "RMS extent".
        self._sk_scale_invariant = bool(_cfg_get(
            cfg, "model", "loss", "sinkhorn", "scale_invariant",
            default=False,
        ))
        # Debiased Sinkhorn divergence. Default True (mathematically
        # clean: S_ε(α, β) = 0 iff α = β). Set to False as a NaN-
        # backward fallback — biased L_ε runs 1 OT problem instead
        # of 3, removing two backward paths that can independently
        # produce log-sum-exp underflow at small blur. See the
        # default.yaml comment for the full hierarchy of fixes.
        self._sk_debias = bool(_cfg_get(
            cfg, "model", "loss", "sinkhorn", "debias", default=True,
        ))
        # Lazy-init: only build the SamplesLoss object on first call,
        # so users who never enable the component never pay the
        # geomloss import cost.
        self._sk_loss_fn = None

        # Coarse-cluster centroid MSE. Auto-enabled by the
        # CoarseToFineWrapper; needs the wrapper to stash
        # ``_predicted_cluster_centroids`` and ``_cluster_ids`` on
        # the pred DataHolder.
        self._cc_enabled = bool(_cfg_get(cfg, "model", "loss",
                                         "coarse_centroid_mse", "enabled",
                                         default=False))
        self._cc_weight = float(_cfg_get(cfg, "model", "loss",
                                         "coarse_centroid_mse", "weight",
                                         default=1.0))

        # Cluster balance regularizer for Gumbel-softmax mode in the
        # c2f wrapper. Pulled in from a stash on the pred DataHolder
        # (the wrapper computes the entropy-based reg term during
        # forward and writes ``_cluster_balance_loss`` /
        # ``_cluster_balance_weight``). Active iff the wrapper has
        # a non-zero ``cluster_balance_weight`` AND we're in
        # training mode (wrapper sets the stash to None at eval).
        # We don't gate this on a config knob in the same way as
        # other components — the wrapper is the source of truth.
        self._cluster_balance_enabled = True

        # EDM (Euclidean Distance Matrix) loss — scGG fundamental
        # method #1. Reads ``masked_pred.edm_D`` (B, N, N squared
        # distances) stashed by EDMOutputWrapper; compares to the
        # SQUARED pairwise-distance matrix of masked_true.positions.
        # Auto-enabled by the wrapper-config-patching in
        # FullDenoisingDiffusion.__init__ when cfg.model.edm.enabled.
        self._edm_enabled = bool(_cfg_get(cfg, "model", "loss",
                                          "edm_distance_mse", "enabled",
                                          default=False))
        self._edm_weight = float(_cfg_get(cfg, "model", "loss",
                                          "edm_distance_mse", "weight",
                                          default=1.0))

        # k-NN graph loss — scGG fundamental method #4. Reads
        # ``masked_pred.knn_logits`` (B, N, N edge logits) stashed by
        # KNNGraphOutputWrapper; supervises with contrastive BCE on
        # positive edges (true k-NN) vs sampled negatives. Auto-
        # enabled by the wrapper-config-patching when
        # cfg.model.knn_graph.enabled.
        self._knn_graph_enabled = bool(_cfg_get(cfg, "model", "loss",
                                                "knn_graph_loss", "enabled",
                                                default=False))
        self._knn_graph_weight = float(_cfg_get(cfg, "model", "loss",
                                                "knn_graph_loss", "weight",
                                                default=1.0))
        self._knn_graph_k = int(_cfg_get(cfg, "model", "loss",
                                         "knn_graph_loss", "k", default=10))
        self._knn_graph_n_neg = int(_cfg_get(cfg, "model", "loss",
                                             "knn_graph_loss", "n_negatives",
                                             default=20))

        # Shape-matching loss — rotation-invariant covariance-eigenvalue
        # MSE on the predicted vs true point cloud. Targets the "scgg
        # predictions look too isotropic" failure mode that pairwise-
        # distance MSE doesn't directly supervise.
        self._shape_enabled = bool(_cfg_get(cfg, "model", "loss",
                                            "shape_matching", "enabled",
                                            default=False))
        self._shape_weight = float(_cfg_get(cfg, "model", "loss",
                                            "shape_matching", "weight",
                                            default=0.1))
        # Which scalar(s) of the covariance to match. See the long
        # docstring in default.yaml for the rationale behind each.
        self._shape_variant = str(_cfg_get(cfg, "model", "loss",
                                           "shape_matching", "variant",
                                           default="eigvals")).lower()
        if self._shape_variant not in ("eigvals", "ratio", "covariance"):
            raise ValueError(
                f"model.loss.shape_matching.variant must be one of "
                f"'eigvals', 'ratio', 'covariance'; got "
                f"{self._shape_variant!r}."
            )
        # Per-batch diagnostic stash populated by
        # ``_compute_shape_matching`` and consumed by ``forward`` /
        # ``log_epoch_metrics`` to emit ``{train,val}_shape/...`` keys
        # to wandb. List of per-slice dicts:
        #   {"eig_min_true", "eig_max_true", "ratio_true",
        #    "eig_min_pred", "eig_max_pred", "ratio_pred"}.
        # Cleared after each log emission so we never publish stale
        # numbers from a prior batch.
        self._last_shape_diagnostics: List[Dict[str, float]] = []

        # Latent diffusion losses — automatically active iff the LDM
        # framework wrote the relevant stashes on the pred DataHolder
        # (pred._ldm_z_0_pred / pred._ldm_z_0_target for the FM-on-z
        # loss; pred._ldm_mu / pred._ldm_logvar for the KL). They
        # return a graph-attached zero on any pred that doesn't have
        # those stashes — so toggling between frameworks just works.
        self._ldm_fm_weight = float(_cfg_get(cfg, "model", "loss",
                                             "latent_fm_mse", "weight",
                                             default=1.0))
        self._ldm_kl_weight = float(_cfg_get(cfg, "model", "loss",
                                             "latent_kl", "weight",
                                             default=0.001))
        # ``always-on`` semantics: registry entry has enabled=True so
        # the component runs whenever the pred carries the stash. For
        # non-LDM runs the component returns 0 instantly.
        self._ldm_fm_enabled = True
        self._ldm_kl_enabled = True

        # Per-component warmup. For the first ``warmup_steps`` training
        # steps, the named component's loss VALUE is still computed
        # (under torch.no_grad — so it shows up on wandb for
        # visibility) but its gradient contribution is skipped
        # entirely. Use for losses whose backward is numerically
        # unstable at training init (Sinkhorn at small ``blur`` is
        # the canonical case — log-sum-exp underflow at random init
        # positions produces NaN in the SamplesLoss backward).
        # Default 0 = no warmup.
        #
        # IMPORTANT — design contract:
        #   1. Loss value is still computed under no_grad and logged
        #      to wandb under ``train_loss/<name>``. The
        #      ``<name>_weighted`` key reports 0 during warmup so the
        #      sum on the wandb chart accurately reflects what's
        #      training the model. ``<name>_warmup_active`` flag
        #      logged as 1.0 / 0.0 so you can SEE on the wandb chart
        #      exactly when each warmup ends — no silent silencing.
        #   2. The autograd graph through the component is NEVER
        #      built during warmup, so the unstable backward can't
        #      poison anything.
        #   3. Step counter increments at each ``forward(...,
        #      train_stage=True)`` call. Val-stage forwards don't
        #      advance the counter.
        self._warmup_steps = {
            name: int(_cfg_get(cfg, "model", "loss", name,
                               "warmup_steps", default=0))
            for name in (
                "sinkhorn", "shape_matching", "persistent_homology",
                "knn_rank", "pairwise_distance_mse", "edm_distance_mse",
                "knn_graph_loss", "coarse_centroid_mse",
                "latent_fm_mse", "latent_kl",
            )
        }
        # Global step counter — advanced only on training-stage forwards.
        self._step_count = 0

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
            ("coarse_centroid_mse", self._cc_enabled, self._cc_weight,
             self._compute_coarse_centroid_mse),
            # cluster_balance is always "enabled" in the registry; the
            # actual value is gated by whether the wrapper stashed
            # anything (returns a graph-attached zero otherwise).
            # Weight 1.0 here because the WRAPPER carries the
            # per-run weight (stashed on the DataHolder) and we just
            # forward it through.
            ("cluster_balance", self._cluster_balance_enabled, 1.0,
             self._compute_cluster_balance),
            # scGG fundamental method #1 (EDM diffusion): MSE on the
            # predicted pairwise-squared-distance matrix.
            ("edm_distance_mse", self._edm_enabled, self._edm_weight,
             self._compute_edm_distance_mse),
            # scGG fundamental method #4 (k-NN graph): contrastive BCE
            # on positive/negative edges drawn from true positions.
            ("knn_graph_loss", self._knn_graph_enabled, self._knn_graph_weight,
             self._compute_knn_graph_loss),
            # Shape-matching loss: covariance-eigenvalue MSE; targets
            # the over-isotropic-prediction failure mode pairwise MSE
            # can't directly see.
            ("shape_matching", self._shape_enabled, self._shape_weight,
             self._compute_shape_matching),
            # Latent-diffusion losses — always in the registry; both
            # return graph-attached zero when the LDM stashes aren't
            # on the pred (so non-LDM runs see no impact).
            ("latent_fm_mse", self._ldm_fm_enabled, self._ldm_fm_weight,
             self._compute_latent_fm_mse),
            ("latent_kl",     self._ldm_kl_enabled, self._ldm_kl_weight,
             self._compute_latent_kl),
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
                if self._ph_dim == 1:
                    extra += f", max_edge={self._ph_max_edge_length:.3g}"
                if self._ph_frequency > 1:
                    extra += f", every {self._ph_frequency} steps"
                if self._ph_cache_true:
                    extra += ", cache_true=on"
                if self._ph_subsample:
                    extra += f", subsample={self._ph_subsample}"
            elif name == "sinkhorn":
                extra = (
                    f", p={self._sk_p}, blur={self._sk_blur:.3g}, "
                    f"scaling={self._sk_scaling:.3g}, "
                    f"backend={self._sk_backend}, "
                    f"procrustes_align={self._sk_procrustes_align}, "
                    f"scale_invariant={self._sk_scale_invariant}, "
                    f"debias={self._sk_debias}"
                )
                if self._sk_subsample:
                    extra += f", subsample={self._sk_subsample}"
            elif name == "coarse_centroid_mse":
                extra = " (auto-wired by CoarseToFineWrapper)"
            elif name == "edm_distance_mse":
                extra = " (auto-wired by EDMOutputWrapper)"
            elif name == "knn_graph_loss":
                extra = (
                    f" (auto-wired by KNNGraphOutputWrapper, "
                    f"k={self._knn_graph_k}, n_neg={self._knn_graph_n_neg})"
                )
            elif name == "shape_matching":
                extra = " (covariance-eigenvalue MSE, rotation-invariant)"
            active.append(f"{name}(w={w:.3g}{extra})")
        print(f"[LossFunction] active components: {', '.join(active) or '(none!)'}")

        # -------- Fail-loud check: silently-silenced auxiliary losses --------
        # These components compute their loss VALUE on
        # ``pred.positions``. Under EDM with ``mds_align=true`` AND
        # ``mds_align_gradient=false`` (legacy default), the EDM head
        # overwrites pred.positions under torch.no_grad() with
        # detached MDS coordinates — making the gradient path through
        # pred.positions IDENTICALLY ZERO. The loss values still
        # move (because edm_distance_mse improves D_sq, which
        # improves the post-MDS positions), but the model never
        # learns from these auxiliary signals. This caused a real
        # silent bug where 5 ablation runs varying ONLY in their
        # auxiliary loss config produced bit-identical Spearman to
        # 16 decimal places.
        #
        # Refuse to start training when this combination is
        # configured. The user must either (a) enable
        # mds_align_gradient=True, (b) disable the listed
        # components, or (c) disable EDM.
        edm_on = bool(_cfg_get(cfg, "model", "edm", "enabled", default=False))
        edm_mds = bool(_cfg_get(cfg, "model", "edm", "mds_align", default=True))
        edm_grad = bool(_cfg_get(cfg, "model", "edm", "mds_align_gradient",
                                 default=False))
        edm_silenced = edm_on and edm_mds and not edm_grad

        # SAME structural bug class for the kNN-graph head: when
        # ``knn_graph.spectral_layout=true`` AND
        # ``knn_graph.spectral_layout_gradient=false``, pred.positions
        # is detached after the Laplacian eigenmaps step. Audit-found
        # by the 2026-05-27 silent-silencing review.
        knn_graph_on = bool(_cfg_get(cfg, "model", "knn_graph", "enabled",
                                     default=False))
        knn_spectral = bool(_cfg_get(cfg, "model", "knn_graph",
                                     "spectral_layout", default=True))
        knn_grad = bool(_cfg_get(cfg, "model", "knn_graph",
                                 "spectral_layout_gradient", default=False))
        knn_silenced = knn_graph_on and knn_spectral and not knn_grad

        if edm_silenced or knn_silenced:
            silenced: List[str] = []
            if self._shape_enabled:
                silenced.append("shape_matching")
            if self._sk_enabled:
                silenced.append("sinkhorn")
            if self._knn_enabled:
                silenced.append("knn_rank")
            if self._ph_enabled:
                silenced.append("persistent_homology")
            # pairwise_distance_mse-on-positions also operates on
            # pred.positions. It's normally AUTO-disabled when EDM or
            # kNN are on (via ``replace_position_loss=true``) — but
            # we should still flag it if the user explicitly
            # re-enabled it under a silenced path.
            if self._pwd_enabled:
                silenced.append("pairwise_distance_mse")
            if silenced:
                # Build the specific reason chain.
                reasons = []
                if edm_silenced:
                    reasons.append(
                        "model.edm.enabled=true AND "
                        "model.edm.mds_align=true AND "
                        "model.edm.mds_align_gradient=false"
                    )
                if knn_silenced:
                    reasons.append(
                        "model.knn_graph.enabled=true AND "
                        "model.knn_graph.spectral_layout=true AND "
                        "model.knn_graph.spectral_layout_gradient=false"
                    )
                # Tailor the fix suggestion to which head is silencing.
                gradient_flag_hint = []
                if edm_silenced:
                    gradient_flag_hint.append(
                        "model.edm.mds_align_gradient=true"
                    )
                if knn_silenced:
                    gradient_flag_hint.append(
                        "model.knn_graph.spectral_layout_gradient=true"
                    )
                raise ValueError(
                    "Silent-silencing combination detected. The "
                    f"following loss components are enabled: {silenced}. "
                    "All of them compute their loss on "
                    "``pred.positions`` — but the following config "
                    f"silences the gradient through that path: "
                    f"{' AND '.join(reasons)}. Result: the loss "
                    "values still move on the wandb chart (because "
                    "the underlying head's gradient-bearing output — "
                    "edm_D or knn_logits — keeps improving), but the "
                    "model is NOT actually trained by these auxiliary "
                    "losses. To proceed, do exactly one of:\n"
                    "  1) Enable the gradient-restoring flag(s): "
                    f"{', '.join(gradient_flag_hint)}. The MDS / "
                    "     spectral-layout path then carries gradient "
                    "     AND a NaN-guard backward hook contains any "
                    "     eigh-degenerate slice.\n"
                    "  2) Disable the listed loss components and "
                    "     rely on the head-native loss alone "
                    "     (edm_distance_mse / knn_graph_loss).\n"
                    "  3) Disable the head (model.edm.enabled=false "
                    "     OR model.knn_graph.enabled=false) — falls "
                    "     back to the inner backbone's direct "
                    "     pred.positions output, gradient flows "
                    "     unconditionally."
                )

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
            # No slice had enough cells. Return a gradient-zero
            # that's safely attachable to the loss-sum graph WITHOUT
            # traversing MDS (see _graph_zero docstring).
            return _graph_zero(masked_pred)

        return torch.mean(torch.stack(slice_losses))

    def _compute_persistent_homology(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> torch.Tensor:
        """Dispatch persistent-homology loss by configured dimension.

        * ``dim=0`` — MST-based, scipy. The 0-dim persistence diagram
          of a point cloud is exactly the multiset of MST edge
          lengths; W2² between two equal-cardinality PD_0's reduces
          to MSE between sorted edge-length vectors.
        * ``dim=1`` — loops, gudhi Vietoris-Rips. The 1-dim PD
          captures loop structure. Birth = filtration value of the
          edge that creates the loop, death = filtration value of
          the triangle that fills it; for VR, both reduce to edge
          lengths between specific cell pairs (differentiable).

        Frequency-skip and per-slice subsample logic apply to both
        dims. The cache key, on the other hand, stores different
        objects (sorted edge lengths for dim=0, simplex-index pairs
        for dim=1).
        """
        if self._ph_dim == 0:
            return self._compute_ph_dim0(masked_pred, masked_true)
        elif self._ph_dim == 1:
            return self._compute_ph_dim1(masked_pred, masked_true)
        else:
            raise NotImplementedError(self._ph_dim)

    def _compute_ph_dim0(
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
                return _graph_zero(masked_pred)
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

    # ------------------------------------------------------------------
    # 1-dim PH (loop structure) via gudhi Vietoris-Rips
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pd1_simplex_pairs(
        positions_np, max_edge_length: float,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int, int]]]:
        """Run gudhi VR on a numpy point cloud, return the list of
        (birth_edge_indices, death_triangle_indices) for 1-dim
        persistence pairs.

        Pure gudhi work — runs on CPU, no gradients.
        Returns a list of ``((i, j), (a, b, c))`` tuples.
        """
        import gudhi
        rips = gudhi.RipsComplex(
            points=positions_np, max_edge_length=float(max_edge_length),
        )
        st = rips.create_simplex_tree(max_dimension=2)
        # ``persistence`` MUST be called before ``persistence_pairs``
        # so gudhi populates the diagram internally.
        st.persistence(homology_coeff_field=2, min_persistence=0.0)
        out: List[Tuple[Tuple[int, int], Tuple[int, int, int]]] = []
        for birth_simplex, death_simplex in st.persistence_pairs():
            # 1-dim feature: birth is an edge (2 vertices), death is
            # a triangle (3 vertices). Skip 0-dim and infinite pairs.
            if len(birth_simplex) != 2 or len(death_simplex) != 3:
                continue
            i, j = int(birth_simplex[0]), int(birth_simplex[1])
            a, b, c = (int(death_simplex[0]),
                       int(death_simplex[1]),
                       int(death_simplex[2]))
            out.append(((i, j), (a, b, c)))
        return out

    @staticmethod
    def _pd1_values_from_pairs(
        d_matrix: torch.Tensor,
        simplex_pairs: List[Tuple[Tuple[int, int], Tuple[int, int, int]]],
    ) -> torch.Tensor:
        """Given a (gradient-bearing) ``cdist`` matrix and a list of
        gudhi persistence-pair simplex indices, return a (M, 2)
        tensor of (birth, death) values where:

          * birth = edge length between the two vertices of the
            birth simplex (the edge that creates the loop in the VR
            filtration);
          * death = MAX edge length among the three edges of the
            death-triangle (VR-complex filtration value of a
            triangle is the max of its edge filtration values).

        Both values are differentiable functions of the input
        positions via the cdist chain rule.
        """
        if not simplex_pairs:
            return d_matrix.new_zeros((0, 2))
        births: List[torch.Tensor] = []
        deaths: List[torch.Tensor] = []
        for (i, j), (a, b, c) in simplex_pairs:
            births.append(d_matrix[i, j])
            # max of the three edges of the death triangle
            e_ab = d_matrix[a, b]
            e_ac = d_matrix[a, c]
            e_bc = d_matrix[b, c]
            deaths.append(torch.stack([e_ab, e_ac, e_bc]).max())
        return torch.stack(
            [torch.stack(births), torch.stack(deaths)], dim=-1
        )

    @staticmethod
    def _pad_pd_with_diagonal(
        pd: torch.Tensor, target_len: int,
    ) -> torch.Tensor:
        """Pad a (M, 2) persistence diagram to (target_len, 2) using
        diagonal points (birth = death = 0). Diagonal points have
        zero lifetime so they sort to the end under lifetime-
        descending order — they only get matched against unmatched
        non-trivial features in the other PD, contributing their
        own (birth-death)² / 2 to the loss (standard Wasserstein
        with diagonal-slack convention).
        """
        m = pd.shape[0]
        if m >= target_len:
            return pd
        pad = pd.new_zeros((target_len - m, 2))
        return torch.cat([pd, pad], dim=0)

    def _compute_ph_dim1(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> torch.Tensor:
        """1-dim persistent homology loss via gudhi Vietoris-Rips.

        Differentiability: gudhi computes the simplex structure of
        the VR filtration non-differentiably; we then re-evaluate
        the (birth, death) VALUES from the gradient-bearing cdist
        matrix of the predicted positions. Gradient flows through
        positions via cdist. Standard "fixed-structure" trick
        (Hofer 2019, Bruel-Gabrielsson 2020) — same approach as
        ``_compute_ph_dim0``.

        Matching: predicted PD vs true PD are matched by sorted
        lifetime (descending), padded to equal length with diagonal
        points (birth=death=0, lifetime=0). The loss is the sum of
        squared (birth, death) coordinate differences across the
        matched pairs — a tractable approximation to true W2²
        bottleneck matching, smooth in the input coordinates.

        Speed: VR is heavier than scipy MST. The ``frequency`` and
        ``cache_true`` knobs apply identically; ``cache_true`` stores
        the simplex-index pairs (which depend ONLY on true positions
        and thus are step-invariant).
        """
        try:
            import gudhi  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "model.loss.persistent_homology.dim=1 requires "
                "the gudhi package. Install via "
                "`pip install gudhi`. Underlying error: " + str(e)
            )

        def _zero():
            if len(masked_pred.positions) > 0:
                return _graph_zero(masked_pred)
            return torch.tensor(0.0)

        # Same frequency-skip semantics as dim=0.
        self._ph_step_counter += 1
        if (self._ph_step_counter - 1) % self._ph_frequency != 0:
            return _zero()

        max_edge = float(self._ph_max_edge_length)
        slice_losses: List[torch.Tensor] = []
        subsample = self._ph_subsample

        for true_pos, pred_pos, mask in zip(
            masked_true.positions, masked_pred.positions, masked_true.node_mask,
        ):
            true_pos_m = true_pos[mask]
            pred_pos_m = pred_pos[mask]
            n = true_pos_m.shape[0]
            # Need at least 4 points to even have a chance of a 1-cycle.
            if n < 4:
                continue

            using_subsample = subsample is not None and n > subsample
            if using_subsample:
                with torch.no_grad():
                    idx = torch.randperm(n, device=true_pos_m.device)[:subsample]
                true_pos_m = true_pos_m[idx]
                pred_pos_m = pred_pos_m[idx]
                n = subsample

            # ---- True PD_1 simplex pairs: cache by content fingerprint ----
            cache_key = None
            true_simplex_pairs = None
            if self._ph_cache_true and not using_subsample:
                with torch.no_grad():
                    cache_key = (
                        "ph1",
                        n,
                        float(true_pos_m[0, 0].item()),
                        float(true_pos_m[-1, -1].item()),
                        float(true_pos_m.sum().item()),
                    )
                true_simplex_pairs = self._ph_true_cache.get(cache_key)

            if true_simplex_pairs is None:
                true_pos_np = true_pos_m.detach().cpu().numpy()
                true_simplex_pairs = self._extract_pd1_simplex_pairs(
                    true_pos_np, max_edge,
                )
                if cache_key is not None:
                    self._ph_true_cache[cache_key] = true_simplex_pairs

            # ---- Pred PD_1 simplex pairs: re-extract every step ----
            pred_pos_np = pred_pos_m.detach().cpu().numpy()
            pred_simplex_pairs = self._extract_pd1_simplex_pairs(
                pred_pos_np, max_edge,
            )

            # If neither cloud has any 1-dim feature within max_edge,
            # there's nothing to compare. Skip this slice — but DON'T
            # let the loss silently become a no-op. Add a tiny detached
            # zero so the loss term is still tracked for logging.
            if not true_simplex_pairs and not pred_simplex_pairs:
                slice_losses.append(_zero())
                continue

            # ---- Compute (birth, death) values from current positions ----
            d_true = torch.cdist(true_pos_m, true_pos_m, p=2)
            d_pred = torch.cdist(pred_pos_m, pred_pos_m, p=2)
            with torch.no_grad():
                true_pd = self._pd1_values_from_pairs(
                    d_true.detach(), true_simplex_pairs,
                )                                                       # (M_true, 2)
            pred_pd = self._pd1_values_from_pairs(
                d_pred, pred_simplex_pairs,
            )                                                           # (M_pred, 2) — grad-bearing

            # Pad to equal length so we can match by sorted lifetime.
            target_len = max(true_pd.shape[0], pred_pd.shape[0])
            true_padded = self._pad_pd_with_diagonal(true_pd, target_len)
            pred_padded = self._pad_pd_with_diagonal(pred_pd, target_len)

            # Sort by lifetime descending. Lifetime = death - birth.
            with torch.no_grad():
                true_lifetimes = true_padded[:, 1] - true_padded[:, 0]
                true_order = torch.argsort(true_lifetimes, descending=True)
            pred_lifetimes = pred_padded[:, 1] - pred_padded[:, 0]
            pred_order = torch.argsort(
                pred_lifetimes.detach(), descending=True,
            )

            true_sorted = true_padded[true_order]
            pred_sorted = pred_padded[pred_order]

            # W2² with diagonal-slack matching.
            #
            # Three cases per matched pair (true_sorted[i], pred_sorted[i]):
            #
            #   (a) Both real (lifetime > 0):
            #         cost = (b_t - b_p)² + (d_t - d_p)²       — standard L2² pair cost
            #
            #   (b) One side is a diagonal pad (b = d = 0 — written by
            #       _pad_pd_with_diagonal), other side is real:
            #         cost = (b - d)² / 2                       — distance² from the
            #         real point to its nearest diagonal projection ((b+d)/2,(b+d)/2).
            #         This is the canonical W₂² diagonal-slack convention.
            #
            #   (c) Both diagonal pads:
            #         cost = 0.
            #
            # The original code used the case-(a) formula EVERYWHERE,
            # penalising case (b) by (b² + d²) instead of (b−d)²/2.
            # For matched pairs where (b, d) lie close to the diagonal
            # this over-penalises by up to 2×, which biased the model
            # toward suppressing all 1-features whenever the predicted
            # PD had higher cardinality than the truth.
            true_lifetimes_s = (true_sorted[:, 1] - true_sorted[:, 0]).detach()
            pred_lifetimes_s = (pred_sorted[:, 1] - pred_sorted[:, 0]).detach()
            # Threshold guards against fp rounding noise — real features
            # have strictly positive lifetimes (gudhi only emits PD
            # points with death > birth), pads have exactly zero.
            eps_pad = 1e-10
            is_pad_true = true_lifetimes_s < eps_pad
            is_pad_pred = pred_lifetimes_s < eps_pad
            both_real = (~is_pad_true) & (~is_pad_pred)
            pad_t_only = is_pad_true & (~is_pad_pred)
            pad_p_only = is_pad_pred & (~is_pad_true)

            # Case (a) — paired real features.
            real_cost = ((true_sorted - pred_sorted) ** 2).sum(dim=-1)
            # Case (b1) — pred real, true pad: penalise pred by its
            # diagonal-projection distance².
            pred_diag_cost = (
                (pred_sorted[:, 0] - pred_sorted[:, 1]) ** 2 * 0.5
            )
            # Case (b2) — true real, pred pad: penalise by true's
            # diagonal-projection distance². This term has no gradient
            # (the true PD is gradient-free anyway) but contributes
            # to the slice loss magnitude, so we include it for
            # numerical comparability with the all-paired case.
            true_diag_cost = (
                (true_sorted[:, 0] - true_sorted[:, 1]) ** 2 * 0.5
            )
            zero_cost = torch.zeros_like(real_cost)

            per_pair = torch.where(
                both_real, real_cost,
                torch.where(
                    pad_t_only, pred_diag_cost,
                    torch.where(pad_p_only, true_diag_cost, zero_cost),
                ),
            )
            slice_loss = per_pair.mean()
            slice_losses.append(slice_loss)

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
        slices. Debiasing makes ``S_ε`` zero when the clouds are equal,
        so the loss landscape behaves like vanilla Wasserstein near
        optimum.

        Why the alignment step matters
        ------------------------------
        Sinkhorn directly compares (x, y) tuples — it's NOT rotation
        or reflection invariant. scgg's predictions live in an
        arbitrary frame:

          * The EDM head does MDS + Procrustes alignment to the
            NOISY x_t (so the frame depends on the noise sample).
          * The FM framework applies rotation+reflection augmentation
            at train time, perturbing the frame every step.

        Two clouds that are identical up to a rotation have Sinkhorn
        divergence proportional to the cloud's extent — so without an
        alignment step, the loss is dominated by rotation noise
        rather than per-cell placement error. The minimum the model
        can find is scale-collapse to zero (which makes any rotation
        the identity).

        This is exactly the failure mode reported: the unaligned
        Sinkhorn doesn't help when ON alone (collapse wins), and only
        helps when stacked WITH pairwise_distance_mse (whose
        rotation-invariant shape signal keeps positions from
        collapsing — at which point the EDM head's Procrustes step
        pulls the prediction into a stable frame and unaligned
        Sinkhorn becomes a usable fine-tuning signal).

        The fix is to make Sinkhorn rotation+reflection-invariant
        directly: run a differentiable Kabsch / orthogonal Procrustes
        alignment of pred onto true BEFORE Sinkhorn. With this on,
        Sinkhorn measures the residual per-cell placement error
        modulo the global frame — which is the signal we actually
        want.

        Implementation
        --------------
        Per slice, in order:

          1. Mean-centre both clouds (translation invariance).
          2. If ``procrustes_align``: rotate+reflect pred onto true
             via SVD of (pred^T · true), with R = U·V^T (Schönemann
             1966 / Kabsch). R is computed UNDER no_grad and detached
             — gradient flows through pred_c as if R were a constant
             rotation. See ``_procrustes_align_2d`` docstring for
             why we detach R rather than flowing gradient through
             SVD's backward.
          3. If ``scale_invariant``: divide both clouds by the
             truth's RMS distance to centroid, so ``blur`` is in
             units of "RMS extent" rather than raw position units.
          4. Call geomloss.SamplesLoss on the aligned (and optionally
             scaled) clouds.

        Gradient: flows through ``pred_pos`` via (a) the (constant-R)
        rotation applied to pred_c, and (b) SamplesLoss's
        differentiable Sinkhorn iteration. ``true_pos`` is explicitly
        detached so no gradient leaks into the dataloader output.
        R itself is non-differentiable by design — sidesteps the
        SVD-backward NaN trap at near-degenerate σ (which is the
        common steady-state regime after the model has learned to
        match true positions up to rotation).
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
                # "tensorized" by default — see __init__ comment for
                # why we don't trust "auto" / KeOps backends here.
                backend=self._sk_backend,
                # Debiased S_ε(α,β) = L_ε(α,β) − ½(L_ε(α,α)+L_ε(β,β))
                # is mathematically cleaner (zero when α=β) but runs
                # 3 OT problems in parallel, so it has 3× the NaN-
                # backward surface area at small blur. Set
                # ``model.loss.sinkhorn.debias=false`` to fall back to
                # plain biased L_ε(α,β) — see default.yaml comment for
                # the full hierarchy of NaN-mitigation knobs.
                debias=self._sk_debias,
            )

        slice_losses: List[torch.Tensor] = []
        for true_pos, pred_pos, mask in zip(
            masked_true.positions, masked_pred.positions, masked_true.node_mask,
        ):
            true_real = true_pos[mask].detach()  # never gradient-bearing
            pred_real = pred_pos[mask]           # (n, 2), gradient-bearing
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

            # (1) Centre both clouds. Translation invariance.
            true_c = true_real - true_real.mean(dim=0, keepdim=True)
            pred_c = pred_real - pred_real.mean(dim=0, keepdim=True)

            # (2) Differentiable Procrustes (rotation + reflection)
            #     of pred onto true. Same closed form as
            #     ``models.edm_head._procrustes_align`` /
            #     ``models.knn_graph_head._procrustes_align`` so the
            #     alignment math is uniform across the codebase.
            if self._sk_procrustes_align:
                pred_c = _procrustes_align_2d(pred_c, true_c)

            # (3) Optional scale invariance: rescale both clouds by
            #     truth's RMS distance to centroid. ``blur`` is then
            #     in units of "RMS extent" rather than raw positions.
            if self._sk_scale_invariant:
                with torch.no_grad():
                    scale = true_c.pow(2).sum(dim=-1).mean().sqrt().clamp_min(1e-6)
                true_c = true_c / scale
                pred_c = pred_c / scale

            # (4) Sinkhorn divergence on the aligned clouds. Gradient
            #     flows through pred_c (via SVD backward + SamplesLoss).
            slice_losses.append(self._sk_loss_fn(pred_c, true_c))

        if not slice_losses:
            return _graph_zero(masked_pred)
        return torch.mean(torch.stack(slice_losses))

    def _compute_coarse_centroid_mse(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> torch.Tensor:
        """Auxiliary coarse-to-fine loss: MSE between the predicted
        cluster centroids (from the CoarseToFineWrapper) and the
        TRUE cluster centroids (computed from the true positions
        and the wrapper-determined cluster assignments).

        The wrapper stashes its outputs on the pred DataHolder:
            masked_pred._predicted_cluster_centroids : (B, K, 2)
            masked_pred._cluster_ids                  : (B, N) long
            masked_pred._n_clusters                   : int K

        We compute true centroids here by scatter-mean of
        ``masked_true.positions`` into the same cluster bins.

        The MSE is on PAIRWISE DISTANCES between centroids (same
        rotation-invariant treatment as the cell-level loss), not
        on raw 2-D coordinates — that keeps the coarse signal
        consistent with the rest of the pipeline's translation/
        rotation invariance.
        """
        # If the wrapper isn't in use, the stashed attributes won't
        # exist. Return a graph-attached zero so downstream summation
        # has something to add into.
        pred_centroids = getattr(masked_pred, "_predicted_cluster_centroids", None)
        cluster_ids = getattr(masked_pred, "_cluster_ids", None)
        K = getattr(masked_pred, "_n_clusters", None)
        if pred_centroids is None or cluster_ids is None or K is None:
            return _graph_zero(masked_pred)

        B, N = cluster_ids.shape
        device = pred_centroids.device

        # Compute true cluster centroids per slice.
        safe_ids = cluster_ids.clamp(min=0)
        valid = masked_true.node_mask & (cluster_ids >= 0)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        combined_id = (batch_idx * K + safe_ids).reshape(-1)
        valid_f = valid.float().reshape(-1, 1)

        pos_flat = masked_true.positions.reshape(B * N, 2) * valid_f
        count_flat = valid.float().reshape(B * N)

        sum_pos = pred_centroids.new_zeros(B * K, 2)
        sum_cnt = pred_centroids.new_zeros(B * K)
        sum_pos.scatter_add_(0, combined_id.unsqueeze(-1).expand(-1, 2), pos_flat)
        sum_cnt.scatter_add_(0, combined_id, count_flat)

        cnt_safe = sum_cnt.clamp(min=1.0)
        true_centroids = (sum_pos / cnt_safe.unsqueeze(-1)).reshape(B, K, 2)
        # Mask out empty clusters from the loss so they don't bias it.
        # An empty cluster has sum_cnt = 0; mark it.
        cluster_present = (sum_cnt > 0).reshape(B, K)                   # (B, K)

        # Pairwise-distance MSE between centroid sets, per slice.
        # Matches the rotation-invariance convention of the cell-level
        # pairwise distance loss. Empty clusters (cluster_present=
        # False) get dropped from the per-slice cdist by masking.
        losses = []
        for b in range(B):
            mask_b = cluster_present[b]
            if mask_b.sum().item() < 2:
                # need at least 2 clusters to have any pairwise distance
                continue
            d_true = torch.cdist(true_centroids[b][mask_b],
                                 true_centroids[b][mask_b], p=2)
            d_pred = torch.cdist(pred_centroids[b][mask_b],
                                 pred_centroids[b][mask_b], p=2)
            losses.append(self.mse(d_pred, d_true))

        primary_loss = (
            torch.mean(torch.stack(losses)) if losses
            else pred_centroids.sum() * 0.0
        )

        # Multi-resolution: if the wrapper also stashed a secondary
        # (K2, 2) centroid set, fold it into this loss component
        # weighted by the per-run config. Same pairwise-distance MSE
        # formulation as primary.
        sec_centroids = getattr(masked_pred, "_predicted_cluster_centroids_secondary", None)
        sec_ids = getattr(masked_pred, "_cluster_ids_secondary", None)
        K_sec = getattr(masked_pred, "_n_clusters_secondary", None)
        sec_weight = float(getattr(masked_pred, "_multi_resolution_loss_weight", 0.5))
        if sec_centroids is None or sec_ids is None or K_sec is None:
            return primary_loss

        # Same scatter-mean true-centroid computation as primary, but
        # at K_sec resolution.
        B2, N2 = sec_ids.shape
        device2 = sec_centroids.device
        safe2 = sec_ids.clamp(min=0)
        valid2 = masked_true.node_mask & (sec_ids >= 0)
        batch_idx2 = torch.arange(B2, device=device2).unsqueeze(1).expand(B2, N2)
        combined_id2 = (batch_idx2 * K_sec + safe2).reshape(-1)
        valid_f2 = valid2.float().reshape(-1, 1)
        pos_flat2 = masked_true.positions.reshape(B2 * N2, 2) * valid_f2
        count_flat2 = valid2.float().reshape(B2 * N2)
        sum_pos2 = sec_centroids.new_zeros(B2 * K_sec, 2)
        sum_cnt2 = sec_centroids.new_zeros(B2 * K_sec)
        sum_pos2.scatter_add_(0, combined_id2.unsqueeze(-1).expand(-1, 2), pos_flat2)
        sum_cnt2.scatter_add_(0, combined_id2, count_flat2)
        cnt_safe2 = sum_cnt2.clamp(min=1.0)
        true_centroids2 = (sum_pos2 / cnt_safe2.unsqueeze(-1)).reshape(B2, K_sec, 2)
        present2 = (sum_cnt2 > 0).reshape(B2, K_sec)

        sec_losses = []
        for b in range(B2):
            mb = present2[b]
            if mb.sum().item() < 2:
                continue
            d_t = torch.cdist(true_centroids2[b][mb], true_centroids2[b][mb], p=2)
            d_p = torch.cdist(sec_centroids[b][mb], sec_centroids[b][mb], p=2)
            sec_losses.append(self.mse(d_p, d_t))

        if sec_losses:
            sec_loss = torch.mean(torch.stack(sec_losses))
            return primary_loss + sec_weight * sec_loss
        return primary_loss

    def _compute_cluster_balance(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> torch.Tensor:
        """Cluster-balance regularizer for Gumbel-softmax mode.

        The CoarseToFineWrapper computes the entropy-based reg term
        during forward and stashes it (plus its weight) on the pred
        DataHolder:
          * ``_cluster_balance_loss``  — scalar tensor (or None)
          * ``_cluster_balance_weight`` — float (per-run config)

        We just multiply and return. Returns zero (graph-attached)
        when the wrapper isn't in use OR when the per-run weight is 0.
        """
        bal = getattr(masked_pred, "_cluster_balance_loss", None)
        weight = float(getattr(masked_pred, "_cluster_balance_weight", 0.0))
        if bal is None or weight <= 0.0:
            return _graph_zero(masked_pred)
        return weight * bal

    # ------------------------------------------------------------------
    # scGG fundamental method #1: EDM (Euclidean-distance) MSE loss
    # ------------------------------------------------------------------
    def _compute_edm_distance_mse(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> torch.Tensor:
        """MSE between predicted and true EUCLIDEAN pairwise distances.

        Reads ``masked_pred.edm_D`` (B, N, N SQUARED distances) stashed
        by EDMOutputWrapper, takes sqrt to get Euclidean distances, and
        compares to ``cdist(masked_true.positions)``. If edm_D isn't
        there (wrapper disabled but loss accidentally enabled), returns
        a graph-attached zero rather than crashing.

        Why Euclidean, not squared
        --------------------------
        v1 of this loss supervised SQUARED distances directly. That's
        the literal "Euclidean Distance Matrix" definition, but on
        position scales where d ~ O(1-100), D² is O(1-10⁴) and the
        MSE on D² is O(10⁸). Gradients blew up on step 1 of training
        and the run NaN'd at step 2. Taking sqrt before the MSE puts
        the loss on the same scale as LUNA's pairwise_distance_mse
        (known-stable with lr=5e-4) while keeping the same information
        content (sqrt is monotonic). pred.edm_D stays as SQUARED
        distances because that's what classical MDS needs in the
        wrapper's inference path.

        Per-slice loop because each slice's valid count differs.
        Upper triangle (i<j) only to avoid double-counting symmetric
        entries.
        """
        D_pred = getattr(masked_pred, "edm_D", None)
        if D_pred is None:
            # Wrapper not wired in. Return graph-attached zero so the
            # backward pass still works through the rest of the loss.
            return _graph_zero(masked_pred)

        losses = []
        # Numerical eps for the sqrt (avoid 0-derivative singularity at
        # the diagonal even though we mask it out below).
        eps = 1e-8
        for b in range(D_pred.shape[0]):
            mask = masked_true.node_mask[b]
            valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            n = int(valid_idx.numel())
            if n < 2:
                continue
            # True Euclidean distances on the valid sub-tensor.
            true_pos_v = masked_true.positions[b].index_select(0, valid_idx)
            d_true_v = torch.cdist(true_pos_v, true_pos_v, p=2)
            # Predicted Euclidean distances from the stashed squared form.
            D_pred_v = D_pred[b].index_select(0, valid_idx).index_select(1, valid_idx)
            d_pred_v = (D_pred_v + eps).sqrt()
            # Upper triangle (k=1 excludes diagonal).
            triu_mask = torch.triu(
                torch.ones(n, n, device=D_pred.device, dtype=torch.bool),
                diagonal=1,
            )
            losses.append(
                self.mse(d_pred_v[triu_mask], d_true_v[triu_mask])
            )
        if not losses:
            return _graph_zero(masked_pred)
        return torch.mean(torch.stack(losses))

    # ------------------------------------------------------------------
    # scGG fundamental method #4: k-NN graph contrastive loss
    # ------------------------------------------------------------------
    def _compute_knn_graph_loss(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> torch.Tensor:
        """Contrastive BCE on the predicted edge logits.

        For each cell, the k nearest TRUE spatial neighbours are
        positive edges; ``n_negatives`` random cells (excluding self
        and positives) are negative edges. The KNNGraphOutputWrapper
        stashed the (B, N, N) edge logits on ``masked_pred.knn_logits``;
        we gather positive and negative entries and apply BCE-with-
        logits.

        The positive set is fixed by the TRUE positions and computed
        without gradient. The negatives are sampled fresh per call —
        adds noise to the loss but is a standard contrastive-learning
        trick that prevents the model from memorising specific
        negatives.
        """
        logits = getattr(masked_pred, "knn_logits", None)
        if logits is None:
            return _graph_zero(masked_pred)

        k = max(1, self._knn_graph_k)
        n_neg = max(1, self._knn_graph_n_neg)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        losses = []
        for b in range(logits.shape[0]):
            mask = masked_true.node_mask[b]
            valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            n = int(valid_idx.numel())
            if n < (k + 2):
                continue
            # True spatial neighbours: top-k by Euclidean distance.
            with torch.no_grad():
                true_pos_v = masked_true.positions[b].index_select(0, valid_idx)
                d_true = torch.cdist(true_pos_v, true_pos_v, p=2)
                # Exclude self (diagonal = 0) by setting it to +inf.
                d_true_self = d_true.clone()
                d_true_self.fill_diagonal_(float("inf"))
                k_eff = min(k, n - 1)
                _, pos_idx_local = torch.topk(
                    d_true_self, k=k_eff, largest=False, dim=1,
                )                                       # (n, k_eff)
                # Random negatives per cell: uniform over non-self,
                # non-positive indices. Cheap rejection: sample 2x,
                # mask out positives + self, take first n_neg.
                cand = torch.randint(
                    0, n, (n, 4 * n_neg), device=logits.device,
                )
                row_idx = torch.arange(n, device=logits.device).unsqueeze(-1)
                is_self = cand == row_idx
                # is_pos: O(n * 4*n_neg * k_eff) — fine for modest k.
                is_pos = (
                    cand.unsqueeze(-1) == pos_idx_local.unsqueeze(1)
                ).any(dim=-1)
                bad = is_self | is_pos
                cand_safe = cand.masked_fill(bad, -1)
                # Take the FIRST n_neg good candidates per row.
                # Sort so valid (-1 sorts last) entries are first.
                ordered, _ = torch.sort(cand_safe, dim=1, descending=True)
                neg_idx_local = ordered[:, :n_neg]      # may include -1
                # Mask out invalid (-1) entries from the BCE
                neg_valid = neg_idx_local >= 0
                # Replace -1 with 0 so the gather is valid; mask out
                # the contribution below.
                neg_idx_local = neg_idx_local.clamp(min=0)

            # Gather logits at positive and negative indices.
            logits_v = logits[b].index_select(0, valid_idx).index_select(1, valid_idx)
            pos_logits = torch.gather(logits_v, 1, pos_idx_local)   # (n, k_eff)
            neg_logits = torch.gather(logits_v, 1, neg_idx_local)   # (n, n_neg)

            pos_targets = torch.ones_like(pos_logits)
            neg_targets = torch.zeros_like(neg_logits)

            # BCE-with-logits per entry. Average positives and
            # negatives separately, then mean (so they're balanced
            # regardless of k vs n_neg).
            pos_loss = bce(pos_logits, pos_targets, reduction="mean")
            # Mask invalid negatives (where we couldn't find enough
            # non-self/non-positive candidates).
            neg_loss_per = bce(neg_logits, neg_targets, reduction="none")
            neg_loss_per = neg_loss_per * neg_valid.to(neg_loss_per.dtype)
            n_valid_neg = neg_valid.sum().clamp_min(1).to(neg_loss_per.dtype)
            neg_loss = neg_loss_per.sum() / n_valid_neg

            losses.append(0.5 * (pos_loss + neg_loss))
        if not losses:
            return _graph_zero(masked_pred)
        return torch.mean(torch.stack(losses))

    # ------------------------------------------------------------------
    # Shape-matching loss: covariance-eigenvalue MSE per slice
    # ------------------------------------------------------------------
    def _compute_shape_matching(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> torch.Tensor:
        """Anisotropic-shape supervision on the predicted point cloud.

        Per slice, computes the 2×2 covariance matrices of the valid
        (non-padding) cell positions on both the prediction and the
        ground truth, then matches one of three scalar summaries
        controlled by ``self._shape_variant``:

          * ``"eigvals"``     — MSE on sorted (λ_min, λ_max). Default
            (matches original behaviour). Rotation-invariant but
            sensitive to absolute scale.

          * ``"ratio"``       — MSE on λ_min/λ_max (the anisotropy
            ratio, in [0, 1]). Scale-invariant: penalises "predicted
            cloud is more circular" regardless of overall extent.

          * ``"covariance"``  — Frobenius² norm on the full 2×2 cov
            matrix. Captures eigenvalues AND principal-axis
            orientation. Rotation-sensitive: only stable once the
            prediction has settled into a consistent frame.

        Per-slice loop because the valid cell count varies per slice
        and the covariance must be computed over the valid subset.

        Side effect: populates ``self._last_shape_diagnostics`` with
        a list of per-slice eigenvalue numerics (true / pred / ratio)
        that ``forward`` then publishes to wandb. This is a diagnostic
        only — independent of which variant drives the gradient.
        """
        # Reset diagnostic stash for this call. Stays empty (and the
        # forward-hook publishes nothing) if the slices are all too
        # small for a non-degenerate cov, matching the early-return
        # path below.
        self._last_shape_diagnostics = []

        losses = []
        for b in range(masked_pred.positions.shape[0]):
            mask = masked_true.node_mask[b]
            valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            n = int(valid_idx.numel())
            # Need at least 3 cells for a non-degenerate 2×2 covariance.
            if n < 3:
                continue
            true_pos = masked_true.positions[b].index_select(0, valid_idx)
            pred_pos = masked_pred.positions[b].index_select(0, valid_idx)
            # Centre (so cov is the second central moment).
            true_c = true_pos - true_pos.mean(dim=0, keepdim=True)
            pred_c = pred_pos - pred_pos.mean(dim=0, keepdim=True)
            # Per-slice 2×2 covariance via Xᵀ X / (n-1).
            cov_true = (true_c.T @ true_c) / float(n - 1)
            cov_pred = (pred_c.T @ pred_c) / float(n - 1)
            # Symmetrise (numerical noise) before eigvalsh.
            cov_true = 0.5 * (cov_true + cov_true.T)
            cov_pred = 0.5 * (cov_pred + cov_pred.T)
            # Tikhonov regularization: add a small NON-uniform diagonal
            # to break eigenvalue degeneracy. At training init,
            # pred ≈ 0 → cov_pred ≈ 0 with λ_1 = λ_2 = 0, and
            # eigvalsh's backward formula contains ``1/(λ_i − λ_j)``
            # terms that go to ±Inf at degenerate eigenvalues. The
            # resulting NaN gradient poisons the position-head's
            # weights on the optimizer step, and the model produces
            # NaN positions on the next forward.
            #
            # Adding a NON-UNIFORM diagonal perturbation breaks the
            # degeneracy: λ_1 and λ_2 differ by at least eps_reg
            # (when no other gap separates them), so the backward
            # is well-defined. The perturbation also shifts the
            # eigenvalues by O(eps_reg), but eps_reg=1e-6 is far
            # below any realistic anisotropy signal — the loss
            # value is essentially unchanged for non-degenerate
            # inputs. Same trick applied to cov_true so its
            # eigenvalues are also broken (their gradient is zero
            # so this is purely defensive).
            eps_reg = 1e-6
            reg = torch.tensor([[0.0, 0.0], [0.0, eps_reg]],
                               device=cov_pred.device, dtype=cov_pred.dtype)
            cov_pred = cov_pred + reg
            cov_true = cov_true + reg
            # Eigenvalues — ascending by default.
            evals_true = torch.linalg.eigvalsh(cov_true)
            evals_pred = torch.linalg.eigvalsh(cov_pred)

            # Diagnostic numerics for wandb (always populated, even
            # under variants that don't touch eigenvalues directly).
            # ``.detach().item()`` because we never want these to
            # carry gradient — they only exist for logging.
            eps = 1e-12
            ratio_true_val = float(
                (evals_true[0] / (evals_true[1] + eps)).detach().item()
            )
            ratio_pred_val = float(
                (evals_pred[0] / (evals_pred[1] + eps)).detach().item()
            )
            self._last_shape_diagnostics.append({
                "eig_min_true": float(evals_true[0].detach().item()),
                "eig_max_true": float(evals_true[1].detach().item()),
                "ratio_true":   ratio_true_val,
                "eig_min_pred": float(evals_pred[0].detach().item()),
                "eig_max_pred": float(evals_pred[1].detach().item()),
                "ratio_pred":   ratio_pred_val,
            })

            # Dispatch the actual gradient-bearing loss.
            if self._shape_variant == "eigvals":
                # Original: MSE on the sorted pair (already ascending
                # out of eigvalsh).
                slice_loss = ((evals_pred - evals_true) ** 2).mean()
            elif self._shape_variant == "ratio":
                # Anisotropy-ratio MSE; scale-invariant. The +eps is
                # ALSO under the gradient graph (it stabilises
                # division by a small eigenvalue at init).
                ratio_true = evals_true[0] / (evals_true[1] + eps)
                ratio_pred = evals_pred[0] / (evals_pred[1] + eps)
                slice_loss = (ratio_pred - ratio_true) ** 2
            else:  # "covariance"
                # Full 2×2 Frobenius². The cov matrices are 2×2 so
                # this is just the sum of squared elementwise
                # differences over four entries.
                slice_loss = ((cov_pred - cov_true) ** 2).sum()

            losses.append(slice_loss)

        if not losses:
            return _graph_zero(masked_pred)
        return torch.mean(torch.stack(losses))

    # ------------------------------------------------------------------
    # Latent diffusion — FM-on-z denoising loss
    # ------------------------------------------------------------------
    def _compute_latent_fm_mse(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> torch.Tensor:
        """MSE between the denoiser's predicted z_0 and the encoder's
        sampled z_0_target.

        Both stashes are set by LatentDiffusionModel.apply_noise +
        LatentDiffusionWrapper.forward. If they're missing (any non-LDM
        framework, or inference where the encoder didn't run), we
        return a graph-attached zero so the registry can keep this
        component "always on" without affecting non-LDM runs.

        Mask is applied so padding cells don't contribute.
        """
        z_0_pred = getattr(masked_pred, "_ldm_z_0_pred", None)
        z_0_target = getattr(masked_pred, "_ldm_z_0_target", None)
        if z_0_pred is None or z_0_target is None:
            return _graph_zero(masked_pred)

        mask = masked_pred.node_mask.unsqueeze(-1).to(z_0_pred.dtype)
        sq_err = (z_0_pred - z_0_target).pow(2) * mask        # (B, N, k)
        # Mean over real (cell, latent-dim) pairs.
        n_real_dims = mask.sum() * float(z_0_pred.shape[-1])
        n_real_dims = n_real_dims.clamp_min(1.0)
        return sq_err.sum() / n_real_dims

    # ------------------------------------------------------------------
    # Latent diffusion — KL prior on the encoder's q(z | x)
    # ------------------------------------------------------------------
    def _compute_latent_kl(
        self, masked_pred: DataHolder, masked_true: DataHolder,
    ) -> torch.Tensor:
        """KL(q(z|x) || N(0, I)) computed from the encoder's mu/logvar
        stashes. Graph-attached zero when the stashes aren't present.

        See ``models.latent_vae.kl_normal_standard`` for the formula
        — the same helper is reused so the loss term and the
        encoder's regularisation use byte-identical math.
        """
        mu = getattr(masked_pred, "_ldm_mu", None)
        logvar = getattr(masked_pred, "_ldm_logvar", None)
        if mu is None or logvar is None:
            return _graph_zero(masked_pred)
        from models.latent_vae import kl_normal_standard
        return kl_normal_standard(mu, logvar, masked_pred.node_mask)

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

        Warmup: for components whose ``warmup_steps > 0`` and the
        current ``self._step_count < warmup_steps``, the loss VALUE
        is computed under ``torch.no_grad()`` (so it logs to wandb)
        but skipped from the autograd graph (so its potentially-
        unstable backward can't poison the rest of training). The
        weighted contribution to total is exactly 0 during warmup;
        full weight resumes once the counter passes the threshold.
        A ``<name>_warmup_active`` scalar is logged so the wandb
        chart shows when each warmup ends.
        """
        total = None
        per_component: Dict[str, float] = {}
        for name, enabled, weight, fn in self._components:
            if not enabled:
                continue
            warmup_n = self._warmup_steps.get(name, 0)
            in_warmup = warmup_n > 0 and self._step_count < warmup_n
            if in_warmup:
                # Compute the value WITHOUT building the autograd graph.
                # Logged-but-not-trained: makes warmup VISIBLE on the
                # wandb chart (value still moves; weighted shows 0).
                with torch.no_grad():
                    val = fn(masked_pred, masked_true)
                per_component[name] = float(val.detach().item())
                per_component[f"{name}_weighted"] = 0.0
                per_component[f"{name}_warmup_active"] = 1.0
                # Intentionally do NOT add to ``total``: total tracks
                # only the gradient-bearing contribution, so the
                # wandb-side "total" matches what the optimizer actually
                # sees. (Adding 0.0 here would mathematically equal
                # this but uselessly bloat the autograd graph.)
            else:
                val = fn(masked_pred, masked_true)
                per_component[name] = float(val.detach().item())
                weighted = weight * val
                per_component[f"{name}_weighted"] = float(weighted.detach().item())
                total = weighted if total is None else (total + weighted)
                # Log the warmup flag as 0 once active, so wandb shows
                # a step-function from 1→0 at the warmup boundary.
                if warmup_n > 0:
                    per_component[f"{name}_warmup_active"] = 0.0
        if total is None:
            # All ACTIVE components disabled (or all in warmup) —
            # define total as 0 attached to the prediction graph so
            # backward is still well-defined. Print a one-shot warning
            # so the user isn't surprised by zero-gradient training
            # during a long warmup.
            total = _graph_zero(masked_pred)
            if not getattr(self, "_warned_empty_total", False):
                print(
                    "[LossFunction] WARNING: total loss has no "
                    "gradient-bearing contribution this step "
                    "(all active components either disabled or in "
                    f"warmup at step {self._step_count}). Training "
                    "will not update weights. This is expected for "
                    "the first ``warmup_steps`` steps if you've "
                    "warmed up every active component."
                )
                self._warned_empty_total = True
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
        # Remember the last batch so log_epoch_metrics can publish the
        # most recent step's metrics WITHOUT calling compute_loss
        # again. Previously log_epoch_metrics re-ran compute_loss,
        # which advanced stateful counters in the components (PH
        # _ph_step_counter, RNG state in Sinkhorn's negative
        # sampling, etc.) and silently shifted their schedules by
        # one extra increment per epoch.
        self.true_positions = masked_true.positions
        self.pred_positions = masked_pred.positions
        self.node_mask = masked_true.node_mask

        loss, per_component = self.compute_loss(masked_pred, masked_true)

        # Advance the warmup step counter on TRAINING-stage forwards
        # only. Val/test stages must not advance it, otherwise a long
        # validation pass could push the counter past warmup
        # thresholds before training has actually reached them.
        if train_stage:
            self._step_count += 1

        # Stash for log_epoch_metrics. Use a snapshot of the FLOAT
        # values (already detached inside compute_loss via .item()
        # on each component) so callers can't accidentally hold on
        # to gradient-bearing tensors.
        self._last_per_component = dict(per_component)
        # Also snapshot the shape-diagnostic stash for the epoch
        # summary — _last_shape_diagnostics is repopulated each
        # _compute_shape_matching call and would be wiped by any
        # future call to compute_loss, so we capture it now.
        self._last_shape_diagnostics_snapshot = list(
            self._last_shape_diagnostics
        )

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
            # Per-slice shape diagnostics (eigenvalue numerics for
            # predicted vs true covariance). Independent of which
            # variant drove the gradient — these tell us WHY the
            # loss is moving (or not). Reported as batch-averaged
            # scalars under ``{train,val}_shape/...`` so they show up
            # next to the loss components.
            shape_prefix = (
                "train_shape" if train_stage else "val_shape"
            )
            to_log.update(self._summarise_shape_diagnostics(shape_prefix))
            if wandb.run:
                wandb.log(to_log, commit=True)

        return loss, to_log

    def reset(self) -> None:
        """Hook kept for parity with LUNA's pl callbacks. No state to clear."""
        pass

    def log_epoch_metrics(self) -> Dict[str, float]:
        """Publish the most recent forward's per-component metrics as
        ``train_epoch/*``. Used by LUNA's training-step wrapper.

        Reads ``self._last_per_component`` (snapshot stashed in
        ``forward()``) rather than re-running ``compute_loss`` on the
        cached tensors. Re-running used to (a) double the wall-clock
        cost of every component once per epoch, (b) advance PH's
        step counter an extra time per epoch, shifting the
        ``frequency`` schedule, and (c) draw fresh random
        subsamples in components like Sinkhorn — meaning the
        published epoch number wasn't the same loss the training
        step used.
        """
        per_component = getattr(self, "_last_per_component", None)
        if not per_component:
            return {}

        to_log = {
            f"train_epoch/{k}": v for k, v in per_component.items()
        }
        # Backward-compat alias.
        if "pairwise_distance_mse" in per_component:
            to_log["train_epoch/position_mse"] = per_component[
                "pairwise_distance_mse"
            ]
        # Epoch-level shape diagnostics. Restore the snapshot
        # captured at forward() time so the values match the step's
        # numbers; _summarise_shape_diagnostics reads
        # _last_shape_diagnostics. Empty dict if shape_matching is
        # disabled or all slices were degenerate at the last step.
        prev_stash = self._last_shape_diagnostics
        self._last_shape_diagnostics = getattr(
            self, "_last_shape_diagnostics_snapshot", []
        )
        try:
            to_log.update(
                self._summarise_shape_diagnostics("train_epoch_shape")
            )
        finally:
            self._last_shape_diagnostics = prev_stash
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log

    # ------------------------------------------------------------------
    # Shape-diagnostic helpers
    # ------------------------------------------------------------------
    def _summarise_shape_diagnostics(
        self, prefix: str,
    ) -> Dict[str, float]:
        """Reduce the per-slice eigenvalue stash to a flat dict of
        wandb scalars under ``{prefix}/...``.

        Reports six MEAN values across the batch:
          * eig_min_true / eig_max_true / ratio_true
          * eig_min_pred / eig_max_pred / ratio_pred

        And three error metrics directly answering "is the prediction
        anisotropic enough":
          * eig_min_mae  — mean abs error on λ_min across slices
          * eig_max_mae  — mean abs error on λ_max
          * ratio_mae    — mean abs error on λ_min/λ_max

        Returns an empty dict (and emits nothing) when there were no
        diagnostic entries — either because shape_matching is
        disabled or because all slices in the batch were too small
        for a non-degenerate covariance.
        """
        stash = self._last_shape_diagnostics
        if not stash:
            return {}
        n = float(len(stash))
        # Aggregate.
        sums = {
            "eig_min_true": 0.0, "eig_max_true": 0.0, "ratio_true": 0.0,
            "eig_min_pred": 0.0, "eig_max_pred": 0.0, "ratio_pred": 0.0,
        }
        mae = {"eig_min": 0.0, "eig_max": 0.0, "ratio": 0.0}
        for entry in stash:
            for k in sums:
                sums[k] += entry[k]
            mae["eig_min"] += abs(entry["eig_min_pred"] - entry["eig_min_true"])
            mae["eig_max"] += abs(entry["eig_max_pred"] - entry["eig_max_true"])
            mae["ratio"]   += abs(entry["ratio_pred"]   - entry["ratio_true"])
        out = {f"{prefix}/{k}_mean": v / n for k, v in sums.items()}
        out[f"{prefix}/eig_min_mae"] = mae["eig_min"] / n
        out[f"{prefix}/eig_max_mae"] = mae["eig_max"] / n
        out[f"{prefix}/ratio_mae"]   = mae["ratio"]   / n
        # Don't clear here — the same stash is re-populated each call
        # to ``_compute_shape_matching``, so a stale read is impossible
        # under normal use. Clearing here would double-empty if a
        # caller hits ``forward`` and then ``log_epoch_metrics``
        # without an intervening compute.
        return out

import pytorch_lightning as pl
import torch
from metrics.loss_function import LossFunction
from models.model import Model
from utils.data.dataholder import DataHolder
from utils.data.misc import setup_wandb
from utils.diffusion_model.diffusion.noise_model import NoiseModel

from utils.diffusion_model.test.test import (
    on_test_epoch_end_func,
    on_test_epoch_start_func,
    test_step_func,
)
from utils.diffusion_model.train.train import (
    on_train_epoch_end_func,
    on_train_epoch_start_func,
    training_step_func,
)
from utils.diffusion_model.validation.val import (
    on_validation_epoch_end_func,
    on_validation_epoch_start_func,
    validation_step_func,
)


class FullDenoisingDiffusion(pl.LightningModule):
    model_dtype = torch.float32
    best_val_nll = 1e8
    val_counter = 0
    start_epoch_time = None
    train_iterations = None
    val_iterations = None

    def __init__(self, cfg, dataset_infos):
        super().__init__()

        self.cfg = cfg
        self.name = cfg.general.name

        # Optional autograd anomaly detection — set
        # ``train.detect_anomaly=true`` to enable. PyTorch then saves
        # the forward stack of every op, and on backward raises an
        # error pinpointing the EXACT op that produced any NaN/Inf
        # gradient (file + line). ~5-10x slower training, so for
        # debugging only. See configs/train/default.yaml for the
        # full usage note. Gated by a global flag (not a context
        # manager) because PL constructs the LightningModule then
        # runs many forwards inside its own training loop — the
        # global flag is the only way to keep anomaly mode active
        # for the whole run.
        if bool(getattr(cfg.train, "detect_anomaly", False)):
            torch.autograd.set_detect_anomaly(True, check_nan=True)
            print(
                "[FullDenoisingDiffusion] WARNING: autograd anomaly "
                "detection is ON (cfg.train.detect_anomaly=true). "
                "Training will be ~5-10x slower. Use only for "
                "debugging NaN/Inf gradient sources."
            )
        # The per-module backward-hook diagnostic. This complements
        # ``torch.autograd.set_detect_anomaly`` in TWO important
        # ways: (a) anomaly_mode only catches NaN — if the gradient
        # is Inf (e.g. from an eigh / SVD backward at near-degenerate
        # spectrum) anomaly_mode misses it because Inf only becomes
        # NaN later (via 0×Inf in chain rule); (b) anomaly_mode's
        # error message points at the op that PRODUCED the NaN but
        # via the FORWARD stack — which is helpful but doesn't
        # directly tell you which nn.Module's backward is responsible.
        # The per-module hook reports both (which module + whether
        # its grad_output was already bad on arrival or its grad_input
        # is what corrupted things). Gated by the same
        # ``train.detect_anomaly`` flag — pure diagnostic, off by
        # default. Hooks installed AFTER self.model is built, see
        # ``_install_per_module_grad_finder`` further down in __init__.
        self._per_module_grad_finder_on = bool(
            getattr(cfg.train, "detect_anomaly", False)
        )
        # ``max_diffusion_steps`` is the outer step count for the
        # sampling loop in utils/diffusion_model/sample/sample.py:47:
        #     for s_int in reversed(range(0, self.max_diffusion_steps, ...))
        # DDPM uses cfg.model.diffusion_steps (typically 1000). FM
        # uses cfg.model.flow_matching.n_sampling_steps (typically
        # 50). We resolve it here so the sampling loop stays
        # framework-agnostic — it just iterates whatever count the
        # current noise model exposes.
        framework = str(getattr(cfg.model, "framework", "diffusion")).lower()
        if framework == "flow_matching":
            fm_cfg = getattr(cfg.model, "flow_matching", None)
            n_steps = int(getattr(fm_cfg, "n_sampling_steps", 50)) if fm_cfg is not None else 50
            self.max_diffusion_steps = n_steps
        elif framework == "regression":
            # Direct supervised regression: the "sampling loop" runs
            # exactly once (one backbone forward = the prediction).
            # RegressionPredictor.max_diffusion_steps is also 1; we
            # mirror it on the Lightning module so sample.py's outer
            # ``reversed(range(0, max_diffusion_steps))`` iterates once.
            self.max_diffusion_steps = 1
        elif framework == "latent_diffusion":
            # LDM has its own n_diffusion_steps knob under
            # cfg.model.latent_diffusion.n_diffusion_steps (default 50,
            # same as FM's default — but the two CAN diverge if the
            # user overrides one). Previously this branch fell through
            # to cfg.model.diffusion_steps (1000, the DDPM default),
            # putting the outer sampling loop on a different grid than
            # the LDM noise model's internal s_val computation
            # (which divides by ldm_cfg.n_diffusion_steps). Mismatch
            # silently distorted the FM-on-z trajectory.
            ldm_cfg = getattr(cfg.model, "latent_diffusion", None)
            n_steps = int(getattr(ldm_cfg, "n_diffusion_steps", 50)) if ldm_cfg is not None else 50
            self.max_diffusion_steps = n_steps
        else:
            self.max_diffusion_steps = cfg.model.diffusion_steps
        self.log_every_steps = True

        self.dataset_infos = dataset_infos
        self.input_dims = dataset_infos.input_dims
        self.output_dims = dataset_infos.output_dims

        # Self-conditioning (Chen 2023). When enabled, each cell's
        # input is augmented with the previous step's x_0_pred (2-D)
        # alongside its gene features. This requires bumping the
        # inner backbone's input_dims BEFORE construction so all
        # downstream wrappers (c2f / hierarchical) see the wider
        # input and add their own augmentations on top.
        #
        # Lives under cfg.model (not cfg.train) so that the
        # inference-side ``load_model_config`` (which only restores
        # cfg.model from the training-time snapshot) picks it up
        # automatically. Otherwise the inference subprocess would
        # build the model with default (off) settings and the
        # state_dict load would fail on shape mismatch.
        sc_cfg = getattr(cfg.model, "self_conditioning", None)
        self._self_cond_enabled = bool(
            getattr(sc_cfg, "enabled", False)
        ) if sc_cfg is not None else False
        self._self_cond_prob = float(
            getattr(sc_cfg, "prob", 0.5)
        ) if sc_cfg is not None else 0.5
        if self._self_cond_enabled:
            # Bump input width by 2 (the self-cond x_0_pred channel).
            # Convert to a mutable dict in case input_dims is frozen.
            self.input_dims = dict(self.input_dims)
            self.input_dims["node_features_dimensions"] = (
                int(self.input_dims["node_features_dimensions"]) + 2
            )

        # Auto-enable the coarse_centroid_mse loss component when the
        # CoarseToFineWrapper is in use, so the user only needs one
        # --override (model.coarse_to_fine.enabled=true). Must happen
        # BEFORE LossFunction construction reads the config.
        _c2f_cfg = getattr(cfg.model, "coarse_to_fine", None)
        _c2f_enabled = bool(getattr(_c2f_cfg, "enabled", False)) if _c2f_cfg else False
        if _c2f_enabled:
            if hasattr(cfg.model, "loss") and hasattr(cfg.model.loss, "coarse_centroid_mse"):
                cfg.model.loss.coarse_centroid_mse.enabled = True

        # Same pre-construction patching pattern for EDM (#1) and k-NN
        # graph (#4) output heads — both auto-wire new loss components
        # and optionally disable the original pairwise_distance_mse.
        # The actual wrapper modules are bolted on AFTER the backbone
        # is built (see the EDM/kNN block far below); here we only
        # patch the loss config so LossFunction sees the right state
        # at construction time.
        _edm_cfg_early = getattr(cfg.model, "edm", None)
        _edm_enabled_early = bool(
            getattr(_edm_cfg_early, "enabled", False)
        ) if _edm_cfg_early is not None else False
        _knn_graph_cfg_early = getattr(cfg.model, "knn_graph", None)
        _knn_graph_enabled_early = bool(
            getattr(_knn_graph_cfg_early, "enabled", False)
        ) if _knn_graph_cfg_early is not None else False
        if _edm_enabled_early and _knn_graph_enabled_early:
            raise ValueError(
                "model.edm.enabled and model.knn_graph.enabled are mutually "
                "exclusive — both replace the position output path. Enable "
                "exactly one."
            )
        # NB: cfg.model.loss is in OmegaConf struct mode (forbids
        # adding new keys at runtime). The default config now declares
        # ``edm_distance_mse`` and ``knn_graph_loss`` blocks (defaulted
        # off), so the hot path is the ``else`` branch — pure attribute
        # writes, no schema mutation. The ``if not hasattr`` branch
        # exists for the edge case where an old training snapshot
        # (from before these blocks were declared) is being restored
        # at inference; that branch enters ``open_dict`` to bypass the
        # struct check so the runtime add succeeds. See the
        # ConfigAttributeError in the run logs from before the schema
        # was added — that's what this guard prevents.
        from omegaconf import OmegaConf as _OC

        if _edm_enabled_early:
            # Auto-disable original pairwise-distance MSE if requested.
            if bool(getattr(_edm_cfg_early, "replace_position_loss", True)):
                if hasattr(cfg.model.loss, "pairwise_distance_mse"):
                    cfg.model.loss.pairwise_distance_mse.enabled = False
            edm_loss_weight = float(getattr(_edm_cfg_early, "loss_weight", 1.0))
            if hasattr(cfg.model.loss, "edm_distance_mse"):
                cfg.model.loss.edm_distance_mse.enabled = True
                cfg.model.loss.edm_distance_mse.weight = edm_loss_weight
            else:
                with _OC.open_dict(cfg.model.loss):
                    cfg.model.loss.edm_distance_mse = _OC.create({
                        "enabled": True,
                        "weight": edm_loss_weight,
                    })

            # Sparse-training guard. When edm.skip_edm_D_train is set,
            # the head emits pred.edm_D = None during training, so EVERY
            # loss that reads edm_D (edm_distance_mse + the dense spatial
            # variants) degrades to its gradient-zero fallback. That is
            # the silent-silencing pattern the codebase rejects — fail
            # LOUD: those terms must be off and the O(N·k)
            # sparse_local_distance must be the active distance loss.
            if bool(getattr(_edm_cfg_early, "skip_edm_D_train", False)):
                _ld = cfg.model.loss
                _edm_block = getattr(_ld, "edm_distance_mse", None)
                _offenders = []
                if (
                    _edm_block is not None
                    and bool(getattr(_edm_block, "enabled", False))
                    and float(getattr(_edm_block, "weight", 0.0)) > 0.0
                ):
                    _offenders.append(
                        "edm_distance_mse (set model.edm.loss_weight=0)"
                    )
                for _nm in (
                    "locality_weighted_distance", "log_distance_mse",
                    "rank_spearman", "knn_neighborhood",
                ):
                    _blk = getattr(_ld, _nm, None)
                    if _blk is not None and bool(getattr(_blk, "enabled", False)):
                        _offenders.append(_nm)
                if _offenders:
                    raise ValueError(
                        "model.edm.skip_edm_D_train=true emits no pred.edm_D "
                        "at train time, but these loss terms read edm_D and "
                        f"would silently become no-ops: {_offenders}. Turn "
                        "them off (set model.edm.loss_weight=0 for the EDM "
                        "MSE) and use model.loss.sparse_local_distance "
                        "instead, which reads edm_h and is O(N*k)."
                    )
                _sld_block = getattr(_ld, "sparse_local_distance", None)
                if _sld_block is None or not bool(
                    getattr(_sld_block, "enabled", False)
                ):
                    raise ValueError(
                        "model.edm.skip_edm_D_train=true removes the full "
                        "(N,N) distance matrix at train time, so the only "
                        "sensible distance loss is the O(N*k) sparse one. "
                        "Set model.loss.sparse_local_distance.enabled=true."
                    )

        if _knn_graph_enabled_early:
            if bool(getattr(_knn_graph_cfg_early, "replace_position_loss", True)):
                if hasattr(cfg.model.loss, "pairwise_distance_mse"):
                    cfg.model.loss.pairwise_distance_mse.enabled = False
            knn_loss_weight = float(getattr(_knn_graph_cfg_early, "loss_weight", 1.0))
            knn_k = int(getattr(_knn_graph_cfg_early, "k", 10))
            knn_n_neg = int(getattr(_knn_graph_cfg_early, "n_negatives", 20))
            if hasattr(cfg.model.loss, "knn_graph_loss"):
                cfg.model.loss.knn_graph_loss.enabled = True
                cfg.model.loss.knn_graph_loss.weight = knn_loss_weight
                cfg.model.loss.knn_graph_loss.k = knn_k
                cfg.model.loss.knn_graph_loss.n_negatives = knn_n_neg
            else:
                with _OC.open_dict(cfg.model.loss):
                    cfg.model.loss.knn_graph_loss = _OC.create({
                        "enabled": True,
                        "weight": knn_loss_weight,
                        "k": knn_k,
                        "n_negatives": knn_n_neg,
                    })

        # Pass the full cfg so the loss can read its `model.loss.*`
        # block and toggle components on/off. Default config keeps only
        # LUNA's pairwise-distance MSE on, so train-from-scratch
        # reproduces the baseline; new ablation components stay opt-in.
        self.train_loss = LossFunction(cfg=cfg)
        self.val_loss = LossFunction(cfg=cfg)

        # FM prediction parameterisation (read here so we can pass
        # ``subtract_input_pos`` into the EGNN constructor and gate
        # the v→x_0 conversion in self.forward).
        if framework == "flow_matching":
            fm_cfg = getattr(cfg.model, "flow_matching", None)
            self._fm_prediction = (
                str(getattr(fm_cfg, "prediction", "x0")).lower()
                if fm_cfg is not None else "x0"
            )
        else:
            self._fm_prediction = "x0"   # ignored for DDPM

        # Multi-scale hierarchical wrapper — toggled by
        # cfg.model.hierarchical.enabled. When on, the wrapper bins
        # cells into spatial patches per forward, runs a small
        # patch-level transformer, and injects each patch's summary
        # as additional input to the main backbone. Off by default
        # for byte-equivalence with the LUNA baseline. Currently
        # supported only with backbone="luna_transformer"; we raise
        # at construction time on egnn+hierarchical so it fails
        # loudly rather than silently dropping the wrapper.
        hier_cfg = getattr(cfg.model, "hierarchical", None)
        hier_enabled = bool(getattr(hier_cfg, "enabled", False)) if hier_cfg else False

        # Coarse-to-fine wrapper — toggled by
        # cfg.model.coarse_to_fine.enabled. K-means clusters cells by
        # gene expression and predicts each cluster's spatial centroid
        # as an auxiliary output; the inner backbone receives the
        # centroids as additional conditioning. Same backbone-
        # constraint as hierarchical (LUNA transformer only for now).
        c2f_cfg = getattr(cfg.model, "coarse_to_fine", None)
        c2f_enabled = bool(getattr(c2f_cfg, "enabled", False)) if c2f_cfg else False
        # If BOTH wrappers are enabled, we route through the combined
        # ``HierarchicalCoarseToFineWrapper`` (in coarse_to_fine.py)
        # which composes both augmentations on a single inner Model.
        # The c2f-only and hier-only branches further below are then
        # short-circuited; we set this flag to drive the routing.
        combined_enabled = c2f_enabled and hier_enabled

        # Backbone selector: LUNA's stock transformer or scgg's
        # SE(2)-equivariant EGNN. The EGNN path is gated behind a
        # config knob so existing runs keep using the LUNA backbone
        # unless explicitly opted in via
        #     --override model.backbone=egnn
        backbone = str(getattr(cfg.model, "backbone", "luna_transformer")).lower()

        if hier_enabled and backbone != "luna_transformer":
            raise ValueError(
                f"model.hierarchical.enabled=true currently only supports "
                f"backbone='luna_transformer' (got backbone={backbone!r}). "
                f"EGNN+hierarchical needs an invariant reformulation of "
                f"the patch centroid features — not yet implemented."
            )
        if c2f_enabled and backbone != "luna_transformer":
            raise ValueError(
                f"model.coarse_to_fine.enabled=true currently only supports "
                f"backbone='luna_transformer' (got backbone={backbone!r}). "
                f"The cluster-centroid conditioning concatenates 2-d "
                f"coordinates to gene features; combining with EGNN "
                f"requires an invariant reformulation."
            )
        if c2f_enabled and framework == "regression":
            raise ValueError(
                f"model.coarse_to_fine.enabled=true is incompatible with "
                f"model.framework='regression' — coarse-to-fine's teacher "
                f"forcing uses true positions during training, but "
                f"regression mode zeros positions throughout. Disable "
                f"one of the two."
            )
        if hier_enabled and framework == "regression":
            # The hierarchical wrapper bins cells by their positions.
            # In regression mode, positions are zeroed at both
            # training and inference time (the network sees no
            # positional info), so all cells would land in the same
            # patch — defeats the purpose of multi-scale processing.
            # A regression-compatible hierarchical would bin cells by
            # GENE features instead; not implemented.
            raise ValueError(
                f"model.hierarchical.enabled=true is incompatible with "
                f"model.framework='regression' — the hierarchical wrapper "
                f"bins cells by positions, but regression feeds the "
                f"backbone zero positions. Disable one of the two "
                f"options, or implement a gene-feature-based patch "
                f"assignment for the regression case."
            )

        # ------------------------------------------------------------------
        # Latent Diffusion short-circuit. When framework=latent_diffusion,
        # self.model becomes the LatentDiffusionWrapper (encoder +
        # DiT-style denoiser + decoder) and the normal backbone +
        # EDM/kNN-wrapper logic is bypassed entirely. The wrapper has
        # its own internal architecture; the user's choice of backbone /
        # c2f / hier / EDM is ignored under this framework (and the
        # checks below validate that those flags aren't mistakenly on).
        if framework == "latent_diffusion":
            # Mutex with the other major output-head wrappers — LDM
            # does its own encode→denoise→decode pipeline and isn't
            # compatible with EDM's MDS-recovery path or kNN-graph's
            # spectral-layout path.
            if _edm_enabled_early:
                raise ValueError(
                    "model.framework='latent_diffusion' is incompatible "
                    "with model.edm.enabled=true. LDM and EDM are "
                    "alternative paradigms — pick one. (LDM trains its "
                    "own learned decoder; EDM uses analytical MDS.) "
                    "Disable one of them."
                )
            if _knn_graph_enabled_early:
                raise ValueError(
                    "model.framework='latent_diffusion' is incompatible "
                    "with model.knn_graph.enabled=true. Both replace the "
                    "position output path; pick one."
                )
            if c2f_enabled or hier_enabled or combined_enabled:
                raise ValueError(
                    "model.framework='latent_diffusion' is incompatible "
                    "with the c2f / hierarchical wrappers. LDM uses its "
                    "own encoder/decoder pipeline; the wrappers expect "
                    "the LUNA Model's position-stream interface."
                )
            from models.latent_diffusion_wrapper import LatentDiffusionWrapper
            self.model = LatentDiffusionWrapper(
                input_dims=self.input_dims,
                n_layers=cfg.model.n_layers,
                hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=self.output_dims,
                ldm_cfg=getattr(cfg.model, "latent_diffusion", None),
            )
        elif backbone == "luna_transformer":
            # Auxiliary-stream ablation flag (severs intermediate-coordinate
            # feedback in the backbone; see models/self_attention.py). Only
            # the DIRECT Model path threads it today — fail loud rather than
            # silently ignore it if a wrapper is active.
            _pfb = str(
                getattr(cfg.model, "position_feedback", "absolute")
            ).lower()
            if _pfb != "absolute" and (
                combined_enabled or hier_enabled or c2f_enabled
            ):
                raise NotImplementedError(
                    "model.position_feedback != 'absolute' is only supported "
                    "on the direct luna_transformer backbone (no hierarchical "
                    "/ coarse-to-fine wrapper). Disable the wrapper or set "
                    "model.position_feedback=absolute."
                )
            if combined_enabled:
                # Resolve the optional input-projection knobs ONCE so
                # the wrappers (hier / c2f / hier+c2f) and the direct
                # Model construction all see the same values. When
                # ``cfg.model.input_projection`` is unset (the default),
                # we produce relu / no-LN / no-dropout — byte-identical
                # to the historic behaviour.
                _ip_cfg = getattr(cfg.model, "input_projection", None)
                _ip_kwargs = dict(
                    input_activation=str(
                        getattr(_ip_cfg, "activation", "relu")
                    ) if _ip_cfg is not None else "relu",
                    input_layernorm=bool(
                        getattr(_ip_cfg, "layernorm", False)
                    ) if _ip_cfg is not None else False,
                    input_dropout=float(
                        getattr(_ip_cfg, "dropout", 0.0)
                    ) if _ip_cfg is not None else 0.0,
                )
                # Composed multi-scale: spatial-hierarchical patches +
                # gene-coarse-to-fine clusters, both feeding ONE inner
                # Model. See HierarchicalCoarseToFineWrapper.
                from models.coarse_to_fine import HierarchicalCoarseToFineWrapper
                self.model = HierarchicalCoarseToFineWrapper(
                    input_dims=self.input_dims,
                    n_layers=cfg.model.n_layers,
                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                    hidden_dims=cfg.model.hidden_dims,
                    output_dims=self.output_dims,
                    hier_cfg=hier_cfg,
                    c2f_cfg=c2f_cfg,
                    **_ip_kwargs,
                )
            elif hier_enabled:
                _ip_cfg = getattr(cfg.model, "input_projection", None)
                _ip_kwargs = dict(
                    input_activation=str(
                        getattr(_ip_cfg, "activation", "relu")
                    ) if _ip_cfg is not None else "relu",
                    input_layernorm=bool(
                        getattr(_ip_cfg, "layernorm", False)
                    ) if _ip_cfg is not None else False,
                    input_dropout=float(
                        getattr(_ip_cfg, "dropout", 0.0)
                    ) if _ip_cfg is not None else 0.0,
                )
                from models.hierarchical import HierarchicalModelWrapper
                self.model = HierarchicalModelWrapper(
                    input_dims=self.input_dims,
                    n_layers=cfg.model.n_layers,
                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                    hidden_dims=cfg.model.hidden_dims,
                    output_dims=self.output_dims,
                    hierarchical_cfg=hier_cfg,
                    **_ip_kwargs,
                )
            elif c2f_enabled:
                _ip_cfg = getattr(cfg.model, "input_projection", None)
                _ip_kwargs = dict(
                    input_activation=str(
                        getattr(_ip_cfg, "activation", "relu")
                    ) if _ip_cfg is not None else "relu",
                    input_layernorm=bool(
                        getattr(_ip_cfg, "layernorm", False)
                    ) if _ip_cfg is not None else False,
                    input_dropout=float(
                        getattr(_ip_cfg, "dropout", 0.0)
                    ) if _ip_cfg is not None else 0.0,
                )
                from models.coarse_to_fine import CoarseToFineWrapper
                self.model = CoarseToFineWrapper(
                    input_dims=self.input_dims,
                    n_layers=cfg.model.n_layers,
                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                    hidden_dims=cfg.model.hidden_dims,
                    output_dims=self.output_dims,
                    c2f_cfg=c2f_cfg,
                    **_ip_kwargs,
                )
            else:
                # Direct Model construction (no wrapper). Same kwargs
                # resolution as the wrapper branches above — kept
                # inline so each branch is self-contained for review.
                _ip_cfg = getattr(cfg.model, "input_projection", None)
                self.model = Model(
                    input_dims=self.input_dims,
                    n_layers=cfg.model.n_layers,
                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                    hidden_dims=cfg.model.hidden_dims,
                    output_dims=self.output_dims,
                    input_activation=str(
                        getattr(_ip_cfg, "activation", "relu")
                    ) if _ip_cfg is not None else "relu",
                    input_layernorm=bool(
                        getattr(_ip_cfg, "layernorm", False)
                    ) if _ip_cfg is not None else False,
                    input_dropout=float(
                        getattr(_ip_cfg, "dropout", 0.0)
                    ) if _ip_cfg is not None else 0.0,
                    position_feedback=_pfb,
                )
        elif backbone == "vn_transformer":
            # scGG fundamental method #3: SE(2)-equivariant vector-
            # neuron transformer. Scaffold only — raises a clear
            # NotImplementedError on construction. The config surface
            # is real (cfg.model.vn_transformer.* is valid) so configs
            # validate; the layers themselves are pending a focused
            # implementation session.
            from models.vn_transformer import VNTransformerBackbone
            self.model = VNTransformerBackbone(
                input_dims=self.input_dims,
                n_layers=cfg.model.n_layers,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=self.output_dims,
                vn_cfg=getattr(cfg.model, "vn_transformer", None),
            )
        elif backbone == "dit":
            # Architectural extension #2 — DiT-style backbone.
            # Per-cell tokens (gene + position) → adaLN-Zero time-
            # conditioned transformer → (node_features, positions).
            # Output contract matches LUNA Model so downstream
            # wrappers (EDM, c2f, gene_recon) work unchanged.
            from models.dit_backbone import DiTBackbone
            self.model = DiTBackbone(
                input_dims=self.input_dims,
                n_layers=cfg.model.n_layers,
                hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=self.output_dims,
                dit_cfg=getattr(cfg.model, "dit", None),
            )
        elif backbone == "perceiver":
            # Architectural extension #3 — Perceiver-IO style backbone.
            # K learnable anchor tokens; cells↔anchors cross-attention
            # replaces O(N²) cell-cell attention with O(N·K).
            from models.perceiver_backbone import PerceiverBackbone
            self.model = PerceiverBackbone(
                input_dims=self.input_dims,
                n_layers=cfg.model.n_layers,
                hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=self.output_dims,
                perceiver_cfg=getattr(cfg.model, "perceiver", None),
            )
        elif backbone == "nystromformer":
            # Nyström-approximated global self-attention. Sub-quadratic
            # O(N·M) cost; preserves all-to-all reachability through M
            # segment-mean landmarks. Architecturally a DiT lookalike
            # but with NystromAttention in place of dense SDPA inside
            # each block, so EDM / c2f / gene_recon wrappers compose
            # identically.
            from models.nystromformer_backbone import NystromformerBackbone
            self.model = NystromformerBackbone(
                input_dims=self.input_dims,
                n_layers=cfg.model.n_layers,
                hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=self.output_dims,
                nystromformer_cfg=getattr(cfg.model, "nystromformer", None),
            )
        elif backbone == "geomattn":
            # Geometry-Coupled Attention — methodological extension.
            # DiT-style backbone whose self-attention carries a
            # learnable, time-gated, SE(2)-invariant geometric bias
            # over the CURRENT coordinate estimate (x_t). The attention
            # pattern and the geometry co-evolve along the FM sampling
            # trajectory: content-driven at high noise, spatially-local
            # at low noise. Zero-init time-gate ⇒ starts identical to
            # DiT, learns the coarse-to-fine schedule. Composes with the
            # EDM / c2f / gene_recon wrappers identically to DiT.
            from models.geometry_coupled_attention import GeometryCoupledBackbone
            self.model = GeometryCoupledBackbone(
                input_dims=self.input_dims,
                n_layers=cfg.model.n_layers,
                hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=self.output_dims,
                geomattn_cfg=getattr(cfg.model, "geomattn", None),
            )
        elif backbone == "geomattn_nystrom":
            # Geometry-Coupled Nyström Attention — scalable GCA.
            # Injects the same learnable, time-gated, SE(2)-invariant
            # geometric bias as `geomattn`, but at the LANDMARK level
            # (cell↔landmark-centroid, centroid↔centroid distances), so
            # both the attention AND the bias are O(N·M) instead of
            # O(N²). Recovers dense geomattn at M ≥ N. Use this for
            # large slices where dense geomattn OOMs.
            from models.geometry_coupled_nystrom import GeometryCoupledNystromBackbone
            self.model = GeometryCoupledNystromBackbone(
                input_dims=self.input_dims,
                n_layers=cfg.model.n_layers,
                hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=self.output_dims,
                geomattn_nystrom_cfg=getattr(cfg.model, "geomattn_nystrom", None),
            )
        elif backbone == "geomattn_localglobal":
            # Geometry Local-Global Attention — sub-quadratic GCA via
            # exact local (top-K spatial neighbours) + landmark global.
            # combine ∈ {gated, unified} (config). O(N·(K+M)) memory and
            # compute; the kNN is computed once per forward (chunked).
            from models.geometry_local_global_attention import LocalGlobalBackbone
            self.model = LocalGlobalBackbone(
                input_dims=self.input_dims,
                n_layers=cfg.model.n_layers,
                hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=self.output_dims,
                geomattn_localglobal_cfg=getattr(cfg.model, "geomattn_localglobal", None),
            )
        elif backbone in (
            "geomattn_dilated", "geomattn_bigbird", "geomattn_axial",
            "geomattn_swin", "geomattn_routing",
        ):
            # Spatial sparse geometry-aware attention zoo — each maps a
            # famous index-based efficient-attention method to 2-D
            # spatial proximity (Sparse Transformer / BigBird / Axial /
            # Swin / Routing Transformer). All sub-quadratic, all carry
            # the SE(2)-invariant radial geometry bias, all DataHolder
            # in / out (EDM / c2f / gene_recon compose unchanged).
            from models.spatial_sparse_attention import SpatialSparseBackbone
            _pattern = backbone[len("geomattn_"):]                # e.g. "swin"
            _cfg = getattr(cfg.model, backbone, None)
            self.model = SpatialSparseBackbone(
                pattern=_pattern,
                input_dims=self.input_dims,
                n_layers=cfg.model.n_layers,
                hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=self.output_dims,
                cfg=_cfg,
            )
        elif backbone == "egnn":
            # Local import so the LUNA-baseline path (which doesn't
            # need EGNN's torch-geometric kNN code) keeps loading
            # fast even if torch-geometric's optional deps are flaky.
            from models.egnn import EGNNModel
            # When the FM wrapper is in v-prediction mode, EGNN
            # outputs the pure cumulative residual R = Σℓ Δxℓ rather
            # than x_t + R. The LightningModule.forward below then
            # interprets that residual as v_pred and converts to
            # x_0_pred = x_t − t·v_pred. At init R ≈ 0 so
            # v_pred ≈ 0 → x_0_pred ≈ x_t (sensible "don't move"
            # starting point). Without this, the network would have
            # to learn to output x_t + (large residual) when its
            # output is interpreted as v_pred — fighting the
            # residual-update inductive bias.
            egnn_subtract = (
                framework == "flow_matching" and self._fm_prediction == "v"
            )
            self.model = EGNNModel(
                input_dims=self.input_dims,
                n_layers=cfg.model.n_layers,
                hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=self.output_dims,
                egnn_cfg=cfg.model.egnn,
                subtract_input_pos=egnn_subtract,
            )
        else:
            raise ValueError(
                f"Unknown model.backbone={backbone!r}. Expected "
                f"'luna_transformer', 'egnn', 'vn_transformer', "
                f"'dit', 'perceiver', 'nystromformer', 'geomattn', "
                f"'geomattn_nystrom', 'geomattn_localglobal', or one of "
                f"'geomattn_{{dilated,bigbird,axial,swin,routing}}'."
            )

        # ------------------------------------------------------------------
        # Output-head wrappers: EDM (#1) and k-NN graph (#4).
        #
        # Mutually exclusive — both replace the position output path.
        # The mutex check + loss-config patching already ran earlier in
        # __init__ (before LossFunction construction). Here we only
        # bolt the wrapper module on top of self.model.
        #
        # Bolted on AFTER any c2f / hierarchical wrapping so EDM/kNN
        # are the OUTERMOST layer and see the c2f-conditioned features.
        # Both wrappers project from pred.node_features (whose width
        # is hidden_dims['output_features_to_pos_dims']; LUNA Model
        # outputs this dim, c2f/hier wrappers preserve it) +
        # pred.positions (2D). The wrappers handle the +2 concat
        # internally.
        # ------------------------------------------------------------------
        edm_cfg = getattr(cfg.model, "edm", None)
        edm_enabled = bool(getattr(edm_cfg, "enabled", False)) if edm_cfg else False
        knn_graph_cfg = getattr(cfg.model, "knn_graph", None)
        knn_graph_enabled = bool(
            getattr(knn_graph_cfg, "enabled", False)
        ) if knn_graph_cfg else False

        inner_out_dim = int(cfg.model.hidden_dims["output_features_to_pos_dims"])

        if edm_enabled:
            # EDM emits pred.positions as the MDS-projected layout, which
            # IS the x_0 estimate. v-prediction would re-interpret that
            # as a velocity field, producing garbage — guard explicitly.
            if framework == "flow_matching" and self._fm_prediction == "v":
                raise ValueError(
                    "model.edm.enabled=true is incompatible with "
                    "model.flow_matching.prediction='v'. The EDM head "
                    "produces x_0-style positions (via MDS); the v-pred "
                    "conversion would mis-interpret them. Use "
                    "model.flow_matching.prediction='x0' with EDM, or "
                    "disable EDM."
                )
            from models.edm_head import EDMOutputWrapper
            self.model = EDMOutputWrapper(
                inner_model=self.model,
                inner_out_dim=inner_out_dim,
                embed_dim=int(getattr(edm_cfg, "embed_dim", 8)),
                mds_align=bool(getattr(edm_cfg, "mds_align", True)),
                anisotropic_gating=bool(
                    getattr(edm_cfg, "anisotropic_gating", False)
                ),
                mds_align_gradient=bool(
                    getattr(edm_cfg, "mds_align_gradient", False)
                ),
                mds_tikhonov_eps=float(
                    getattr(edm_cfg, "mds_tikhonov_eps", 1e-6)
                ),
                # Default True preserves historic behaviour (MDS runs
                # every training step). Set False to skip during
                # training when no loss reads pred.positions — major
                # speedup at large N because eigh is O(N³).
                mds_align_train=bool(
                    getattr(edm_cfg, "mds_align_train", True)
                ),
                # MDS solver precision + algorithm — see EDMOutputWrapper
                # docstring for the safety guards. Defaults preserve
                # historic behaviour (fp64 + eigh).
                mds_dtype=str(getattr(edm_cfg, "mds_dtype", "fp64")),
                mds_solver=str(getattr(edm_cfg, "mds_solver", "eigh")),
                # Sparse-training: skip the (B,N,N) D_sq at train time so
                # the O(N·k) sparse_local_distance loss is the only
                # distance term. Default False (historic full-matrix).
                skip_edm_D_train=bool(
                    getattr(edm_cfg, "skip_edm_D_train", False)
                ),
                # Keep the geometry head (projector→distances→MDS) in fp32
                # even under bf16-mixed; default True (no-op in fp32).
                fp32_geometry=bool(
                    getattr(edm_cfg, "fp32_geometry", True)
                ),
                # Geometric decoder: 'mds' (classical, default) or 'smacof'
                # (MDS warm-start + differentiable stress-minimising Guttman
                # refinement). Default preserves the classical read-out.
                decoder=str(getattr(edm_cfg, "decoder", "mds")),
                smacof_iters=int(getattr(edm_cfg, "smacof_iters", 30)),
                # v2: train end-to-end through the SMACOF decoder (detached
                # MDS init + grad through the last smacof_grad_iters steps).
                # Pair with model.loss.edm_coord_mse + mds_align_train=true.
                decoder_grad=bool(getattr(edm_cfg, "decoder_grad", False)),
                smacof_grad_iters=int(getattr(edm_cfg, "smacof_grad_iters", 5)),
                # Backward through the decode: "unroll" (BPTT, default) or
                # "jfb" (Jacobian-Free Backprop — constant memory, no unroll).
                smacof_backward=str(getattr(edm_cfg, "smacof_backward", "unroll")),
            )

        if knn_graph_enabled:
            if framework == "flow_matching" and self._fm_prediction == "v":
                raise ValueError(
                    "model.knn_graph.enabled=true is incompatible with "
                    "model.flow_matching.prediction='v'. The k-NN graph "
                    "head produces x_0-style positions (via spectral "
                    "layout); the v-pred conversion would mis-interpret "
                    "them. Use prediction='x0' or disable knn_graph."
                )
            from models.knn_graph_head import KNNGraphOutputWrapper
            self.model = KNNGraphOutputWrapper(
                inner_model=self.model,
                inner_out_dim=inner_out_dim,
                embed_dim=int(getattr(knn_graph_cfg, "embed_dim", 16)),
                spectral_layout=bool(getattr(knn_graph_cfg, "spectral_layout", True)),
                k_for_layout=int(getattr(knn_graph_cfg, "k", 10)),
                temperature=float(getattr(knn_graph_cfg, "temperature", 0.1)),
                spectral_layout_gradient=bool(
                    getattr(knn_graph_cfg, "spectral_layout_gradient", False)
                ),
            )

        # Auxiliary gene-reconstruction head. Active iff
        # cfg.model.gene_reconstruction.enabled. Takes the inner
        # backbone's output node_features (out_node_dim) and predicts
        # back to gene space. Trained jointly with the main loss.
        # Lives under cfg.model (not cfg.train) for the same reason
        # as self-conditioning above — the head is a Lightning module
        # parameter, so the checkpoint format depends on this flag,
        # so it has to survive the inference-side config restore.
        recon_cfg = getattr(cfg.model, "gene_reconstruction", None)
        self._gene_recon_enabled = bool(
            getattr(recon_cfg, "enabled", False)
        ) if recon_cfg is not None else False
        if self._gene_recon_enabled and framework == "latent_diffusion":
            # Mutex: gene reconstruction masks gene features in the
            # forward pass and asks the model to recover the masked
            # values. Under framework=latent_diffusion, the VAE
            # encoder runs inside apply_noise on the SAME (masked)
            # gene features, so the encoder learns
            # "masked-gene → latent" at training, but sees UNMASKED
            # genes at inference (no gene-recon at sample time).
            # The resulting distribution shift on the encoder's
            # input degrades z_T → x_0 quality silently. If you
            # want to combine them, the gene_recon path needs to
            # be redesigned to feed UNmasked features to the
            # encoder and only mask for the recon head. Until that
            # exists, refuse the combination.
            raise ValueError(
                "model.gene_reconstruction.enabled=true is "
                "incompatible with model.framework='latent_diffusion'. "
                "Gene-recon masks gene features into the backbone, "
                "but the LDM encoder runs on those same features in "
                "apply_noise — train sees masked, inference sees "
                "unmasked, and the encoder's input distribution "
                "silently drifts. Disable one of the two."
            )
        if self._gene_recon_enabled:
            self._gene_recon_mask_ratio = float(getattr(recon_cfg, "mask_ratio", 0.15))
            self._gene_recon_weight = float(getattr(recon_cfg, "weight", 0.05))
            recon_hidden = int(getattr(recon_cfg, "hidden_dim", 256))
            # pred.node_features (from the inner backbone) has width
            # ``hidden_dims.output_features_to_pos_dims`` (default 4),
            # NOT ``output_dims["node_features_dimensions"]`` which is
            # the gene-input-side dim used elsewhere in the LUNA
            # config. The head's input is that 4-D output concatenated
            # with the 2-D predicted position = 6-D.
            inner_out_dim = int(cfg.model.hidden_dims["output_features_to_pos_dims"])
            # ORIGINAL gene dim (pre-self-cond-bump). The head
            # predicts back to the masked input genes, which live
            # in dataset_infos.input_dims, NOT the possibly-bumped
            # self.input_dims.
            n_genes_for_recon = int(dataset_infos.input_dims["node_features_dimensions"])
            import torch.nn as _nn_recon
            self.gene_recon_head = _nn_recon.Sequential(
                _nn_recon.Linear(inner_out_dim + 2, recon_hidden),
                _nn_recon.SiLU(),
                _nn_recon.Linear(recon_hidden, n_genes_for_recon),
            )
        else:
            self.gene_recon_head = None

        # Training paradigm: DDPM (LUNA default), rectified-flow FM,
        # or pure supervised regression. All three classes share the
        # SAME apply_noise / sample_limit_dist /
        # sample_zs_from_zt_and_pred interface so the training step
        # and sampling loop are framework-agnostic.
        if framework == "flow_matching":
            # Architectural extension #4 — True EDM diffusion (FM in
            # the k-D embedding space). When BOTH edm.enabled=true
            # AND edm.diffuse_in_embed_space=true, we route through
            # EDMFlowMatchingModel instead of FlowMatchingModel. The
            # interface is identical so the rest of the framework
            # machinery is unchanged; only the noise application and
            # FM Euler step are reinterpreted in h-space.
            _edm_cfg_fm = getattr(cfg.model, "edm", None)
            _edm_hspace = (
                _edm_cfg_fm is not None
                and bool(getattr(_edm_cfg_fm, "enabled", False))
                and bool(getattr(_edm_cfg_fm, "diffuse_in_embed_space", False))
            )
            if _edm_hspace:
                _fmc = getattr(cfg.model, "flow_matching", None)
                if str(getattr(_fmc, "prior_mode", "gaussian")).lower() == \
                        "learned_regression":
                    raise NotImplementedError(
                        "prior_mode='learned_regression' is implemented for "
                        "coordinate-space FM only, not h-space "
                        "(edm.diffuse_in_embed_space=true). Use "
                        "diffuse_in_embed_space=false."
                    )
                from utils.diffusion_model.diffusion.edm_fm_model import (
                    EDMFlowMatchingModel,
                )
                self.noise_model = EDMFlowMatchingModel(cfg)
            else:
                # Local import keeps the DDPM path import-cost identical
                # for runs that don't opt into FM (parity with how the
                # EGNN backbone is imported on demand above).
                from utils.diffusion_model.diffusion.flow_matching_model import (
                    FlowMatchingModel,
                )
                self.noise_model = FlowMatchingModel(cfg)
                # MixFlow-style informed prior: a learnable head predicts
                # each cell's coarse CANONICAL-frame position from its
                # features (genes, or Nicheformer embeddings if those are
                # the node_features); the FM then starts from that prior
                # instead of N(0,I). The head trains via a geodesic MSE in
                # training_step_func; its (detached) output re-centres the
                # source Gaussian in apply_noise / sample_limit_dist.
                # Gated by prior_mode='learned_regression', default off.
                _fm_cfg = getattr(cfg.model, "flow_matching", None)
                _prior_mode = str(
                    getattr(_fm_cfg, "prior_mode", "gaussian")
                ).lower() if _fm_cfg is not None else "gaussian"
                if _prior_mode == "learned_regression":
                    if not bool(getattr(_fm_cfg, "canonicalize_target", False)):
                        raise ValueError(
                            "flow_matching.prior_mode='learned_regression' "
                            "requires flow_matching.canonicalize_target=true: "
                            "a per-cell point prior is only well-defined in a "
                            "fixed canonical frame (otherwise the predicted "
                            "position carries an arbitrary SE(2) gauge)."
                        )
                    _ph_hidden = int(getattr(_fm_cfg, "prior_hidden", 128))
                    # Concrete Linear (NOT LazyLinear): configure_optimizers
                    # runs before the first forward, so a lazy/uninitialised
                    # param would be missed by the optimizer and never train.
                    # node_features width = the gene-column count.
                    _gene_dim = int(cfg.dataset.gene_columns_end) - int(
                        cfg.dataset.gene_columns_start
                    )
                    # Optionally feed the prior head ONLY the last-K feature
                    # columns (e.g. an appended scVI latent) so the coarse
                    # positional prior is regressed from the cell-state
                    # embedding alone. 0 -> full features (byte-identical).
                    self._prior_input_last_k = max(
                        0, int(getattr(_fm_cfg, "prior_input_last_k", 0))
                    )
                    _ph_in = (
                        self._prior_input_last_k
                        if 0 < self._prior_input_last_k <= _gene_dim
                        else _gene_dim
                    )
                    if self._prior_input_last_k > _gene_dim:
                        raise ValueError(
                            f"flow_matching.prior_input_last_k="
                            f"{self._prior_input_last_k} exceeds the feature "
                            f"width {_gene_dim} (gene_columns). It must be the "
                            f"size of the appended conditioning block (e.g. the "
                            f"scVI dim)."
                        )
                    self.prior_head = torch.nn.Sequential(
                        torch.nn.Linear(_ph_in, _ph_hidden),
                        torch.nn.SiLU(),
                        torch.nn.Linear(_ph_hidden, 2),
                    )
                    self._prior_geo_weight = float(
                        getattr(_fm_cfg, "prior_geo_weight", 1.0)
                    )
        elif framework == "regression":
            # Direct supervised regression — no noise, no iterative
            # sampling. See module docstring of regression_predictor.
            from utils.diffusion_model.diffusion.regression_predictor import (
                RegressionPredictor,
            )
            self.noise_model = RegressionPredictor(cfg)
        elif framework == "energy":
            # scGG fundamental method #2: cell-cell-potentials / energy-
            # based generative model. Scaffold only — raises a clear
            # NotImplementedError on instantiation. The config surface
            # (cfg.model.energy.*) is real so configs validate; the
            # actual score-matching loop + annealed-Langevin sampler
            # are pending a focused implementation session.
            from utils.diffusion_model.diffusion.energy_predictor import (
                EnergyPredictor,
            )
            self.noise_model = EnergyPredictor(cfg)
        elif framework == "latent_diffusion":
            # Architectural extension #5: Stable-Diffusion-style
            # latent diffusion. Encoder + DiT-style denoiser + decoder
            # are all in ``self.model`` (the LatentDiffusionWrapper
            # constructed in the backbone-dispatch block far above);
            # the noise model here drives apply_noise / sampling in
            # latent space and reaches into the wrapper for
            # encode/decode operations via ``_ldm_wrapper``.
            from utils.diffusion_model.diffusion.latent_diffusion_model import (
                LatentDiffusionModel,
            )
            self.noise_model = LatentDiffusionModel(cfg)
            # Bind the noise model to the wrapper so apply_noise can
            # call wrapper.encode() / wrapper.decode(). The wrapper
            # was constructed in the LDM branch of the backbone
            # dispatch — if this attribute is missing the user hit a
            # config validation we should have caught upstream.
            from models.latent_diffusion_wrapper import LatentDiffusionWrapper
            if not isinstance(self.model, LatentDiffusionWrapper):
                raise RuntimeError(
                    "framework='latent_diffusion' built self.noise_model "
                    "= LatentDiffusionModel but self.model is "
                    f"{type(self.model).__name__}, not "
                    "LatentDiffusionWrapper. The backbone-dispatch "
                    "block above should have routed this — bug in "
                    "diffusion_model.py construction order."
                )
            self.noise_model._ldm_wrapper = self.model
        elif framework == "diffusion":
            self.noise_model = NoiseModel(cfg)
        else:
            raise ValueError(
                f"Unknown model.framework={framework!r}. Expected "
                f"'diffusion', 'flow_matching', 'regression', 'energy', "
                f"or 'latent_diffusion'."
            )

        # One-time trainable-parameter count, printed at construction so it
        # lands in every run's log (no behavioural effect). Read it with
        # grep on "[params]" in the run log.
        try:
            _n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[params] trainable={_n_params:,}", flush=True)
        except Exception:
            pass

    def _install_per_module_grad_finder(self) -> None:
        """Install register_full_backward_hook on every nn.Module so we
        can pinpoint WHICH module's backward first produces a
        non-finite gradient. Complements
        ``torch.autograd.set_detect_anomaly`` in two important ways:

          1. anomaly_mode only catches NaN. If the gradient is Inf
             (the canonical signature of an eigh/SVD backward at
             near-degenerate spectrum), Inf only becomes NaN later
             via 0×Inf in chain rule — by which point anomaly_mode
             has missed it. We check for BOTH NaN and Inf.

          2. anomaly_mode tells you the FORWARD op that produced the
             bad gradient (via saved stack), which is useful but
             abstract. The per-module hook reports the nn.Module by
             name, which is what you actually grep for in source.

        The hook fires AFTER each module's backward computes
        ``grad_input``. By comparing:
          - ``grad_output`` (gradient flowing INTO this module's
            backward; the gradient produced by later modules)
          - ``grad_input`` (gradient produced BY this module's
            backward; flows to earlier modules)
        we classify:
          - grad_output non-finite → bug is later in forward (we
            received a bad gradient already corrupted)
          - grad_output finite but grad_input non-finite → THIS
            module's backward is the producer
        The FIRST module (in backward order) to report this is the
        culprit. We raise immediately so the user gets a stack trace
        at the moment of detection rather than running through the
        rest of the backward.

        Only installed when ``cfg.train.detect_anomaly=true`` to
        keep the hot path free of overhead in production runs.
        """
        if not self._per_module_grad_finder_on:
            return
        # Track the first reporter so we don't spam — once any
        # module reports a bad gradient, we raise.
        self._first_bad_grad_reported = False

        def make_hook(name: str):
            def hook(module, grad_input, grad_output):
                if self._first_bad_grad_reported:
                    return
                # Helpers: tuple-of-tensors-or-None safe checks.
                def _any_nonfinite(t_tuple):
                    if t_tuple is None:
                        return False
                    for t in t_tuple:
                        if t is None:
                            continue
                        if not torch.isfinite(t).all():
                            return True
                    return False
                def _stats(t_tuple):
                    if t_tuple is None:
                        return "(no tensors)"
                    parts = []
                    for i, t in enumerate(t_tuple):
                        if t is None:
                            parts.append(f"[{i}]=None")
                            continue
                        n_nan = torch.isnan(t).sum().item()
                        n_inf = torch.isinf(t).sum().item()
                        finite_mask = torch.isfinite(t)
                        if finite_mask.any():
                            finite_vals = t[finite_mask]
                            mx = finite_vals.abs().max().item()
                        else:
                            mx = float("nan")
                        parts.append(
                            f"[{i}] shape={tuple(t.shape)} "
                            f"nan={n_nan} inf={n_inf} max|finite|={mx:.3e}"
                        )
                    return "; ".join(parts)

                bad_out = _any_nonfinite(grad_output)
                bad_in  = _any_nonfinite(grad_input)
                if not (bad_in or bad_out):
                    return
                self._first_bad_grad_reported = True
                if bad_out and not bad_in:
                    verdict = "received-bad-from-later"
                elif bad_in and not bad_out:
                    verdict = "PRODUCED-BAD-IN-THIS-MODULE"
                else:
                    verdict = "both-bad (this module amplified an "
                    verdict += "already-corrupted gradient)"
                msg = (
                    f"\n[per-module grad finder] non-finite gradient "
                    f"detected during backward.\n"
                    f"  module:       {name} ({type(module).__name__})\n"
                    f"  verdict:      {verdict}\n"
                    f"  grad_output:  {_stats(grad_output)}\n"
                    f"  grad_input:   {_stats(grad_input)}\n"
                    f"To dig deeper:\n"
                    f"  - 'PRODUCED-BAD-IN-THIS-MODULE' verdict → the "
                    f"backward of THIS module's forward op is\n"
                    f"    where the bug lives. Examples: eigh/SVD on "
                    f"degenerate spectrum; sqrt at d=0; clamp_min\n"
                    f"    with too-small floor; log at 0; cdist(p=2) "
                    f"at coincident points.\n"
                    f"  - 'received-bad-from-later' → walk BACK up "
                    f"the call chain; the bug is in a module that\n"
                    f"    runs AFTER this one in forward (earlier "
                    f"in backward).\n"
                )
                # Print first so the message is visible even if the
                # subsequent raise gets caught somewhere.
                print(msg, flush=True)
                raise RuntimeError(msg)
            return hook

        for name, mod in self.named_modules():
            # Skip the LightningModule root (named_modules includes
            # self with name='').
            if name == "":
                continue
            mod.register_full_backward_hook(make_hook(name))
        print(
            "[FullDenoisingDiffusion] per-module grad-finder armed "
            "on all nn.Module children. Will raise on the first "
            "non-finite gradient during backward."
        )

    def on_fit_start(self) -> None:
        # Install per-module hooks lazily here — the model has been
        # fully constructed by now (DDP rank-aware), so named_modules
        # walks the right tree. Calling from __init__ would miss
        # any module added later by Lightning (rare but possible).
        if getattr(self, "_per_module_grad_finder_on", False):
            if not hasattr(self, "_per_module_hooks_installed"):
                self._install_per_module_grad_finder()
                self._per_module_hooks_installed = True
        # Original on_fit_start continues below.
        self.train_iterations = 100
        if self.local_rank == 0:
            setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        on_train_epoch_start_func(self)

    def training_step(self, data, i) -> torch.Tensor:
        loss = training_step_func(self, data, i)
        return loss

    def on_after_backward(self) -> None:
        """Detect NaN/Inf in any parameter's gradient and fail loud.

        The principled stabilisations elsewhere (Tikhonov on eigh /
        eigvalsh, degenerate-skip on Procrustes) eliminate the
        known NaN-prone gradient sites. This hook is the
        belt-and-suspenders: if a NEW NaN source ever appears (a new
        loss component, a new model wrapper, a corner case the
        stabilisations don't cover), training halts immediately
        with a clear pointer at the loss value at that step —
        rather than silently writing NaN into the weights and
        producing nonsense for the rest of the run.

        Replaces the implicit "NaN poison propagates through
        autograd → next forward crashes 200 lines deep in a model
        module" failure mode with an explicit, debuggable error.
        Cost: one extra reduction across all parameters per step
        (~milliseconds on H100). The detector is unconditional
        because the cost is negligible and the safety guarantee is
        worth it.
        """
        # Scan param.grad once. We don't dump every parameter — the
        # error message just names the first offender so the user
        # has somewhere to start. Skip parameters without a grad
        # (e.g. frozen layers, parameters that didn't participate
        # in the loss this step).
        bad_param = None
        for name, p in self.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad_param = name
                break
        if bad_param is not None:
            # Surface the most recent loss components so the user
            # can correlate "which loss spiked" with "which gradient
            # NaN'd". self.train_loss._last_per_component is set
            # inside LossFunction.forward() (the stash that
            # log_epoch_metrics reads).
            recent = getattr(
                getattr(self, "train_loss", None),
                "_last_per_component", None,
            )
            recent_str = ", ".join(
                f"{k}={v:.4g}" for k, v in (recent or {}).items()
            ) if recent else "(no loss snapshot)"
            raise RuntimeError(
                f"NaN / Inf gradient detected on parameter "
                f"{bad_param!r} after backward. Training halted to "
                f"prevent silent weight corruption. Last-step loss "
                f"components: {recent_str}. Likely culprits: a "
                f"newly-enabled auxiliary loss with an unstable "
                f"backward path (e.g. an op whose Jacobian diverges "
                f"at degenerate spectra — eigh / eigvalsh / SVD); a "
                f"learning rate too high for the active loss "
                f"weights; or fp16 underflow in a kernel. To "
                f"diagnose: (1) check which loss spiked just before "
                f"this step; (2) reproduce with "
                f"torch.autograd.set_detect_anomaly(True) to find "
                f"the exact op; (3) if it's a known unstable op, "
                f"add Tikhonov regularization at that site (see "
                f"models/edm_head.py::_classical_mds_2d for the "
                f"pattern). DO NOT add a torch.nan_to_num hook here "
                f"— that would silently produce wrong gradients."
            )

    def on_train_epoch_end(self) -> None:
        on_train_epoch_end_func(self)

    def on_validation_epoch_start(self) -> None:
        on_validation_epoch_start_func(self=self)

    def validation_step(self, data: DataHolder, i: int) -> torch.Tensor:
        loss = validation_step_func(self, data, i)
        return loss

    def on_validation_epoch_end(self):
        on_validation_epoch_end_func(self=self)

    def on_test_epoch_start(self):
        on_test_epoch_start_func(self=self)

    def test_step(self, data: DataHolder, i: int):
        test_step_func(self, data, i)

    def on_test_epoch_end(self) -> None:
        """Measure likelihood on a test set and compute stability metrics."""
        on_test_epoch_end_func(self=self)

    def forward(self, z_t: DataHolder) -> DataHolder:
        assert z_t.node_mask is not None
        model_input = z_t.copy()

        # DataHolder.copy() is field-based — it only copies the
        # explicit constructor args (positions/node_features/etc.) and
        # silently drops any extra attributes set via direct
        # assignment. The Latent Diffusion noise model stashes the
        # noisy latent z_t and the encoder outputs onto the DataHolder
        # via setattr (``out._ldm_z_t = ...`` etc.) — those would be
        # lost by the .copy() above, so we re-attach them here. Same
        # for the self-conditioning channel and EDM/kNN auxiliaries
        # (the latter are read by the LightningModule itself rather
        # than the inner model, but propagating defensively makes the
        # behaviour uniform).
        for attr in (
            "_ldm_z_t",
            "_ldm_z_0_target",
            "_ldm_mu",
            "_ldm_logvar",
            "_self_cond_x0",
            # EDM-FM h-space state. Currently consumed only by the
            # noise model's sampler (which reads it off the ORIGINAL
            # z_t, not model_input), so omitting it would not break
            # anything today — but propagating it defensively keeps
            # the stash contract uniform with the LDM stashes and
            # protects against future refactors that route the noise-
            # model call through model_input instead of z_t.
            "_edm_h_t",
            "_edm_h_0",
        ):
            val = getattr(z_t, attr, None)
            if val is not None:
                setattr(model_input, attr, val)

        # Self-conditioning: prepend the 2-D x_0_pred channel (from
        # the previous training-step inner pass OR previous sampling
        # step) to node_features. The inner backbone's input_dims
        # was bumped by 2 at construction time so the gene encoder
        # accepts the wider input. When no self-cond has been
        # stashed yet (very first forward of a sampling chain),
        # default to zeros.
        if self._self_cond_enabled:
            sc = getattr(z_t, "_self_cond_x0", None)
            if sc is None:
                sc = torch.zeros_like(z_t.positions)
            # Mask out padding cells so they contribute zeros.
            sc = sc * z_t.node_mask.unsqueeze(-1).to(sc.dtype)
            model_input.node_features = torch.cat(
                [model_input.node_features, sc], dim=-1,
            )

        # Coarse-to-fine wrapper needs the TRUE positions to compute
        # true cluster centroids for teacher-forcing during training.
        # ``training_step_func`` (utils/diffusion_model/train/train.py)
        # stashes them on ``self._c2f_true_positions`` before calling
        # forward. At inference / validation / test the attribute
        # isn't set and the wrapper falls back to predicted centroids.
        if hasattr(self.model, "_c2f_uses_true_positions") or "CoarseToFineWrapper" in type(self.model).__name__:
            true_pos = getattr(self, "_c2f_true_positions", None)
            pred = self.model(model_input, true_positions=true_pos)
        else:
            pred = self.model(model_input)

        # Parameterisation conversion at the FM/LightningModule
        # boundary. When ``model.flow_matching.prediction == "v"``,
        # the backbone's output (``pred.positions``) is interpreted
        # as the velocity field v_pred; we convert to clean-position
        # prediction here so the LOSS and SAMPLER stay
        # parameterisation-agnostic (both still see x_0_pred vs
        # x_0). The conversion uses the linear FM schedule
        # x_t = (1−t)·x_0 + t·x_1  →  v = (x_t − x_0)/t  →
        # x_0_pred = x_t − t·v_pred.
        #
        # For EGNN with this mode, the backbone has been built with
        # ``subtract_input_pos=True`` so pred.positions is the pure
        # cumulative residual R = Σℓ Δxℓ — exactly the right shape
        # to interpret as v_pred (R ≈ 0 at init → v_pred ≈ 0 →
        # x_0_pred = x_t). For the LUNA transformer in v-pred mode
        # no architectural change is needed; its output magnitude is
        # set by mlp_out_pos_norm and the network learns v directly.
        if self._fm_prediction == "v":
            # z_t.t shape: (B, 1). Broadcast to (B, 1, 1) for positions.
            t_b = z_t.t.unsqueeze(-1)
            v_pred = pred.positions
            pred.positions = z_t.positions - t_b * v_pred
            pred = pred.mask()
        return pred

    # NOTE: on_fit_start is defined ABOVE (near the per-module grad
    # finder hook). The original here was a duplicate — Python would
    # call the LATER definition (this one) and the grad-finder
    # installation would never run. Removed.

    @property
    def BS(self) -> int:
        return self.cfg.train.batch_size

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )
        # Optional LR schedule. Default "none" → bare AdamW at constant
        # LR (byte-identical to the historic behaviour). The other modes
        # add a linear WARMUP (ramp LR 0→target over warmup_steps), which
        # is the standard transformer recipe and the safe way to use a
        # higher LR / larger batch: without it the scaled LR slams in at
        # step 1 and destabilises the early (NaN-prone) distance-loss
        # steps. "warmup_cosine" then anneals to ``min_lr_ratio``×LR;
        # "warmup_constant" holds at the target after warmup.
        schedule = str(getattr(self.cfg.train, "lr_schedule", "none")).lower()
        if schedule == "none":
            return {"optimizer": optimizer}
        if schedule not in ("warmup_cosine", "warmup_constant"):
            raise ValueError(
                f"train.lr_schedule must be 'none', 'warmup_cosine' or "
                f"'warmup_constant'; got {schedule!r}."
            )
        import math
        warmup_steps = max(0, int(getattr(self.cfg.train, "warmup_steps", 0)))
        min_lr_ratio = float(getattr(self.cfg.train, "min_lr_ratio", 0.1))
        # Total optimiser steps over the whole run (epochs × batches/epoch
        # ÷ accumulation), provided by Lightning once the trainer is set
        # up. Falls back to a large constant if unavailable so cosine
        # still decays gently rather than crashing.
        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            total_steps = max(warmup_steps + 1, 100_000)

        def _lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if schedule == "warmup_constant":
                return 1.0
            # warmup_cosine: decay 1 → min_lr_ratio over the remaining steps
            denom = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

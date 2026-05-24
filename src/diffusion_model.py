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
        else:
            self.max_diffusion_steps = cfg.model.diffusion_steps
        self.log_every_steps = True

        self.dataset_infos = dataset_infos
        self.input_dims = dataset_infos.input_dims
        self.output_dims = dataset_infos.output_dims
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

        if backbone == "luna_transformer":
            if hier_enabled:
                from models.hierarchical import HierarchicalModelWrapper
                self.model = HierarchicalModelWrapper(
                    input_dims=self.input_dims,
                    n_layers=cfg.model.n_layers,
                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                    hidden_dims=cfg.model.hidden_dims,
                    output_dims=self.output_dims,
                    hierarchical_cfg=hier_cfg,
                )
            else:
                self.model = Model(
                    input_dims=self.input_dims,
                    n_layers=cfg.model.n_layers,
                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                    hidden_dims=cfg.model.hidden_dims,
                    output_dims=self.output_dims,
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
                f"Unknown model.backbone={backbone!r}. "
                f"Expected 'luna_transformer' or 'egnn'."
            )

        # Generative framework: DDPM (LUNA default) or rectified-flow
        # FM. Both classes share the same apply_noise /
        # sample_limit_dist / sample_zs_from_zt_and_pred interface so
        # the training step and sampling loop are unchanged.
        if framework == "flow_matching":
            # Local import keeps the DDPM path import-cost identical
            # for runs that don't opt into FM (parity with how the
            # EGNN backbone is imported on demand above).
            from utils.diffusion_model.diffusion.flow_matching_model import (
                FlowMatchingModel,
            )
            self.noise_model = FlowMatchingModel(cfg)
        elif framework == "diffusion":
            self.noise_model = NoiseModel(cfg)
        else:
            raise ValueError(
                f"Unknown model.framework={framework!r}. "
                f"Expected 'diffusion' or 'flow_matching'."
            )

    def on_train_epoch_start(self) -> None:
        on_train_epoch_start_func(self)

    def training_step(self, data, i) -> torch.Tensor:
        loss = training_step_func(self, data, i)
        return loss

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

    def on_fit_start(self) -> None:
        self.train_iterations = 100
        if self.local_rank == 0:
            setup_wandb(self.cfg)

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
        return {"optimizer": optimizer}

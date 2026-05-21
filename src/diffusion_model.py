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
        self.max_diffusion_steps = cfg.model.diffusion_steps
        self.log_every_steps = True

        self.dataset_infos = dataset_infos
        self.input_dims = dataset_infos.input_dims
        self.output_dims = dataset_infos.output_dims
        self.train_loss = LossFunction()
        self.val_loss = LossFunction()

        self.model = Model(
            input_dims=self.input_dims,
            n_layers=cfg.model.n_layers,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
        )

        self.noise_model = NoiseModel(cfg)

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
        return self.model(model_input)

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

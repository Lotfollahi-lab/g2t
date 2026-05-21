
import os
import pathlib
import omegaconf
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from datasets.data_module import DataModule, Infos
from diffusion_model import FullDenoisingDiffusion
from utils.data.abstract_datatype import AbstractDataModule, AbstractDatasetInfos
from utils.data.misc import setup_wandb

def get_resume(
    cfg: omegaconf.DictConfig,
    dataset_infos: AbstractDatasetInfos,
    checkpoint_path: pathlib.Path,
) -> tuple:
    """
    Loads a model from a checkpoint and updates the experiment configuration.

    Parameters:
    cfg: Configuration object containing settings and parameters for the experiment.
    dataset_infos: Dataset-specific information required for model initialization.
    checkpoint_path: The file path to the saved model checkpoint.
    test: A boolean flag indicating if the model is being loaded for testing (True) or resuming training (False).

    Returns:
    Tuple containing the updated configuration object and the loaded model.
    """
    # Load the model from the specified checkpoint
    model = FullDenoisingDiffusion.load_from_checkpoint(
        checkpoint_path, dataset_infos=dataset_infos, cfg=cfg
    )

    # Return the updated configuration and the loaded model
    return cfg, model

def create_model_checkpoint_callbacks(cfg: omegaconf.DictConfig) -> list:
    """Create model checkpoint callbacks based on configuration."""
    
    callbacks = []
    
    if cfg.validation.if_validate:
        assert cfg.dataset.validation_data_path is not None, "Validation data path is not provided."
        # Validation enabled: use specific validation settings
        save_top_k = cfg.validation.save_top_k_models
        monitor_metric = cfg.validation.check_val_monitor
        check_every_n_epochs = cfg.validation.check_val_every_n_epochs
        

        callbacks.append(
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="{epoch}",
                monitor=monitor_metric,
                save_top_k=save_top_k,
                mode="min",
                every_n_epochs=check_every_n_epochs,
            )
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath="checkpoints", filename="last_epoch", every_n_epochs=1
            )
        )
    else:
        print("[INFO]: Validation is disabled.")
        # Validation disabled: save model at a defined interval
        save_top_k = cfg.validation.save_top_k_models
        save_model_every_n_epochs = cfg.validation.save_model_every_n_epochs
        # Validation disabled: save model at a defined interval
        callbacks.append(
            ModelCheckpoint(
                dirpath="checkpoints", 
                filename="{epoch}",
                save_top_k=-1,     
                save_on_train_epoch_end=True,
                every_n_epochs=save_model_every_n_epochs  # Save model every m epochs, or set as needed
            )
        )
    
    return callbacks


def create_early_stopping_callback(cfg: omegaconf.DictConfig, monitor_metric: str) -> EarlyStopping:
    """Create an early stopping callback based on configuration."""
    
    if cfg.validation.if_validate and cfg.validation.early_stopping:
        return EarlyStopping(
            monitor=monitor_metric,
            patience=cfg.validation.early_stopping_patience,
            mode="min",
        )
    else:
        return None  # No early stopping when validation is disabled

def setup_dataset(
    cfg: omegaconf.DictConfig,
) -> tuple:
    """Set up the dataset based on the configuration provided."""
    datamodule = DataModule(cfg)
    dataset_infos = Infos(datamodule, cfg)
    return datamodule, dataset_infos


def setup_model(
    cfg: omegaconf.DictConfig,
    dataset_infos: AbstractDatasetInfos,
    checkpoint_path: pathlib.Path = None,
) -> FullDenoisingDiffusion:
    """
    Set up the model based on the configuration and dataset information.

    Parameters:
    cfg: Configuration object containing model settings and other parameters.
    dataset_infos: Information specific to the dataset.

    Returns:
    model: Initialized model based on the provided configuration.
    """
    if cfg.general.mode == "train_and_test":
        pass
    else:
        # Set up for testing only
        cfg, _ = get_resume(cfg, dataset_infos, checkpoint_path)

    # Initialize the model
    model = FullDenoisingDiffusion(cfg=cfg, dataset_infos=dataset_infos)

    return model


def setup_callbacks(cfg: omegaconf.DictConfig, datamodule: AbstractDataModule) -> list:
    """Set up training callbacks based on the configuration."""
    callbacks = create_model_checkpoint_callbacks(cfg)
    
    lr_monitor = create_lr_monitor_callback()
    callbacks.append(lr_monitor)

    if cfg.validation.if_validate and cfg.validation.early_stopping:
        monitor_metric = cfg.validation.check_val_monitor
        early_stopping = create_early_stopping_callback(cfg, monitor_metric)
        if early_stopping:
            callbacks.append(early_stopping)

    return callbacks

def create_lr_monitor_callback() -> LearningRateMonitor:
    """Create a learning rate monitor callback."""
    return LearningRateMonitor(logging_interval="epoch")


def setup_trainer(cfg: omegaconf.DictConfig, callbacks: list) -> Trainer:
    """Set up the PyTorch Lightning Trainer based on the configuration and callbacks."""
    
    fast_dev_run = cfg.train.fast_dev_run
    if fast_dev_run:
        print("[WARNING]: The model will run with fast_dev_run.")

    gpus = cfg.distribute.gpus_per_node
    max_epochs = cfg.train.n_epochs
    check_val_every_n_epochs = 0 if not cfg.validation.if_validate else cfg.validation.check_val_every_n_epochs

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if wandb.run and local_rank == 0:
        setup_wandb(cfg)

    return Trainer(
        devices=gpus,
        max_epochs=max_epochs,
        check_val_every_n_epoch=check_val_every_n_epochs,
        fast_dev_run=fast_dev_run,
        callbacks=callbacks,
        strategy='ddp_find_unused_parameters_true',
        log_every_n_steps=50 if fast_dev_run else 1,
        enable_progress_bar=cfg.general.enable_progress_bar,
    )


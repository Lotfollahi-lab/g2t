import hydra
import random
import numpy as np
import torch
import os
import pathlib
from omegaconf import DictConfig
from yaml import safe_load
import pytorch_lightning as pl
from utils.diffusion_model.setup.setup import (
    setup_callbacks,
    setup_dataset,
    setup_model,
    setup_trainer,
)

@hydra.main(version_base="1.3", config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    # Set seed for reproducibility
    set_seed(cfg.general.seed)
    # Set output path for local saving
    cfg.general.local_saved_path = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    # Set up the dataset
    datamodule, dataset_infos = setup_dataset(cfg)

    # Run training or testing based on mode.
    #
    # `train_only` (scgg addition) — train the model on the train CSV
    # only; DataModule below skips the test split entirely, so the
    # test CSV (which may not even be valid for training, e.g. CNS
    # scRNA cells with string IDs) never gets touched. Pairs with the
    # `run_*_train.py` entry points.
    if cfg.general.mode == "train_and_test":
        train_model(cfg, datamodule, dataset_infos)
        test_model(cfg, datamodule, dataset_infos)
    elif cfg.general.mode == "test_only":
        test_model(cfg, datamodule, dataset_infos)
    elif cfg.general.mode == "train_only":
        train_model(cfg, datamodule, dataset_infos)
    else:
        raise ValueError(
            f"Unknown general.mode={cfg.general.mode!r}. "
            f"Expected one of: train_and_test, test_only, train_only."
        )


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(cfg: DictConfig, datamodule, dataset_infos):
    """Train the model from scratch."""
    model = setup_model(cfg, dataset_infos)
    callbacks = setup_callbacks(cfg, datamodule)
    trainer = setup_trainer(cfg, callbacks)

    trainer.fit(model, datamodule=datamodule)

    checkpoints_parent_dir = os.path.join(os.getcwd(), "checkpoints")
    cfg.test.checkpoints_parent_dir = checkpoints_parent_dir
    return checkpoints_parent_dir


def test_model(cfg: DictConfig, datamodule, dataset_infos):
    """Test the model using saved checkpoints."""
    checkpoints_parent_dir = pathlib.Path(cfg.test.checkpoints_parent_dir)
    print("Directory:", checkpoints_parent_dir)

    checkpoints_name_list = get_checkpoints_list(cfg, checkpoints_parent_dir)
    checkpoints_paths = [
        os.path.join(checkpoints_parent_dir, item) for item in checkpoints_name_list
    ]

    dataloaders_test = datamodule.test_dataloader()

    for checkpoint_path in checkpoints_paths:
        test_single_checkpoint(
            cfg, datamodule, dataset_infos, checkpoint_path, dataloaders_test
        )


def get_checkpoints_list(cfg: DictConfig, checkpoints_parent_dir: pathlib.Path):
    """Get the list of checkpoints to test."""
    if cfg.test.checkpoints_name_list == "all":
        checkpoints_name_list = os.listdir(checkpoints_parent_dir)
        checkpoints_name_list = list(
            set(checkpoints_name_list) - set("last_epoch.ckpt")
        )
    else:
        checkpoints_name_list = cfg.test.checkpoints_name_list
    return checkpoints_name_list


def test_single_checkpoint(
    cfg: DictConfig, datamodule, dataset_infos, checkpoint_path, dataloader_test
):
    """Test the model using a single checkpoint."""
    cfg.test.checkpoint_path = checkpoint_path
    cfg.test.test_save_parent_path = os.path.join(cfg.test.save_dir, cfg.general.name)

    print("Testing checkpoint:", checkpoint_path)
    try:
        cfg.test.epoch_index = int(checkpoint_path.split("=")[-1].split(".")[0])
    except ValueError:
        return
    print("Epoch index:", cfg.test.epoch_index)

    if cfg.general.mode == "test_only":
        load_model_config(cfg, checkpoint_path)

    model = setup_model(cfg, dataset_infos, checkpoint_path=checkpoint_path)
    callbacks = setup_callbacks(cfg, datamodule)
    trainer = setup_trainer(cfg, callbacks)

    trainer.test(model, ckpt_path=checkpoint_path, dataloaders=dataloader_test)


def load_model_config(cfg: DictConfig, checkpoint_path: str):
    """Load model configuration from a previous training session.

    Tolerant of missing snapshots: training runs from before scgg
    started writing ``.hydra/config.yaml`` (or runs launched via
    ``compose()`` instead of ``@hydra.main``) won't have this file. In
    that case we just keep the currently-composed ``cfg.model``,
    which matches the training-time model config as long as the user
    didn't override it. The checkpoint load will fail at the next
    step if there's a real mismatch.
    """
    config_file = "/".join(checkpoint_path.split("/")[:-2])
    config_yaml = f"{config_file}/.hydra/config.yaml"
    if not os.path.exists(config_yaml):
        print(
            f"[load_model_config] no snapshot at {config_yaml} — keeping "
            f"the currently composed cfg.model (works if model config "
            f"is unchanged from training)."
        )
        return
    loading_model_cfg = safe_load(open(config_yaml))
    cfg["model"] = loading_model_cfg["model"]


if __name__ == "__main__":
    main()

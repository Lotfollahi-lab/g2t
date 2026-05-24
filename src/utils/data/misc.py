import math

import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from torch_geometric.utils import to_dense_batch

from utils.data.dataholder import DataHolder


def to_batch(data: DataHolder, device=None) -> DataHolder:
    """
    Convert data to dense representation.
    I.e. it converts the node features and positions to a batch of tensors of
    same dimensions. (i.e. it pads the data to the maximum number of nodes in the batch).

    Args:
        data: Input data.
        device (torch.device, optional): Device for the dense data. Defaults to None.

    Returns:
        DataHolder: Batch representation of the input data.
    """

    node_features, node_mask = to_dense_batch(x=data.node_features, batch=data.batch)
    pos, _ = to_dense_batch(x=data.positions, batch=data.batch)
    cell_class, _ = to_dense_batch(x=data.cell_class, batch=data.batch)
    if cell_class.dim() == 2:
        cell_class = cell_class.unsqueeze(-1)
    elif cell_class.dim() == 3:
        pass
    else:
        raise ValueError("cell_class has wrong dimensionality")
    try:
        cell_ID, _ = to_dense_batch(x=data.cell_ID, batch=data.batch)
        if cell_ID.dim() == 2:
            cell_ID = cell_ID.unsqueeze(-1)
        elif cell_ID.dim() == 3:
            pass
        else:
            raise ValueError("cell_ID has wrong dimensionality")
    except AttributeError:
        cell_ID = None
    pos = pos.float()

    if device is not None:
        node_features = node_features.to(device)
        pos = pos.to(device)
        node_mask = node_mask.to(device)
        cell_class = cell_class.to(device)
        cell_ID = cell_ID.to(device) if cell_ID is not None else None

    data = DataHolder(
        node_features=node_features,
        positions=pos,
        node_mask=node_mask,
        cell_class=cell_class,
        cell_ID=cell_ID,
        diffusion_time=None,
    ).mask()

    return data


def setup_wandb(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """
    Initializes the Weights & Biases (wandb) environment for experiment tracking.

    This function converts the OmegaConf configuration object to a dictionary, sets up the
    wandb environment with specified settings (including project name, configuration, and other
    wandb settings), and then initializes wandb. It also saves any .txt files to the wandb dashboard.

    Parameters:
    cfg (OmegaConf): An OmegaConf configuration object containing the setup parameters for wandb.

    Returns:
    OmegaConf: The configuration object (unchanged).
    """
    # Convert OmegaConf configuration to a dictionary
    config_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    # Project name: prefer the explicit cfg.general.wandb_project setting
    # (our run_scgg_train.py / run_luna_train.py wrappers default it to
    # "scgg" / "luna" respectively). Fall back to upstream LUNA's
    # MolDiffusion_<dataset> when the field is unset (null) so
    # bare-LUNA invocations without a wrapper still work.
    # OmegaConf DictConfig with a declared-null field returns None
    # from both `getattr` and key access — we want that to trigger
    # the fallback, hence the `or` chain.
    explicit_project = getattr(cfg.general, "wandb_project", None)
    fallback_project = f'MolDiffusion_{cfg.dataset["dataset_name"]}'
    project = explicit_project or fallback_project

    # Entity override (team or username). Hydra null → wandb's default
    # entity (= the user's personal account from `wandb login`).
    entity = getattr(cfg.general, "wandb_entity", None) or None

    # Setup wandb initialization arguments
    kwargs = {
        "name": cfg.general.name,
        "project": project,
        "entity": entity,
        "config": config_dict,
        "reinit": True,
        "mode": cfg.general.wandb,
    }
    print(
        f"[setup_wandb] init: project={project!r} "
        f"(explicit={explicit_project!r}, fallback={fallback_project!r}), "
        f"entity={entity!r}, mode={cfg.general.wandb!r}, "
        f"name={cfg.general.name!r}"
    )

    # Initialize wandb
    wandb.init(**kwargs)

    # Save .txt files to wandb
    wandb.save("*.txt")

    return cfg


class GradientMagnitudeCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        """
        Callback function executed after the backward pass to log gradient magnitudes and statistics.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer object.
            pl_module (Module): PyTorch Lightning Module object.

        Raises:
            ValueError: If gradient magnitude is NaN.
        """
        list_of_nans = []  # List to store parameters with NaN gradients
        check_nan_global = False  # Flag to check if any gradient magnitude is NaN
        for name, param in pl_module.named_parameters():
            check_nan = False
            if param.grad is not None:
                # Calculate gradient magnitude and check for NaN
                gradient_magnitude = param.grad.abs().mean().item()
                check_nan = math.isnan(gradient_magnitude)
                if check_nan:
                    list_of_nans.append(name)
                    check_nan_global = True
                else:
                    wandb.log({f"gradient_norm/{name}": gradient_magnitude})

        if not check_nan_global:
            # If no NaN gradients, log statistics of all gradients
            all_gradients = torch.cat(
                [
                    param.grad.view(-1)
                    for param in pl_module.parameters()
                    if param.grad is not None
                ]
            )
            wandb.log(
                {
                    "cumulative_gradient_norms/cumulative_gradient_histogram": wandb.Histogram(
                        all_gradients.cpu().detach().numpy()
                    )
                }
            )
            wandb.log(
                {
                    "cumulative_gradient_norms/cumulative_gradient_statistics": {
                        "mean": all_gradients.mean().item(),
                        "std": all_gradients.std().item(),
                        "min": all_gradients.min().item(),
                        "max": all_gradients.max().item(),
                    }
                }
            )
        else:
            # If any NaN gradients, log the parameters with NaN gradients
            nan_data = {
                "list_of_nans": list_of_nans,
                "sl_no": list(range(len(list_of_nans))),
            }
            df = pd.DataFrame(nan_data)
            wandb.log({"Nan_layers": wandb.Table(dataframe=df)})
            raise ValueError("Gradient magnitude is NaN")

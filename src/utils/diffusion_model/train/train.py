import torch
import wandb

from utils.data.dataholder import DataHolder
from utils.data.misc import to_batch


def training_step_func(self, data: DataHolder, i: int) -> torch.Tensor:
    """
    Training step for a single batch.

    Parameters:
    - data: Batch of input data.
    - i: Index of the current batch.

    Returns:
    - torch.Tensor: Loss for the current batch.
    """
    # Get the current learning rate and log it if using WandB
    lr = self.optimizers().param_groups[0]["lr"]
    if wandb.run:
        wandb.log({"LR": lr}, commit=False)

    # Set the model to train mode
    self.model.train()

    # Preprocess the input data
    batched_data = to_batch(data)
    z_t = self.noise_model.apply_noise(batched_data)

    # Forward pass through the model

    pred = self.forward(z_t)

    # Compute the training loss
    loss, tl_log_dict = self.train_loss(
        masked_pred=pred, masked_true=batched_data, log=i % self.log_every_steps == 0
    )
    loss = loss

    # Log the training loss and metrics if available
    if tl_log_dict is not None:
        self.log_dict(tl_log_dict, batch_size=self.BS)

    # Log epoch metrics for training loss
    tle_log = self.train_loss.log_epoch_metrics()
    self.log_dict(tle_log, batch_size=self.BS)

    # Log the epoch number if using WandB
    if wandb.run:
        wandb.log({"epoch": self.current_epoch}, commit=False)
    return loss


def on_train_epoch_end_func(self) -> None:
    """
    Callback function called at the end of each training epoch.

    Returns:
    - None
    """
    pass


def on_train_epoch_start_func(self) -> None:
    """
    Callback function called at the start of each training epoch.

    Returns:
    - None
    """
    # Reset training loss and metrics for the new epoch
    self.train_loss.reset()

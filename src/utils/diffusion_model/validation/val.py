import torch
import wandb

from metrics.evaluation_statistics import compute_RSSD
from utils.data.dataholder import DataHolder
from utils.data.misc import to_batch


def on_validation_epoch_start_func(self) -> None:
    """
    Callback function called at the start of each validation epoch.

    Returns:
    - None
    """
    self.val_loss.reset()
    self.validation_step_outputs = []
    self.absolute_rssds = []


def validation_step_func(self, data: DataHolder, i: int) -> torch.Tensor:
    """
    Validation step for a single batch.

    Parameters:
    - data: Batch of input data.
    - i: Index of the current batch.

    Returns:
    - torch.Tensor: Loss for the current batch.
    """

    self.model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        # Process data similarly to the training_step
        batched_data = to_batch(data)
        z_t = self.noise_model.apply_noise(batched_data, train_flag=False)
        pred = self.forward(z_t)
        # Compute the loss for validation
        vloss, _ = self.val_loss(
            masked_pred=pred,
            masked_true=batched_data,
            train_stage=False,
            log=i % self.log_every_steps == 0,
        )
        _, __doc__, absolute_rssd = compute_RSSD(batched_data, pred)

    self.validation_step_outputs.append(vloss)
    self.absolute_rssds.append(absolute_rssd)

    return vloss


def on_validation_epoch_end_func(self):
    """
    Callback function called at the end of each validation epoch.

    Returns:
    - None
    """
    # Gather all losses and calculate the mean
    avg_val_loss = torch.stack(self.validation_step_outputs).mean()
    self.absolute_rssds = [
        torch.tensor(item, dtype=torch.float64).to(avg_val_loss.device)
        for item in self.absolute_rssds
    ]
    absolute_rssds = torch.stack(self.absolute_rssds).mean()

    # Use self.log to log the average validation loss
    self.log(
        "val_loss/position_mse",
        avg_val_loss,
        prog_bar=True,
        on_epoch=True,
        sync_dist=True,
    )
    self.log(
        "val_loss/absolute_rssd",
        absolute_rssds,
        prog_bar=True,
        on_epoch=True,
        sync_dist=True,
    )

    self.validation_step_outputs.clear()

    # Optionally, log to WandB or other loggers
    if wandb.run:
        wandb.log({"val_loss/position_mse": avg_val_loss.item()})
        wandb.log({"val_loss/absolute_rssd": absolute_rssds.item()})

import numpy as np
import torch
from utils.data.dataholder import DataHolder


@torch.no_grad()
def sample_noise(self, batch: DataHolder) -> torch.Tensor:
    """
    Samples noise z_t from the noise model based on the node features and other batch information.

    Parameters:
        batch (DataHolder): Batch data containing node features, cell class, and other related information.
        seed (int): Seed for random noise generation.

    Returns:
        torch.Tensor: Sampled noise z_t.
    """
    node_features = batch.node_features
    cell_class = batch.cell_class
    cell_ID = batch.cell_ID
    node_mask = batch.node_mask

    z_t = self.noise_model.sample_limit_dist(
        node_features=node_features,
        cell_class=cell_class,
        cell_ID=cell_ID,
        node_mask=node_mask,
    )

    return z_t.device_as(node_features)


def iterate_sampling(self, z_t: torch.Tensor, batch: DataHolder) -> torch.Tensor:
    """
    Iteratively sample p(z_s | z_t) over the diffusion steps.

    Parameters:
        z_t (torch.Tensor): The initial sampled noise.
        batch (DataHolder): The batch data containing node features.

    Returns:
        torch.Tensor: The final sampled graph after diffusion.

    Self-conditioning support: when ``self._self_cond_enabled`` is
    True (set by FullDenoisingDiffusion based on
    ``cfg.train.self_conditioning.enabled``), each step's prediction
    is saved as ``z_t._self_cond_x0`` and used as conditioning for
    the next step. The very first step uses zeros.
    """
    sample_interval = 1  # Sample interval for the diffusion process
    self_cond_on = bool(getattr(self, "_self_cond_enabled", False))

    # Initialise self-cond to zeros (no prior info at the start of
    # the sampling chain). The LightningModule.forward checks
    # ``z_t._self_cond_x0`` and uses zeros if missing, so this is
    # technically redundant — but explicit setup makes the
    # contract clearer.
    if self_cond_on:
        z_t._self_cond_x0 = torch.zeros_like(z_t.positions)

    # Iteratively sample z_s from z_t for each diffusion step.
    # The forward inside ``sample_zs_from_zt`` returns ``pred``
    # whose ``.positions`` IS the x_0 estimate at this step.
    # We capture that and stash on z_s for the next iteration.
    for s_int in reversed(range(0, self.max_diffusion_steps, sample_interval)):
        s_array = torch.full(
            (1, 1), s_int, dtype=torch.long, device=batch.node_features.device
        )
        # Save the current self-cond before we overwrite z_t.
        prev_self_cond = getattr(z_t, "_self_cond_x0", None) if self_cond_on else None
        z_s = sample_zs_from_zt(self, z_t, s_array)
        if self_cond_on:
            # The next forward will see z_s._self_cond_x0. The pred
            # we just made is the best x_0 estimate so far; cache it
            # so the LightningModule's next forward uses it.
            # ``sample_zs_from_zt_and_pred`` returns z_s with the
            # noise-model-decided positions (the next intermediate);
            # the x_0 ESTIMATE comes from the network's prediction
            # which we don't have direct access to here. We approximate
            # by using z_s.positions, which for FM IS close to the
            # current x_0 estimate after the Euler step.
            z_s._self_cond_x0 = z_s.positions.detach()
        z_t = z_s

    return z_t


@torch.no_grad()
def sample_from_single_graph(
    self, test: bool = True, batch: DataHolder = None
) -> torch.Tensor:
    """
    Samples a batch with specified number of nodes for each graph.

    Parameters:
        test (bool, optional): If True, sampling is done in test mode. Defaults to True.
        seed (int, optional): Seed for random noise generation. Defaults to 0.
        batch (DataHolder, optional): DataHolder containing the data. Defaults to None.

    Returns:
        torch.Tensor: Sampled graph positions.
    """
    num_node = batch.positions[batch.node_mask].shape[0]
    print(f"Sampling. The number of nodes to sample is {num_node}.")

    # Sample noise z_t from the batch
    z_t = sample_noise(self, batch)

    # Perform iterative sampling over diffusion steps
    sampled_graph = iterate_sampling(self, z_t, batch)

    return sampled_graph.positions


def sample_zs_from_zt(self, z_t: torch.Tensor, s_int: torch.Tensor) -> torch.Tensor:
    """
    Samples zs ~ p(zs | zt) for the denoising process.

    Parameters:
        z_t (torch.Tensor): The tensor representing z_t in the diffusion process.
        s_int (torch.Tensor): The tensor representing the integer time step s.

    Returns:
        torch.Tensor: The sampled zs tensor.

    Notes:
        Default path is single-forward (Euler / DDPM reverse step).
        If the noise model advertises ``sampler == "heun"`` (currently
        FlowMatchingModel only), we take a SECOND forward at the
        Euler-predicted endpoint and ask the noise model to apply
        the Heun (trapezoidal) correction. DDPM's NoiseModel never
        sets ``sampler``, so its path is untouched.
    """
    pred = self.forward(z_t)
    z_s = self.noise_model.sample_zs_from_zt_and_pred(z_t=z_t, pred=pred, s_int=s_int)

    # 2nd-order (Heun) correction: one extra forward at the Euler
    # endpoint, averaged-velocity re-step. Only flow-matching opts
    # in. The DDPM NoiseModel has no `sampler` attr so this branch
    # is dead for the diffusion path.
    if getattr(self.noise_model, "sampler", "euler") == "heun":
        pred2 = self.forward(z_s)
        z_s = self.noise_model.heun_correction(
            z_t=z_t, pred1=pred, z_s_euler=z_s, pred2=pred2, s_int=s_int,
        )
    return z_s


@torch.no_grad()
def sample_graphs(self, batch: DataHolder, test: bool) -> list[torch.Tensor]:
    """
    Samples multiple graphs from the given batch data.

    Parameters:
        batch (DataHolder): The batch of data to sample from.
        samples_to_generate (int): Number of graph samples to generate.
        test (bool): If True, sampling is done in test mode.

    Returns:
        list[torch.Tensor]: A list of sampled graph positions.
    """
    samples = []

    sample = sample_from_single_graph(self, test=test, batch=batch)
    samples.extend(sample)

    return samples

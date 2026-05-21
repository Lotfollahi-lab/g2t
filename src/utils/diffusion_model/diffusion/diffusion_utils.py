import math

import numpy as np
import torch
from torch.nn import functional as F


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Sum the elements of the input tensor along all dimensions except the batch dimension.

    Parameters:
        - x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Resulting tensor after summing along dimensions except the batch dimension.
    """
    # Reshape the tensor, keeping the batch dimension unchanged
    reshaped_x = x.reshape(x.size(0), -1)

    # Sum along the reshaped dimensions (excluding the batch dimension)
    result = reshaped_x.sum(dim=-1)

    return result


def assert_correctly_masked(
    variable: torch.Tensor, node_mask: torch.Tensor
) -> torch.Tensor:
    """
    Assert that the given tensor is correctly masked by the binary node_mask.

    Parameters:
        - variable (torch.Tensor): Input tensor.
        - node_mask (torch.Tensor): Binary mask tensor.

    Raises:
        AssertionError: If NaN values are present in the tensor or if unmasked elements are not close to zero.
    """
    # Check for NaN values in the tensor
    assert not torch.isnan(variable).any(), f"Shape:{variable.shape}"

    # Check that unmasked elements are close to zero
    masked_variable = variable * (1 - node_mask.long())
    assert (
        masked_variable.abs().max().item() < 1e-4
    ), f"Variables not masked properly. {masked_variable}"


def sample_gaussian_with_mask(size: int, node_mask: torch.Tensor) -> torch.Tensor:
    x = torch.randn(size).to(node_mask.device)
    x_masked = x * node_mask
    return x_masked


def clip_noise_schedule(alphas2, clip_value=0.001) -> np.ndarray:
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = alphas2[1:] / alphas2[:-1]
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)
    return alphas2


def cosine_beta_schedule(
    timesteps: int, s: float = 0.008, raise_to_power: float = 1
) -> np.ndarray:
    """
    Generate a cosine schedule for betas in a diffusion model.

    Parameters:
        - timesteps (int): Number of diffusion steps.
        - s (float, optional): Scaling factor for the cosine schedule.
        - raise_to_power (float, optional): Exponentiation factor for the cumulative alphas.

    Returns:
        numpy.ndarray: Cosine schedule for betas.

    Notes:
        - The cosine schedule is based on the formula proposed in the referenced paper.
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    # Calculate cumulative alphas using a cosine schedule
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # Calculate betas from cumulative alphas
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)

    # Calculate alphas from betas
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    # Raise cumulative alphas to the specified power
    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def cosine_beta_schedule_discrete(
    timesteps: int, nu_arr: np.ndarray, s=0.008
) -> np.ndarray:
    """
    Generate a cosine schedule for betas in a diffusion model with discrete components.

    Parameters:
        - timesteps (int): Number of diffusion steps.
        - nu_arr (list or array): List or array specifying the components for each nu.
        - s (float, optional): Scaling factor for the cosine schedule.

    Returns:
        numpy.ndarray: Cosine schedule for betas.

    Notes:
        - The cosine schedule is based on the formula proposed in the referenced paper.
        - The function considers different components specified by nu_arr.

    Example:
        cosine_beta_schedule_discrete(10, [1, 2, 3, 4, 5], s=0.008)
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    x = np.expand_dims(x, 0)  # ((1, steps))

    nu_arr = np.array(nu_arr)  # (components, )  # X, charges, E, y, pos
    nu_arr = np.expand_dims(nu_arr, 1)  # ((components, 1))

    alphas_cumprod = (
        np.cos(0.5 * np.pi * (((x / steps) ** nu_arr) + s) / (1 + s)) ** 2
    )  # ((components, steps))
    # divide every element of alphas_cumprod by the first element of alphas_cumprod
    alphas_cumprod_new = alphas_cumprod / np.expand_dims(alphas_cumprod[:, 0], 1)
    # remove the first element of alphas_cumprod and then multiply every element by the one before it
    alphas = alphas_cumprod_new[:, 1:] / alphas_cumprod_new[:, :-1]

    betas = 1 - alphas  # ((components, steps)) # X, charges, E, y, pos
    betas = np.swapaxes(betas, 0, 1)

    return betas


def gaussian_KL(q_mu: torch.Tensor, q_sigma: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL distance between a normal distribution and the standard normal.

    Args:
        q_mu (torch.Tensor): Mean of distribution q.
        q_sigma (torch.Tensor): Standard deviation of distribution q.

    Returns:
        torch.Tensor: The KL distance, summed over all dimensions except the batch dim.
    """
    return torch.log(1 / q_sigma) + 0.5 * (q_sigma**2 + q_mu**2) - 0.5


def cdf_std_gaussian(x) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def SNR(gamma) -> torch.Tensor:
    """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
    return torch.exp(-gamma)


def inflate_batch_array(array, target_shape) -> torch.Tensor:
    """
    Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape = (array.size(0),) + (1,) * (len(target_shape) - 1)
    return array.view(target_shape)


def sigma(gamma, target_shape) -> torch.Tensor:
    """Computes sigma given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_shape)


def alpha(gamma, target_shape) -> torch.Tensor:
    """Computes alpha given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_shape)


def check_mask_correct(variables, node_mask) -> None:
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def check_tensor_same_size(*args) -> None:
    for i, arg in enumerate(args):
        if i == 0:
            continue
        assert args[0].size() == arg.size()


def sigma_and_alpha_t_given_s(
    gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_size: torch.Size
) -> torch.Tensor:
    """
    Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

    These are defined as:
        alpha t given s = alpha t / alpha s,
        sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
    """
    sigma2_t_given_s = inflate_batch_array(
        -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_size
    )

    # alpha_t_given_s = alpha_t / alpha_s
    log_alpha2_t = F.logsigmoid(-gamma_t)
    log_alpha2_s = F.logsigmoid(-gamma_s)
    log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

    alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
    alpha_t_given_s = inflate_batch_array(alpha_t_given_s, target_size)

    sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

    return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s


def reverse_tensor(x) -> torch.Tensor:
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def check_issues_norm_values(gamma, norm_val1, norm_val2, num_stdevs=8) -> None:
    """Check if 1 / norm_value is still larger than 10 * standard deviation."""
    zeros = torch.zeros((1, 1))
    gamma_0 = gamma(zeros)
    sigma_0 = sigma(gamma_0, target_shape=zeros.size()).item()
    max_norm_value = max(norm_val1, norm_val2)
    if sigma_0 * num_stdevs > 1.0 / max_norm_value:
        raise ValueError(
            f"Value for normalization value {max_norm_value} probably too "
            f"large with sigma_0 {sigma_0:.5f}*{num_stdevs} and "
            f"1 / norm_value = {1. / max_norm_value}"
        )

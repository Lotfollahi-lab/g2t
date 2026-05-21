import torch
import wandb
from utils.data.dataholder import DataHolder
from utils.data.load import remove_mean_with_mask
from utils.diffusion_model.diffusion.diffusion_utils import (
    cosine_beta_schedule_discrete,
)


class NoiseModel:
    def __init__(self, cfg):
        """
        Initialize the NoiseModel with configuration parameters.

        Parameters:
            - cfg: Configuration object containing model parameters.

        Notes:
            - The NoiseModel is designed for modeling diffusion noise.
            - It supports a specified diffusion noise schedule, such as "cosine".
        """
        # Mapping for discrete features
        self.mapping = ["p"]
        self.inverse_mapping = {m: i for i, m in enumerate(self.mapping)}

        # Extract nu values from the configuration for each discrete feature
        nu = cfg.model.nu
        self.nu_arr = [nu[m] for m in self.mapping]

        # Define diffusion model parameters
        self.noise_schedule = cfg.model.diffusion_noise_schedule
        self.timesteps = cfg.model.diffusion_steps
        self.max_diffusion_steps = cfg.model.diffusion_steps

        # Initialize beta values based on the diffusion noise schedule
        if self.noise_schedule == "cosine":
            betas = cosine_beta_schedule_discrete(self.timesteps, self.nu_arr)
        else:
            raise NotImplementedError(self.noise_schedule)

        # Initialize alpha and log alpha values
        self._betas = torch.from_numpy(betas)
        self._alphas = 1 - torch.clamp(self._betas, min=0, max=0.9999)
        log_alpha = torch.log(self._alphas)

        # Calculate cumulative sum and exponential of log alpha for further computations
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self._log_alpha_bar = log_alpha_bar
        self._alphas_bar = torch.exp(log_alpha_bar)

        # Calculate sigma_bar and gamma values
        self._sigma2_bar = -torch.expm1(2 * log_alpha_bar)
        self._sigma_bar = torch.sqrt(self._sigma2_bar)
        self._gamma = (
            torch.log(-torch.special.expm1(2 * log_alpha_bar)) - 2 * log_alpha_bar
        )

    def get_alpha_bar(
        self, t_normalized: int = None, t_int: int = None, key: int = None
    ) -> torch.Tensor:
        """
        Get denoising parameter alpha_bar based on normalized or integer time.

        Parameters:
            - t_normalized (float, optional): Normalized diffusion time.
            - t_int (torch.Tensor, optional): Integer diffusion time.
            - key (str, optional): Key for accessing specific components.

        Returns:
            torch.Tensor: Denoising parameter alpha_bar.

        Notes:
            - Either t_normalized or t_int should be provided.
            - If key is specified, return the component of alpha_bar based on the inverse mapping.

        Raises:
            AssertionError: If both t_normalized and t_int are provided or none of them are provided.
        """
        assert int(t_normalized is None) + int(t_int is None) == 1

        # If t_int is not provided, calculate it from t_normalized
        if t_int is None:
            t_int = torch.round(t_normalized * self.max_diffusion_steps)

        # Retrieve alpha_bar values based on the calculated t_int
        a = self._alphas_bar.to(t_int.device)[t_int.long()]

        # If key is not specified, return the entire alpha_bar as a float
        if key is None:
            return a.float()
        else:
            # If key is specified, return the component based on the inverse mapping
            return a[..., self.inverse_mapping[key]].float()

    def get_sigma_bar(
        self,
        t_normalized: torch.Tensor = None,  # Expected dtype: torch.int
        t_int: torch.Tensor = None,  # Expected dtype: torch.int
        key=None,
    ) -> torch.Tensor:  # Expected dtype: torch.float
        """
        Get denoising parameter sigma_bar based on normalized or integer time.

        Parameters:
            - t_normalized (float, optional): Normalized diffusion time.
            - t_int (torch.Tensor, optional): Integer diffusion time.
            - key (str, optional): Key for accessing specific components.

        Returns:
            torch.Tensor: Denoising parameter sigma_bar.

        Notes:
            - Either t_normalized or t_int should be provided.
            - If key is specified, return the component of sigma_bar based on the inverse mapping.

        Raises:
            AssertionError: If both t_normalized and t_int are provided or none of them are provided.
        """
        assert int(t_normalized is None) + int(t_int is None) == 1

        # If t_int is not provided, calculate it from t_normalized
        if t_int is None:
            t_int = torch.round(t_normalized * self.max_diffusion_steps)

        # Retrieve sigma_bar values based on the calculated t_int
        s = self._sigma_bar.to(t_int.device)[t_int]

        # If key is not specified, return the entire sigma_bar as a float
        if key is None:
            return s.float()
        else:
            # If key is specified, return the component based on the inverse mapping
            return s[..., self.inverse_mapping[key]].float()

    def get_alpha_pos_ts(
        self,
        s_int: torch.Tensor,  # Expected dtype: torch.int
        t_int: torch.Tensor,  # Expected dtype: torch.int
    ) -> torch.Tensor:  # Expected dtype: torch.float
        """
        Calculate the ratio of alpha values for position denoising.

        Parameters:
            - s_int (torch.Tensor): Integer target diffusion time.
            - t_int (torch.Tensor): Integer source diffusion time.

        Returns:
            torch.Tensor: Ratio of alpha values for position denoising.

        Notes:
            - The formula used is exp(log(a_t) - log(a_s)).
            - The log alpha_bar values are retrieved based on the inverse mapping.

        Raises:
            AssertionError: If either t_int or s_int is not provided.
        """
        log_a_bar = self._log_alpha_bar[..., self.inverse_mapping["p"]].to(t_int.device)
        ratio = torch.exp(log_a_bar[t_int] - log_a_bar[s_int])
        return ratio.float()

    def get_alpha_pos_ts_sq(
        self,
        s_int: torch.Tensor,  # Expected dtype: torch.int
        t_int: torch.Tensor,  # Expected dtype: torch.int
    ) -> torch.Tensor:  # Expected dtype: torch.float
        """
        Calculate the squared ratio of alpha values for position denoising.

        Parameters:
            - t_int (torch.Tensor): Integer target diffusion time.
            - s_int (torch.Tensor): Integer source diffusion time.

        Returns:
            torch.Tensor: Squared ratio of alpha values for position denoising.

        Notes:
            - The formula used is exp(2 * log(a_t) - 2 * log(a_s)).
            - The log alpha_bar values are retrieved based on the inverse mapping.

        Raises:
            AssertionError: If either t_int or s_int is not provided.
        """
        log_a_bar = self._log_alpha_bar[..., self.inverse_mapping["p"]].to(t_int.device)
        ratio = torch.exp(2 * log_a_bar[t_int] - 2 * log_a_bar[s_int])
        return ratio.float()

    def get_sigma_pos_sq_ratio(
        self,
        s_int: torch.Tensor,  # Expected dtype: torch.int
        t_int: torch.Tensor,  # Expected dtype: torch.int
    ) -> torch.Tensor:  # Expected dtype: torch.float
        """
        Calculate the ratio of sigma squared values for position denoising.

        Parameters:
            - s_int (torch.Tensor): Integer source diffusion time.
            - t_int (torch.Tensor): Integer target diffusion time.

        Returns:
            torch.Tensor: Ratio of sigma squared values for position denoising.

        Notes:
            - The formula used is exp(log(s2_s) - log(s2_t)).
            - The log alpha_bar values are retrieved based on the inverse mapping.
            - The sigma squared values are calculated using expm1.

        Raises:
            AssertionError: If either s_int or t_int is not provided.
        """
        log_a_bar = self._log_alpha_bar[..., self.inverse_mapping["p"]].to(t_int.device)
        s2_s = -torch.expm1(2 * log_a_bar[s_int])
        s2_t = -torch.expm1(2 * log_a_bar[t_int])
        ratio = torch.exp(torch.log(s2_s) - torch.log(s2_t))
        return ratio.float()

    def get_positions_prefactor(
        self,
        s_int: torch.Tensor,  # Expected dtype: torch.int
        t_int: torch.Tensor,  # Expected dtype: torch.int
    ) -> torch.Tensor:  # Expected dtype: torch.float
        """
        Calculate the position prefactor for denoising.

        Parameters:
            - s_int (torch.Tensor): Integer source diffusion time.
            - t_int (torch.Tensor): Integer target diffusion time.

        Returns:
            torch.Tensor: Position prefactor for denoising.

        Notes:
            - The formula used is (a_s * (1 - a_t_s^2 * s_s^2 / s_t^2)).
            - The denoising parameter a_s is obtained using get_alpha_bar().
            - The ratios of alpha and sigma values are calculated using other methods.

        Raises:
            AssertionError: If either s_int or t_int is not provided.
        """
        a_s = self.get_alpha_bar(t_int=s_int, key="p")
        alpha_ratio_sq = self.get_alpha_pos_ts_sq(s_int=s_int, t_int=t_int)
        sigma_ratio_sq = self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
        prefactor = a_s * (1 - alpha_ratio_sq * sigma_ratio_sq)
        return prefactor.float()

    def apply_noise(self, data: DataHolder, train_flag=True) -> DataHolder:
        """
        Apply noise to the input data.

        Parameters:
            - data (DataHolder): Input data containing node features, positions, node mask, and diffusion time.

        Returns:
            DataHolder: Data with noise applied to positions and updated diffusion time.

        Notes:
            - The noise is generated using random sampling and then modified based on the node mask.
            - The denoising parameters beta_t and alpha_s_bar are used for loss computation.

        Raises:
            AssertionError: If the diffusion time tensor is not within the expected range.
        """
        # Sample a random integer for time manipulation
        t_int = torch.randint(
            1,
            self.max_diffusion_steps + 1,
            size=(data.node_features.size(0), 1),
            device=data.node_features.device,
        )
        t_float = t_int.float() / self.max_diffusion_steps

        if wandb.run is not None and train_flag:
            wandb.log({"t_int/histogram": wandb.Histogram(t_int[0].cpu().numpy())})
            wandb.log({"t_int/record": t_int[0].cpu().numpy()})

        # Generate random noise for positions and apply mean removal
        noise_pos = torch.randn(data.positions.shape, device=data.node_features.device)
        noise_positions_masked = noise_pos * data.node_mask.unsqueeze(-1)
        noise_positions_masked = remove_mean_with_mask(
            x=noise_positions_masked, node_mask=data.node_mask
        )

        # Calculate denoising parameters for position modification
        a = self.get_alpha_bar(t_int=t_int, key="p").unsqueeze(-1)
        s = self.get_sigma_bar(t_int=t_int, key="p").unsqueeze(-1)

        # Apply noise to positions based on denoising parameters
        pos_t = a * data.positions + s * noise_positions_masked

        # Create a new DataHolder with updated positions and diffusion time
        z_t = DataHolder(
            node_features=data.node_features,
            positions=pos_t,
            cell_class=data.cell_class,
            node_mask=data.node_mask,
            t_int=t_int,
            t=t_float,
            diffusion_time=t_float,
        ).mask()

        return z_t

    def sample_limit_dist(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        cell_ID: torch.Tensor,
        cell_class: torch.Tensor,
    ) -> DataHolder:
        """
        Sample from the limit distribution of the diffusion process.

        Parameters:
            - node_features (torch.Tensor): Node features tensor.
            - node_mask (torch.Tensor): Boolean mask tensor.
            - cell_class (torch.Tensor): Cell class tensor.

        Returns:
            DataHolder: Data with positions sampled from the limit distribution and appropriate time information.

        Notes:
            - Randomly sample positions from a normal distribution.
            - Apply node mask to the sampled positions.
            - Remove mean from the sampled positions using a utility function (utils.remove_mean_with_mask).
            - Create tensors for time information.
            - Create a DataHolder object with node features, positions, node mask, and time information.
            - Apply a mask to the resulting DataHolder.

        Raises:
            AssertionError: If the shape of node_mask is inconsistent.
        """
        # Randomly sample positions from a normal distribution
        positions = torch.randn(
            node_mask.shape[0], node_mask.shape[1], 2, device=node_mask.device
        )

        torch.manual_seed(0)

        # Apply node mask to the sampled positions
        positions = positions * node_mask.unsqueeze(-1)
        # Remove mean from the sampled positions using a utility function
        # @TODO: should nomalize the positions or not? (-.5 to .5)
        positions = remove_mean_with_mask(positions, node_mask)
        # Create tensors for time information
        t_array = positions.new_ones((positions.shape[0], 1))
        t_int_array = self.max_diffusion_steps * t_array.long()

        # Create a DataHolder object with node features, positions, node mask, and time information
        result_data = DataHolder(
            node_features=node_features,
            positions=positions,
            node_mask=node_mask,
            cell_class=cell_class,
            cell_ID=cell_ID,
            t_int=t_int_array,
            t=t_array,
            diffusion_time=t_array,
        )

        # Apply a mask to the resulting DataHolder
        return result_data.mask()

    def sample_zs_from_zt_and_pred(
        self,
        z_t: DataHolder,
        pred: DataHolder,
        s_int: torch.Tensor,  # Expected dtype: torch.int
    ) -> DataHolder:
        """
        Sample from zs ~ p(zs | zt). Only used during sampling.

        Parameters:
            - z_t (DataHolder): Input DataHolder representing the state at time t (zt).
            - pred (DataHolder): Input DataHolder representing predictions for the next time step.
            - s_int (torch.Tensor): Integer source diffusion time.

        Returns:
            DataHolder: Sampled DataHolder representing the state at time s (zs).

        Notes:
            - This function generates synthetic samples for the next diffusion time step based on the current state (zt) and predictions.
            - The positions of zs are sampled using a combination of the previous state (zt), predicted positions, and added noise.
            - The sampled positions are influenced by the previous state, predictions, and noise drawn from a normal distribution.
            - The resulting DataHolder includes node features, sampled positions, node mask, and time information.

        Raises:
            AssertionError: If the shapes of z_t.positions and pred.positions are inconsistent.
        """
        # Extract relevant information from the current state at time t (zt)
        node_mask = z_t.node_mask
        t_int = z_t.t_int

        # Sample the positions for zs
        sigma_sq_ratio = self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
        z_t_prefactor = (
            self.get_alpha_pos_ts(s_int=s_int, t_int=t_int) * sigma_sq_ratio
        ).unsqueeze(-1)
        positions_prefactor = self.get_positions_prefactor(
            s_int=s_int, t_int=t_int
        ).unsqueeze(-1)

        # Calculate the mean positions for zs
        mu = z_t_prefactor * z_t.positions + positions_prefactor * pred.positions

        # Sample noise for zs
        sampled_pos = torch.randn(
            z_t.positions.shape, device=z_t.positions.device
        ) * node_mask.unsqueeze(-1)
        noise = remove_mean_with_mask(sampled_pos, node_mask=node_mask)

        # Calculate prefactors for noise
        prefactor1 = self.get_sigma_bar(t_int=t_int, key="p")
        prefactor2 = self.get_sigma_bar(
            t_int=s_int, key="p"
        ) * self.get_alpha_pos_ts_sq(s_int=s_int, t_int=t_int)
        sigma2_t_s = prefactor1 - prefactor2
        noise_prefactor_sq = sigma2_t_s * sigma_sq_ratio
        noise_prefactor = torch.sqrt(noise_prefactor_sq).unsqueeze(-1)

        # Combine mean and noise to get the final sampled positions for zs
        positions = mu + noise_prefactor * noise

        # Create a DataHolder for zs with relevant information
        z_s = DataHolder(
            node_features=z_t.node_features,
            positions=positions,
            node_mask=node_mask,
            t_int=s_int,
            t=s_int / self.max_diffusion_steps,
            diffusion_time=s_int / self.max_diffusion_steps,
        ).mask()

        return z_s

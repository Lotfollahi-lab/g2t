"""
Conditional flow matching for spatial embedding generation.

Implements optimal transport conditional flow matching (OT-CFM) following
Lipman et al. (2023) "Flow Matching for Generative Modeling". The flow
learns to transport samples from a Gaussian prior to spatial coordinate
distributions, conditioned on gene expression features.

Key design choices:
- OT conditional paths (linear interpolation) for stable training.
- Per-cell flow (no cell-cell interaction in the generative process itself;
  cell context enters only through conditioning).
- Multiple ODE solvers for inference (Euler, midpoint, RK4).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal

from .velocity_net import VelocityNetwork


class ConditionalFlowMatching(nn.Module):
    """Conditional flow matching module for spatial embedding generation.

    During training, samples random times t ~ U(0,1), constructs interpolated
    states z_t = (1 - (1-sigma_min)*t) * z_0 + t * z_1, and trains the
    velocity network to predict the conditional velocity u_t = z_1 - (1-sigma_min)*z_0.

    During inference, integrates the learned velocity field from t=0 to t=1
    starting from z_0 ~ N(0, I) to generate spatial embeddings.

    Args:
        velocity_net: The velocity prediction network.
        sigma_min: Minimum noise scale for OT conditional paths.
    """

    def __init__(
        self,
        velocity_net: VelocityNetwork,
        sigma_min: float = 1e-4,
        pairwise_dist_weight: float = 1.0,
        velocity_mse_weight: float = 0.1,
        translation_equivariant: bool = True,
        pairwise_dist_max_cells: int = 4096,
    ):
        """Conditional flow matching with LUNA-style pairwise-distance loss.

        Two loss terms run jointly:

          * **velocity MSE**: standard OT-CFM target ``||v_pred - u_t||^2``.
            Kept as a small regulariser; this is what the prior version
            optimised exclusively.

          * **pairwise-distance MSE**: LUNA's
            ``metrics.loss_function.LossFunction`` target —
            ``MSE(cdist(x_hat_0), cdist(z_1))`` on the implied x_0
            estimate. Rotation-, translation-, and reflection-invariant,
            which means we're directly optimising the structural property
            the Spearman headline metric measures. This is the primary
            training signal in the LUNA-aligned setup.

        ``translation_equivariant``: when True, subtract the slice mean
        from the predicted velocity before computing losses (mirrors
        LUNA's ``new_pos - mean(new_pos, dim=1)`` step). With centred
        targets this removes the model's incentive to chase the centroid.

        ``pairwise_dist_max_cells``: cap on cells used for the cdist
        loss. For a slice of N cells the cdist matrix is N×N, so >4k
        cells per slice would blow up GPU memory. Subsampling cells
        keeps the loss stochastic but unbiased.

        Args:
            velocity_net: The velocity prediction network.
            sigma_min: Minimum noise scale for OT conditional paths.
            pairwise_dist_weight: weight on the pairwise-distance MSE.
            velocity_mse_weight: weight on the velocity MSE.
            translation_equivariant: subtract per-batch velocity mean.
            pairwise_dist_max_cells: subsample cells for cdist when batch
                exceeds this.
        """
        super().__init__()
        self.velocity_net = velocity_net
        self.sigma_min = sigma_min
        self.spatial_dim = velocity_net.spatial_dim
        self.pairwise_dist_weight = float(pairwise_dist_weight)
        self.velocity_mse_weight = float(velocity_mse_weight)
        self.translation_equivariant = bool(translation_equivariant)
        self.pairwise_dist_max_cells = int(pairwise_dist_max_cells)

    @staticmethod
    def _center(x: torch.Tensor) -> torch.Tensor:
        """Subtract the batch mean from x (translation equivariance)."""
        return x - x.mean(dim=0, keepdim=True)

    def compute_loss(
        self,
        z_1: torch.Tensor,
        cell_embed: torch.Tensor,
        section_embed: torch.Tensor,
        k_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute flow matching training loss.

        Args:
            z_1: Target spatial coordinates, shape (batch_size, spatial_dim).
            cell_embed: Cell expression embeddings, shape (batch_size, cell_embed_dim).
            section_embed: Section embedding, shape (embed_dim,).
            k_target: Target k values, shape (batch_size,) of ints.

        Returns:
            loss: Scalar flow matching loss.
            metrics: Dict with additional training metrics.
        """
        batch_size = z_1.shape[0]
        device = z_1.device

        # ----- 1. Centre the target (translation equivariance) ---------
        # The dataset normalisation is supposed to be zero-mean per
        # section already, but each mini-batch is a random subset of the
        # slice and so its empirical mean is not exactly zero. Removing
        # the batch mean here makes the targets the model is asked to
        # predict invariant to translation, and matches LUNA's
        # `remove_mean_with_mask` step.
        if self.translation_equivariant:
            z_1 = self._center(z_1)

        # ----- 2. Sample t and z_0, build z_t / u_t --------------------
        t = torch.rand(batch_size, device=device)
        z_0 = torch.randn_like(z_1)
        if self.translation_equivariant:
            z_0 = self._center(z_0)

        t_expand = t.unsqueeze(-1)  # (batch_size, 1)
        mu_t = t_expand * z_1
        sigma_t = 1.0 - (1.0 - self.sigma_min) * t_expand
        z_t = sigma_t * z_0 + mu_t

        # OT-CFM conditional velocity target
        u_t = z_1 - (1.0 - self.sigma_min) * z_0

        # ----- 3. Network prediction -----------------------------------
        v_t = self.velocity_net(z_t, t, cell_embed, section_embed, k_target)
        if self.translation_equivariant:
            # Subtract per-batch mean of the velocity prediction. This
            # makes the velocity field translation-equivariant: the model
            # cannot predict a net drift away from the slice centroid.
            v_t = self._center(v_t)

        # ----- 4. Velocity MSE (the original FM objective) -------------
        velocity_mse = torch.mean((v_t - u_t) ** 2)

        # ----- 5. Implied x_0 estimate ---------------------------------
        # For OT-CFM with sigma_min ≈ 0:
        #   z_t = (1-t)*z_0 + t*z_1   and   v_t = z_1 - z_0
        #   ⇒  z_1 = z_t + (1-t)*v_t
        # With sigma_min > 0 the correction is O(sigma_min) and negligible
        # (sigma_min defaults to 1e-4).
        one_minus_t = (1.0 - t).unsqueeze(-1)
        x_hat_0 = z_t + one_minus_t * v_t
        if self.translation_equivariant:
            x_hat_0 = self._center(x_hat_0)

        # ----- 6. Pairwise-distance MSE (LUNA's loss) ------------------
        # Subsample cells for the cdist to keep memory bounded. For
        # cortex slices of ~5–7k cells with the default cap of 4096,
        # this is a no-op when n <= cap. cdist(n, n) is n^2 floats.
        if (
            self.pairwise_dist_max_cells > 0
            and batch_size > self.pairwise_dist_max_cells
        ):
            idx = torch.randperm(batch_size, device=device)[
                : self.pairwise_dist_max_cells
            ]
            x_hat_0_sub = x_hat_0[idx]
            z_1_sub = z_1[idx]
        else:
            x_hat_0_sub = x_hat_0
            z_1_sub = z_1

        d_pred = torch.cdist(x_hat_0_sub, x_hat_0_sub, p=2.0)
        d_true = torch.cdist(z_1_sub, z_1_sub, p=2.0)
        pairwise_dist_mse = torch.mean((d_pred - d_true) ** 2)

        # ----- 7. Combined loss ----------------------------------------
        loss = (
            self.velocity_mse_weight * velocity_mse
            + self.pairwise_dist_weight * pairwise_dist_mse
        )

        metrics = {
            "fm_loss": loss.item(),
            "fm_velocity_mse": velocity_mse.item(),
            "fm_pairwise_dist_mse": pairwise_dist_mse.item(),
            "v_norm": torch.mean(torch.norm(v_t, dim=-1)).item(),
            "u_norm": torch.mean(torch.norm(u_t, dim=-1)).item(),
            "x_hat_0_norm": torch.mean(torch.norm(x_hat_0, dim=-1)).item(),
        }

        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        cell_embed: torch.Tensor,
        section_embed: torch.Tensor,
        k_target: torch.Tensor,
        n_steps: int = 100,
        solver: Literal["euler", "midpoint", "rk4"] = "euler",
    ) -> torch.Tensor:
        """Generate spatial embeddings by integrating the learned flow.

        Args:
            cell_embed: Cell expression embeddings, shape (n_cells, cell_embed_dim).
            section_embed: Section embedding, shape (embed_dim,).
            k_target: Target k values, shape (n_cells,) of ints.
            n_steps: Number of ODE integration steps.
            solver: ODE solver type.

        Returns:
            Generated spatial embeddings, shape (n_cells, spatial_dim).
        """
        n_cells = cell_embed.shape[0]
        device = cell_embed.device

        # Sample from prior, centred (matches training).
        z = torch.randn(n_cells, self.spatial_dim, device=device)
        if self.translation_equivariant:
            z = self._center(z)

        dt = 1.0 / n_steps

        for step in range(n_steps):
            t_val = step * dt
            t = torch.full((n_cells,), t_val, device=device)

            if solver == "euler":
                v = self.velocity_net(z, t, cell_embed, section_embed, k_target)
                if self.translation_equivariant:
                    v = self._center(v)
                z = z + dt * v

            elif solver == "midpoint":
                # Half step
                v1 = self.velocity_net(z, t, cell_embed, section_embed, k_target)
                if self.translation_equivariant:
                    v1 = self._center(v1)
                z_mid = z + 0.5 * dt * v1
                t_mid = torch.full((n_cells,), t_val + 0.5 * dt, device=device)
                v2 = self.velocity_net(z_mid, t_mid, cell_embed, section_embed, k_target)
                if self.translation_equivariant:
                    v2 = self._center(v2)
                z = z + dt * v2

            elif solver == "rk4":
                t1 = t
                t2 = torch.full((n_cells,), t_val + 0.5 * dt, device=device)
                t3 = t2
                t4 = torch.full((n_cells,), t_val + dt, device=device)

                def _v(zz, tt):
                    out = self.velocity_net(zz, tt, cell_embed, section_embed, k_target)
                    return self._center(out) if self.translation_equivariant else out

                k1 = _v(z, t1)
                k2 = _v(z + 0.5 * dt * k1, t2)
                k3 = _v(z + 0.5 * dt * k2, t3)
                k4 = _v(z + dt * k3, t4)
                z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return z

    @torch.no_grad()
    def sample_trajectory(
        self,
        cell_embed: torch.Tensor,
        section_embed: torch.Tensor,
        k_target: torch.Tensor,
        n_steps: int = 100,
        save_every: int = 10,
    ) -> torch.Tensor:
        """Sample and return intermediate states for visualization.

        Returns:
            Trajectory tensor, shape (n_saved, n_cells, spatial_dim).
        """
        n_cells = cell_embed.shape[0]
        device = cell_embed.device

        z = torch.randn(n_cells, self.spatial_dim, device=device)
        dt = 1.0 / n_steps

        trajectory = [z.clone()]

        for step in range(n_steps):
            t_val = step * dt
            t = torch.full((n_cells,), t_val, device=device)
            v = self.velocity_net(z, t, cell_embed, section_embed, k_target)
            z = z + dt * v

            if (step + 1) % save_every == 0:
                trajectory.append(z.clone())

        return torch.stack(trajectory, dim=0)

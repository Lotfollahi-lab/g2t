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
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.sigma_min = sigma_min
        self.spatial_dim = velocity_net.spatial_dim

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

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Sample noise from prior
        z_0 = torch.randn_like(z_1)

        # Construct interpolated state (OT conditional path)
        # z_t = (1 - (1 - sigma_min) * t) * z_0 + t * z_1
        t_expand = t.unsqueeze(-1)  # (batch_size, 1)
        mu_t = t_expand * z_1
        sigma_t = 1.0 - (1.0 - self.sigma_min) * t_expand
        z_t = sigma_t * z_0 + mu_t

        # Conditional velocity target
        # u_t = z_1 - (1 - sigma_min) * z_0
        u_t = z_1 - (1.0 - self.sigma_min) * z_0

        # Predict velocity
        v_t = self.velocity_net(z_t, t, cell_embed, section_embed, k_target)

        # Flow matching loss (MSE)
        loss = torch.mean((v_t - u_t) ** 2)

        metrics = {
            "fm_loss": loss.item(),
            "v_norm": torch.mean(torch.norm(v_t, dim=-1)).item(),
            "u_norm": torch.mean(torch.norm(u_t, dim=-1)).item(),
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

        # Sample from prior
        z = torch.randn(n_cells, self.spatial_dim, device=device)

        dt = 1.0 / n_steps

        for step in range(n_steps):
            t_val = step * dt
            t = torch.full((n_cells,), t_val, device=device)

            if solver == "euler":
                v = self.velocity_net(z, t, cell_embed, section_embed, k_target)
                z = z + dt * v

            elif solver == "midpoint":
                # Half step
                v1 = self.velocity_net(z, t, cell_embed, section_embed, k_target)
                z_mid = z + 0.5 * dt * v1
                t_mid = torch.full((n_cells,), t_val + 0.5 * dt, device=device)
                v2 = self.velocity_net(z_mid, t_mid, cell_embed, section_embed, k_target)
                z = z + dt * v2

            elif solver == "rk4":
                t1 = t
                t2 = torch.full((n_cells,), t_val + 0.5 * dt, device=device)
                t3 = t2
                t4 = torch.full((n_cells,), t_val + dt, device=device)

                k1 = self.velocity_net(z, t1, cell_embed, section_embed, k_target)
                k2 = self.velocity_net(z + 0.5 * dt * k1, t2, cell_embed, section_embed, k_target)
                k3 = self.velocity_net(z + 0.5 * dt * k2, t3, cell_embed, section_embed, k_target)
                k4 = self.velocity_net(z + dt * k3, t4, cell_embed, section_embed, k_target)
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

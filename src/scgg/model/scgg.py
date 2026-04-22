"""
ScGG: the full model combining encoder, section encoder, flow matching,
and graph construction.

This is the top-level module that provides a clean interface for training,
inference, and graph generation.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from typing import Optional, Dict, Tuple, Literal, Union

from .encoder import GeneExpressionEncoder, SectionEncoder
from .velocity_net import VelocityNetwork
from .flow_matching import ConditionalFlowMatching
from .graph_constructor import GraphConstructor


class ScGG(nn.Module):
    """Spatial Graph Generation via Conditional Flow Matching.

    Given gene expression data, generates spatial embeddings and constructs
    kNN/radius graphs with controllable sparsity.

    Args:
        n_genes: Number of input genes.
        config: Model configuration dict (from YAML config).
    """

    def __init__(self, n_genes: int, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config["model"]

        # Gene expression encoder
        self.encoder = GeneExpressionEncoder(
            n_genes=n_genes,
            hidden_dims=model_cfg["encoder"]["hidden_dims"],
            embed_dim=model_cfg["encoder"]["embed_dim"],
            dropout=model_cfg["encoder"]["dropout"],
            norm=model_cfg["encoder"]["norm"],
        )

        # Section-level encoder
        self.section_encoder = SectionEncoder(
            cell_embed_dim=model_cfg["encoder"]["embed_dim"],
            hidden_dim=model_cfg["section_encoder"]["hidden_dim"],
            embed_dim=model_cfg["section_encoder"]["embed_dim"],
            n_attention_heads=model_cfg["section_encoder"]["n_attention_heads"],
            subsample_size=model_cfg["section_encoder"]["subsample_size"],
        )

        # Velocity network
        self.velocity_net = VelocityNetwork(
            spatial_dim=model_cfg["spatial_dim"],
            cell_embed_dim=model_cfg["encoder"]["embed_dim"],
            section_embed_dim=model_cfg["section_encoder"]["embed_dim"],
            hidden_dims=model_cfg["velocity_net"]["hidden_dims"],
            time_embed_dim=model_cfg["velocity_net"]["time_embed_dim"],
            k_embed_dim=model_cfg["velocity_net"]["k_embed_dim"],
            k_max=config["graph"].get("k_train_range", [5, 30])[1] + 10,
            dropout=model_cfg["velocity_net"]["dropout"],
        )

        # Flow matching
        self.flow = ConditionalFlowMatching(
            velocity_net=self.velocity_net,
            sigma_min=model_cfg["flow"]["sigma_min"],
        )

        # Graph constructor (not a nn.Module, no parameters)
        self.graph_constructor = GraphConstructor(
            backend=config["graph"]["backend"],
            metric=config["graph"]["metric"],
        )

    def encode(
        self,
        gene_expr: torch.Tensor,
        section_gene_expr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode gene expression into cell and section embeddings.

        Args:
            gene_expr: Gene expression for batch cells, shape (batch_size, n_genes).
            section_gene_expr: Gene expression for ALL cells in the section
                (used for section-level embedding). If None, uses gene_expr.

        Returns:
            cell_embed: Shape (batch_size, cell_embed_dim).
            section_embed: Shape (section_embed_dim,).
        """
        cell_embed = self.encoder(gene_expr)

        # Section embedding from all cells in section (or batch as fallback)
        if section_gene_expr is not None:
            with torch.no_grad():
                section_cell_embed = self.encoder(section_gene_expr)
            section_embed = self.section_encoder(section_cell_embed)
        else:
            section_embed = self.section_encoder(cell_embed.detach())

        return cell_embed, section_embed

    def training_step(
        self,
        gene_expr: torch.Tensor,
        coords: torch.Tensor,
        section_gene_expr: Optional[torch.Tensor] = None,
        k_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step.

        Args:
            gene_expr: Gene expression, shape (batch_size, n_genes).
            coords: Ground truth spatial coordinates, shape (batch_size, spatial_dim).
            section_gene_expr: Expression for all section cells (for section embedding).
            k_target: Target k values, shape (batch_size,). If None, sampled randomly.

        Returns:
            loss: Scalar loss for backpropagation.
            metrics: Dict of training metrics.
        """
        batch_size = gene_expr.shape[0]
        device = gene_expr.device

        # Encode
        cell_embed, section_embed = self.encode(gene_expr, section_gene_expr)

        # Sample k_target if not provided
        if k_target is None:
            k_range = self.config["graph"].get("k_train_range", [5, 30])
            k_target = torch.randint(
                k_range[0], k_range[1] + 1, (batch_size,), device=device
            )

        # Flow matching loss
        loss, metrics = self.flow.compute_loss(
            z_1=coords,
            cell_embed=cell_embed,
            section_embed=section_embed,
            k_target=k_target,
        )

        return loss, metrics

    @torch.no_grad()
    def generate_embeddings(
        self,
        gene_expr: torch.Tensor,
        k: int = 10,
        section_gene_expr: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        solver: Optional[str] = None,
        batch_size: int = 16384,
    ) -> torch.Tensor:
        """Generate spatial embeddings for a set of cells.

        Handles batching for large cell sets.

        Args:
            gene_expr: Gene expression, shape (n_cells, n_genes).
            k: Target k for graph construction.
            section_gene_expr: Expression for section embedding context.
            n_steps: ODE integration steps (default from config).
            solver: ODE solver (default from config).
            batch_size: Max cells per inference batch.

        Returns:
            Spatial embeddings, shape (n_cells, spatial_dim).
        """
        self.eval()
        n_cells = gene_expr.shape[0]
        device = gene_expr.device
        flow_cfg = self.config["model"]["flow"]
        n_steps = n_steps or flow_cfg["n_steps"]
        solver = solver or flow_cfg["solver"]

        # Compute section embedding once from all cells
        if section_gene_expr is not None:
            section_cell_embed = self.encoder(section_gene_expr)
        else:
            # Use subsample for large datasets
            if n_cells > 8192:
                idx = torch.randperm(n_cells, device=device)[:8192]
                section_cell_embed = self.encoder(gene_expr[idx])
            else:
                section_cell_embed = self.encoder(gene_expr)
        section_embed = self.section_encoder(section_cell_embed)

        # Generate embeddings in batches
        all_embeddings = []
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            batch_expr = gene_expr[start:end]
            batch_cell_embed = self.encoder(batch_expr)
            batch_k = torch.full(
                (end - start,), k, dtype=torch.long, device=device
            )

            batch_embeddings = self.flow.sample(
                cell_embed=batch_cell_embed,
                section_embed=section_embed,
                k_target=batch_k,
                n_steps=n_steps,
                solver=solver,
            )
            all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def generate_graph(
        self,
        gene_expr: torch.Tensor,
        k: int = 10,
        section_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        section_gene_expr: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        solver: Optional[str] = None,
        n_samples: int = 1,
    ) -> Union[sparse.csr_matrix, Tuple[sparse.csr_matrix, np.ndarray]]:
        """Generate a spatial kNN graph from gene expression.

        This is the main inference method. It generates spatial embeddings
        via flow matching and constructs a kNN graph.

        Args:
            gene_expr: Gene expression, shape (n_cells, n_genes).
            k: Number of nearest neighbors.
            section_ids: Section labels for block-diagonal construction.
            section_gene_expr: Expression for section embedding context.
            n_steps: ODE integration steps.
            solver: ODE solver.
            n_samples: Number of independent flow samples. If >1, returns
                edge confidence scores.

        Returns:
            If n_samples == 1:
                Sparse adjacency matrix, shape (n_cells, n_cells).
            If n_samples > 1:
                Tuple of (consensus adjacency, edge confidence matrix).
        """
        self.eval()

        if n_samples == 1:
            embeddings = self.generate_embeddings(
                gene_expr, k, section_gene_expr, n_steps, solver
            )
            adj = self.graph_constructor.build_knn_graph(
                embeddings, k, section_ids, symmetric=True
            )
            return adj
        else:
            # Multiple samples for uncertainty quantification
            adjs = []
            for _ in range(n_samples):
                embeddings = self.generate_embeddings(
                    gene_expr, k, section_gene_expr, n_steps, solver
                )
                adj = self.graph_constructor.build_knn_graph(
                    embeddings, k, section_ids, symmetric=True
                )
                adjs.append(adj)

            # Consensus: edge frequency across samples
            confidence = sum(adjs) / n_samples
            # Threshold at 0.5 for consensus adjacency
            consensus = confidence.copy()
            consensus.data[consensus.data < 0.5] = 0
            consensus.eliminate_zeros()
            consensus.data[:] = 1.0

            return consensus, confidence

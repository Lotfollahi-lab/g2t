"""
Visualization utilities for ScGG.

Provides plotting functions for spatial embeddings, graph comparisons,
flow trajectories, and edge confidence maps.
"""

import numpy as np
from scipy import sparse
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def plot_spatial_embedding(
    embeddings: np.ndarray,
    cell_types: Optional[np.ndarray] = None,
    title: str = "Spatial Embeddings",
    figsize: tuple = (8, 8),
    point_size: float = 0.5,
    save_path: Optional[str] = None,
):
    """Plot 2D spatial embeddings colored by cell type.

    Args:
        embeddings: Shape (n_cells, 2).
        cell_types: Integer cell type labels.
        title: Plot title.
        figsize: Figure size.
        point_size: Scatter point size.
        save_path: If provided, save figure to this path.
    """
    if not MPL_AVAILABLE:
        logger.warning("matplotlib not available for plotting")
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if cell_types is not None:
        scatter = ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=cell_types,
            cmap="tab20",
            s=point_size,
            alpha=0.6,
            rasterized=True,
        )
        plt.colorbar(scatter, ax=ax, label="Cell type")
    else:
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            s=point_size,
            alpha=0.3,
            c="steelblue",
            rasterized=True,
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


def plot_graph_comparison(
    pred_embeddings: np.ndarray,
    true_coords: np.ndarray,
    pred_adj: sparse.csr_matrix,
    true_adj: sparse.csr_matrix,
    cell_types: Optional[np.ndarray] = None,
    n_cells_plot: int = 2000,
    figsize: tuple = (16, 8),
    save_path: Optional[str] = None,
):
    """Side-by-side comparison of predicted and ground truth spatial graphs.

    Args:
        pred_embeddings: Predicted spatial embeddings, shape (n_cells, 2).
        true_coords: Ground truth coordinates, shape (n_cells, 2).
        pred_adj: Predicted adjacency.
        true_adj: Ground truth adjacency.
        cell_types: Optional cell type labels.
        n_cells_plot: Max cells to plot (subsamples for clarity).
        save_path: If provided, save figure.
    """
    if not MPL_AVAILABLE:
        return

    n_cells = pred_embeddings.shape[0]
    if n_cells > n_cells_plot:
        idx = np.random.choice(n_cells, n_cells_plot, replace=False)
        pred_emb = pred_embeddings[idx]
        true_c = true_coords[idx]
        pred_a = pred_adj[idx][:, idx]
        true_a = true_adj[idx][:, idx]
        ct = cell_types[idx] if cell_types is not None else None
    else:
        pred_emb, true_c, pred_a, true_a, ct = (
            pred_embeddings, true_coords, pred_adj, true_adj, cell_types,
        )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, coords, adj, title in [
        (axes[0], true_c, true_a, "Ground Truth"),
        (axes[1], pred_emb, pred_a, "Predicted"),
    ]:
        # Draw edges
        coo = adj.tocoo()
        for i, j in zip(coo.row, coo.col):
            if i < j:  # avoid drawing twice
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    c="lightgray",
                    linewidth=0.2,
                    alpha=0.3,
                )

        # Draw nodes
        if ct is not None:
            ax.scatter(
                coords[:, 0], coords[:, 1], c=ct, cmap="tab20",
                s=5, alpha=0.8, zorder=2,
            )
        else:
            ax.scatter(
                coords[:, 0], coords[:, 1], c="steelblue",
                s=5, alpha=0.8, zorder=2,
            )

        ax.set_title(title)
        ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


def plot_flow_trajectory(
    trajectory: np.ndarray,
    cell_types: Optional[np.ndarray] = None,
    n_cells_plot: int = 1000,
    figsize: tuple = (20, 4),
    save_path: Optional[str] = None,
):
    """Visualize the flow trajectory from noise to spatial embeddings.

    Args:
        trajectory: Shape (n_timesteps, n_cells, 2).
        cell_types: Cell type labels.
        n_cells_plot: Max cells to show.
        save_path: If provided, save figure.
    """
    if not MPL_AVAILABLE:
        return

    n_steps = trajectory.shape[0]
    n_cells = trajectory.shape[1]

    if n_cells > n_cells_plot:
        idx = np.random.choice(n_cells, n_cells_plot, replace=False)
        trajectory = trajectory[:, idx]
        if cell_types is not None:
            cell_types = cell_types[idx]

    n_show = min(n_steps, 6)
    step_indices = np.linspace(0, n_steps - 1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=figsize)

    for i, step in enumerate(step_indices):
        ax = axes[i]
        emb = trajectory[step]
        t_val = step / (n_steps - 1) if n_steps > 1 else 1.0

        if cell_types is not None:
            ax.scatter(emb[:, 0], emb[:, 1], c=cell_types, cmap="tab20",
                      s=2, alpha=0.6)
        else:
            ax.scatter(emb[:, 0], emb[:, 1], c="steelblue", s=2, alpha=0.6)

        ax.set_title(f"t = {t_val:.2f}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Flow Trajectory: Noise → Spatial Embedding", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


def plot_edge_confidence(
    embeddings: np.ndarray,
    confidence_adj: sparse.csr_matrix,
    n_cells_plot: int = 2000,
    figsize: tuple = (8, 8),
    save_path: Optional[str] = None,
):
    """Plot edges colored by confidence from multi-sample inference.

    Args:
        embeddings: Spatial embeddings, shape (n_cells, 2).
        confidence_adj: Edge confidence matrix (values in [0, 1]).
        n_cells_plot: Max cells to plot.
        save_path: If provided, save figure.
    """
    if not MPL_AVAILABLE:
        return

    n_cells = embeddings.shape[0]
    if n_cells > n_cells_plot:
        idx = np.random.choice(n_cells, n_cells_plot, replace=False)
        emb = embeddings[idx]
        conf = confidence_adj[idx][:, idx]
    else:
        emb = embeddings
        conf = confidence_adj

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw edges colored by confidence
    coo = conf.tocoo()
    cmap = plt.cm.RdYlGn

    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i < j:
            color = cmap(v)
            ax.plot(
                [emb[i, 0], emb[j, 0]],
                [emb[i, 1], emb[j, 1]],
                c=color, linewidth=0.5, alpha=0.5,
            )

    ax.scatter(emb[:, 0], emb[:, 1], c="black", s=3, zorder=2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label="Edge Confidence")

    ax.set_title("Edge Confidence Map")
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig

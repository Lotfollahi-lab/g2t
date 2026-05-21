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


# ---------------------------------------------------------------------------
# Pred vs truth comparison plot (used by inference_luna.py + inference_scgg.py)
# ---------------------------------------------------------------------------


def umeyama_align(
    src: np.ndarray, dst: np.ndarray, allow_reflection: bool = True,
) -> np.ndarray:
    """Best similarity transform (rotate + scale + translate, plus
    optional reflection) mapping ``src`` onto ``dst``. Returns the
    transformed ``src``.

    Used as a visual A/B aid before plotting predictions next to GT —
    the prediction frame may be rotated / scaled differently from GT
    coordinates even when the spatial structure is correct (the loss
    is rotation-invariant). Applying Umeyama puts both panels in the
    same frame so the eye can compare layouts directly.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    if finite.sum() < 3:
        return src.astype(np.float32)
    a = src[finite]
    b = dst[finite]
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    ac, bc = a - mu_a, b - mu_b
    var_a = (ac ** 2).sum() / a.shape[0]
    if var_a < 1e-12:
        return src.astype(np.float32)
    cov = (bc.T @ ac) / a.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    d = np.eye(cov.shape[0])
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        d[-1, -1] = -1
    R = U @ d @ Vt
    s = (S * np.diag(d)).sum() / var_a
    t = mu_b - s * R @ mu_a
    return ((s * (src @ R.T)) + t).astype(np.float32)


def _palette_for(cats: List[str], scheme: str = "glasbey") -> list:
    """Return a list of RGB(A) colors for the given categories.

    Prefers ``colorcet.glasbey`` (high-distinctness, used by LUNA's
    paper figures); falls back to matplotlib's tab20 if colorcet
    isn't installed.
    """
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib is required for _palette_for")
    n = max(len(cats), 1)
    if scheme == "glasbey":
        try:
            import colorcet as cc  # type: ignore
            return list(cc.glasbey[:n])
        except ImportError:
            logger.info(
                "colorcet not installed; falling back to tab20. "
                "Install with: pip install colorcet"
            )
    cmap = plt.get_cmap("tab20", n)
    return [cmap(i) for i in range(n)]


def plot_pred_vs_truth(
    coords_true: np.ndarray,
    coords_pred: np.ndarray,
    cell_class: Optional[np.ndarray],
    out_path,
    title_prefix: str = "",
    align_for_plot: bool = True,
    spot_size: Optional[float] = None,
    palette: str = "glasbey",
    method_label: str = "prediction",
) -> None:
    """Side-by-side scatter of ground-truth vs predicted spatial coords.

    Args:
        coords_true: (n, 2) ground-truth XY.
        coords_pred: (n, 2) predicted XY (same row order as coords_true).
        cell_class: (n,) optional categorical labels for coloring;
            None falls back to a single neutral color.
        out_path: where to write the figure (svg / png — inferred from
            the suffix).
        title_prefix: prepended to each panel's title (e.g. section name).
        align_for_plot: if True (default) apply a Umeyama similarity
            transform to ``coords_pred`` so both panels share a frame.
            Set False when you want to inspect the raw predicted frame.
        spot_size: matplotlib ``s=`` for scatter. Auto-scales from
            n_cells if omitted.
        palette: ``"glasbey"`` (colorcet, used by LUNA) or any other
            value to fall back to tab20.
        method_label: label for the prediction panel
            (``"LUNA prediction"``, ``"scgg prediction"``, etc.).

    Writes the figure with dpi=150 and closes it. No return value.
    """
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plot_pred_vs_truth")
    from pathlib import Path as _Path
    from matplotlib.patches import Patch

    out_path = _Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coords_true = np.asarray(coords_true, dtype=np.float64)
    coords_pred = np.asarray(coords_pred, dtype=np.float64)
    if coords_true.shape != coords_pred.shape:
        raise ValueError(
            f"coords_true and coords_pred shape mismatch: "
            f"{coords_true.shape} vs {coords_pred.shape}"
        )
    n = coords_true.shape[0]
    if spot_size is None:
        spot_size = max(1.0, min(20.0, 1500.0 / np.sqrt(max(n, 1))))

    # Optional similarity alignment for visual comparability.
    coords_pred_plot = (
        umeyama_align(coords_pred, coords_true, allow_reflection=True)
        if align_for_plot else coords_pred
    )
    aligned_suffix = " (aligned)" if align_for_plot else ""

    # Resolve colors: per-class palette if cell_class is given, else
    # a single muted grey.
    if cell_class is not None:
        cell_class = np.asarray(cell_class).astype(str)
        cats = sorted(set(cell_class))
        colors = _palette_for(cats, scheme=palette)
        cat_to_color = dict(zip(cats, colors))
        point_colors = [cat_to_color[c] for c in cell_class]
    else:
        cats = []
        cat_to_color = {}
        point_colors = "#666666"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, xy, title in (
        (axes[0], coords_true, f"{title_prefix}Ground truth"),
        (axes[1], coords_pred_plot, f"{title_prefix}{method_label}{aligned_suffix}"),
    ):
        ax.scatter(xy[:, 0], xy[:, 1], c=point_colors, s=spot_size, linewidths=0)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#999")

    # Shared legend at the bottom (only when we actually colored by class).
    if cats:
        n_cats = len(cats)
        ncol = min(max(1, (n_cats + 3) // 4), 6)
        patches = [
            Patch(facecolor=cat_to_color[c], label=str(c)) for c in cats
        ]
        fig.legend(
            handles=patches, loc="lower center",
            bbox_to_anchor=(0.5, 0.0), ncol=ncol,
            frameon=False, fontsize="small",
        )
        n_rows = (n_cats + ncol - 1) // ncol
        bottom = min(0.30, 0.05 + 0.04 * n_rows)
        fig.tight_layout(rect=(0, bottom, 1, 1))
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved pred-vs-truth plot: {out_path}")

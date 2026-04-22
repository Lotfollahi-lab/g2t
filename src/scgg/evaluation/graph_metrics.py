"""
Graph-level evaluation metrics.

These metrics compare predicted spatial graphs against ground truth spatial
graphs built from real coordinates. They are the primary evaluation criteria
for ScGG, distinguishing it from coordinate-based methods that only
evaluate Moran's I or coordinate MSE.
"""

import numpy as np
from scipy import sparse
from scipy.stats import wasserstein_distance
from typing import Dict, Optional, Tuple


def edge_f1(
    pred_adj: sparse.csr_matrix,
    true_adj: sparse.csr_matrix,
) -> Dict[str, float]:
    """Compute precision, recall, and F1 score of predicted edges.

    Treats edges as binary predictions. This is the most direct measure
    of spatial graph quality.

    Args:
        pred_adj: Predicted adjacency (binary, sparse).
        true_adj: Ground truth adjacency (binary, sparse).

    Returns:
        Dict with 'precision', 'recall', 'f1'.
    """
    # Binarize
    pred = pred_adj.copy()
    pred.data[:] = 1.0
    true = true_adj.copy()
    true.data[:] = 1.0

    # True positives: edges in both
    tp = pred.multiply(true).nnz

    # Predicted positives
    pp = pred.nnz

    # Actual positives
    ap = true.nnz

    precision = tp / pp if pp > 0 else 0.0
    recall = tp / ap if ap > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def neighborhood_composition_accuracy(
    pred_adj: sparse.csr_matrix,
    true_adj: sparse.csr_matrix,
    cell_types: np.ndarray,
) -> Dict[str, float]:
    """Compare cell-type composition of predicted vs. true neighborhoods.

    For each cell, computes the cell-type distribution among its neighbors
    in the predicted and true graphs, then measures the average similarity.
    This captures whether the predicted graph places cells in biologically
    correct microenvironments, even if the exact edges differ.

    Args:
        pred_adj: Predicted adjacency.
        true_adj: Ground truth adjacency.
        cell_types: Integer cell type labels, shape (n_cells,).

    Returns:
        Dict with 'composition_cosine' (avg cosine similarity of neighborhood
        type distributions) and 'composition_jaccard' (avg Jaccard of neighbor types).
    """
    n_cells = pred_adj.shape[0]
    unique_types = np.unique(cell_types)
    n_types = len(unique_types)
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    type_indices = np.array([type_to_idx[t] for t in cell_types])

    cosine_sims = []
    jaccard_sims = []

    pred_csr = pred_adj.tocsr()
    true_csr = true_adj.tocsr()

    for i in range(n_cells):
        # Predicted neighbor types
        pred_neighbors = pred_csr[i].indices
        true_neighbors = true_csr[i].indices

        if len(pred_neighbors) == 0 or len(true_neighbors) == 0:
            continue

        # Type distributions
        pred_dist = np.zeros(n_types)
        for j in pred_neighbors:
            pred_dist[type_indices[j]] += 1
        pred_dist = pred_dist / pred_dist.sum() if pred_dist.sum() > 0 else pred_dist

        true_dist = np.zeros(n_types)
        for j in true_neighbors:
            true_dist[type_indices[j]] += 1
        true_dist = true_dist / true_dist.sum() if true_dist.sum() > 0 else true_dist

        # Cosine similarity
        dot = np.dot(pred_dist, true_dist)
        norm = np.linalg.norm(pred_dist) * np.linalg.norm(true_dist)
        cosine_sims.append(dot / norm if norm > 0 else 0.0)

        # Jaccard of unique types present
        pred_types = set(type_indices[pred_neighbors])
        true_types = set(type_indices[true_neighbors])
        intersection = len(pred_types & true_types)
        union = len(pred_types | true_types)
        jaccard_sims.append(intersection / union if union > 0 else 0.0)

    return {
        "composition_cosine": np.mean(cosine_sims) if cosine_sims else 0.0,
        "composition_jaccard": np.mean(jaccard_sims) if jaccard_sims else 0.0,
    }


def degree_distribution_divergence(
    pred_adj: sparse.csr_matrix,
    true_adj: sparse.csr_matrix,
) -> Dict[str, float]:
    """Compare degree distributions of predicted and true graphs.

    Uses Wasserstein distance (earth mover's distance) between the degree
    distributions. For a well-predicted kNN graph, this should be near zero
    since both graphs should have similar degree distributions.

    Args:
        pred_adj: Predicted adjacency.
        true_adj: Ground truth adjacency.

    Returns:
        Dict with 'degree_wasserstein'.
    """
    pred_degrees = np.array(pred_adj.sum(axis=1)).flatten()
    true_degrees = np.array(true_adj.sum(axis=1)).flatten()

    wd = wasserstein_distance(pred_degrees, true_degrees)

    return {
        "degree_wasserstein": wd,
        "pred_mean_degree": pred_degrees.mean(),
        "true_mean_degree": true_degrees.mean(),
    }


def spectral_distance(
    pred_adj: sparse.csr_matrix,
    true_adj: sparse.csr_matrix,
    n_eigenvalues: int = 50,
) -> Dict[str, float]:
    """Compare spectral properties of predicted and true graphs.

    Computes the L2 distance between the top-k eigenvalues of the normalized
    Laplacians. This captures global structural similarity.

    Args:
        pred_adj: Predicted adjacency.
        true_adj: Ground truth adjacency.
        n_eigenvalues: Number of eigenvalues to compare.

    Returns:
        Dict with 'spectral_l2' distance.
    """
    from scipy.sparse.linalg import eigsh

    def _normalized_laplacian_eigenvalues(adj, k):
        n = adj.shape[0]
        k = min(k, n - 2)
        if k < 1:
            return np.array([])

        # Degree matrix
        degrees = np.array(adj.sum(axis=1)).flatten()
        degrees = np.maximum(degrees, 1e-12)
        d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))

        # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
        L_norm = sparse.eye(n) - d_inv_sqrt @ adj @ d_inv_sqrt

        try:
            eigenvalues = eigsh(L_norm, k=k, which="SM", return_eigenvectors=False)
            return np.sort(eigenvalues)
        except Exception:
            return np.zeros(k)

    pred_eigs = _normalized_laplacian_eigenvalues(pred_adj, n_eigenvalues)
    true_eigs = _normalized_laplacian_eigenvalues(true_adj, n_eigenvalues)

    # Pad to same length
    max_len = max(len(pred_eigs), len(true_eigs))
    pred_eigs = np.pad(pred_eigs, (0, max_len - len(pred_eigs)))
    true_eigs = np.pad(true_eigs, (0, max_len - len(true_eigs)))

    l2_dist = np.linalg.norm(pred_eigs - true_eigs)

    return {"spectral_l2": l2_dist}


def evaluate_graph(
    pred_adj: sparse.csr_matrix,
    true_adj: sparse.csr_matrix,
    cell_types: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Run all graph evaluation metrics.

    Args:
        pred_adj: Predicted adjacency.
        true_adj: Ground truth adjacency.
        cell_types: Optional cell type labels for composition metrics.

    Returns:
        Dict with all metric results.
    """
    results = {}

    results.update(edge_f1(pred_adj, true_adj))
    results.update(degree_distribution_divergence(pred_adj, true_adj))

    # Spectral distance is expensive; skip for very large graphs
    if pred_adj.shape[0] < 50000:
        results.update(spectral_distance(pred_adj, true_adj))

    if cell_types is not None:
        results.update(
            neighborhood_composition_accuracy(pred_adj, true_adj, cell_types)
        )

    return results

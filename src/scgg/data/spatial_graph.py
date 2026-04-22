"""
Ground truth spatial graph construction from spatial coordinates.

Builds kNN or radius graphs from ground truth spatial coordinates,
used as training targets and evaluation references.
"""

import numpy as np
from scipy import sparse
from typing import Optional, Union

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from sklearn.neighbors import NearestNeighbors


def build_ground_truth_graph(
    coords: np.ndarray,
    k: int = 10,
    section_ids: Optional[np.ndarray] = None,
    graph_type: str = "knn",
    radius: Optional[float] = None,
    symmetric: bool = True,
) -> sparse.csr_matrix:
    """Build ground truth spatial graph from coordinates.

    Args:
        coords: Spatial coordinates, shape (n_cells, 2) or (n_cells, 3).
        k: Number of neighbors for kNN graph.
        section_ids: Section labels for per-section graph construction.
        graph_type: 'knn' or 'radius'.
        radius: Connection radius (required if graph_type='radius').
        symmetric: Whether to symmetrize the graph.

    Returns:
        Sparse adjacency matrix, shape (n_cells, n_cells).
    """
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    n_cells = coords.shape[0]

    if section_ids is None:
        if graph_type == "knn":
            adj = _knn_graph(coords, k)
        else:
            assert radius is not None, "radius required for radius graph"
            adj = _radius_graph(coords, radius)
    else:
        unique_sections = np.unique(section_ids)
        rows, cols, vals = [], [], []

        for sec_id in unique_sections:
            mask = section_ids == sec_id
            sec_coords = coords[mask]
            sec_idx = np.where(mask)[0]

            k_sec = min(k, len(sec_coords) - 1)
            if k_sec < 1:
                continue

            if graph_type == "knn":
                sec_adj = _knn_graph(sec_coords, k_sec)
            else:
                sec_adj = _radius_graph(sec_coords, radius)

            sec_coo = sec_adj.tocoo()
            for i, j, v in zip(sec_coo.row, sec_coo.col, sec_coo.data):
                rows.append(sec_idx[i])
                cols.append(sec_idx[j])
                vals.append(v)

        adj = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_cells, n_cells)
        )

    if symmetric:
        adj = adj + adj.T
        adj.data[:] = 1.0

    return adj


def _knn_graph(coords: np.ndarray, k: int) -> sparse.csr_matrix:
    """Build kNN graph using FAISS or sklearn."""
    n = coords.shape[0]
    d = coords.shape[1]

    if FAISS_AVAILABLE and n > 1000:
        index = faiss.IndexFlatL2(d)
        index.add(coords)
        _, indices = index.search(coords, k + 1)
        indices = indices[:, 1:]  # remove self
    else:
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        indices = indices[:, 1:]

    rows = np.repeat(np.arange(n), k)
    cols = indices.flatten()
    vals = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def _radius_graph(coords: np.ndarray, radius: float) -> sparse.csr_matrix:
    """Build radius graph using sklearn BallTree."""
    from sklearn.neighbors import BallTree

    n = coords.shape[0]
    tree = BallTree(coords, metric="euclidean")
    indices_list = tree.query_radius(coords, r=radius)

    rows, cols = [], []
    for i, neighbors in enumerate(indices_list):
        for j in neighbors:
            if j != i:
                rows.append(i)
                cols.append(j)

    vals = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))

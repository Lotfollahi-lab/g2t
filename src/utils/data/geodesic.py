"""Geodesic (intrinsic-manifold) distances on a 2D point cloud — the
target metric for the gauge-invariant variant of the sparse-local-distance
loss (``model.loss.sparse_local_distance.landmark_metric=geodesic``).

Motivation
----------
The sparse loss's STRUCTURED global term supervises each cell's predicted
embedding distance to a set of spread-out landmark cells. By default the
TRUE target is the straight-line (Euclidean) distance to each landmark.
Euclidean distance "cuts across" empty space / concavities, so on a
non-convex slice (a folded cortex, a ventricle, a C-shaped section) it
mixes two manifold-distant regions that merely happen to be close in the
plane — and that Euclidean shortcut is morphology-specific, so it doesn't
transfer across slices with different shapes.

The GEODESIC distance — shortest path along the cell-to-cell kNN graph —
respects the tissue's intrinsic manifold: it follows the sheet of cells
rather than jumping across gaps. It is the classic Isomap target, and as
a relative/intrinsic quantity it is exactly the gauge-invariant structure
we argue transfers across slices (it is invariant to the per-slice
similarity gauge up to the global scale the loss already handles).

This module is numpy + scipy only (no torch) so the numeric core is unit
testable off-cluster; the loss imports it lazily and feeds it detached
true positions (the target path carries no gradient).
"""
from __future__ import annotations

import numpy as np


def geodesic_landmark_distances(
    pos: np.ndarray,
    land_idx: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    """Geodesic distance from every point to each landmark.

    Args:
        pos: (n, d) point coordinates (d typically 2).
        land_idx: (M,) integer indices into ``pos`` marking the landmarks.
        k: number of nearest neighbours used to build the manifold graph
            (edge weights = Euclidean distance between adjacent points).

    Returns:
        (n, M) float64 matrix of geodesic (shortest-path) distances along
        the symmetric kNN graph. Pairs in disconnected components (no path
        to the landmark) are capped at the maximum *finite* geodesic
        distance in the matrix, so the result is always finite and
        monotone-sensible (a degenerate "all-disconnected" matrix falls
        back to a flat 1.0).

    The graph is built once on the true positions; the loss caches the
    result per slice (true positions are fixed across epochs).
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    pos = np.asarray(pos, dtype=np.float64)
    n = pos.shape[0]
    land_idx = np.asarray(land_idx, dtype=np.int64).reshape(-1)
    M = int(land_idx.size)
    if n == 0 or M == 0:
        return np.zeros((n, M), dtype=np.float64)
    if n == 1:
        return np.zeros((1, M), dtype=np.float64)

    keff = min(int(k) + 1, n)  # +1: query returns self at column 0
    tree = cKDTree(pos)
    dist, idx = tree.query(pos, k=keff)
    # query returns (n,) when keff==1; normalise to 2-D
    dist = np.atleast_2d(dist.reshape(n, -1))
    idx = np.atleast_2d(idx.reshape(n, -1))

    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.reshape(-1)
    w = dist.reshape(-1)
    keep = rows != cols  # drop self-loops (0-weight, harmless but tidy)
    rows, cols, w = rows[keep], cols[keep], w[keep]
    # A directed weighted graph, then symmetrise. Euclidean weights are
    # symmetric, so .maximum(A.T) just makes every observed edge undirected
    # without changing any weight.
    A = csr_matrix((w, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)

    # Multi-source Dijkstra from the landmark nodes -> (M, n); transpose.
    D = dijkstra(A, directed=False, indices=land_idx)  # (M, n)
    D = np.asarray(D, dtype=np.float64).T              # (n, M)

    finite = np.isfinite(D)
    if not finite.all():
        cap = float(D[finite].max()) if finite.any() else 1.0
        D = np.where(finite, D, cap)
    return D

"""
Scalable graph construction using FAISS.

Builds kNN or radius graphs from spatial embeddings, optionally respecting
section boundaries (no cross-section edges). This is the key component that
converts continuous spatial embeddings into discrete graphs with controllable
sparsity.

Design rationale:
- FAISS provides O(n log n) approximate kNN for millions of points.
- Section-aware construction ensures no edges cross section boundaries
  (block-diagonal adjacency).
- The kNN parameter k is specified at construction time, enabling controllable
  graph generation without retraining.
"""

import numpy as np
import torch
from scipy import sparse
from typing import Optional, Tuple, Union

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from sklearn.neighbors import NearestNeighbors, BallTree


class GraphConstructor:
    """Builds spatial graphs from embeddings using kNN or radius search.

    Supports FAISS (GPU/CPU) for scalability and falls back to sklearn
    for smaller datasets or when FAISS is unavailable.

    Args:
        backend: 'faiss' or 'sklearn'.
        metric: Distance metric ('euclidean' or 'cosine').
        use_gpu: Whether to use GPU for FAISS.
    """

    def __init__(
        self,
        backend: str = "faiss",
        metric: str = "euclidean",
        use_gpu: bool = False,
    ):
        if backend == "faiss" and not FAISS_AVAILABLE:
            print("Warning: FAISS not available, falling back to sklearn.")
            backend = "sklearn"
        self.backend = backend
        self.metric = metric
        self.use_gpu = use_gpu

    def build_knn_graph(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        k: int,
        section_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        symmetric: bool = True,
    ) -> sparse.csr_matrix:
        """Build a kNN graph from embeddings.

        Args:
            embeddings: Node embeddings, shape (n_nodes, dim).
            k: Number of nearest neighbors per node.
            section_ids: Section labels, shape (n_nodes,). If provided,
                only connects nodes within the same section.
            symmetric: If True, symmetrize the adjacency (union of directed edges).

        Returns:
            Sparse adjacency matrix, shape (n_nodes, n_nodes).
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        if isinstance(section_ids, torch.Tensor):
            section_ids = section_ids.cpu().numpy()

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        n_nodes = embeddings.shape[0]

        if section_ids is None:
            # Single section: build one graph
            distances, indices = self._knn_search(embeddings, k)
            adj = self._build_adjacency(n_nodes, distances, indices, k)
        else:
            # Multi-section: build per-section graphs, combine as block diagonal
            unique_sections = np.unique(section_ids)
            rows, cols, vals = [], [], []

            for sec_id in unique_sections:
                mask = section_ids == sec_id
                sec_emb = embeddings[mask]
                sec_indices_global = np.where(mask)[0]

                if len(sec_emb) <= k:
                    # Section too small, connect all pairs
                    k_sec = len(sec_emb) - 1
                else:
                    k_sec = k

                if k_sec < 1:
                    continue

                distances, indices = self._knn_search(sec_emb, k_sec)

                for i in range(len(sec_emb)):
                    for j_idx in range(k_sec):
                        j_local = indices[i, j_idx]
                        if j_local == i:
                            continue
                        rows.append(sec_indices_global[i])
                        cols.append(sec_indices_global[j_local])
                        vals.append(1.0)

            adj = sparse.csr_matrix(
                (vals, (rows, cols)), shape=(n_nodes, n_nodes)
            )

        if symmetric:
            adj = adj + adj.T
            adj.data[:] = 1.0  # binarize

        return adj

    def build_radius_graph(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        radius: float,
        section_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        max_neighbors: int = 100,
    ) -> sparse.csr_matrix:
        """Build a radius graph from embeddings.

        Args:
            embeddings: Node embeddings, shape (n_nodes, dim).
            radius: Connection radius.
            section_ids: Section labels for block-diagonal structure.
            max_neighbors: Maximum neighbors per node (for memory safety).

        Returns:
            Sparse adjacency matrix, shape (n_nodes, n_nodes).
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        if isinstance(section_ids, torch.Tensor):
            section_ids = section_ids.cpu().numpy()

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        n_nodes = embeddings.shape[0]

        if section_ids is None:
            tree = BallTree(embeddings, metric="euclidean")
            indices_list = tree.query_radius(embeddings, r=radius)
            rows, cols = [], []
            for i, neighbors in enumerate(indices_list):
                for j in neighbors[:max_neighbors]:
                    if j != i:
                        rows.append(i)
                        cols.append(j)
            adj = sparse.csr_matrix(
                (np.ones(len(rows)), (rows, cols)),
                shape=(n_nodes, n_nodes),
            )
        else:
            unique_sections = np.unique(section_ids)
            rows, cols = [], []
            for sec_id in unique_sections:
                mask = section_ids == sec_id
                sec_emb = embeddings[mask]
                sec_idx = np.where(mask)[0]

                tree = BallTree(sec_emb, metric="euclidean")
                indices_list = tree.query_radius(sec_emb, r=radius)
                for i, neighbors in enumerate(indices_list):
                    for j in neighbors[:max_neighbors]:
                        if j != i:
                            rows.append(sec_idx[i])
                            cols.append(sec_idx[j])

            adj = sparse.csr_matrix(
                (np.ones(len(rows)), (rows, cols)),
                shape=(n_nodes, n_nodes),
            )

        return adj

    def _knn_search(
        self, embeddings: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform kNN search.

        Returns:
            distances: Shape (n, k).
            indices: Shape (n, k).
        """
        n, d = embeddings.shape

        if self.backend == "faiss":
            # Build FAISS index
            if self.metric == "cosine":
                faiss.normalize_L2(embeddings)

            # Use IVF for large datasets, flat for small
            if n > 50000:
                nlist = min(int(np.sqrt(n)), 4096)
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist)
                index.train(embeddings)
                index.nprobe = min(nlist, 64)
            else:
                index = faiss.IndexFlatL2(d)

            if self.use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)

            index.add(embeddings)
            # Search for k+1 because the first neighbor is the point itself
            distances, indices = index.search(embeddings, k + 1)
            # Remove self-connections
            distances = distances[:, 1:]
            indices = indices[:, 1:]

        else:
            nn_model = NearestNeighbors(
                n_neighbors=k + 1, metric=self.metric, algorithm="auto"
            )
            nn_model.fit(embeddings)
            distances, indices = nn_model.kneighbors(embeddings)
            distances = distances[:, 1:]
            indices = indices[:, 1:]

        return distances, indices

    def _build_adjacency(
        self,
        n_nodes: int,
        distances: np.ndarray,
        indices: np.ndarray,
        k: int,
    ) -> sparse.csr_matrix:
        """Build sparse adjacency from kNN results."""
        rows = np.repeat(np.arange(n_nodes), k)
        cols = indices.flatten()
        vals = np.ones(len(rows), dtype=np.float32)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))

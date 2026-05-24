"""Coarse-to-fine factorisation of the cell-position prediction.

Toggleable via ``model.coarse_to_fine.enabled``. See the config block
in ``scgg/src/configs/model/default.yaml`` for the motivation.

Three pieces glued together:

1. ``GeneClusterModule`` — k-means cluster cells by gene expression
   per forward pass. Returns cluster assignments (B, N) and
   per-cluster aggregated features (B, K, H).

2. ``CoarseRegressor`` — small transformer over K cluster tokens
   that predicts each cluster's spatial centroid as a 2-D point.

3. ``CoarseToFineWrapper`` — orchestrates clustering, runs the
   coarse regressor, conditions the inner backbone on cluster
   centroids, calls the inner backbone for cell-level prediction.

Training uses TRUE cluster centroids (teacher forcing) for the
cell-level conditioning. Inference uses PREDICTED centroids.
Both centroids and cluster IDs are stashed on the returned
``DataHolder`` so the LossFunction can compute the auxiliary
coarse-centroid MSE.

Combines with the hierarchical wrapper as a separate axis of
multi-scale processing — hierarchical bins by current x_t spatial
positions; coarse-to-fine clusters by gene expression (fixed per
slice).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data.dataholder import DataHolder


# ---------------------------------------------------------------------------
# Gene-clustering module
# ---------------------------------------------------------------------------


class GeneClusterModule(nn.Module):
    """Cluster cells by gene expression + per-cluster feature aggregation.

    Two clustering modes are supported via ``cluster_mode``:

    * ``"kmeans"`` (default, backward-compatible). Hard k-means on
      the projected gene embeddings, recomputed per forward.
      Non-differentiable; gradients flow into ``gene_proj`` only
      through the per-cluster aggregation, not the cluster
      assignment itself.

    * ``"gumbel"``. A small learnable ``cluster_head`` predicts soft
      logits per cell over the K clusters. Gumbel-softmax with
      ``hard=True`` produces a one-hot assignment in the forward
      pass (matches k-means semantics for downstream computation),
      and the straight-through estimator routes gradients through
      the soft weights back into the cluster head + ``gene_proj``.
      Cluster boundaries can then co-adapt with the downstream
      spatial-prediction loss.

    Both modes return the SAME (cluster_ids, cluster_features) tuple
    so the wrapper code is unchanged:

      * ``cluster_ids``: ``(B, N)`` long, ``[0, K)`` for real cells,
        ``-1`` for masked.
      * ``cluster_features``: ``(B, K, proj_dim)`` mean per-cluster
        embedding, gradient-bearing.
    """

    def __init__(
        self,
        gene_input_dim: int,
        proj_dim: int,
        n_clusters: int,
        kmeans_n_iters: int = 10,
        cluster_mode: str = "kmeans",
        gumbel_tau: float = 1.0,
    ):
        super().__init__()
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be ≥ 2; got {n_clusters}.")
        cluster_mode = str(cluster_mode).lower()
        if cluster_mode not in ("kmeans", "gumbel"):
            raise ValueError(
                f"cluster_mode must be 'kmeans' or 'gumbel'; got {cluster_mode!r}"
            )
        self.n_clusters = int(n_clusters)
        self.kmeans_n_iters = int(kmeans_n_iters)
        self.cluster_mode = cluster_mode
        self.gumbel_tau = float(gumbel_tau)

        # Light projection of raw gene expression for clustering AND
        # downstream coarse-stage features. Kept small (default 128)
        # because k-means cost is O(N · K · proj_dim) per iter and we
        # run this every forward pass.
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_input_dim, proj_dim),
            nn.SiLU(),
        )

        # Learned cluster head — only used when cluster_mode='gumbel'.
        # We allocate it unconditionally so the parameter set is stable
        # across runs that switch modes, but it stays untouched (zero
        # gradient) in kmeans mode. Tiny cost — proj_dim × K ≈ 128 × 16
        # = 2k params.
        self.cluster_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, n_clusters),
        )

    @torch.no_grad()
    def _kmeans_per_slice(
        self, emb: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """K-means on a single slice's embeddings.

        Args:
            emb: (n, H) — projected gene embeddings (real cells only).
            mask: unused (already filtered to real cells); kept for
                interface symmetry.

        Returns:
            cluster_ids: (n,) long — assignment in [0, K).
        """
        n, H = emb.shape
        K = self.n_clusters
        if n == 0:
            return torch.zeros(0, dtype=torch.long, device=emb.device)

        # Initialise centroids: K random cells. Use a per-call seed
        # so cluster assignments are stable across forward passes on
        # the same input (important for inference reproducibility).
        # We seed off a content hash of the embedding tensor.
        with torch.no_grad():
            content_seed = int(
                (emb.abs().sum().item() * 1e6) % (2**31)
            )
        g = torch.Generator(device="cpu").manual_seed(content_seed)
        init_idx = torch.randperm(n, generator=g)[: min(K, n)].to(emb.device)
        centroids = emb[init_idx].clone()                  # (K_eff, H)

        # Pad centroids up to K with random duplicates if n < K
        # (degenerate case for very small slices).
        if centroids.shape[0] < K:
            extra = K - centroids.shape[0]
            centroids = torch.cat(
                [centroids, centroids[torch.randint(0, centroids.shape[0], (extra,), generator=g).to(emb.device)]],
                dim=0,
            )

        # Iterate Lloyd's algorithm.
        for _ in range(self.kmeans_n_iters):
            # Pairwise sq-distances cell→centroid: (n, K)
            d = torch.cdist(emb, centroids, p=2)
            cluster_ids = d.argmin(dim=1)                  # (n,)
            # Recompute centroids = mean of assigned cells.
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(K, device=emb.device)
            new_centroids.scatter_add_(
                0,
                cluster_ids.unsqueeze(-1).expand(-1, H),
                emb,
            )
            counts.scatter_add_(0, cluster_ids, torch.ones_like(cluster_ids, dtype=emb.dtype))
            # Empty clusters: retain previous centroid (avoid div by 0).
            empty = counts == 0
            counts_safe = counts.clamp(min=1.0).unsqueeze(-1)
            new_centroids = new_centroids / counts_safe
            new_centroids[empty] = centroids[empty]
            centroids = new_centroids

        # Final assignment with the converged centroids.
        d = torch.cdist(emb, centroids, p=2)
        cluster_ids = d.argmin(dim=1)
        return cluster_ids

    def forward(
        self, node_features: torch.Tensor, node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: (B, N, gene_input_dim) — raw cell features.
            node_mask: (B, N) bool.

        Returns:
            cluster_ids: (B, N) long — assignment in [0, K), -1 for
                masked cells.
            cluster_features: (B, K, proj_dim) — mean-aggregated
                cell embeddings per cluster (gradient-bearing).
        """
        B, N = node_features.shape[:2]
        K = self.n_clusters
        device = node_features.device

        cell_emb = self.gene_proj(node_features)                    # (B, N, H)
        H = cell_emb.shape[-1]

        if self.cluster_mode == "gumbel":
            cluster_ids, cluster_features = self._cluster_gumbel(
                cell_emb, node_mask, B, N, K, H,
            )
        else:  # kmeans
            cluster_ids, cluster_features = self._cluster_kmeans(
                cell_emb, node_mask, B, N, K, H, device,
            )

        return cluster_ids, cluster_features

    # ------------------------------------------------------------------
    # Hard k-means (default; backward-compatible)
    # ------------------------------------------------------------------
    def _cluster_kmeans(
        self, cell_emb, node_mask, B, N, K, H, device,
    ):
        # Per-slice k-means; loop over B is fine for typical B≤8.
        cluster_ids = torch.full((B, N), -1, dtype=torch.long, device=device)
        for b in range(B):
            mask_b = node_mask[b]
            n_real = int(mask_b.sum().item())
            if n_real == 0:
                continue
            emb_b = cell_emb[b][mask_b].detach()                    # detach OK — clustering is no-grad
            ids_b = self._kmeans_per_slice(emb_b, mask_b[mask_b])
            cluster_ids[b][mask_b] = ids_b

        # Per-cluster mean of cell embeddings (gradient-bearing on
        # cell_emb). Same scatter-mean pattern as the hierarchical
        # wrapper.
        safe_ids = cluster_ids.clamp(min=0)
        valid = node_mask & (cluster_ids >= 0)

        batch_idx = torch.arange(B, device=cell_emb.device).unsqueeze(1).expand(B, N)
        combined_id = (batch_idx * K + safe_ids).reshape(-1)
        valid_f = valid.float().reshape(-1, 1)

        emb_flat = cell_emb.reshape(B * N, H) * valid_f
        count_flat = valid.float().reshape(B * N)

        sum_emb = cell_emb.new_zeros(B * K, H)
        sum_cnt = cell_emb.new_zeros(B * K)
        sum_emb.scatter_add_(0, combined_id.unsqueeze(-1).expand(-1, H), emb_flat)
        sum_cnt.scatter_add_(0, combined_id, count_flat)

        cnt_safe = sum_cnt.clamp(min=1.0)
        cluster_features = (sum_emb / cnt_safe.unsqueeze(-1)).reshape(B, K, H)

        return cluster_ids, cluster_features

    # ------------------------------------------------------------------
    # Gumbel-softmax (learnable, differentiable)
    # ------------------------------------------------------------------
    def _cluster_gumbel(
        self, cell_emb, node_mask, B, N, K, H,
    ):
        """Soft cluster assignment with straight-through Gumbel-softmax.

        Forward pass: ``F.gumbel_softmax(..., hard=True)`` returns a
        one-hot ``(B, N, K)`` tensor. The "hard" output makes the
        cluster_features computation behave exactly like a hard-
        clustering scatter-mean (each cell contributes its full weight
        to exactly ONE cluster), so the downstream coarse_regressor
        sees the same kind of input as in k-means mode.

        Backward pass: the straight-through estimator routes gradients
        through the soft probabilities, so the cluster head and
        ``gene_proj`` learn to choose cluster boundaries that minimise
        the downstream spatial-prediction loss. This is the whole
        point — cluster boundaries that are right for THE TASK rather
        than for unsupervised gene-space K-means.

        Inference: deterministic argmax (no Gumbel noise) for
        reproducibility. We still build a one-hot to keep the
        aggregation code path identical.
        """
        # (B, N, K) logits — pure function of the projected gene embeddings.
        logits = self.cluster_head(cell_emb)

        if self.training:
            # ``hard=True`` straight-through: forward is one-hot,
            # backward uses soft weights.
            weights = F.gumbel_softmax(
                logits, tau=self.gumbel_tau, hard=True, dim=-1,
            )                                                       # (B, N, K)
        else:
            # Deterministic argmax for inference. We still build a
            # one-hot tensor so the aggregation einsum below is
            # identical to the training path.
            argmax_idx = logits.argmax(dim=-1)                      # (B, N)
            weights = F.one_hot(argmax_idx, num_classes=K).to(cell_emb.dtype)

        # Zero out padding cells so they don't contribute to any
        # cluster. This is the right thing to do regardless of mode —
        # padded cells have no real gene content.
        weights = weights * node_mask.unsqueeze(-1).to(cell_emb.dtype)

        # Aggregate per cluster: cluster_features[b, k] = sum_n
        # cell_emb[b, n] · weights[b, n, k] / sum_n weights[b, n, k].
        # einsum here is differentiable end-to-end.
        sum_emb = torch.einsum("bnk,bnh->bkh", weights, cell_emb)    # (B, K, H)
        counts = weights.sum(dim=1)                                  # (B, K)
        # Cluster might be empty (all cells assigned to other clusters);
        # clamp avoids div-by-zero. The cluster's downstream centroid
        # prediction will then just inherit whatever the cluster head
        # was producing — fine because the loss has nothing to compare
        # against (the true centroid for that cluster has count=0 too).
        counts_safe = counts.clamp(min=1e-6).unsqueeze(-1)
        cluster_features = sum_emb / counts_safe

        # Hard cluster_ids (argmax, gradient-free) for downstream
        # uses that need an integer per cell: true-centroid scatter
        # in CoarseToFineWrapper._compute_true_centroids, and the
        # broadcast in _broadcast_per_cluster. Padded cells get -1.
        with torch.no_grad():
            cluster_ids = logits.argmax(dim=-1)
            cluster_ids = torch.where(
                node_mask, cluster_ids,
                torch.full_like(cluster_ids, -1),
            )

        return cluster_ids, cluster_features


# ---------------------------------------------------------------------------
# Coarse regressor — predicts cluster centroids
# ---------------------------------------------------------------------------


class CoarseRegressor(nn.Module):
    """Small transformer over K cluster tokens predicting each
    cluster's spatial centroid as a 2-D point."""

    def __init__(
        self,
        cluster_feature_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(cluster_feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )
        self.centroid_head = nn.Linear(hidden_dim, 2)

    def forward(self, cluster_features: torch.Tensor) -> torch.Tensor:
        """Backward-compatible wrapper: returns only centroids.

        Prefer ``forward_with_embeddings`` for callers that also want
        the rich per-cluster hidden state as conditioning input —
        keeping that 128-D signal (rather than throwing it away after
        projecting to the 2-D centroid) is a substantial improvement
        in the c2f wrapper's effective conditioning capacity.
        """
        centroids, _ = self.forward_with_embeddings(cluster_features)
        return centroids

    def forward_with_embeddings(
        self, cluster_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cluster_features: (B, K, cluster_feature_dim)

        Returns:
            predicted_centroids:  (B, K, 2) — through the centroid head.
            cluster_embeddings:   (B, K, hidden_dim) — pre-head
                transformer output. Encodes the same info the
                centroid prediction depends on plus richer context
                that the centroid projection compresses away. Used
                by the c2f wrappers as a per-cell conditioning
                signal (broadcast to cells via cluster_ids).
        """
        h = self.input_proj(cluster_features)                       # (B, K, hidden)
        h = self.transformer(h)                                     # (B, K, hidden)
        centroids = self.centroid_head(h)                           # (B, K, 2)
        return centroids, h


# ---------------------------------------------------------------------------
# Coarse-to-fine wrapper
# ---------------------------------------------------------------------------


class CoarseToFineWrapper(nn.Module):
    """Wraps the LUNA Model with a coarse-stage cluster-centroid
    regression. Drop-in for ``models.model.Model``: same DataHolder-in
    / DataHolder-out interface so the training loop / sampling loop /
    loss function don't need to know coarse-to-fine is on.

    Per forward:
      1. Cluster cells by gene expression (k-means).
      2. Aggregate per-cluster gene features.
      3. CoarseRegressor predicts cluster centroids (B, K, 2).
      4. Broadcast each cell's cluster centroid as a 2-D feature.
      5. Conditioning vector (true at training time, predicted at
         inference time) is concatenated to ``data.node_features``.
      6. Inner Model forward on the augmented input.
      7. Stash predicted_centroids + cluster_ids on the returned
         DataHolder so the LossFunction can read them for the
         auxiliary coarse-centroid MSE term.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        n_layers: int,
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        output_dims: Dict[str, int],
        c2f_cfg,
    ):
        super().__init__()

        def _g(k, default):
            return (
                c2f_cfg.get(k, default)
                if hasattr(c2f_cfg, "get")
                else getattr(c2f_cfg, k, default)
            )

        self.n_clusters = int(_g("n_clusters", 16))
        self.kmeans_n_iters = int(_g("kmeans_n_iters", 10))
        self.coarse_hidden_dim = int(_g("coarse_hidden_dim", 128))
        self.coarse_n_layers = int(_g("coarse_n_layers", 2))
        self.coarse_n_heads = int(_g("coarse_n_heads", 4))
        self.gene_proj_dim = int(_g("gene_proj_dim", 128))
        self.cluster_mode = str(_g("cluster_mode", "kmeans")).lower()
        self.gumbel_tau = float(_g("gumbel_tau", 1.0))

        gene_in = int(input_dims["node_features_dimensions"])

        self.cluster_module = GeneClusterModule(
            gene_input_dim=gene_in,
            proj_dim=self.gene_proj_dim,
            n_clusters=self.n_clusters,
            kmeans_n_iters=self.kmeans_n_iters,
            cluster_mode=self.cluster_mode,
            gumbel_tau=self.gumbel_tau,
        )

        self.coarse_regressor = CoarseRegressor(
            cluster_feature_dim=self.gene_proj_dim,
            hidden_dim=self.coarse_hidden_dim,
            n_layers=self.coarse_n_layers,
            n_heads=self.coarse_n_heads,
        )

        # Inner Model takes augmented input: original gene features
        # + 2-d cluster-centroid + ``coarse_hidden_dim``-d cluster
        # embedding (the pre-centroid-head hidden state of the
        # coarse transformer, broadcast to each cell of the cluster).
        # The richer embedding signal is the main improvement over a
        # bare 2-D centroid: at gene_in≈2000, raw centroid is only
        # ~0.1% of input width and gets drowned out, but the 128-D
        # embedding is ~6% — substantial enough that the inner
        # encoder can't ignore it.
        self.cond_dim = 2 + self.coarse_hidden_dim
        augmented_input_dims = dict(input_dims)
        augmented_input_dims["node_features_dimensions"] = gene_in + self.cond_dim

        from models.model import Model
        self.inner = Model(
            input_dims=augmented_input_dims,
            n_layers=n_layers,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_true_centroids(
        positions: torch.Tensor,
        cluster_ids: torch.Tensor,
        node_mask: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Compute true cluster centroids (mean of TRUE cell positions
        within each cluster). Returns (B, K, 2).

        Used at training time for teacher forcing — we pass these
        as conditioning to the inner backbone instead of the
        possibly-wrong predicted centroids, so error doesn't
        compound while the coarse regressor is still learning.
        """
        B, N = positions.shape[:2]
        device = positions.device
        safe_ids = cluster_ids.clamp(min=0)
        valid = node_mask & (cluster_ids >= 0)

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        combined_id = (batch_idx * K + safe_ids).reshape(-1)
        valid_f = valid.float().reshape(-1, 1)

        pos_flat = positions.reshape(B * N, 2) * valid_f
        count_flat = valid.float().reshape(B * N)

        sum_pos = positions.new_zeros(B * K, 2)
        sum_cnt = positions.new_zeros(B * K)
        sum_pos.scatter_add_(0, combined_id.unsqueeze(-1).expand(-1, 2), pos_flat)
        sum_cnt.scatter_add_(0, combined_id, count_flat)

        cnt_safe = sum_cnt.clamp(min=1.0)
        centroids = (sum_pos / cnt_safe.unsqueeze(-1)).reshape(B, K, 2)
        return centroids

    @staticmethod
    def _broadcast_centroids(
        centroids: torch.Tensor,
        cluster_ids: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Each cell gets its cluster's centroid as a 2-d feature.
        Masked cells get a zero vector. Thin wrapper around
        ``_broadcast_per_cluster`` for backward compatibility (the
        2-D-specific helper that used hardcoded ``expand(B, N, 2)``).
        """
        return CoarseToFineWrapper._broadcast_per_cluster(
            centroids, cluster_ids, node_mask,
        )

    @staticmethod
    def _broadcast_per_cluster(
        features: torch.Tensor,
        cluster_ids: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generic per-cluster → per-cell broadcast. Takes a
        ``(B, K, F)`` tensor and returns ``(B, N, F)`` where each
        cell receives its cluster's feature vector. Masked cells
        get zeros. Used for BOTH the 2-D centroid AND the
        ``coarse_hidden_dim``-D cluster-embedding broadcasts.
        """
        B, N = cluster_ids.shape
        F = features.shape[-1]
        safe_ids = cluster_ids.clamp(min=0).unsqueeze(-1).expand(B, N, F)
        broadcast = torch.gather(features, 1, safe_ids)             # (B, N, F)
        broadcast = broadcast * node_mask.unsqueeze(-1).float()
        return broadcast

    # ------------------------------------------------------------------
    def forward(
        self,
        data: DataHolder,
        true_positions: Optional[torch.Tensor] = None,
    ) -> DataHolder:
        """Args:
            data: standard DataHolder with possibly-noisy ``positions``.
            true_positions: optional (B, N, 2) — TRUE positions used
                for teacher forcing at training time. The
                ``training_step_func`` wrapper is responsible for
                passing these in; we fall back to the predicted
                centroids if not provided (i.e. at inference time).

        Returns:
            DataHolder from the inner backbone, with two extra
            attributes stashed on it:
              * ``_predicted_cluster_centroids`` : (B, K, 2)
              * ``_cluster_ids``                  : (B, N) long, -1 for masked

            Read by the LossFunction's ``_compute_coarse_centroid_mse``
            component.
        """
        # 1+2. Cluster + aggregate.
        cluster_ids, cluster_features = self.cluster_module(
            data.node_features, data.node_mask,
        )

        # 3. Coarse regressor → predicted cluster centroids AND the
        # 128-D pre-head embeddings (the transformer hidden states
        # that produced the centroids). We use BOTH as conditioning:
        # the centroid for the auxiliary loss + a thin spatial
        # anchor signal, the embeddings for rich per-cluster
        # identity context that the gene encoder can't drown out.
        predicted_centroids, cluster_embeddings = (
            self.coarse_regressor.forward_with_embeddings(cluster_features)
        )                                                              # (B, K, 2), (B, K, H)

        # 4+5. Centroid conditioning: teacher-forced (true) at
        # training time, predicted at inference. Cluster embeddings
        # are ALWAYS the regressor's output (no ground truth exists
        # for them — they're a learned representation).
        if self.training and true_positions is not None:
            cond_centroids = self._compute_true_centroids(
                true_positions, cluster_ids, data.node_mask, self.n_clusters,
            )
        else:
            cond_centroids = predicted_centroids

        cell_centroids = self._broadcast_per_cluster(
            cond_centroids, cluster_ids, data.node_mask,
        )                                                              # (B, N, 2)
        cell_cluster_emb = self._broadcast_per_cluster(
            cluster_embeddings, cluster_ids, data.node_mask,
        )                                                              # (B, N, H)

        # 6. Concatenate genes + centroid + cluster_embedding and
        # call the inner backbone. The inner Model was built with
        # input_dims bumped by ``self.cond_dim`` = 2 + H.
        data_aug = data.copy()
        data_aug.node_features = torch.cat(
            [data.node_features, cell_centroids, cell_cluster_emb], dim=-1,
        )
        out = self.inner(data_aug)

        # 7. Stash coarse outputs on the returned DataHolder so the
        # loss function can find them.
        out._predicted_cluster_centroids = predicted_centroids
        out._cluster_ids = cluster_ids
        out._n_clusters = self.n_clusters
        return out


# ---------------------------------------------------------------------------
# Combined hierarchical + coarse-to-fine wrapper
# ---------------------------------------------------------------------------


class HierarchicalCoarseToFineWrapper(nn.Module):
    """Composes spatial-hierarchical and gene-coarse-to-fine into a
    SINGLE wrapper around one inner LUNA Model.

    Why both at once
    ----------------
    Hierarchical mixes SPATIAL neighborhoods (bins cells by current
    x_t positions → per-patch context). Coarse-to-fine factorizes
    by GENE similarity (k-means clusters cells by gene expression →
    per-cluster spatial centroid). These are conceptually orthogonal:

        hierarchical  : "I know what's spatially near me"
        coarse-to-fine: "I know which biological group I belong to
                          and where that group should go"

    Each individually gave a measurable bump on cortex. Stacking
    them lets the network use BOTH signals simultaneously without
    one swallowing the other.

    Composition
    -----------
    We can't simply nest the two existing wrappers because each one
    re-instantiates its inner Model with its own input_dims bump,
    so naive nesting would produce a Model that doesn't know about
    the other wrapper's augmentation. Instead, this class:

      1. Constructs both module pieces (PatchContextModule from
         the hierarchical wrapper, GeneClusterModule + CoarseRegressor
         from the c2f wrapper) on the SAME raw input dim.
      2. Constructs ONE inner Model with input_dims bumped by BOTH
         output dims (centroid_2 + patch_output_dim).
      3. Per forward, runs both augmentations on the raw features,
         concatenates everything, and calls the inner Model once.

    Stash semantics
    ---------------
    The output DataHolder carries the c2f stash attributes
    (predicted cluster centroids, cluster IDs, n_clusters) so the
    LossFunction's ``_compute_coarse_centroid_mse`` works without
    modification. The hierarchical wrapper has no auxiliary loss
    component, so nothing extra needs to be stashed for it.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        n_layers: int,
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        output_dims: Dict[str, int],
        hier_cfg,
        c2f_cfg,
    ):
        super().__init__()

        # Local import to avoid circular import at module load time
        # (hierarchical.py is in the same package).
        from models.hierarchical import PatchContextModule

        def _g(cfg_obj, k, default):
            return (
                cfg_obj.get(k, default)
                if hasattr(cfg_obj, "get")
                else getattr(cfg_obj, k, default)
            )

        # -------- Hierarchical pieces --------
        n_patches_axis = int(_g(hier_cfg, "n_patches_per_axis", 4))
        patch_hidden = int(_g(hier_cfg, "patch_hidden_dim", 128))
        patch_layers = int(_g(hier_cfg, "patch_n_layers", 2))
        patch_heads = int(_g(hier_cfg, "patch_n_heads", 4))
        patch_out_dim = int(_g(hier_cfg, "output_dim", 64))

        gene_in = int(input_dims["node_features_dimensions"])
        self.patch_module = PatchContextModule(
            cell_input_dim=gene_in,
            n_patches_per_axis=n_patches_axis,
            patch_hidden_dim=patch_hidden,
            patch_n_layers=patch_layers,
            patch_n_heads=patch_heads,
            output_dim=patch_out_dim,
        )
        self._patch_out_dim = patch_out_dim

        # -------- Coarse-to-fine pieces --------
        self.n_clusters = int(_g(c2f_cfg, "n_clusters", 16))
        self.kmeans_n_iters = int(_g(c2f_cfg, "kmeans_n_iters", 10))
        coarse_hidden = int(_g(c2f_cfg, "coarse_hidden_dim", 128))
        coarse_layers = int(_g(c2f_cfg, "coarse_n_layers", 2))
        coarse_heads = int(_g(c2f_cfg, "coarse_n_heads", 4))
        gene_proj_dim = int(_g(c2f_cfg, "gene_proj_dim", 128))
        self.cluster_mode = str(_g(c2f_cfg, "cluster_mode", "kmeans")).lower()
        self.gumbel_tau = float(_g(c2f_cfg, "gumbel_tau", 1.0))

        self.cluster_module = GeneClusterModule(
            gene_input_dim=gene_in,
            proj_dim=gene_proj_dim,
            n_clusters=self.n_clusters,
            kmeans_n_iters=self.kmeans_n_iters,
            cluster_mode=self.cluster_mode,
            gumbel_tau=self.gumbel_tau,
        )
        self.coarse_regressor = CoarseRegressor(
            cluster_feature_dim=gene_proj_dim,
            hidden_dim=coarse_hidden,
            n_layers=coarse_layers,
            n_heads=coarse_heads,
        )
        self._coarse_hidden = coarse_hidden  # used in forward()

        # -------- Inner Model: receives [genes, centroid_2, cluster_emb_H, patch_ctx_64] --------
        # The cluster_emb_H term is the c2f regressor's pre-head
        # transformer hidden state — much richer conditioning than
        # the 2-D centroid alone (see CoarseToFineWrapper comment).
        augmented_input_dims = dict(input_dims)
        augmented_input_dims["node_features_dimensions"] = (
            gene_in + 2 + coarse_hidden + patch_out_dim
        )

        from models.model import Model
        self.inner = Model(
            input_dims=augmented_input_dims,
            n_layers=n_layers,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
        )

    def forward(
        self,
        data: DataHolder,
        true_positions: Optional[torch.Tensor] = None,
    ) -> DataHolder:
        # --- Coarse-to-fine stage ---
        cluster_ids, cluster_features = self.cluster_module(
            data.node_features, data.node_mask,
        )
        # Get BOTH centroids (for aux loss + thin spatial anchor)
        # AND the 128-D cluster embeddings (rich identity signal).
        predicted_centroids, cluster_embeddings = (
            self.coarse_regressor.forward_with_embeddings(cluster_features)
        )

        if self.training and true_positions is not None:
            cond_centroids = CoarseToFineWrapper._compute_true_centroids(
                true_positions, cluster_ids, data.node_mask, self.n_clusters,
            )
        else:
            cond_centroids = predicted_centroids

        cell_centroids = CoarseToFineWrapper._broadcast_per_cluster(
            cond_centroids, cluster_ids, data.node_mask,
        )                                                              # (B, N, 2)
        cell_cluster_emb = CoarseToFineWrapper._broadcast_per_cluster(
            cluster_embeddings, cluster_ids, data.node_mask,
        )                                                              # (B, N, H)

        # --- Hierarchical stage ---
        # Patch module takes the RAW gene features + CURRENT positions
        # (not the c2f-augmented features). This keeps the two
        # augmentations conceptually independent: hierarchical sees
        # spatial neighborhood structure on the genes, c2f sees
        # gene-similarity clusters with spatial centroids + rich
        # cluster identity — and the inner Model gets to combine
        # both via attention.
        patch_ctx = self.patch_module(
            data.node_features, data.positions, data.node_mask,
        )                                                              # (B, N, patch_out_dim)

        # --- Concatenate everything and call the inner Model ---
        # Layout: [genes, centroid_2, cluster_emb_H, patch_ctx_64]
        data_aug = data.copy()
        data_aug.node_features = torch.cat(
            [data.node_features, cell_centroids, cell_cluster_emb, patch_ctx], dim=-1,
        )
        out = self.inner(data_aug)

        # Stash for the loss (only c2f has an auxiliary loss term).
        out._predicted_cluster_centroids = predicted_centroids
        out._cluster_ids = cluster_ids
        out._n_clusters = self.n_clusters
        return out

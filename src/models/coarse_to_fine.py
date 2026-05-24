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
        gumbel_anneal_steps: int = 0,
        gumbel_tau_final: float = 0.1,
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
        self.gumbel_tau_initial = float(gumbel_tau)
        self.gumbel_tau_final = float(gumbel_tau_final)
        self.gumbel_anneal_steps = int(gumbel_anneal_steps)
        # Step counter incremented per training forward — used for
        # temperature annealing. Buffered so it persists in
        # state_dict (resumed training resumes the annealing schedule).
        self.register_buffer(
            "_step_count", torch.zeros(1, dtype=torch.long),
            persistent=True,
        )
        # Most recent cluster-balance regularizer value, stashed
        # here for the wrapper to read and forward to LossFunction.
        # Updated each forward call when cluster_mode='gumbel'.
        self._last_balance_loss: Optional[torch.Tensor] = None

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
    def _current_tau(self) -> float:
        """Effective Gumbel temperature at the current training step.
        Linearly interpolates from ``gumbel_tau_initial`` to
        ``gumbel_tau_final`` over ``gumbel_anneal_steps`` forward
        passes. After annealing completes, holds at the final value.
        At inference (when ``_step_count`` is not advanced), always
        returns the initial tau — the deterministic-argmax inference
        path doesn't use tau anyway.
        """
        if self.gumbel_anneal_steps <= 0:
            return self.gumbel_tau_initial
        step = int(self._step_count.item())
        progress = min(1.0, step / float(self.gumbel_anneal_steps))
        return (
            self.gumbel_tau_initial * (1.0 - progress)
            + self.gumbel_tau_final * progress
        )

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
            # Advance the step counter (used by tau annealing).
            with torch.no_grad():
                self._step_count += 1
            # ``hard=True`` straight-through: forward is one-hot,
            # backward uses soft weights. Tau may be annealed from
            # gumbel_tau_initial to gumbel_tau_final over
            # gumbel_anneal_steps steps; see ``_current_tau``.
            tau = self._current_tau()
            weights = F.gumbel_softmax(
                logits, tau=tau, hard=True, dim=-1,
            )                                                       # (B, N, K)
        else:
            # Deterministic argmax for inference. We still build a
            # one-hot tensor so the aggregation einsum below is
            # identical to the training path.
            argmax_idx = logits.argmax(dim=-1)                      # (B, N)
            weights = F.one_hot(argmax_idx, num_classes=K).to(cell_emb.dtype)

        # Cluster balance regularizer. Computed from SOFT
        # probabilities (not the hard weights) so the regularizer
        # has meaningful gradient on the cluster head's logits, not
        # just on the masked-out hard counts.
        # L_balance = -H(p_avg) where p_avg = mean over real cells
        # of softmax(logits). Maximising H ↔ minimising -H pushes
        # the average cluster usage toward uniform 1/K. The wrapper
        # reads ``self._last_balance_loss`` after this forward.
        if self.training:
            soft_probs = F.softmax(logits, dim=-1)                  # (B, N, K)
            valid_mask = node_mask.unsqueeze(-1).to(soft_probs.dtype)
            soft_probs_masked = soft_probs * valid_mask
            n_real = valid_mask.sum().clamp_min(1.0)
            p_avg = soft_probs_masked.sum(dim=(0, 1)) / n_real      # (K,)
            # Negative entropy (small when uniform); add a small
            # epsilon for numerical safety against log(0).
            eps = 1e-9
            neg_entropy = (p_avg * (p_avg + eps).log()).sum()
            # Minimum (most-balanced) value is log(1/K)*1 = -log(K).
            # Subtract that floor so the term is non-negative and
            # zero when perfectly balanced.
            self._last_balance_loss = neg_entropy + float(torch.log(torch.tensor(float(K))))
        else:
            self._last_balance_loss = None

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
# Two-stage FM coarse stage (option #4)
# ---------------------------------------------------------------------------


class ClusterFlowMatchingStage(nn.Module):
    """Wraps ``CoarseRegressor`` with a flow-matching trajectory on
    the K cluster centroids.

    The underlying ``CoarseRegressor`` is unchanged — it's still a
    small transformer that maps cluster_features → predicted
    centroids. The wrapper adds:

    * **Training**: for each batch, sample t ∼ U(eps, 1), construct
      a noisy centroid intermediate ``c_t = (1-t)·c_0 + t·c_1``
      (where c_0 = true centroid, c_1 ~ N(0, I)), feed the noisy
      centroid as ADDITIONAL input to the regressor alongside the
      cluster features, and have it predict the clean ``c_0``.
      The training step gathers a single FM loss on this prediction.

    * **Inference**: run an internal Euler ODE over
      ``n_sampling_steps`` to refine the centroid prediction from
      pure noise to the converged cluster centroid. The outer
      c2f wrapper sees the converged centroids and conditions the
      cell-level FM on them.

    Concretely, the regressor's input gets ONE EXTRA per-cluster
    feature (the current noisy centroid, padded into the cluster
    feature vector). Output stays (B, K, 2) — predicted clean
    centroid.

    The loss on the coarse stage is the same
    ``_compute_coarse_centroid_mse`` as the regression path; the
    only difference is HOW we got predicted_centroids (one FM
    forward at training, many at inference).
    """

    def __init__(
        self,
        cluster_feature_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        n_sampling_steps: int = 25,
        eps_t: float = 1.0e-3,
    ):
        super().__init__()
        # The inner regressor takes cluster_features + noisy_centroid
        # (2D) + t_embedding (8D), so input width is
        # cluster_feature_dim + 2 + 8.
        self._time_dim = 8
        self.time_proj = nn.Sequential(
            nn.Linear(1, self._time_dim),
            nn.SiLU(),
        )
        self.regressor = CoarseRegressor(
            cluster_feature_dim=cluster_feature_dim + 2 + self._time_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        self.n_sampling_steps = int(n_sampling_steps)
        self.eps_t = float(eps_t)

    def _step_forward(
        self,
        cluster_features: torch.Tensor,
        c_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One regressor forward at (cluster_features, c_t, t).
        Returns (predicted_centroid_x0, cluster_hidden_state)."""
        B, K = cluster_features.shape[:2]
        t_emb = self.time_proj(t.view(-1, 1)).view(B, 1, self._time_dim).expand(B, K, self._time_dim)
        # Concatenate cluster_features + noisy centroid (c_t) + time.
        aug = torch.cat([cluster_features, c_t, t_emb], dim=-1)
        # forward_with_embeddings returns (centroids, hidden_state).
        return self.regressor.forward_with_embeddings(aug)

    def forward_train(
        self,
        cluster_features: torch.Tensor,
        true_centroids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training-time forward. One FM step:
          1. Sample t ∈ U(eps, 1).
          2. Build c_t = (1-t)·true + t·noise.
          3. Predict x0 from c_t.
        Returns (predicted_x0, cluster_hidden_state).
        """
        B = cluster_features.shape[0]
        device = cluster_features.device
        t = self.eps_t + (1.0 - self.eps_t) * torch.rand(B, 1, device=device)
        # noise: same shape as centroids (B, K, 2).
        noise = torch.randn_like(true_centroids)
        t_b = t.unsqueeze(-1)                                          # (B, 1, 1)
        c_t = (1.0 - t_b) * true_centroids + t_b * noise
        return self._step_forward(cluster_features, c_t, t.squeeze(-1))

    @torch.no_grad()
    def forward_sample(
        self,
        cluster_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference-time forward. Runs an internal Euler ODE from
        pure noise to converged centroid. Returns (final_centroids,
        cluster_hidden_state from the last step).
        """
        B, K = cluster_features.shape[:2]
        device = cluster_features.device
        c_t = torch.randn(B, K, 2, device=device)                       # (B, K, 2)
        n = max(1, self.n_sampling_steps)
        last_hidden = None
        # Euler reverse from t=1 → t=0.
        for s_int in reversed(range(0, n)):
            t = torch.full((B,), float(s_int + 1) / n, device=device)
            x0_pred, last_hidden = self._step_forward(cluster_features, c_t, t)
            # Euler step toward x0: x_{t-Δ} ≈ x_t + (Δ/t)·(x0 - x_t)
            s_next = float(s_int) / n
            t_val = float(s_int + 1) / n
            dt = t_val - s_next
            c_t = c_t + (dt / max(t_val, self.eps_t)) * (x0_pred - c_t)
        # Final pass with t≈0 to get the cleanest x0 estimate.
        t_final = torch.full((B,), self.eps_t, device=device)
        x0_pred, last_hidden = self._step_forward(cluster_features, c_t, t_final)
        return x0_pred, last_hidden


# ---------------------------------------------------------------------------
# Cell → cluster cross-attention (option #1 conditioning mode)
# ---------------------------------------------------------------------------


class CellToClusterCrossAttention(nn.Module):
    """Each cell soft-queries the K cluster tokens.

    Used in place of the static "broadcast argmax cluster's
    embedding to each cell" path when
    ``coarse_to_fine.conditioning_mode == "cross_attention"``.

    Forward signature matches what the wrapper needs: takes the
    raw cell features, the K cluster tokens (the coarse regressor's
    pre-head hidden states), and the cell mask. Returns a single
    ``(B, N, output_dim)`` per-cell vector that gets concatenated to
    gene features just like the concat-broadcast does today.

    Architecturally trivial: standard scaled-dot-product attention
    with Q from cell features, K/V from cluster tokens. No
    self-attention among cells (we let LUNA's main transformer
    handle that downstream).
    """

    def __init__(
        self,
        cell_input_dim: int,
        cluster_input_dim: int,
        hidden_dim: int,
        n_heads: int,
        output_dim: int,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, (
            f"hidden_dim ({hidden_dim}) must divide n_heads ({n_heads})"
        )
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.output_dim = output_dim

        # Q/K/V projections. K and V share their input (cluster tokens).
        self.q_proj = nn.Linear(cell_input_dim, hidden_dim)
        self.k_proj = nn.Linear(cluster_input_dim, hidden_dim)
        self.v_proj = nn.Linear(cluster_input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        cell_features: torch.Tensor,      # (B, N, cell_input_dim)
        cluster_tokens: torch.Tensor,      # (B, K, cluster_input_dim)
        node_mask: torch.Tensor,           # (B, N) bool
    ) -> torch.Tensor:
        """Returns (B, N, output_dim)."""
        B, N, _ = cell_features.shape
        K = cluster_tokens.shape[1]
        H = self.hidden_dim
        Hh = H // self.n_heads

        Q = self.q_proj(cell_features).reshape(B, N, self.n_heads, Hh).transpose(1, 2)
        Kt = self.k_proj(cluster_tokens).reshape(B, K, self.n_heads, Hh).transpose(1, 2)
        V = self.v_proj(cluster_tokens).reshape(B, K, self.n_heads, Hh).transpose(1, 2)
        # Standard scaled dot-product attention.
        # attn: (B, n_heads, N, K)
        attn_scores = torch.matmul(Q, Kt.transpose(-2, -1)) / (Hh ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        # ctx: (B, n_heads, N, Hh) → (B, N, H)
        ctx = torch.matmul(attn_weights, V).transpose(1, 2).reshape(B, N, H)
        out = self.out_proj(ctx)
        # Zero out padding cells.
        out = out * node_mask.unsqueeze(-1).to(out.dtype)
        return out


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
        self.gumbel_anneal_steps = int(_g("gumbel_anneal_steps", 0))
        self.gumbel_tau_final = float(_g("gumbel_tau_final", 0.1))
        self.cluster_balance_weight = float(_g("cluster_balance_weight", 0.0))
        self.conditioning_mode = str(_g("conditioning_mode", "concat")).lower()
        self.cross_attn_hidden_dim = int(_g("cross_attn_hidden_dim", 128))
        self.cross_attn_n_heads = int(_g("cross_attn_n_heads", 4))
        self.cross_attn_output_dim = int(_g("cross_attn_output_dim", 128))
        if self.conditioning_mode not in ("concat", "cross_attention"):
            raise ValueError(
                f"conditioning_mode must be 'concat' or 'cross_attention'; "
                f"got {self.conditioning_mode!r}"
            )
        self.multi_resolution_enabled = bool(_g("multi_resolution_enabled", False))
        self.multi_resolution_k = int(_g("multi_resolution_k", 64))
        self.multi_resolution_loss_weight = float(_g("multi_resolution_loss_weight", 0.5))
        self.coarse_stage = str(_g("coarse_stage", "regression")).lower()
        self.coarse_stage_n_steps = int(_g("coarse_stage_n_steps", 25))
        if self.coarse_stage not in ("regression", "flow_matching"):
            raise ValueError(
                f"coarse_stage must be 'regression' or 'flow_matching'; "
                f"got {self.coarse_stage!r}"
            )

        gene_in = int(input_dims["node_features_dimensions"])

        self.cluster_module = GeneClusterModule(
            gene_input_dim=gene_in,
            proj_dim=self.gene_proj_dim,
            n_clusters=self.n_clusters,
            kmeans_n_iters=self.kmeans_n_iters,
            cluster_mode=self.cluster_mode,
            gumbel_tau=self.gumbel_tau,
            gumbel_anneal_steps=self.gumbel_anneal_steps,
            gumbel_tau_final=self.gumbel_tau_final,
        )

        if self.coarse_stage == "regression":
            self.coarse_regressor = CoarseRegressor(
                cluster_feature_dim=self.gene_proj_dim,
                hidden_dim=self.coarse_hidden_dim,
                n_layers=self.coarse_n_layers,
                n_heads=self.coarse_n_heads,
            )
            self.coarse_fm = None
        else:  # flow_matching
            self.coarse_fm = ClusterFlowMatchingStage(
                cluster_feature_dim=self.gene_proj_dim,
                hidden_dim=self.coarse_hidden_dim,
                n_layers=self.coarse_n_layers,
                n_heads=self.coarse_n_heads,
                n_sampling_steps=self.coarse_stage_n_steps,
            )
            # The downstream code reads ``self.coarse_regressor`` to
            # get embeddings — alias the FM's inner regressor so the
            # same code path works.
            self.coarse_regressor = self.coarse_fm.regressor

        # Inner Model takes augmented input. Two conditioning modes:
        #
        # 1. ``concat`` (default): each cell gets [centroid (2-d) +
        #    cluster_embedding (coarse_hidden_dim)] concatenated to
        #    its gene features. Static one-hot view (cell argmax
        #    determines which cluster's embedding it receives).
        #
        # 2. ``cross_attention``: each cell uses its gene features
        #    as Query against the K cluster tokens as Key/Value.
        #    The cross-attn output is concatenated to gene features.
        #    Lets boundary cells soft-mix multiple cluster
        #    representations.
        if self.conditioning_mode == "concat":
            self.cond_dim = 2 + self.coarse_hidden_dim
            self.cross_attn = None
        else:  # cross_attention
            self.cond_dim = self.cross_attn_output_dim
            self.cross_attn = CellToClusterCrossAttention(
                cell_input_dim=gene_in,
                cluster_input_dim=self.coarse_hidden_dim,
                hidden_dim=self.cross_attn_hidden_dim,
                n_heads=self.cross_attn_n_heads,
                output_dim=self.cross_attn_output_dim,
            )

        # Multi-resolution: a SECOND parallel cluster_module +
        # coarse_regressor + cross_attn (if applicable) at a
        # different K. Provides per-cell conditioning at two
        # gene-similarity resolutions simultaneously. Off by default.
        if self.multi_resolution_enabled:
            self.cluster_module_secondary = GeneClusterModule(
                gene_input_dim=gene_in,
                proj_dim=self.gene_proj_dim,
                n_clusters=self.multi_resolution_k,
                kmeans_n_iters=self.kmeans_n_iters,
                cluster_mode=self.cluster_mode,
                gumbel_tau=self.gumbel_tau,
                gumbel_anneal_steps=self.gumbel_anneal_steps,
                gumbel_tau_final=self.gumbel_tau_final,
            )
            self.coarse_regressor_secondary = CoarseRegressor(
                cluster_feature_dim=self.gene_proj_dim,
                hidden_dim=self.coarse_hidden_dim,
                n_layers=self.coarse_n_layers,
                n_heads=self.coarse_n_heads,
            )
            if self.conditioning_mode == "concat":
                self.cond_dim_secondary = 2 + self.coarse_hidden_dim
                self.cross_attn_secondary = None
            else:
                self.cond_dim_secondary = self.cross_attn_output_dim
                self.cross_attn_secondary = CellToClusterCrossAttention(
                    cell_input_dim=gene_in,
                    cluster_input_dim=self.coarse_hidden_dim,
                    hidden_dim=self.cross_attn_hidden_dim,
                    n_heads=self.cross_attn_n_heads,
                    output_dim=self.cross_attn_output_dim,
                )
            total_cond_dim = self.cond_dim + self.cond_dim_secondary
        else:
            self.cluster_module_secondary = None
            self.coarse_regressor_secondary = None
            self.cross_attn_secondary = None
            self.cond_dim_secondary = 0
            total_cond_dim = self.cond_dim

        augmented_input_dims = dict(input_dims)
        augmented_input_dims["node_features_dimensions"] = gene_in + total_cond_dim

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

        # 3. Coarse stage → predicted cluster centroids AND
        # cluster embeddings.
        #
        # Two paths depending on ``coarse_stage`` config:
        # * ``regression`` (default): one-shot regressor forward.
        # * ``flow_matching``: at training, ONE FM step (random t);
        #   at inference, an internal ODE refines from noise to the
        #   converged centroid.
        if self.coarse_stage == "regression":
            predicted_centroids, cluster_embeddings = (
                self.coarse_regressor.forward_with_embeddings(cluster_features)
            )
        else:  # flow_matching
            if self.training and true_positions is not None:
                # Need true centroids to build the FM noisy interpolant.
                true_centroids_for_fm = self._compute_true_centroids(
                    true_positions, cluster_ids, data.node_mask, self.n_clusters,
                )
                predicted_centroids, cluster_embeddings = (
                    self.coarse_fm.forward_train(cluster_features, true_centroids_for_fm)
                )
            else:
                predicted_centroids, cluster_embeddings = (
                    self.coarse_fm.forward_sample(cluster_features)
                )

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

        # 6. Build per-cell conditioning. Two paths:
        if self.conditioning_mode == "concat":
            # Static path: each cell gets its argmax cluster's
            # centroid (2-D) + embedding (coarse_hidden_dim) via
            # broadcast.
            cell_centroids = self._broadcast_per_cluster(
                cond_centroids, cluster_ids, data.node_mask,
            )                                                          # (B, N, 2)
            cell_cluster_emb = self._broadcast_per_cluster(
                cluster_embeddings, cluster_ids, data.node_mask,
            )                                                          # (B, N, H)
            cond_per_cell = torch.cat(
                [cell_centroids, cell_cluster_emb], dim=-1,
            )                                                          # (B, N, 2+H)
        else:  # cross_attention
            # Dynamic path: cells soft-query the K cluster tokens.
            # Note: we feed the raw gene features (not the projected
            # ones from cluster_module.gene_proj) as Query, because
            # the projected representation is too narrow to express
            # diverse query intents. The cluster_embeddings here are
            # ``coarse_hidden_dim``-wide and serve as both K and V.
            cond_per_cell = self.cross_attn(
                cell_features=data.node_features,
                cluster_tokens=cluster_embeddings,
                node_mask=data.node_mask,
            )                                                          # (B, N, cross_attn_output_dim)

        # Secondary resolution (if enabled): compute a SECOND cluster
        # assignment + coarse regression at multi_resolution_k clusters,
        # broadcast its conditioning, concatenate alongside the primary.
        secondary_cond = None
        secondary_predicted_centroids = None
        secondary_cluster_ids = None
        if self.multi_resolution_enabled:
            cluster_ids2, cluster_features2 = self.cluster_module_secondary(
                data.node_features, data.node_mask,
            )
            predicted_centroids2, cluster_embeddings2 = (
                self.coarse_regressor_secondary.forward_with_embeddings(cluster_features2)
            )
            if self.training and true_positions is not None:
                cond_centroids2 = self._compute_true_centroids(
                    true_positions, cluster_ids2, data.node_mask, self.multi_resolution_k,
                )
            else:
                cond_centroids2 = predicted_centroids2

            if self.conditioning_mode == "concat":
                cell_centroids2 = self._broadcast_per_cluster(
                    cond_centroids2, cluster_ids2, data.node_mask,
                )
                cell_cluster_emb2 = self._broadcast_per_cluster(
                    cluster_embeddings2, cluster_ids2, data.node_mask,
                )
                secondary_cond = torch.cat([cell_centroids2, cell_cluster_emb2], dim=-1)
            else:
                secondary_cond = self.cross_attn_secondary(
                    cell_features=data.node_features,
                    cluster_tokens=cluster_embeddings2,
                    node_mask=data.node_mask,
                )
            secondary_predicted_centroids = predicted_centroids2
            secondary_cluster_ids = cluster_ids2

        # Call the inner backbone. The inner Model was built with
        # input_dims bumped by ``self.cond_dim`` (+ secondary if on).
        data_aug = data.copy()
        if secondary_cond is not None:
            data_aug.node_features = torch.cat(
                [data.node_features, cond_per_cell, secondary_cond], dim=-1,
            )
        else:
            data_aug.node_features = torch.cat(
                [data.node_features, cond_per_cell], dim=-1,
            )
        out = self.inner(data_aug)

        # 7. Stash coarse outputs on the returned DataHolder so the
        # loss function can find them.
        out._predicted_cluster_centroids = predicted_centroids
        out._cluster_ids = cluster_ids
        out._n_clusters = self.n_clusters
        # Cluster balance regularizer (Gumbel mode + training only).
        out._cluster_balance_loss = self.cluster_module._last_balance_loss
        out._cluster_balance_weight = self.cluster_balance_weight
        # Secondary-resolution outputs for the multi-resolution loss
        # term. None when multi_resolution_enabled=False.
        out._predicted_cluster_centroids_secondary = secondary_predicted_centroids
        out._cluster_ids_secondary = secondary_cluster_ids
        out._n_clusters_secondary = (
            self.multi_resolution_k if self.multi_resolution_enabled else None
        )
        out._multi_resolution_loss_weight = self.multi_resolution_loss_weight
        # Combined balance loss from BOTH cluster modules.
        if self.multi_resolution_enabled and self.cluster_module_secondary._last_balance_loss is not None:
            sec_balance = self.cluster_module_secondary._last_balance_loss
            primary_balance = out._cluster_balance_loss
            if primary_balance is not None:
                out._cluster_balance_loss = primary_balance + sec_balance
            else:
                out._cluster_balance_loss = sec_balance
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
        self.gumbel_anneal_steps = int(_g(c2f_cfg, "gumbel_anneal_steps", 0))
        self.gumbel_tau_final = float(_g(c2f_cfg, "gumbel_tau_final", 0.1))
        self.cluster_balance_weight = float(_g(c2f_cfg, "cluster_balance_weight", 0.0))
        self.conditioning_mode = str(_g(c2f_cfg, "conditioning_mode", "concat")).lower()
        self.cross_attn_hidden_dim = int(_g(c2f_cfg, "cross_attn_hidden_dim", 128))
        self.cross_attn_n_heads = int(_g(c2f_cfg, "cross_attn_n_heads", 4))
        self.cross_attn_output_dim = int(_g(c2f_cfg, "cross_attn_output_dim", 128))

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

        # -------- Inner Model conditioning width depends on conditioning_mode --------
        # concat:           [genes, centroid_2, cluster_emb_H, patch_ctx_64]
        # cross_attention:  [genes, cross_attn_out, patch_ctx_64]
        if self.conditioning_mode == "concat":
            self.cond_dim = 2 + coarse_hidden
            self.cross_attn = None
        else:  # cross_attention
            self.cond_dim = self.cross_attn_output_dim
            self.cross_attn = CellToClusterCrossAttention(
                cell_input_dim=gene_in,
                cluster_input_dim=coarse_hidden,
                hidden_dim=self.cross_attn_hidden_dim,
                n_heads=self.cross_attn_n_heads,
                output_dim=self.cross_attn_output_dim,
            )
        augmented_input_dims = dict(input_dims)
        augmented_input_dims["node_features_dimensions"] = (
            gene_in + self.cond_dim + patch_out_dim
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

        # Conditioning per cell — concat-broadcast or cross-attention.
        if self.conditioning_mode == "concat":
            cell_centroids = CoarseToFineWrapper._broadcast_per_cluster(
                cond_centroids, cluster_ids, data.node_mask,
            )                                                          # (B, N, 2)
            cell_cluster_emb = CoarseToFineWrapper._broadcast_per_cluster(
                cluster_embeddings, cluster_ids, data.node_mask,
            )                                                          # (B, N, H)
            c2f_cond = torch.cat([cell_centroids, cell_cluster_emb], dim=-1)
        else:  # cross_attention
            c2f_cond = self.cross_attn(
                cell_features=data.node_features,
                cluster_tokens=cluster_embeddings,
                node_mask=data.node_mask,
            )                                                          # (B, N, cross_attn_output_dim)

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
        # Layout: [genes, c2f_cond, patch_ctx]
        # where c2f_cond is either [centroid_2, cluster_emb_H] (concat)
        # or cross_attn_output (cross_attention).
        data_aug = data.copy()
        data_aug.node_features = torch.cat(
            [data.node_features, c2f_cond, patch_ctx], dim=-1,
        )
        out = self.inner(data_aug)

        # Stash for the loss (only c2f has an auxiliary loss term).
        out._predicted_cluster_centroids = predicted_centroids
        out._cluster_ids = cluster_ids
        out._n_clusters = self.n_clusters
        out._cluster_balance_loss = self.cluster_module._last_balance_loss
        out._cluster_balance_weight = self.cluster_balance_weight
        return out

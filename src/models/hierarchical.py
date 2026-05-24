"""Multi-scale hierarchical wrapper for the LUNA transformer backbone.

Toggleable via ``model.hierarchical.enabled``. See the config block in
``scgg/src/configs/model/default.yaml`` for the motivation and full
behavior contract.

Two classes:

* ``PatchContextModule`` — the architectural piece. Spatially bins
  cells into patches by quantile, aggregates per-patch features,
  runs a small transformer over the patches, broadcasts each
  patch's output back to its member cells.

* ``HierarchicalModelWrapper`` — wraps a freshly-constructed
  ``models.model.Model`` so the patch context module runs once per
  forward, the patch context is concatenated to ``node_features``,
  and the inner Model runs on the augmented input. ``input_dims``
  is bumped by ``output_dim`` so the inner Model's
  ``mlp_in_node_features`` accepts the wider input.

The wrapper preserves LUNA's ``DataHolder``-in / ``DataHolder``-out
interface, so the rest of the pipeline (training step, sampling
loop, loss function) doesn't care whether hierarchical is on or off.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from utils.data.dataholder import DataHolder


# ---------------------------------------------------------------------------
# Patch context module
# ---------------------------------------------------------------------------


class PatchContextModule(nn.Module):
    """Per-forward patch-level summarisation + broadcast.

    Pipeline
    --------
    1. Encode each cell with a small MLP from ``cell_input_dim`` to
       ``patch_hidden_dim``. This is a SHALLOW pre-encoder used only
       for patch aggregation — the main backbone has its own deeper
       gene encoder that consumes the raw cell features in parallel.

    2. Assign each cell a patch ID by quantile-binning the cell's
       (x, y) position within the slice. With
       ``n_patches_per_axis=K``, the slice is split into K×K patches
       of approximately equal cell count along each axis.

    3. Aggregate per patch: mean of cell embeddings, patch centroid
       (mean of cell positions), log(cell count). Concatenate.

    4. Run an ``nn.TransformerEncoder`` with ``patch_n_layers`` layers
       over the ``n_patches²`` patch tokens. This is where
       tissue-level context mixing happens.

    5. Project each patch's output to ``output_dim``.

    6. Broadcast each patch's output to its constituent cells.
       Masked cells (node_mask=False) receive a zero vector.
    """

    def __init__(
        self,
        cell_input_dim: int,
        n_patches_per_axis: int,
        patch_hidden_dim: int,
        patch_n_layers: int,
        patch_n_heads: int,
        output_dim: int,
    ):
        super().__init__()
        if n_patches_per_axis < 2:
            raise ValueError(
                f"n_patches_per_axis must be ≥ 2; got {n_patches_per_axis}."
            )
        self.n_patches_per_axis = int(n_patches_per_axis)
        self.n_patches = self.n_patches_per_axis ** 2
        self.output_dim = int(output_dim)

        # Cell pre-encoder. Lightweight — the main backbone has its
        # own gene encoder. This is just for the patch aggregation
        # pathway so we don't pool 2000-dim raw counts.
        self.cell_enc = nn.Sequential(
            nn.Linear(cell_input_dim, patch_hidden_dim),
            nn.SiLU(),
        )

        # Project the patch summary tensor [mean_emb (H), centroid
        # (2), log_count (1)] back to ``patch_hidden_dim`` before
        # feeding the patch-level transformer.
        self.patch_in = nn.Linear(patch_hidden_dim + 3, patch_hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=patch_hidden_dim,
            nhead=patch_n_heads,
            dim_feedforward=patch_hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.patch_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=patch_n_layers,
        )

        # Per-cell output projection from the patch transformer's
        # hidden width to the downstream injection width.
        self.patch_out = nn.Linear(patch_hidden_dim, output_dim)

    # ------------------------------------------------------------------
    # Patch assignment (no gradient through this — pure indexing)
    # ------------------------------------------------------------------
    def _assign_patches(
        self, positions: torch.Tensor, node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Quantile-bin cells into ``K²`` patches based on their
        current positions. Returns a (B, N) Long tensor with values
        in ``[0, K²)`` for real cells and ``-1`` for masked-out cells.

        Per slice we rank cells by x, bin into K equal-count groups,
        then within each x-bin we rank by y and bin again. This is
        more uniform than a grid based on the bounding box because
        it doesn't waste capacity on cell-sparse regions of the
        slice.
        """
        B, N = positions.shape[:2]
        K_axis = self.n_patches_per_axis
        device = positions.device
        patch_ids = torch.full((B, N), -1, dtype=torch.long, device=device)

        for b in range(B):
            mask_b = node_mask[b]
            n_real = int(mask_b.sum().item())
            if n_real == 0:
                continue
            pos_b = positions[b][mask_b]    # (n_real, 2)

            # Rank in [0, n_real). argsort().argsort() is the
            # canonical "rank" trick.
            x_rank = pos_b[:, 0].argsort().argsort()
            # Bin in [0, K_axis).
            x_bin = (x_rank.float() * K_axis / max(n_real, 1)).long().clamp(max=K_axis - 1)

            # Within each x-bin, rank by y. We do this per-bin to
            # ensure y-binning is computed at the x-bin sub-population
            # scale (so within an upper-cortex x-bin the y-bins still
            # span its full y range).
            y_bin = torch.zeros_like(x_bin)
            for k in range(K_axis):
                in_bin = (x_bin == k)
                if in_bin.sum() == 0:
                    continue
                y_pos = pos_b[in_bin, 1]
                y_rank_in = y_pos.argsort().argsort()
                y_bin_in = (y_rank_in.float() * K_axis / max(y_pos.shape[0], 1)).long().clamp(max=K_axis - 1)
                y_bin[in_bin] = y_bin_in

            ids = x_bin * K_axis + y_bin     # (n_real,) in [0, K²)
            patch_ids[b][mask_b] = ids

        return patch_ids

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (B, N, cell_input_dim) — raw cell features
                (typically gene expression, the same tensor the main
                backbone sees).
            positions: (B, N, 2) — current cell positions (x_t at
                training time; the noisy intermediate at sampling
                time).
            node_mask: (B, N) bool — true for real cells.

        Returns:
            patch_context: (B, N, output_dim) — per-cell vector to
                concatenate to ``node_features`` before the main
                backbone. Masked cells get a zero vector.
        """
        B, N = node_features.shape[:2]
        device = node_features.device
        K = self.n_patches

        # 1. Cell-level pre-encoding for patch aggregation.
        cell_emb = self.cell_enc(node_features)        # (B, N, H)
        H = cell_emb.shape[-1]

        # 2. Patch assignment from current positions. No grad needed
        # — these are integer bin indices that change per forward but
        # we don't want to propagate gradients through the assignment
        # (standard hard-clustering convention).
        with torch.no_grad():
            patch_ids = self._assign_patches(positions, node_mask)

        # 3. Vectorised scatter-based aggregation across (B, K) bins.
        #    We treat (b, k) as a single flat index "b * K + k" so a
        #    single scatter_add_ over a (B*K, ·) buffer does the
        #    whole batch.
        safe_ids = patch_ids.clamp(min=0)              # (B, N) ≥ 0
        valid = node_mask & (patch_ids >= 0)           # (B, N) bool

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        combined_id = (batch_idx * K + safe_ids).reshape(-1)        # (B*N,)
        valid_f = valid.float().reshape(-1, 1)                      # (B*N, 1)

        cell_emb_flat = cell_emb.reshape(B * N, H) * valid_f        # (B*N, H)
        pos_flat = positions.reshape(B * N, 2) * valid_f            # (B*N, 2)
        count_flat = valid.float().reshape(B * N)                   # (B*N,)

        sum_emb = cell_emb.new_zeros(B * K, H)
        sum_pos = positions.new_zeros(B * K, 2)
        sum_cnt = positions.new_zeros(B * K)

        sum_emb.scatter_add_(
            0, combined_id.unsqueeze(-1).expand(-1, H), cell_emb_flat,
        )
        sum_pos.scatter_add_(
            0, combined_id.unsqueeze(-1).expand(-1, 2), pos_flat,
        )
        sum_cnt.scatter_add_(0, combined_id, count_flat)

        # Means (with safe div for empty patches — they'll get zero
        # mean_emb / zero centroid, which is fine because their
        # log_count will also be ~0 and the patch transformer can
        # learn to deprioritise them).
        cnt_safe = sum_cnt.clamp(min=1.0)
        mean_emb = (sum_emb / cnt_safe.unsqueeze(-1)).reshape(B, K, H)
        mean_pos = (sum_pos / cnt_safe.unsqueeze(-1)).reshape(B, K, 2)
        log_cnt = (sum_cnt + 1.0).log().reshape(B, K, 1)

        # 4. Patch summary tensor: [mean_emb, centroid, log_count].
        patch_summary = torch.cat([mean_emb, mean_pos, log_cnt], dim=-1)  # (B, K, H+3)
        patch_feats = self.patch_in(patch_summary)                         # (B, K, H)

        # 5. Patch-level transformer. No padding mask needed — empty
        # patches have zero features and the transformer will learn
        # to handle them (we could pass a per-batch mask but K is
        # small enough that wasted compute is negligible).
        patch_out = self.patch_transformer(patch_feats)                    # (B, K, H)
        patch_out = self.patch_out(patch_out)                              # (B, K, output_dim)

        # 6. Broadcast each patch's output to its member cells.
        # Gather along dim=1 of patch_out using safe_ids (B, N).
        # Masked cells point to patch 0 by the clamp; we zero them
        # out at the end so the inner backbone doesn't see garbage
        # at padded positions.
        broadcast = torch.gather(
            patch_out,
            1,
            safe_ids.unsqueeze(-1).expand(B, N, self.output_dim),
        )                                                                   # (B, N, output_dim)
        broadcast = broadcast * node_mask.unsqueeze(-1).float()

        return broadcast


# ---------------------------------------------------------------------------
# Wrapper around the LUNA Model that injects patch context
# ---------------------------------------------------------------------------


class HierarchicalModelWrapper(nn.Module):
    """Wraps a LUNA ``Model`` so each forward injects patch-level
    context into ``data.node_features`` before the inner forward.

    Drop-in for ``models.model.Model``: same ``DataHolder``-in,
    ``DataHolder``-out interface so the training step / sampling loop
    / loss function don't need to know hierarchical is on.

    Constructed with the SAME args as ``Model`` plus the hierarchical
    config block. We re-construct the inner Model here with a bumped
    ``input_dims["node_features_dimensions"]`` so its
    ``mlp_in_node_features`` accepts the wider concatenated input.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        n_layers: int,
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        output_dims: Dict[str, int],
        hierarchical_cfg,
    ):
        super().__init__()

        # Resolve hierarchical config with safe defaults so older
        # cfgs missing this block don't blow up during Hydra
        # composition.
        def _g(k, default):
            return (
                hierarchical_cfg.get(k, default)
                if hasattr(hierarchical_cfg, "get")
                else getattr(hierarchical_cfg, k, default)
            )

        n_axes = int(_g("n_patches_per_axis", 4))
        patch_h = int(_g("patch_hidden_dim", 128))
        patch_l = int(_g("patch_n_layers", 2))
        patch_n_heads = int(_g("patch_n_heads", 4))
        out_dim = int(_g("output_dim", 64))
        self.output_dim = out_dim

        cell_in_dim = int(input_dims["node_features_dimensions"])
        self.patch_module = PatchContextModule(
            cell_input_dim=cell_in_dim,
            n_patches_per_axis=n_axes,
            patch_hidden_dim=patch_h,
            patch_n_layers=patch_l,
            patch_n_heads=patch_n_heads,
            output_dim=out_dim,
        )

        # Bump input_dims so the inner Model's mlp_in_node_features
        # accepts the concatenated [genes, patch_context] input. All
        # other dims pass through unchanged.
        augmented_input_dims: Dict[str, int] = dict(input_dims)
        augmented_input_dims["node_features_dimensions"] = cell_in_dim + out_dim

        # Local import to avoid a circular dependency at module
        # import time (models/__init__ doesn't re-export Model, but
        # the import path resolves cleanly here).
        from models.model import Model
        self.inner = Model(
            input_dims=augmented_input_dims,
            n_layers=n_layers,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
        )

    def forward(self, data: DataHolder) -> DataHolder:
        # Patch context is computed from the current cell features
        # (the SAME tensor the inner backbone will see) and current
        # positions (x_t — noisy at training, evolving during
        # sampling). The assignment is recomputed every forward pass
        # so patches track the model's coarse-to-fine refinement.
        patch_ctx = self.patch_module(
            data.node_features, data.positions, data.node_mask,
        )

        # Concatenate and call the inner model on a COPY of the
        # DataHolder so the caller's tensor isn't mutated (the
        # sampling loop, in particular, reuses the input).
        data_aug = data.copy()
        data_aug.node_features = torch.cat(
            [data.node_features, patch_ctx], dim=-1,
        )
        return self.inner(data_aug)

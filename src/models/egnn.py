"""SE(2)-equivariant graph neural network backbone.

Drop-in replacement for ``models.model.Model``: takes the same
``DataHolder`` in, returns the same ``DataHolder`` out, with predicted
clean positions in ``out.positions`` and (a projection of) the latent
node features in ``out.node_features``. Selected via
``cfg.model.backbone = "egnn"`` in scgg's Hydra config.

Why
---
LUNA's transformer is approximately translation-invariant (mean-subtraction
at the input + output) but is NOT rotation-equivariant — it has to
learn rotation invariance from the pairwise-distance loss alone. An
SE(2)-equivariant architecture bakes both symmetries into the
weights, so:

  * sample efficiency improves (we don't burn capacity learning
    something we could enforce by construction),
  * the output frame is meaningful relative to inputs (the prediction
    rotates / translates with the input, no degenerate solutions
    where the model collapses everything to the origin), and
  * the eval metric is also rotation/translation invariant, so the
    inductive bias matches the metric.

Architecture
------------
Per layer ``l`` (Satorras et al. 2021, "E(n) Equivariant GNNs",
adapted to a kNN-graph + diffusion-time conditioning)::

    m_ij = φ_e([h_i, h_j, ||x_i - x_j||², t_i])
    c_ij = φ_x(m_ij)                                   # scalar
    Δx_i = Σ_{j ∈ N(i)} (x_i - x_j) / ||x_i - x_j||  ·  c_ij
    m_i  = Σ_{j ∈ N(i)} m_ij
    Δh_i = φ_h([h_i, m_i])
    x_i ← x_i + Δx_i           (skip if cfg.model.egnn.update_coords=false)
    h_i ← h_i + Δh_i

The position update is a sum of RELATIVE vectors scaled by SCALAR
coefficients, and the messages depend only on SE(2)-invariant
quantities (h's, ||x_i - x_j||², t). So::

    EGNN(R·x + b, h, t) = (R·EGNN_x(x, h, t) + b,  EGNN_h(x, h, t))

The kNN graph is rebuilt per forward pass from the current positions.
Padding cells are pushed to a far position so they never appear in any
real cell's neighbourhood, and any edge that does involve a padding
endpoint is zeroed out as belt-and-braces.

Memory
------
With ``B`` slices, ``N`` cells per slice (padded), ``K`` neighbours,
and edge hidden ``E``, the dominant per-layer tensor is
``(B·N·K, E)`` = ``B·N·K·E`` floats. For cortex defaults
(B=6, N=7500, K=50, E=64) that's ~600 MB per layer activations,
well within an 80 GB H100. Scaling to ABCA-size sections needs a
``knn_k`` of ~32 or hierarchical message passing — out of scope for
this first cut.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from utils.data.dataholder import DataHolder


# ---------------------------------------------------------------------------
# Single EGNN layer
# ---------------------------------------------------------------------------


class EGNNLayer(nn.Module):
    """One SE(2)-equivariant message-passing layer.

    Inputs are flat (B*N-shape) tensors with an explicit ``edge_index``
    so we can ride on torch-geometric's kNN/scatter primitives —
    cheaper than materialising a dense (B, N, N) edge tensor.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        update_coords: bool = True,
    ):
        super().__init__()
        self.update_coords = bool(update_coords)

        # φ_e: edge MLP on [h_i, h_j, ||x_i-x_j||², t_i]
        in_edge = 2 * node_dim + 1 + time_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
        )

        # φ_x: edge feature → scalar coordinate-update coefficient.
        # SMALL-GAIN Xavier init on the last layer (Satorras et al. 2021,
        # https://github.com/vgsatorras/egnn). Keeps the initial
        # position update tiny — predictions start near the identity
        # so positions don't blow up — but NON-ZERO so gradients flow
        # through every upstream parameter from step 0.
        #
        # The Tanh on the LAST activation bounds the per-edge
        # coefficient to (-1, 1). Without it, ``c`` can grow
        # unboundedly during training: combined with ``scatter_sum``
        # over ``knn_k`` neighbours, the per-cell position update
        # accumulates roughly as ``k · c`` per layer, and 8 layers
        # compound multiplicatively. On the first MMC run the loss
        # exploded to ~1e10 around epoch 50 — this is what fixed it.
        # See:
        #   - github.com/vgsatorras/egnn (canonical implementation;
        #     uses tanh on coord head for stability)
        #   - Brandstetter et al. 2022 "Geometric and Physical
        #     Quantities Improve E(3) Equivariant Message Passing"
        #     (discusses the explosion failure mode explicitly).
        coord_last = nn.Linear(edge_dim, 1)
        nn.init.xavier_uniform_(coord_last.weight, gain=1e-3)
        nn.init.zeros_(coord_last.bias)
        self.coord_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
            coord_last,
            nn.Tanh(),
        )

        # φ_h: [h_i, Σ_j m_ij] → Δh_i (residual added below).
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(
        self,
        h: torch.Tensor,        # (B*N, node_dim)
        x: torch.Tensor,        # (B*N, 2)
        t: torch.Tensor,        # (B*N, time_dim)
        edge_index: torch.Tensor,  # (2, E)
        edge_valid: torch.Tensor,  # (E,) bool — both endpoints are real cells
        mask: torch.Tensor,     # (B*N,) bool
    ):
        from torch_scatter import scatter_mean, scatter_sum  # noqa: WPS433

        src, dst = edge_index[0], edge_index[1]

        # Relative vectors and their squared norms. The norm is the only
        # SE(2)-invariant scalar feature of (x_i, x_j).
        x_diff = x[src] - x[dst]                          # (E, 2)
        d_sq = (x_diff * x_diff).sum(dim=-1, keepdim=True)  # (E, 1)

        # Edge messages.
        edge_input = torch.cat([h[src], h[dst], d_sq, t[src]], dim=-1)
        m = self.edge_mlp(edge_input)                     # (E, edge_dim)

        # Zero out messages on edges with a padding endpoint so they
        # don't pollute the per-node aggregations downstream.
        ev = edge_valid.float().unsqueeze(-1)
        m = m * ev

        # Coordinate update: this is the stability-critical block.
        # Three lessons baked in (from the first MMC run that blew up
        # to loss ~1e10 around epoch 50):
        #
        #   1. Use RAW ``x_diff``, NOT ``x_diff / ‖d‖``. Unit-vector
        #      normalisation gives each edge a magnitude-1 direction
        #      times ``c``, regardless of how close the points really
        #      are. Once ``c`` drifts from init, the update has no
        #      geometric anchor and positions explode.
        #
        #   2. Use ``scatter_mean``, NOT ``scatter_sum``. With
        #      ``knn_k=50`` neighbours, a sum makes per-cell updates
        #      ~50× larger than the per-edge contribution. mean
        #      decouples the update scale from ``knn_k`` entirely.
        #
        #   3. The Tanh inside ``coord_mlp`` (set up in __init__)
        #      bounds ``c ∈ (-1, 1)``. Without it, ``c`` can grow
        #      arbitrarily during training and compound across the
        #      8-layer stack.
        #
        # This combination matches the canonical Satorras et al.
        # EGNN implementation at github.com/vgsatorras/egnn.
        if self.update_coords:
            c = self.coord_mlp(m)                         # (E, 1)
            x_update = x_diff * c                         # (E, 2) -- raw, not /‖d‖
            x_agg = scatter_mean(
                x_update, src, dim=0, dim_size=x.shape[0],
            )                                             # (B*N, 2)
        else:
            x_agg = torch.zeros_like(x)

        # Message aggregation for the node-feature update. Sum is fine
        # here because m is bounded by SiLU's range AND multiplied by
        # the edge_valid mask AND followed by an MLP that re-scales.
        m_agg = scatter_sum(m, src, dim=0, dim_size=h.shape[0])  # (B*N, edge_dim)

        # Residual updates. Cells with mask=False stay at 0 (we re-mask
        # below in the wrapping EGNNModel; here we just keep the
        # tensors valid for downstream layers).
        x_new = x + x_agg
        h_new = h + self.node_mlp(torch.cat([h, m_agg], dim=-1))
        if mask is not None:
            mf = mask.float().unsqueeze(-1)
            x_new = x_new * mf
            h_new = h_new * mf

        return h_new, x_new


# ---------------------------------------------------------------------------
# Full model wrapper (drop-in for models.model.Model)
# ---------------------------------------------------------------------------


class EGNNModel(nn.Module):
    """SE(2)-equivariant alternative to ``models.model.Model``.

    Same DataHolder-in / DataHolder-out interface so it's a drop-in
    when ``cfg.model.backbone == "egnn"``.
    """

    # Sentinel far-away coordinate used to push padding cells out of
    # every real cell's k-NN neighbourhood when we build the graph.
    _PAD_DIST_SENTINEL = 1e6

    def __init__(
        self,
        input_dims,
        n_layers: int,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims,
        egnn_cfg,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_genes_in = int(input_dims["node_features_dimensions"])
        self.time_in = int(input_dims["diffusion_time_dimensions"])
        self.out_node_dim = int(hidden_dims["output_features_to_pos_dims"])

        # Read EGNN-specific config with safe defaults.
        def _g(k, default):
            return egnn_cfg.get(k, default) if hasattr(egnn_cfg, "get") else getattr(egnn_cfg, k, default)
        self.node_dim = int(_g("node_dim", hidden_dims["dx"]))
        self.edge_dim = int(_g("edge_dim", 64))
        self.time_dim = int(_g("time_dim", 32))
        self.knn_k = int(_g("knn_k", 50))
        self.update_coords = bool(_g("update_coords", True))

        # Input gene-expression encoder: (B*N, n_genes) -> (B*N, node_dim).
        # Scalar features → SE(2)-invariant by construction.
        self.gene_encoder = nn.Sequential(
            nn.Linear(self.n_genes_in, hidden_mlp_dims["X"]),
            nn.SiLU(),
            nn.Linear(hidden_mlp_dims["X"], self.node_dim),
            nn.SiLU(),
        )

        # Diffusion-time encoder: (B*N, time_in) -> (B*N, time_dim).
        # Scalar → SE(2)-invariant.
        self.time_encoder = nn.Sequential(
            nn.Linear(self.time_in, hidden_mlp_dims["y"]),
            nn.SiLU(),
            nn.Linear(hidden_mlp_dims["y"], self.time_dim),
            nn.SiLU(),
        )

        self.layers = nn.ModuleList([
            EGNNLayer(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                time_dim=self.time_dim,
                update_coords=self.update_coords,
            )
            for _ in range(n_layers)
        ])

        # Output node-feature projection — matches LUNA's
        # `mlp_out_node_features` output width, so downstream code
        # that consumes pred.node_features sees the same shape.
        self.out_node_proj = nn.Sequential(
            nn.Linear(self.node_dim, hidden_mlp_dims["X"]),
            nn.SiLU(),
            nn.Linear(hidden_mlp_dims["X"], self.out_node_dim),
        )

    def forward(self, data: DataHolder) -> DataHolder:
        # We import here so torch_geometric stays an optional dep — if
        # someone never selects backbone="egnn", they don't pay this
        # import cost at scgg import time.
        from torch_geometric.nn import knn_graph

        node_features = data.node_features    # (B, N, n_genes)
        positions = data.positions            # (B, N, 2)
        diffusion_time = data.diffusion_time  # (B, N, time_in)
        node_mask = data.node_mask            # (B, N)

        B, N = node_features.shape[:2]
        device = positions.device
        flat = lambda t: t.reshape(B * N, -1)

        # The noise model emits `diffusion_time` as one scalar per
        # slice — shape (B, 1) — not per-cell. LUNA's stock model
        # handles that implicitly: its time MLP runs on (B, 1) then
        # the transformer block broadcasts internally. Our EGNN
        # flattens to (B*N, ·) up-front, so we have to broadcast
        # explicitly. Accept (B,), (B, 1), (B, time_in), and the
        # already-broadcast (B, N, time_in) cases — all produce
        # the same downstream shape.
        if diffusion_time.dim() == 1:                 # (B,)
            diffusion_time = diffusion_time.view(B, 1, 1).expand(B, N, 1)
        elif diffusion_time.dim() == 2:               # (B, T)
            diffusion_time = diffusion_time.unsqueeze(1).expand(B, N, -1)
        # else: already (B, N, T) — leave it alone.

        # Initial encodings.
        h = self.gene_encoder(flat(node_features))   # (B*N, node_dim)
        x = flat(positions)                          # (B*N, 2)
        t_emb = self.time_encoder(flat(diffusion_time))  # (B*N, time_dim)
        mask_flat = node_mask.reshape(B * N)         # (B*N,) bool

        # Push padding cells out of every real cell's neighbourhood
        # for kNN — they sit at the origin (we explicitly mask them
        # at the boundary). Without this, padding cells crowd the
        # kNN of real cells in slices with high padding ratio.
        x_for_knn = torch.where(
            mask_flat.unsqueeze(-1),
            x,
            x.new_full(x.shape, self._PAD_DIST_SENTINEL),
        )

        batch_idx = torch.arange(B, device=device).repeat_interleave(N)

        # EGNN stack. We rebuild the kNN graph per layer so it tracks
        # the updated positions; the inductive bias here is that local
        # neighbourhoods refine as the model converges on better
        # coordinates.
        for layer in self.layers:
            edge_index = knn_graph(
                x_for_knn, k=self.knn_k, batch=batch_idx,
                loop=False, flow="source_to_target",
            )
            edge_valid = mask_flat[edge_index[0]] & mask_flat[edge_index[1]]
            h, x = layer(h, x, t_emb, edge_index, edge_valid, mask_flat)
            # Refresh the kNN-graph "padded" view so subsequent layers
            # see the updated x but with padding still pushed out.
            x_for_knn = torch.where(
                mask_flat.unsqueeze(-1),
                x,
                x.new_full(x.shape, self._PAD_DIST_SENTINEL),
            )

        # Reshape + final centering (translation invariance — same as
        # LUNA's Model.forward post-process). The mean-subtraction is
        # masked so padding cells don't bias it.
        x_dense = x.reshape(B, N, -1)
        h_dense = h.reshape(B, N, -1)
        mask_d = node_mask.unsqueeze(-1).float()
        x_dense = x_dense * mask_d
        # Masked mean over real cells per slice.
        valid_count = mask_d.sum(dim=1, keepdim=True).clamp_min(1.0)
        x_mean = (x_dense * mask_d).sum(dim=1, keepdim=True) / valid_count
        x_dense = (x_dense - x_mean) * mask_d

        # Project node features to LUNA's downstream-expected width
        # so the rest of the pipeline (which may consume
        # pred.node_features) sees the same shape.
        h_out = self.out_node_proj(h_dense.reshape(B * N, -1)).reshape(B, N, -1)

        return DataHolder(
            node_features=h_out,
            positions=x_dense,
            diffusion_time=diffusion_time,
            node_mask=node_mask,
        ).mask()

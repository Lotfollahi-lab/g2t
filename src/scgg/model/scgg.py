"""
ScGG: top-level model.

Two training objectives are supported, picked via config["training"]["objective"]:

  - "contrastive" (default, primary): learn a d-dim metric embedding and
    optimize a supervised contrastive / InfoNCE loss against the ground-truth
    spatial kNN graph (PinSage-style retrieval). Inference is a single forward
    pass + FAISS kNN. The velocity / flow modules are NOT instantiated.

  - "flow_matching" (ablation): retain the legacy conditional flow matching
    objective that generates 2-D coordinates and induces a kNN graph at the
    end. The metric head is NOT used.

The encoder + section encoder are shared across both objectives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from typing import Optional, Tuple, Union, List

from .encoder import GeneExpressionEncoder, SectionEncoder
from .metric_head import MetricHead, CrossCellMetricHead
from .velocity_net import VelocityNetwork
from .velocity_net_attention import CrossAttentionVelocityNetwork
from .luna_model import LunaTransformerNet
from .flow_matching import ConditionalFlowMatching
from .graph_constructor import GraphConstructor


VALID_OBJECTIVES = ("contrastive", "flow_matching")
VALID_METRIC_HEAD_TYPES = ("mlp", "cross_attention")


class ScGG(nn.Module):
    """Spatial graph generator over gene expression.

    Args:
        n_genes: Number of input genes.
        config: Full configuration dict.
    """

    def __init__(self, n_genes: int, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        train_cfg = config.get("training", {})

        # ---- Training objective (controls which heads are built) ------------
        self.objective: str = train_cfg.get("objective", "contrastive")
        if self.objective not in VALID_OBJECTIVES:
            raise ValueError(
                f"training.objective={self.objective!r} not in {VALID_OBJECTIVES}"
            )

        # ---- Encoder ---------------------------------------------------------
        enc_cfg = model_cfg["encoder"]
        self.encoder = GeneExpressionEncoder(
            n_genes=n_genes,
            hidden_dims=enc_cfg["hidden_dims"],
            embed_dim=enc_cfg["embed_dim"],
            dropout=enc_cfg["dropout"],
            norm=enc_cfg["norm"],
        )

        # ---- Section encoder -------------------------------------------------
        sec_cfg = model_cfg["section_encoder"]
        self.section_encoder = SectionEncoder(
            cell_embed_dim=enc_cfg["embed_dim"],
            hidden_dim=sec_cfg["hidden_dim"],
            embed_dim=sec_cfg["embed_dim"],
            n_attention_heads=sec_cfg["n_attention_heads"],
            subsample_size=sec_cfg["subsample_size"],
        )

        # ---- Metric head (contrastive mode) ---------------------------------
        self.metric_head: Optional[nn.Module] = None
        if self.objective == "contrastive":
            mh_cfg = model_cfg.get("metric_head", {})
            mh_type = mh_cfg.get("type", "cross_attention")
            if mh_type not in VALID_METRIC_HEAD_TYPES:
                raise ValueError(
                    f"model.metric_head.type={mh_type!r} not in "
                    f"{VALID_METRIC_HEAD_TYPES}"
                )
            if mh_type == "mlp":
                self.metric_head = MetricHead(
                    cell_embed_dim=enc_cfg["embed_dim"],
                    section_embed_dim=sec_cfg["embed_dim"],
                    hidden_dims=mh_cfg.get("hidden_dims", [256, 128]),
                    embed_dim=mh_cfg.get("embed_dim", 64),
                    normalize=mh_cfg.get("normalize", True),
                    dropout=mh_cfg.get("dropout", 0.1),
                )
            else:  # cross_attention
                self.metric_head = CrossCellMetricHead(
                    cell_embed_dim=enc_cfg["embed_dim"],
                    section_embed_dim=sec_cfg["embed_dim"],
                    embed_dim=mh_cfg.get("embed_dim", 64),
                    n_layers=mh_cfg.get("n_layers", 2),
                    n_heads=mh_cfg.get("n_heads", 4),
                    ff_mult=mh_cfg.get("ff_mult", 4),
                    dropout=mh_cfg.get("dropout", 0.1),
                    normalize=mh_cfg.get("normalize", True),
                    use_section_embed=mh_cfg.get("use_section_embed", False),
                )
            self.metric_head_type = mh_type
        else:
            self.metric_head_type = None

        # ---- Flow matching (primary mode) ----------------------------------
        # Velocity network has two flavours:
        #   * "mlp"  (legacy): per-cell MLP with section_embed as global
        #     context. No cell-cell information flow — empirically
        #     produces noise on reconstruction tasks because the model
        #     can only learn marginal positions, not spatial
        #     arrangement.
        #   * "cross_attention"  (default, LUNA-style): transformer
        #     encoder with multi-head self-attention OVER cells. Each
        #     cell's velocity depends on every other cell's current
        #     position at each denoising step. This is the architectural
        #     ingredient LUNA uses (see their models/transformer.py).
        self.velocity_net = None
        self.flow: Optional[ConditionalFlowMatching] = None
        if self.objective == "flow_matching":
            vn_cfg = model_cfg["velocity_net"]
            fl_cfg = model_cfg["flow"]
            vn_type = str(vn_cfg.get("type", "cross_attention")).lower()
            k_max = config["graph"].get("k_train_range", [5, 30])[1] + 10

            # Default prediction target depends on the net flavour.
            # LUNA predicts x_0; the MLP and cross-attention velocity
            # nets predict velocity.
            default_prediction_target = (
                "x_0" if vn_type == "luna" else "velocity"
            )
            prediction_target = str(
                fl_cfg.get("prediction_target", default_prediction_target)
            ).lower()

            # Default velocity_mse_weight depends on prediction_target.
            # When predicting x_0, the derived velocity is
            # v = (x_hat_0 - z_t) / (1-t) which blows up at t≈1 — the
            # velocity MSE term can dominate the loss by 2500x at init.
            # LUNA's actual setup uses ONLY the pairwise-distance loss,
            # so we follow suit: default vel-MSE weight to 0 when
            # x_0 is predicted; leave it as the configured value
            # (typically 0.1) for explicit velocity prediction.
            default_velocity_mse_weight = (
                0.0 if prediction_target == "x_0" else 0.1
            )
            velocity_mse_weight = float(
                fl_cfg.get("velocity_mse_weight", default_velocity_mse_weight)
            )

            if vn_type == "mlp":
                self.velocity_net = VelocityNetwork(
                    spatial_dim=model_cfg["spatial_dim"],
                    cell_embed_dim=enc_cfg["embed_dim"],
                    section_embed_dim=sec_cfg["embed_dim"],
                    hidden_dims=vn_cfg["hidden_dims"],
                    time_embed_dim=vn_cfg["time_embed_dim"],
                    k_embed_dim=vn_cfg["k_embed_dim"],
                    k_max=k_max,
                    dropout=vn_cfg["dropout"],
                )
            elif vn_type == "cross_attention":
                self.velocity_net = CrossAttentionVelocityNetwork(
                    spatial_dim=model_cfg["spatial_dim"],
                    cell_embed_dim=enc_cfg["embed_dim"],
                    hidden_dim=vn_cfg.get("hidden_dim", 256),
                    n_layers=vn_cfg.get("n_layers", 8),
                    n_heads=vn_cfg.get("n_heads", 16),
                    time_embed_dim=vn_cfg["time_embed_dim"],
                    k_embed_dim=vn_cfg["k_embed_dim"],
                    k_max=k_max,
                    dropout=vn_cfg["dropout"],
                    ff_mult=vn_cfg.get("ff_mult", 4),
                )
            elif vn_type == "luna":
                self.velocity_net = LunaTransformerNet(
                    spatial_dim=model_cfg["spatial_dim"],
                    cell_embed_dim=enc_cfg["embed_dim"],
                    time_embed_dim=vn_cfg["time_embed_dim"],
                    node_dim=vn_cfg.get("node_dim", 256),
                    time_dim_hidden=vn_cfg.get("time_dim_hidden", 128),
                    delta_dim=vn_cfg.get("delta_dim", 64),
                    n_layers=vn_cfg.get("n_layers", 8),
                    n_heads=vn_cfg.get("n_heads", 8),
                    ff_mult=vn_cfg.get("ff_mult", 4),
                    dropout=vn_cfg["dropout"],
                )
            else:
                raise ValueError(
                    f"model.velocity_net.type must be 'mlp', "
                    f"'cross_attention', or 'luna'; got {vn_type!r}"
                )
            self.flow = ConditionalFlowMatching(
                velocity_net=self.velocity_net,
                sigma_min=fl_cfg["sigma_min"],
                pairwise_dist_weight=float(
                    fl_cfg.get("pairwise_dist_weight", 1.0)
                ),
                velocity_mse_weight=velocity_mse_weight,
                translation_equivariant=bool(
                    fl_cfg.get("translation_equivariant", True)
                ),
                pairwise_dist_max_cells=int(
                    fl_cfg.get("pairwise_dist_max_cells", 4096)
                ),
                prediction_target=prediction_target,
            )

        # ---- Graph constructor (always present, parameter-free) -------------
        self.graph_constructor = GraphConstructor(
            backend=config["graph"]["backend"],
            metric=config["graph"]["metric"],
        )

    # ------------------------------------------------------------------ encode

    def encode(
        self,
        gene_expr: torch.Tensor,
        section_gene_expr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-cell and section embeddings.

        Args:
            gene_expr: Gene expression for the batch cells, shape (B, n_genes).
            section_gene_expr: Gene expression for ALL cells in the section
                (used for the section-level embedding). If None, uses gene_expr.

        Returns:
            cell_embed: shape (B, cell_embed_dim).
            section_embed: shape (section_embed_dim,).
        """
        cell_embed = self.encoder(gene_expr)

        if section_gene_expr is not None:
            with torch.no_grad():
                section_cell_embed = self.encoder(section_gene_expr)
            section_embed = self.section_encoder(section_cell_embed)
        else:
            section_embed = self.section_encoder(cell_embed.detach())

        return cell_embed, section_embed

    # ------------------------------------------------------------------- embed

    def embed(
        self,
        gene_expr: torch.Tensor,
        section_gene_expr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute d-dim metric embedding (contrastive mode only).

        Returns:
            (B, embed_dim) metric embedding, L2-normalized if the head is
            configured to normalize.
        """
        if self.objective != "contrastive" or self.metric_head is None:
            raise RuntimeError(
                f"embed() is only available in contrastive mode; "
                f"current objective={self.objective}"
            )
        cell_embed, section_embed = self.encode(gene_expr, section_gene_expr)
        return self.metric_head(cell_embed, section_embed)

    # ------------------------------------------------------------- batched embed

    @torch.no_grad()
    def embed_batched(
        self,
        gene_expr: torch.Tensor,
        section_gene_expr: Optional[torch.Tensor] = None,
        batch_size: int = 16384,
    ) -> torch.Tensor:
        """Compute metric embeddings for many cells with memory-safe batching.

        The section embedding is computed once from `section_gene_expr` (or a
        subsample of `gene_expr` if not provided) and reused for every batch.
        """
        if self.objective != "contrastive" or self.metric_head is None:
            raise RuntimeError("embed_batched() requires contrastive mode")

        self.eval()
        device = gene_expr.device
        n = gene_expr.shape[0]

        # Section embedding once. If the section is large, take a
        # deterministic equidistant subsample so the same cells are picked
        # across repeated inference passes (so the benchmark headline
        # number doesn't wobble between runs of the same checkpoint).
        if section_gene_expr is not None:
            sec_in = self.encoder(section_gene_expr)
        else:
            if n > 8192:
                idx = torch.linspace(0, n - 1, 8192, dtype=torch.long, device=device)
                sec_in = self.encoder(gene_expr[idx])
            else:
                sec_in = self.encoder(gene_expr)
        section_embed = self.section_encoder(sec_in)

        # NOTE: For metric_head_type == "cross_attention", cells in different
        # chunks won't see each other through attention. If the section is
        # larger than `batch_size`, this gives chunk-local attention which is
        # NOT what we want. For LUNA cortex (sections < 7,500 cells) and the
        # default batch_size=16384, this never triggers. We emit a one-time
        # warning if a larger section comes in.
        if self.metric_head_type == "cross_attention" and n > batch_size:
            import warnings
            warnings.warn(
                f"embed_batched: section has {n} cells but batch_size is "
                f"{batch_size}. With cross_attention metric head this gives "
                "chunk-local attention. Increase batch_size to fit the whole "
                "section in one pass, or switch to metric_head.type='mlp'.",
                RuntimeWarning,
            )

        outs: List[torch.Tensor] = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ce = self.encoder(gene_expr[start:end])
            outs.append(self.metric_head(ce, section_embed))
        return torch.cat(outs, dim=0)

    # ----------------------------------------------------- flow-matching path

    @torch.no_grad()
    def generate_embeddings(
        self,
        gene_expr: torch.Tensor,
        k: int = 10,
        section_gene_expr: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        solver: Optional[str] = None,
        batch_size: int = 16384,
    ) -> torch.Tensor:
        """Generate 2-D spatial embeddings via the flow matching ODE.

        Only available when `objective == "flow_matching"`. For the default
        contrastive mode, call `embed_batched()` instead.
        """
        if self.objective != "flow_matching" or self.flow is None:
            raise RuntimeError(
                f"generate_embeddings() requires flow_matching mode; "
                f"current objective={self.objective}"
            )

        self.eval()
        device = gene_expr.device
        n = gene_expr.shape[0]
        flow_cfg = self.config["model"]["flow"]
        n_steps = n_steps or flow_cfg["n_steps"]
        solver = solver or flow_cfg["solver"]

        if section_gene_expr is not None:
            sec_in = self.encoder(section_gene_expr)
        else:
            if n > 8192:
                idx = torch.randperm(n, device=device)[:8192]
                sec_in = self.encoder(gene_expr[idx])
            else:
                sec_in = self.encoder(gene_expr)
        section_embed = self.section_encoder(sec_in)

        outs: List[torch.Tensor] = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ce = self.encoder(gene_expr[start:end])
            kk = torch.full((end - start,), k, dtype=torch.long, device=device)
            outs.append(
                self.flow.sample(
                    cell_embed=ce,
                    section_embed=section_embed,
                    k_target=kk,
                    n_steps=n_steps,
                    solver=solver,
                )
            )
        return torch.cat(outs, dim=0)

    # ----------------------------------------------------- unified generation

    @torch.no_grad()
    def generate_graph(
        self,
        gene_expr: torch.Tensor,
        k: int = 10,
        section_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        section_gene_expr: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        solver: Optional[str] = None,
        n_samples: int = 1,
    ) -> Union[sparse.csr_matrix, Tuple[sparse.csr_matrix, sparse.csr_matrix]]:
        """Generate a spatial kNN graph.

        In contrastive mode this is one forward pass + FAISS kNN. In flow
        matching mode this is ODE integration + FAISS kNN; `n_samples > 1`
        returns a consensus / confidence pair as before.
        """
        self.eval()

        if self.objective == "contrastive":
            emb = self.embed_batched(gene_expr, section_gene_expr)
            adj = self.graph_constructor.build_knn_graph(
                emb, k, section_ids, symmetric=True
            )
            return adj

        # flow_matching
        if n_samples == 1:
            emb = self.generate_embeddings(
                gene_expr, k, section_gene_expr, n_steps, solver
            )
            return self.graph_constructor.build_knn_graph(
                emb, k, section_ids, symmetric=True
            )

        adjs = []
        for _ in range(n_samples):
            emb = self.generate_embeddings(
                gene_expr, k, section_gene_expr, n_steps, solver
            )
            adjs.append(
                self.graph_constructor.build_knn_graph(
                    emb, k, section_ids, symmetric=True
                )
            )
        confidence = sum(adjs) / n_samples
        consensus = confidence.copy()
        consensus.data[consensus.data < 0.5] = 0
        consensus.eliminate_zeros()
        consensus.data[:] = 1.0
        return consensus, confidence

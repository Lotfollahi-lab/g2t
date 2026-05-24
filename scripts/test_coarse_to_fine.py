#!/usr/bin/env python
"""Smoke tests for the coarse-to-fine wrapper.

Run inside the scgg env::

    source /nfs/team361/sb75/.venvs/scgg/bin/activate
    python /nfs/team361/sb75/scgg/scripts/test_coarse_to_fine.py

Exits 0 on pass, non-zero on the first failure. ~5 seconds.

Four properties checked:

  1. Gene clustering partitions real cells into K groups; masked
     cells get cluster_id = -1.
  2. CoarseRegressor outputs (B, K, 2) — one centroid per cluster.
  3. Wrapper preserves the inner Model's DataHolder-in / DataHolder-
     out contract; stashed coarse outputs are readable from the
     returned object.
  4. Gradient flows through BOTH the wrapper's own params (cluster
     module + coarse regressor) AND the inner backbone params under
     a joint position+centroid loss.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _setup_path() -> None:
    here = Path(__file__).resolve()
    scgg_src = here.parent.parent / "src"
    if not scgg_src.exists():
        raise FileNotFoundError(f"scgg/src not found at {scgg_src}")
    sys.path.insert(0, str(scgg_src))


_setup_path()

import torch  # noqa: E402

from utils.data.dataholder import DataHolder  # noqa: E402
from models.coarse_to_fine import (  # noqa: E402
    GeneClusterModule,
    CoarseRegressor,
    CoarseToFineWrapper,
    HierarchicalCoarseToFineWrapper,
)


def test_gene_clustering() -> None:
    """Cluster IDs are in [0, K) for real cells, -1 for masked.
    Tests both 'kmeans' (default) and 'gumbel' modes."""
    torch.manual_seed(0)
    B, N, F = 2, 64, 16
    K = 8

    for mode in ("kmeans", "gumbel"):
        module = GeneClusterModule(
            gene_input_dim=F,
            proj_dim=32,
            n_clusters=K,
            kmeans_n_iters=5,
            cluster_mode=mode,
            gumbel_tau=1.0,
        ).eval()

        node_features = torch.randn(B, N, F)
        node_mask = torch.ones(B, N, dtype=torch.bool)
        node_mask[0, 50:] = False

        cluster_ids, cluster_features = module(node_features, node_mask)

        if not (cluster_ids[0, 50:] == -1).all():
            raise AssertionError(
                f"[{mode}] Masked cells must get cluster_id=-1."
            )
        for b in range(B):
            real = cluster_ids[b][node_mask[b]]
            if not (real >= 0).all() or not (real < K).all():
                raise AssertionError(
                    f"[{mode}] Slice {b}: cluster ids out of [0, {K})."
                )
        if cluster_features.shape != (B, K, 32):
            raise AssertionError(
                f"[{mode}] Cluster features shape mismatch: got "
                f"{tuple(cluster_features.shape)}, expected ({B}, {K}, 32)."
            )
        print(f"[gene-clustering/{mode}]  cluster_ids ∈ [0, {K}) ✓  "
              f"cluster_features {tuple(cluster_features.shape)} ✓")
    print("[gene-clustering]  PASS\n")


def test_gumbel_gradient_flow() -> None:
    """In Gumbel mode, gradients MUST reach the cluster_head (which
    is dead in k-means mode). This is the whole point of the
    soft-clustering option."""
    torch.manual_seed(0)
    B, N, F_in = 2, 32, 16
    K = 8

    module = GeneClusterModule(
        gene_input_dim=F_in,
        proj_dim=32,
        n_clusters=K,
        cluster_mode="gumbel",
        gumbel_tau=1.0,
    ).train()

    node_features = torch.randn(B, N, F_in, requires_grad=True)
    node_mask = torch.ones(B, N, dtype=torch.bool)

    _, cluster_features = module(node_features, node_mask)
    loss = cluster_features.pow(2).mean()
    loss.backward()

    head_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in module.cluster_head.parameters()
    )
    proj_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in module.gene_proj.parameters()
    )
    if not head_has_grad:
        raise AssertionError(
            "Gumbel mode: cluster_head received zero gradient. "
            "Likely the straight-through Gumbel-softmax isn't routing "
            "gradients (check F.gumbel_softmax hard=True call)."
        )
    if not proj_has_grad:
        raise AssertionError(
            "Gumbel mode: gene_proj received zero gradient — wrapper bug."
        )
    print("[gumbel-grad]  cluster_head ✓  gene_proj ✓")
    print("[gumbel-grad]  PASS\n")


def test_coarse_regressor() -> None:
    """Outputs (B, K, 2) — one 2-D centroid per cluster.
    Also verifies the new ``forward_with_embeddings`` returns
    both centroids AND the pre-head hidden state."""
    torch.manual_seed(0)
    B, K, H = 2, 8, 32
    hidden = 64

    regressor = CoarseRegressor(
        cluster_feature_dim=H,
        hidden_dim=hidden,
        n_layers=2,
        n_heads=4,
    ).eval()

    cluster_features = torch.randn(B, K, H)

    # Backward-compat: forward() returns just centroids.
    centroids = regressor(cluster_features)
    if centroids.shape != (B, K, 2):
        raise AssertionError(
            f"Coarse regressor output shape mismatch: {tuple(centroids.shape)}."
        )

    # New: forward_with_embeddings returns (centroids, hidden).
    centroids2, embeddings = regressor.forward_with_embeddings(cluster_features)
    if centroids2.shape != (B, K, 2):
        raise AssertionError(
            f"forward_with_embeddings centroid shape mismatch: "
            f"{tuple(centroids2.shape)}."
        )
    if embeddings.shape != (B, K, hidden):
        raise AssertionError(
            f"forward_with_embeddings hidden shape mismatch: "
            f"{tuple(embeddings.shape)}, expected (B, K, hidden={hidden})."
        )
    # Centroids returned by both paths must agree exactly (same
    # graph + eval mode = same numerics).
    err = (centroids - centroids2).abs().max().item()
    if err > 1e-7:
        raise AssertionError(
            f"forward() and forward_with_embeddings() disagree on "
            f"centroids: max-err {err:.2e}."
        )
    print(f"[coarse-regressor]  forward {tuple(centroids.shape)} ✓ "
          f"forward_with_embeddings hidden {tuple(embeddings.shape)} ✓")
    print("[coarse-regressor]  PASS\n")


def _build_wrapper():
    input_dims = {
        "node_features_dimensions": 16,
        "diffusion_time_dimensions": 1,
    }
    hidden_mlp_dims = {"X": 32, "y": 32, "pos": 8}
    hidden_dims = {
        "dx": 32, "dy": 1, "num_heads": 4,
        "dim_ffX": 32, "dim_ffy": 32, "dd": 16,
        "output_features_to_pos_dims": 4,
    }
    output_dims = {
        "node_features_dimensions": 16,
        "diffusion_time_dimensions": 0,
    }

    class _C2FCfg:
        n_clusters = 8
        kmeans_n_iters = 5
        coarse_hidden_dim = 32
        coarse_n_layers = 1
        coarse_n_heads = 2
        gene_proj_dim = 32

        def get(self, k, default=None):
            return getattr(self, k, default)

    return CoarseToFineWrapper(
        input_dims=input_dims,
        n_layers=2,
        hidden_mlp_dims=hidden_mlp_dims,
        hidden_dims=hidden_dims,
        output_dims=output_dims,
        c2f_cfg=_C2FCfg(),
    )


def test_wrapper_contract() -> None:
    """The wrapper returns a DataHolder with the inner model's
    expected positions shape, plus stashed coarse outputs."""
    torch.manual_seed(0)
    B, N, F = 2, 32, 16

    model = _build_wrapper().eval()
    data = DataHolder(
        node_features=torch.randn(B, N, F),
        positions=torch.randn(B, N, 2),
        diffusion_time=torch.rand(B, 1),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )

    with torch.no_grad():
        out = model(data)

    if not isinstance(out, DataHolder):
        raise AssertionError(f"Expected DataHolder; got {type(out)}.")
    if out.positions.shape != (B, N, 2):
        raise AssertionError(
            f"Output positions shape mismatch: {tuple(out.positions.shape)}."
        )
    # Stashed outputs.
    if not hasattr(out, "_predicted_cluster_centroids"):
        raise AssertionError("Missing _predicted_cluster_centroids on output.")
    if out._predicted_cluster_centroids.shape != (B, model.n_clusters, 2):
        raise AssertionError(
            f"Centroids shape mismatch: {tuple(out._predicted_cluster_centroids.shape)}."
        )
    if not hasattr(out, "_cluster_ids"):
        raise AssertionError("Missing _cluster_ids on output.")
    if out._cluster_ids.shape != (B, N):
        raise AssertionError(
            f"cluster_ids shape mismatch: {tuple(out._cluster_ids.shape)}."
        )
    print(f"[wrapper-contract]  positions {tuple(out.positions.shape)} ✓ "
          f"centroids {tuple(out._predicted_cluster_centroids.shape)} ✓")
    print("[wrapper-contract]  PASS\n")


def test_gradient_flow() -> None:
    """Joint loss (positions + centroid) reaches BOTH wrapper-side
    and inner-side parameters."""
    torch.manual_seed(0)
    B, N, F = 2, 32, 16

    model = _build_wrapper().train()
    data = DataHolder(
        node_features=torch.randn(B, N, F),
        positions=torch.randn(B, N, 2),
        diffusion_time=torch.rand(B, 1),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )
    # Provide true_positions so teacher-forcing path is exercised.
    out = model(data, true_positions=data.positions)

    target_pos = torch.randn_like(out.positions)
    target_centroids = torch.randn_like(out._predicted_cluster_centroids)
    loss = (
        ((out.positions - target_pos) ** 2).mean()
        + ((out._predicted_cluster_centroids - target_centroids) ** 2).mean()
        + (out.node_features ** 2).mean()   # also touch the inner-feature output
    )
    loss.backward()

    coarse_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in list(model.cluster_module.parameters())
        + list(model.coarse_regressor.parameters())
    )
    inner_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.inner.parameters()
    )
    if not coarse_has_grad:
        raise AssertionError(
            "Coarse stage received zero gradient — wrapper not "
            "propagating through cluster_module / coarse_regressor."
        )
    if not inner_has_grad:
        raise AssertionError("Inner backbone received zero gradient.")
    print("[grad]  coarse params ✓  inner params ✓")
    print("[grad]  PASS\n")


def _build_combined_wrapper():
    """Smallest possible HierarchicalCoarseToFineWrapper for testing.
    Mirrors the dim contract from diffusion_model.py."""
    input_dims = {
        "node_features_dimensions": 16,
        "diffusion_time_dimensions": 1,
    }
    hidden_mlp_dims = {"X": 32, "y": 32, "pos": 8}
    hidden_dims = {
        "dx": 32, "dy": 1, "num_heads": 4,
        "dim_ffX": 32, "dim_ffy": 32, "dd": 16,
        "output_features_to_pos_dims": 4,
    }
    output_dims = {
        "node_features_dimensions": 16,
        "diffusion_time_dimensions": 0,
    }

    class _HierCfg:
        n_patches_per_axis = 4
        patch_hidden_dim = 32
        patch_n_layers = 1
        patch_n_heads = 2
        output_dim = 16

        def get(self, k, default=None):
            return getattr(self, k, default)

    class _C2FCfg:
        n_clusters = 8
        kmeans_n_iters = 5
        coarse_hidden_dim = 32
        coarse_n_layers = 1
        coarse_n_heads = 2
        gene_proj_dim = 32

        def get(self, k, default=None):
            return getattr(self, k, default)

    return HierarchicalCoarseToFineWrapper(
        input_dims=input_dims,
        n_layers=2,
        hidden_mlp_dims=hidden_mlp_dims,
        hidden_dims=hidden_dims,
        output_dims=output_dims,
        hier_cfg=_HierCfg(),
        c2f_cfg=_C2FCfg(),
    )


def test_combined_wrapper_contract() -> None:
    """HierarchicalCoarseToFineWrapper preserves the DataHolder
    contract AND stashes the c2f outputs for the loss."""
    torch.manual_seed(0)
    B, N, F = 2, 32, 16

    model = _build_combined_wrapper().eval()
    data = DataHolder(
        node_features=torch.randn(B, N, F),
        positions=torch.randn(B, N, 2),
        diffusion_time=torch.rand(B, 1),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )
    with torch.no_grad():
        out = model(data)

    if not isinstance(out, DataHolder):
        raise AssertionError(f"Expected DataHolder; got {type(out)}.")
    if out.positions.shape != (B, N, 2):
        raise AssertionError(
            f"Combined output positions shape mismatch: "
            f"{tuple(out.positions.shape)}."
        )
    # c2f stash MUST survive composition.
    if not hasattr(out, "_predicted_cluster_centroids"):
        raise AssertionError("Combined wrapper missing _predicted_cluster_centroids.")
    if out._predicted_cluster_centroids.shape != (B, 8, 2):
        raise AssertionError(
            f"Centroids shape mismatch: {tuple(out._predicted_cluster_centroids.shape)}."
        )
    print(f"[combined-contract]  positions {tuple(out.positions.shape)} ✓ "
          f"centroids {tuple(out._predicted_cluster_centroids.shape)} ✓")
    print("[combined-contract]  PASS\n")


def test_combined_gradient_flow() -> None:
    """Gradient must reach all THREE components: patch_module,
    coarse pieces (cluster_module + coarse_regressor), and the
    inner Model."""
    torch.manual_seed(0)
    B, N, F = 2, 32, 16

    model = _build_combined_wrapper().train()
    data = DataHolder(
        node_features=torch.randn(B, N, F),
        positions=torch.randn(B, N, 2),
        diffusion_time=torch.rand(B, 1),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )
    out = model(data, true_positions=data.positions)

    target_pos = torch.randn_like(out.positions)
    target_centroids = torch.randn_like(out._predicted_cluster_centroids)
    loss = (
        ((out.positions - target_pos) ** 2).mean()
        + ((out._predicted_cluster_centroids - target_centroids) ** 2).mean()
        + (out.node_features ** 2).mean()
    )
    loss.backward()

    patch_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.patch_module.parameters()
    )
    coarse_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in list(model.cluster_module.parameters())
        + list(model.coarse_regressor.parameters())
    )
    inner_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.inner.parameters()
    )
    if not patch_has_grad:
        raise AssertionError(
            "patch_module received zero gradient in combined wrapper."
        )
    if not coarse_has_grad:
        raise AssertionError(
            "Coarse stage received zero gradient in combined wrapper."
        )
    if not inner_has_grad:
        raise AssertionError(
            "Inner Model received zero gradient in combined wrapper."
        )
    print("[combined-grad]  patch ✓  coarse ✓  inner ✓")
    print("[combined-grad]  PASS\n")


def main() -> int:
    test_gene_clustering()
    test_gumbel_gradient_flow()
    test_coarse_regressor()
    test_wrapper_contract()
    test_gradient_flow()
    test_combined_wrapper_contract()
    test_combined_gradient_flow()
    print("All coarse-to-fine tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

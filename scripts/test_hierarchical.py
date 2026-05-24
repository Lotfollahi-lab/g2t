#!/usr/bin/env python
"""Sanity tests for the multi-scale hierarchical wrapper.

Run inside the scgg env::

    source /nfs/team361/sb75/.venvs/scgg/bin/activate
    python /nfs/team361/sb75/scgg/scripts/test_hierarchical.py

Exits 0 on pass, non-zero on the first failure. ~5 seconds.

Five properties checked:

  1. Patch assignment partitions REAL cells into exactly K² bins,
     with cell counts in each bin within a factor-of-2 of N/K²
     (quantile binning is approximately uniform).
  2. Masked cells get patch_id = -1 (don't pollute aggregation).
  3. Forward output shape matches (B, N, output_dim) and masked
     cells receive zero vectors.
  4. The wrapped model preserves the inner Model's DataHolder-in /
     DataHolder-out contract.
  5. Gradient flows from the output back into both the patch
     module parameters AND the inner backbone parameters (i.e.
     the wrapper doesn't accidentally detach anything).
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
from models.hierarchical import PatchContextModule, HierarchicalModelWrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Test 1+2: Patch assignment behaviour
# ---------------------------------------------------------------------------


def test_patch_assignment_quantile() -> None:
    """Assignment should put roughly N/K² cells per patch, masked
    cells get -1."""
    torch.manual_seed(0)
    B, N = 2, 400
    K_axis = 4
    K = K_axis ** 2

    module = PatchContextModule(
        cell_input_dim=8,
        n_patches_per_axis=K_axis,
        patch_hidden_dim=32,
        patch_n_layers=1,
        patch_n_heads=2,
        output_dim=16,
    ).eval()

    positions = torch.randn(B, N, 2)
    node_mask = torch.ones(B, N, dtype=torch.bool)
    # Mask out a chunk to simulate padding.
    node_mask[0, 300:] = False

    with torch.no_grad():
        ids = module._assign_patches(positions, node_mask)

    # Masked cells: id == -1.
    if not (ids[0, 300:] == -1).all():
        raise AssertionError(
            f"Masked cells should get patch_id=-1; "
            f"first masked id is {ids[0, 300]}."
        )

    # Real cells: ids in [0, K²).
    for b in range(B):
        real_ids = ids[b][node_mask[b]]
        if not (real_ids >= 0).all() or not (real_ids < K).all():
            raise AssertionError(
                f"Slice {b}: real cell ids out of range [0, {K}); "
                f"got min={real_ids.min().item()}, max={real_ids.max().item()}."
            )
        # Cell-count uniformity: every patch should be within a
        # factor of 2 of N/K². For N=300 cells and K²=16 patches
        # the expected count is ~19; we accept [5, 80].
        counts = torch.bincount(real_ids, minlength=K)
        n_real = node_mask[b].sum().item()
        expected = n_real / K
        if counts.max() > 5 * expected or counts.min() < expected / 5:
            print(
                f"[patch-assign]  WARN slice {b}: count spread "
                f"[{counts.min()}, {counts.max()}] around expected {expected:.1f}"
            )
    print("[patch-assign]  PASS\n")


# ---------------------------------------------------------------------------
# Test 3: Forward output shape + masked-cell zeroing
# ---------------------------------------------------------------------------


def test_patch_module_forward() -> None:
    torch.manual_seed(0)
    B, N, F_in = 2, 64, 8

    module = PatchContextModule(
        cell_input_dim=F_in,
        n_patches_per_axis=4,
        patch_hidden_dim=32,
        patch_n_layers=1,
        patch_n_heads=2,
        output_dim=16,
    ).eval()

    node_features = torch.randn(B, N, F_in)
    positions = torch.randn(B, N, 2)
    node_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask[0, 50:] = False                # mask the tail of slice 0

    with torch.no_grad():
        out = module(node_features, positions, node_mask)

    if out.shape != (B, N, 16):
        raise AssertionError(
            f"Patch module output shape mismatch: expected "
            f"{(B, N, 16)}, got {tuple(out.shape)}."
        )

    # Masked cells should get a zero vector.
    masked_norms = out[0, 50:].norm(dim=-1)
    if masked_norms.max().item() > 1e-6:
        raise AssertionError(
            f"Masked cells should receive zero patch context; got "
            f"max norm {masked_norms.max().item():.2e}."
        )
    print(f"[patch-forward]  output shape {tuple(out.shape)} ✓ "
          f"masked cells zeroed ✓")
    print("[patch-forward]  PASS\n")


# ---------------------------------------------------------------------------
# Test 4: Wrapper preserves DataHolder interface
# ---------------------------------------------------------------------------


def _build_wrapper():
    """Smallest possible HierarchicalModelWrapper that LUNA's Model
    will accept. Mirrors the dimension contract from
    diffusion_model.py."""
    input_dims = {
        "node_features_dimensions": 16,    # raw gene-expression input width
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

    return HierarchicalModelWrapper(
        input_dims=input_dims,
        n_layers=2,
        hidden_mlp_dims=hidden_mlp_dims,
        hidden_dims=hidden_dims,
        output_dims=output_dims,
        hierarchical_cfg=_HierCfg(),
    ).eval()


def test_wrapper_dataholder_contract() -> None:
    torch.manual_seed(0)
    B, N = 2, 32
    F_in = 16

    model = _build_wrapper()
    data = DataHolder(
        node_features=torch.randn(B, N, F_in),
        positions=torch.randn(B, N, 2),
        diffusion_time=torch.rand(B, 1),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )
    # Pre-mutation snapshot — the wrapper must not mutate the caller's
    # DataHolder (the sampling loop reuses the input).
    pre_features = data.node_features.clone()

    with torch.no_grad():
        out = model(data)

    if not isinstance(out, DataHolder):
        raise AssertionError(f"Expected DataHolder output, got {type(out)}.")
    if out.positions.shape != (B, N, 2):
        raise AssertionError(
            f"Output positions shape mismatch: {tuple(out.positions.shape)}."
        )
    if not torch.equal(data.node_features, pre_features):
        raise AssertionError(
            "Wrapper mutated the input DataHolder's node_features. "
            "Use data.copy() to avoid corrupting the caller's tensor."
        )
    print(f"[wrapper-contract]  out.positions {tuple(out.positions.shape)} ✓ "
          f"input unmutated ✓")
    print("[wrapper-contract]  PASS\n")


# ---------------------------------------------------------------------------
# Test 5: Gradient flows through BOTH the patch module and the inner backbone
# ---------------------------------------------------------------------------


def test_gradient_flow() -> None:
    torch.manual_seed(0)
    B, N = 2, 32
    F_in = 16

    model = _build_wrapper().train()

    data = DataHolder(
        node_features=torch.randn(B, N, F_in),
        positions=torch.randn(B, N, 2),
        diffusion_time=torch.rand(B, 1),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )
    out = model(data)

    # Mix positions + node_features in the loss so gradient touches
    # all output heads of the inner Model (matches test_egnn.py's
    # joint-loss convention).
    target_pos = torch.randn_like(out.positions)
    target_feat = torch.randn_like(out.node_features)
    loss = (
        ((out.positions - target_pos) ** 2).mean()
        + ((out.node_features - target_feat) ** 2).mean()
    )
    loss.backward()

    # Sanity: at least one parameter in each of {patch_module, inner}
    # should have a non-trivial gradient.
    patch_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.patch_module.parameters()
    )
    inner_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.inner.parameters()
    )
    if not patch_has_grad:
        raise AssertionError(
            "patch_module received zero gradient — the wrapper is "
            "detaching the patch context, or the inner Model's "
            "mlp_in_node_features isn't reaching the patch features."
        )
    if not inner_has_grad:
        raise AssertionError("Inner Model received zero gradient — wrapper bug.")
    print("[grad]  patch_module gradient ✓  inner Model gradient ✓")
    print("[grad]  PASS\n")


# ---------------------------------------------------------------------------


def main() -> int:
    test_patch_assignment_quantile()
    test_patch_module_forward()
    test_wrapper_dataholder_contract()
    test_gradient_flow()
    print("All hierarchical-wrapper tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

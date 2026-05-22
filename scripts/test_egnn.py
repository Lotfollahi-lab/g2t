#!/usr/bin/env python
"""Sanity tests for the SE(2)-equivariant EGNN backbone.

Run inside the scgg env (where torch, torch_geometric, torch_scatter
are available)::

    source /nfs/team361/sb75/.venvs/scgg/bin/activate
    python /nfs/team361/sb75/scgg/scripts/test_egnn.py

Exits 0 on pass, non-zero on the first failure. Useful before you
queue a long training run on a new EGNN ablation — these checks take
<5 seconds and catch the obvious wiring bugs (shape mismatches,
equivariance violations, gradient flow) cheaply.

Three tests:

  1. EGNNLayer SE(2) equivariance — one forward, check
     ``layer(R·x + b, h, t) = (R·layer_x + b, layer_h)`` on random
     points.
  2. EGNNModel diffusion-time shape handling — feed (B,), (B, 1),
     (B, N, 1), confirm all three give identical predictions (the
     model should auto-broadcast).
  3. EGNNModel gradient flow — call backward on a dummy loss against
     the predicted positions, confirm every learnable parameter
     receives a non-zero gradient.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


def _setup_path() -> None:
    """Make scgg's vendored LUNA tree importable so EGNNModel can
    import its DataHolder + torch-geometric helpers.
    """
    here = Path(__file__).resolve()
    scgg_src = here.parent.parent / "src"
    if not scgg_src.exists():
        raise FileNotFoundError(f"scgg/src not found at {scgg_src}")
    sys.path.insert(0, str(scgg_src))


_setup_path()

import torch                                          # noqa: E402
from torch_geometric.nn import knn_graph              # noqa: E402

from models.egnn import EGNNLayer, EGNNModel          # noqa: E402
from utils.data.dataholder import DataHolder          # noqa: E402


# ---------------------------------------------------------------------------
# Test 1: EGNNLayer SE(2) equivariance
# ---------------------------------------------------------------------------


def test_layer_equivariance(tol: float = 1e-3) -> None:
    """layer(R·x + b, h, t) = (R·layer_x + b, layer_h)."""
    torch.manual_seed(0)
    B, N, F, E, T, K = 2, 64, 16, 16, 8, 10
    layer = EGNNLayer(node_dim=F, edge_dim=E, time_dim=T, update_coords=True).eval()

    h = torch.randn(B * N, F)
    x = torch.randn(B * N, 2)
    t = torch.randn(B * N, T)
    batch_idx = torch.arange(B).repeat_interleave(N)
    mask = torch.ones(B * N, dtype=torch.bool)

    edge_index = knn_graph(x, k=K, batch=batch_idx, loop=False)
    edge_valid = mask[edge_index[0]] & mask[edge_index[1]]
    with torch.no_grad():
        h_base, x_base = layer(h, x, t, edge_index, edge_valid, mask)

    # Random SE(2) transform.
    theta = 0.7
    R = torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)],
    ])
    b = torch.tensor([0.4, -1.2])

    x_t = x @ R.T + b
    edge_index_t = knn_graph(x_t, k=K, batch=batch_idx, loop=False)
    edge_valid_t = mask[edge_index_t[0]] & mask[edge_index_t[1]]
    with torch.no_grad():
        h_trans, x_trans = layer(h, x_t, t, edge_index_t, edge_valid_t, mask)

    # h is SE(2)-invariant; x is SE(2)-equivariant.
    expected_x = x_base @ R.T + b
    h_err = (h_trans - h_base).abs().max().item()
    x_err = (x_trans - expected_x).abs().max().item()
    print(f"[layer]  h invariance max-err: {h_err:.2e}  (tol={tol})")
    print(f"[layer]  x equivariance max-err: {x_err:.2e}  (tol={tol})")
    if h_err > tol:
        raise AssertionError(f"EGNNLayer h is NOT SE(2)-invariant (err={h_err})")
    if x_err > tol:
        raise AssertionError(f"EGNNLayer x is NOT SE(2)-equivariant (err={x_err})")
    print("[layer]  PASS\n")


# ---------------------------------------------------------------------------
# Test 2: EGNNModel handles all diffusion_time shapes
# ---------------------------------------------------------------------------


def _build_egnn_model(time_in: int = 1) -> EGNNModel:
    """Construct an EGNNModel with small dims for fast tests."""
    input_dims = {
        "node_features_dimensions": 16,
        "diffusion_time_dimensions": time_in,
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

    class _EgnnCfg:
        node_dim, edge_dim, time_dim, knn_k = 32, 16, 8, 10
        update_coords = True
        def get(self, k, default=None):
            return getattr(self, k, default)

    return EGNNModel(
        input_dims=input_dims,
        n_layers=2,
        hidden_mlp_dims=hidden_mlp_dims,
        hidden_dims=hidden_dims,
        output_dims=output_dims,
        egnn_cfg=_EgnnCfg(),
    ).eval()


def test_time_shape_broadcast(tol: float = 1e-5) -> None:
    """The noise model emits diffusion_time as (B, 1). The model must
    handle that AND (B,) AND (B, N, 1) identically — all three should
    produce the same predicted positions.
    """
    torch.manual_seed(0)
    B, N = 3, 50
    F = 16

    node_features = torch.randn(B, N, F)
    positions = torch.randn(B, N, 2)
    node_mask = torch.ones(B, N, dtype=torch.bool)

    # Three equivalent representations of one t per slice.
    t_per_slice = torch.rand(B, 1)
    t_shapes = {
        "(B,)": t_per_slice.squeeze(-1),
        "(B, 1)": t_per_slice,
        "(B, N, 1)": t_per_slice.unsqueeze(1).expand(B, N, 1).contiguous(),
    }

    model = _build_egnn_model(time_in=1)

    outputs = {}
    for tag, dt in t_shapes.items():
        data = DataHolder(
            node_features=node_features.clone(),
            positions=positions.clone(),
            node_mask=node_mask,
            diffusion_time=dt,
        )
        with torch.no_grad():
            out = model(data)
        outputs[tag] = out.positions

    # All three should match.
    base = outputs["(B, 1)"]
    for tag, x in outputs.items():
        err = (x - base).abs().max().item()
        print(f"[time-shape]  {tag:>10s}  max-err vs (B, 1): {err:.2e}")
        if err > tol:
            raise AssertionError(
                f"EGNNModel produced different output for diffusion_time "
                f"shape {tag} vs (B, 1): max-err {err}"
            )
    print("[time-shape]  PASS\n")


# ---------------------------------------------------------------------------
# Test 3: EGNNModel gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flow() -> None:
    """Every learnable parameter should receive a non-zero gradient
    when the loss touches BOTH outputs of the model (positions AND
    node features). This is a smoke test for wiring — the diffusion
    training loss is positions-only, which *correctly* gives zero
    gradient on output-side `h` branches (layers[-1].node_mlp and
    out_node_proj feed only into out.node_features, which the
    position loss ignores). Those branches are still part of the
    architecture for future auxiliary losses on cell features, so we
    want them trainable in principle — hence the joint loss here.

    Catches actual wiring bugs (e.g., the zero-init coord_mlp issue
    that blocked gradient flow through every upstream parameter).
    """
    torch.manual_seed(0)
    B, N, F = 2, 32, 16

    model = _build_egnn_model(time_in=1).train()

    data = DataHolder(
        node_features=torch.randn(B, N, F),
        positions=torch.randn(B, N, 2),
        node_mask=torch.ones(B, N, dtype=torch.bool),
        diffusion_time=torch.rand(B, 1),
    )
    out = model(data)

    # Joint loss: positions + node_features. The diffusion loss only
    # uses positions in real training; we add the node_features term
    # here solely to exercise the h-output branches in the gradient
    # check. Both halves are simple MSEs against random targets — the
    # values are meaningless, the only thing being tested is that
    # gradient propagates everywhere.
    target_pos = torch.randn_like(out.positions)
    target_feat = torch.randn_like(out.node_features)
    loss = (
        ((out.positions - target_pos) ** 2).mean()
        + ((out.node_features - target_feat) ** 2).mean()
    )
    loss.backward()

    zero_names = []
    n_total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_total += 1
        # Use absolute-sum > 0 as the "any gradient flowed" predicate.
        # The previous zero-init coord_mlp bug specifically produced
        # exact-zero gradients on a long list of parameters; this is
        # the cheapest check that catches it.
        if p.grad is None or p.grad.abs().sum().item() == 0.0:
            zero_names.append(name)
    if zero_names:
        print(f"[grad]  {n_total - len(zero_names)}/{n_total} params received gradient")
        print("[grad]  parameters with ZERO gradient (likely wiring bug):")
        for name in zero_names:
            print(f"          {name}")
        raise AssertionError(
            f"{len(zero_names)}/{n_total} parameters got NO gradient under "
            f"a JOINT positions + node_features loss. This means some "
            f"learnable branch is completely disconnected from BOTH "
            f"outputs — a real wiring bug, not just a position-loss "
            f"dead-end. Likely culprits:\n"
            f"  - a coord_mlp / node_mlp last layer that's strictly "
            f"zero-init (use xavier_uniform_(gain=1e-3) instead)\n"
            f"  - a layer's output not being passed forward / written "
            f"to the returned DataHolder\n"
            f"  - a module created but never called in forward()"
        )
    print(f"[grad]  {n_total}/{n_total} params received gradient")
    print("[grad]  PASS\n")


# ---------------------------------------------------------------------------


def main() -> int:
    test_layer_equivariance()
    test_time_shape_broadcast()
    test_gradient_flow()
    print("All EGNN tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

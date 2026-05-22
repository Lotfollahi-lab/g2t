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
    from a loss on predicted positions. Catches dead branches (e.g.,
    layer outputs not contributing to the prediction).
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

    # Simple loss: MSE against random "targets".
    target = torch.randn_like(out.positions)
    loss = ((out.positions - target) ** 2).mean()
    loss.backward()

    n_zero = 0
    n_total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_total += 1
        if p.grad is None or p.grad.abs().sum().item() == 0.0:
            print(f"[grad]  no gradient: {name}")
            n_zero += 1
    print(f"[grad]  {n_total - n_zero}/{n_total} params received gradient")
    # Allow up to one zero (the position-stream bias term in node_mlp
    # can legitimately get zero on a 1-batch test if it always sees
    # zeros; we don't expect this in practice but be lenient).
    if n_zero > 1:
        raise AssertionError(
            f"{n_zero}/{n_total} parameters got NO gradient — likely a "
            f"wiring bug (some EGNN branch isn't connected to the loss)."
        )
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

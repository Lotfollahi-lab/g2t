#!/usr/bin/env python
"""Lite smoke test for EDM and k-NN graph output heads.

Lighter than ``test_edm_and_knn_graph.py``: does NOT import the LUNA
Model, so it avoids the scanpy/torch_geometric dependency chain.
Suitable for laptop validation; the full Model-based test belongs on
the GPU box where the env is complete.

Strategy: build a *minimal* nn.Module stub that emulates the
DataHolder-in / DataHolder-out interface of the inner backbone, and
exercise the EDM / KNNGraph wrappers against it.
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
import torch.nn as nn  # noqa: E402

# Bypass models/__init__.py and the heavy load chain by reading edm_head /
# knn_graph_head directly. Both modules only depend on torch + the
# DataHolder dataclass, which itself only needs the load helpers if you
# call .mask() — we construct DataHolders manually here.

from utils.data.dataholder import DataHolder  # noqa: E402
from models.edm_head import EDMOutputWrapper, _classical_mds_2d, _procrustes_align  # noqa: E402
from models.knn_graph_head import KNNGraphOutputWrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Stub inner model — mimics Model's DataHolder-in / DataHolder-out
# interface with a trivial linear projection.
# ---------------------------------------------------------------------------


class _StubInner(nn.Module):
    def __init__(self, in_features: int, out_features: int = 4):
        super().__init__()
        self.feat_proj = nn.Linear(in_features, out_features)
        self.pos_proj = nn.Linear(in_features, 2)

    def forward(self, data: DataHolder, **kwargs):  # ignores kwargs
        out_features = self.feat_proj(data.node_features)
        out_positions = self.pos_proj(data.node_features)
        # Apply node_mask to mimic Model's behaviour.
        m = data.node_mask.unsqueeze(-1).to(out_features.dtype)
        out_features = out_features * m
        out_positions = out_positions * m
        # Mean-centre positions (matches Model.forward).
        out_positions = out_positions - out_positions.mean(dim=1, keepdim=True)
        out_positions = out_positions * m
        return DataHolder(
            node_features=out_features,
            positions=out_positions,
            diffusion_time=data.diffusion_time,
            cell_class=data.cell_class,
            cell_ID=data.cell_ID,
            t_int=data.t_int,
            t=data.t,
            node_mask=data.node_mask,
        )


def _build_data(B: int = 1, N: int = 30, gene_dim: int = 16) -> DataHolder:
    torch.manual_seed(0)
    return DataHolder(
        node_features=torch.randn(B, N, gene_dim),
        positions=torch.randn(B, N, 2) * 0.3,
        diffusion_time=torch.rand(B, 1),
        cell_class=torch.zeros(B, N, dtype=torch.long),
        cell_ID=torch.arange(N).repeat(B, 1),
        t_int=torch.zeros(B, 1, dtype=torch.long),
        t=torch.rand(B, 1),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_classical_mds_recovers_pairwise_distances() -> None:
    """Sanity check on the differentiable MDS helper itself."""
    torch.manual_seed(0)
    n = 20
    # True positions in 2D.
    X = torch.randn(n, 2) * 2.0
    X = X - X.mean(dim=0, keepdim=True)
    # Squared-distance matrix.
    diff = X.unsqueeze(0) - X.unsqueeze(1)
    D = (diff * diff).sum(dim=-1)
    # Run MDS.
    X_mds = _classical_mds_2d(D)
    # Distances should match (the layout itself is in eigenframe).
    diff_mds = X_mds.unsqueeze(0) - X_mds.unsqueeze(1)
    D_mds = (diff_mds * diff_mds).sum(dim=-1)
    err = (D - D_mds).abs().max().item()
    assert err < 1e-4, f"MDS recovered distances differ by {err}"


def test_procrustes_finds_inverse_rotation() -> None:
    torch.manual_seed(0)
    n = 30
    X = torch.randn(n, 2)
    X = X - X.mean(dim=0, keepdim=True)
    # Rotate by 30 degrees.
    theta = 0.5
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    R = torch.tensor([[c, -s], [s, c]])
    X_rot = X @ R.T
    # Procrustes-align X_rot back to X.
    X_aligned = _procrustes_align(X_rot, X)
    err = (X_aligned - X).abs().max().item()
    assert err < 1e-4, f"Procrustes alignment error: {err}"


def test_edm_forward_shapes_and_finite() -> None:
    B, N = 1, 25
    inner = _StubInner(in_features=16, out_features=4)
    wrap = EDMOutputWrapper(inner_model=inner, inner_out_dim=4, embed_dim=6, mds_align=True)
    data = _build_data(B=B, N=N)
    pred = wrap(data)
    assert pred.positions.shape == (B, N, 2)
    assert hasattr(pred, "edm_D")
    assert pred.edm_D.shape == (B, N, N)
    assert pred.edm_h.shape == (B, N, 6)
    assert torch.isfinite(pred.positions).all(), "non-finite positions"
    assert torch.isfinite(pred.edm_D).all(), "non-finite edm_D"
    assert torch.allclose(pred.edm_D, pred.edm_D.transpose(-1, -2), atol=1e-5)
    assert (pred.edm_D >= -1e-6).all()


def test_edm_gradient_flows_through_projector_and_inner() -> None:
    """EDM-only loss path: gradient should reach both the projector
    (wrapper) and the inner backbone."""
    B, N = 1, 24
    inner = _StubInner(in_features=16, out_features=4)
    wrap = EDMOutputWrapper(inner_model=inner, inner_out_dim=4, embed_dim=8, mds_align=False)
    data = _build_data(B=B, N=N)
    pred = wrap(data)
    # Hand-build the squared-distance loss (mirrors LossFunction's
    # _compute_edm_distance_mse).
    diff_t = data.positions.unsqueeze(2) - data.positions.unsqueeze(1)
    D_true = (diff_t * diff_t).sum(dim=-1)
    loss = ((pred.edm_D - D_true) ** 2).mean()
    loss.backward()
    proj_grad = sum(p.grad.abs().sum().item() for p in wrap.projector.parameters() if p.grad is not None)
    inner_grad = sum(p.grad.abs().sum().item() for p in inner.parameters() if p.grad is not None)
    assert proj_grad > 0, "projector got zero gradient"
    assert inner_grad > 0, "inner got zero gradient"


def test_edm_zero_ref_no_crash() -> None:
    """When framework=regression zeroes positions, Procrustes ref is
    zero — the guard should kick in and use the raw MDS frame."""
    B, N = 1, 20
    inner = _StubInner(in_features=16, out_features=4)
    wrap = EDMOutputWrapper(inner_model=inner, inner_out_dim=4, embed_dim=6, mds_align=True)
    data = _build_data(B=B, N=N)
    # Zero out positions (mimics regression framework).
    data.positions = torch.zeros_like(data.positions)
    pred = wrap(data)
    assert torch.isfinite(pred.positions).all(), "non-finite positions on zero ref"


def test_knn_graph_forward() -> None:
    B, N = 1, 30
    inner = _StubInner(in_features=16, out_features=4)
    wrap = KNNGraphOutputWrapper(
        inner_model=inner, inner_out_dim=4, embed_dim=8,
        spectral_layout=True, k_for_layout=5,
    )
    data = _build_data(B=B, N=N)
    pred = wrap(data)
    assert pred.positions.shape == (B, N, 2)
    assert hasattr(pred, "knn_logits")
    assert pred.knn_logits.shape == (B, N, N)
    assert torch.isfinite(pred.positions).all()


def test_knn_graph_gradient() -> None:
    B, N = 1, 28
    inner = _StubInner(in_features=16, out_features=4)
    wrap = KNNGraphOutputWrapper(
        inner_model=inner, inner_out_dim=4, embed_dim=8,
        spectral_layout=False, k_for_layout=5,
    )
    data = _build_data(B=B, N=N)
    pred = wrap(data)

    # Hand-build BCE on top-k true-spatial neighbours (mirrors
    # LossFunction._compute_knn_graph_loss).
    logits = pred.knn_logits[0]                    # (N, N)
    with torch.no_grad():
        d_true = torch.cdist(data.positions[0], data.positions[0], p=2)
        d_true.fill_diagonal_(float("inf"))
        _, pos_idx = torch.topk(d_true, k=4, largest=False, dim=1)
    pos_logits = torch.gather(logits, 1, pos_idx)
    targets = torch.ones_like(pos_logits)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_logits, targets)
    loss.backward()
    proj_grad = sum(p.grad.abs().sum().item() for p in wrap.projector.parameters() if p.grad is not None)
    inner_grad = sum(p.grad.abs().sum().item() for p in inner.parameters() if p.grad is not None)
    assert proj_grad > 0
    assert inner_grad > 0


def test_c2f_marker_propagation() -> None:
    """If the inner model has _c2f_uses_true_positions, the EDM/kNN
    wrapper should also expose it so the LightningModule's c2f
    detection survives the wrapping."""
    inner = _StubInner(in_features=16, out_features=4)
    inner._c2f_uses_true_positions = True
    wrap_edm = EDMOutputWrapper(inner_model=inner, inner_out_dim=4, embed_dim=4)
    assert hasattr(wrap_edm, "_c2f_uses_true_positions")
    wrap_knn = KNNGraphOutputWrapper(inner_model=inner, inner_out_dim=4, embed_dim=4)
    assert hasattr(wrap_knn, "_c2f_uses_true_positions")


def test_stubs_raise() -> None:
    from models.vn_transformer import VNTransformerBackbone
    from utils.diffusion_model.diffusion.energy_predictor import EnergyPredictor
    try:
        _ = VNTransformerBackbone()
        raise AssertionError("VN stub did not raise")
    except NotImplementedError:
        pass
    try:
        _ = EnergyPredictor()
        raise AssertionError("Energy stub did not raise")
    except NotImplementedError:
        pass


def main() -> int:
    tests = [
        ("classical MDS recovers pairwise distances", test_classical_mds_recovers_pairwise_distances),
        ("Procrustes finds inverse rotation",          test_procrustes_finds_inverse_rotation),
        ("EDM forward shapes + finiteness",            test_edm_forward_shapes_and_finite),
        ("EDM gradient flow",                          test_edm_gradient_flows_through_projector_and_inner),
        ("EDM zero-ref guard (regression framework)",  test_edm_zero_ref_no_crash),
        ("kNN graph forward",                          test_knn_graph_forward),
        ("kNN graph gradient flow",                    test_knn_graph_gradient),
        ("c2f marker propagation through wrappers",    test_c2f_marker_propagation),
        ("stubs raise NotImplementedError",            test_stubs_raise),
    ]
    n_pass = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            n_pass += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    print(f"\n{n_pass}/{len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

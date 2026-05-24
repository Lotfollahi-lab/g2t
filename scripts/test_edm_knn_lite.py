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
    (wrapper) and the inner backbone. Mirrors the EUCLIDEAN-distance
    MSE formulation in LossFunction._compute_edm_distance_mse."""
    B, N = 1, 24
    inner = _StubInner(in_features=16, out_features=4)
    wrap = EDMOutputWrapper(inner_model=inner, inner_out_dim=4, embed_dim=8, mds_align=False)
    data = _build_data(B=B, N=N)
    pred = wrap(data)
    # Hand-build the Euclidean-distance loss (matches LossFunction):
    # MSE on sqrt(D) — same scale as LUNA's pairwise_distance_mse.
    d_true = torch.cdist(data.positions[0], data.positions[0], p=2)
    d_pred = (pred.edm_D[0] + 1e-8).sqrt()
    loss = ((d_pred - d_true) ** 2).mean()
    loss.backward()
    proj_grad = sum(p.grad.abs().sum().item() for p in wrap.projector.parameters() if p.grad is not None)
    inner_grad = sum(p.grad.abs().sum().item() for p in inner.parameters() if p.grad is not None)
    assert proj_grad > 0, "projector got zero gradient"
    assert inner_grad > 0, "inner got zero gradient"


def test_edm_multi_step_stable() -> None:
    """Regression check for the NaN-divergence we saw at training step ~2
    with the v1 (MSE on SQUARED distances) of this loss. Now uses MSE
    on Euclidean distances which puts the loss on the same scale as
    LUNA's pairwise_distance_mse — known-stable with lr=5e-4.

    Runs 30 optimizer steps with the EUCLIDEAN-distance EDM loss.
    Tests with positions in a realistic raw-coordinate scale (×50)
    so squared distances would be ~2500× larger than Euclidean — the
    failure regime from the GPU run.
    """
    B, N = 1, 32
    inner = _StubInner(in_features=16, out_features=4)
    wrap = EDMOutputWrapper(inner_model=inner, inner_out_dim=4, embed_dim=8, mds_align=False)
    data = _build_data(B=B, N=N)
    # Scale positions to "raw cortex" magnitude — exposes the
    # squared-vs-euclidean scale issue. With v1 squared-MSE this would
    # explode immediately.
    data.positions = data.positions * 50.0
    optimizer = torch.optim.AdamW(wrap.parameters(), lr=5e-4)
    eps = 1e-8

    for step in range(30):
        optimizer.zero_grad()
        pred = wrap(data)
        d_true = torch.cdist(data.positions[0], data.positions[0], p=2)
        d_pred = (pred.edm_D[0] + eps).sqrt()
        loss = ((d_pred - d_true) ** 2).mean()
        assert torch.isfinite(loss), f"Loss went non-finite at step {step}: {loss}"
        loss.backward()
        optimizer.step()
        for p in wrap.parameters():
            assert torch.isfinite(p).all(), f"Parameter went NaN at step {step}"


def test_edm_mds_align_with_full_loss_chain_stable() -> None:
    """Regression for the eigh-backward NaN poisoning bug.

    With ``mds_align=True``, the wrapper sets pred.positions to the MDS
    eigendecomposition output. If autograd traces backward through
    that path (e.g. via cluster_balance's
    ``pred.positions.sum() * 0.0`` fallback), the eigh backward
    formula's ``1/(λ_i − λ_j)`` term produces inf for near-degenerate
    eigenvalues, and ``0 × inf = NaN`` poisons every parameter's
    gradient.

    This test simulates that exact chain: full EDM forward with
    mds_align=True, then a loss that includes BOTH the EDM term AND
    a ``pred.positions.sum() * 0.0`` term (mimicking
    cluster_balance's graph-attached zero). All parameters must stay
    finite for 30 steps.

    With the detach() fix in EDMOutputWrapper.forward, the MDS path
    is removed from the gradient graph — the ``0 × ...`` term sees
    a non-tracked tensor and contributes ZERO gradient.
    """
    B, N = 1, 32
    inner = _StubInner(in_features=16, out_features=4)
    wrap = EDMOutputWrapper(inner_model=inner, inner_out_dim=4, embed_dim=8, mds_align=True)
    data = _build_data(B=B, N=N)
    data.positions = data.positions * 50.0
    optimizer = torch.optim.AdamW(wrap.parameters(), lr=5e-4)
    eps = 1e-8

    for step in range(30):
        optimizer.zero_grad()
        pred = wrap(data)
        d_true = torch.cdist(data.positions[0], data.positions[0], p=2)
        d_pred = (pred.edm_D[0] + eps).sqrt()
        edm_loss = ((d_pred - d_true) ** 2).mean()
        # Mimic the cluster_balance fallback: a graph-attached zero
        # through pred.positions. Without the detach fix this path
        # injects NaN via eigh backward.
        fallback_zero = pred.positions[0].sum() * 0.0
        loss = edm_loss + fallback_zero
        assert torch.isfinite(loss), f"Loss non-finite at step {step}: {loss}"
        loss.backward()
        # Critical assertion: gradients should also be finite. The bug
        # produced finite loss but NaN gradients via eigh backward.
        for name, p in wrap.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f"NaN gradient at step {step} on parameter {name}"
                )
        optimizer.step()
        for name, p in wrap.named_parameters():
            assert torch.isfinite(p).all(), (
                f"Parameter {name} went NaN at step {step}"
            )


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


def test_knn_graph_multi_step_stable() -> None:
    """Regression check for the NaN-divergence we saw at training step ~9
    with the v1 (no normalisation, no temperature) of this module.

    Runs 30 optimizer steps with both positive AND negative edges
    (the negative-edge BCE was the unbounded direction). Asserts
    that no parameter goes NaN and the loss stays finite. The unit-
    norm + temperature recipe in the wrapper should make this trivially
    pass; if anyone removes it, this test catches the regression before
    GPU time is wasted.
    """
    B, N = 1, 32
    inner = _StubInner(in_features=16, out_features=4)
    wrap = KNNGraphOutputWrapper(
        inner_model=inner, inner_out_dim=4, embed_dim=8,
        spectral_layout=False, k_for_layout=5, temperature=0.1,
    )
    data = _build_data(B=B, N=N)
    # `wrap.parameters()` recursively includes inner because
    # `wrap.inner_model = inner`, so passing only wrap.parameters()
    # covers both modules without the duplicate-parameter warning.
    optimizer = torch.optim.AdamW(wrap.parameters(), lr=5e-4)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    for step in range(30):
        optimizer.zero_grad()
        pred = wrap(data)
        logits = pred.knn_logits[0]
        with torch.no_grad():
            d_true = torch.cdist(data.positions[0], data.positions[0], p=2)
            d_true_self = d_true.clone()
            d_true_self.fill_diagonal_(float("inf"))
            _, pos_idx = torch.topk(d_true_self, k=4, largest=False, dim=1)
            # Random negatives.
            neg_idx = torch.randint(0, N, (N, 8))
        pos_logits = torch.gather(logits, 1, pos_idx)
        neg_logits = torch.gather(logits, 1, neg_idx)
        pos_loss = bce(pos_logits, torch.ones_like(pos_logits))
        neg_loss = bce(neg_logits, torch.zeros_like(neg_logits))
        loss = 0.5 * (pos_loss + neg_loss)
        assert torch.isfinite(loss), f"Loss went non-finite at step {step}: {loss}"
        loss.backward()
        optimizer.step()
        # Check parameters stayed finite.
        for p in wrap.parameters():
            assert torch.isfinite(p).all(), f"Parameter went NaN at step {step}"


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
        ("EDM 30-step stability with raw-scale positions", test_edm_multi_step_stable),
        ("EDM mds_align + cluster_balance fallback (eigh-backward NaN regression)",
                                                       test_edm_mds_align_with_full_loss_chain_stable),
        ("EDM zero-ref guard (regression framework)",  test_edm_zero_ref_no_crash),
        ("kNN graph forward",                          test_knn_graph_forward),
        ("kNN graph gradient flow",                    test_knn_graph_gradient),
        ("kNN graph 30-step stability (no NaN)",       test_knn_graph_multi_step_stable),
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

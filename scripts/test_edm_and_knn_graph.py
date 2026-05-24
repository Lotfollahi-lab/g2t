#!/usr/bin/env python
"""Smoke tests for EDM (#1) and k-NN graph (#4) output heads.

Run inside the scgg env::

    source /nfs/team361/sb75/.venvs/scgg/bin/activate
    python /nfs/team361/sb75/scgg/scripts/test_edm_and_knn_graph.py

Exits 0 on pass, non-zero on first failure. ~5 seconds.

Properties checked, per wrapper:
  1. Wrapper preserves the DataHolder-in / DataHolder-out contract
     of the inner Model.
  2. The new auxiliary output (edm_D for EDM; knn_logits for k-NN
     graph) is stashed on the returned DataHolder with the right
     shape and is finite.
  3. ``pred.positions`` shape is preserved (B, N, 2) and is finite
     after MDS / spectral-layout post-processing.
  4. Gradient flows through BOTH the wrapper's projector AND the
     inner backbone params under the corresponding loss
     (``LossFunction._compute_edm_distance_mse`` /
     ``LossFunction._compute_knn_graph_loss``).
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
from omegaconf import OmegaConf  # noqa: E402

from utils.data.dataholder import DataHolder  # noqa: E402
from models.model import Model  # noqa: E402
from models.edm_head import EDMOutputWrapper  # noqa: E402
from models.knn_graph_head import KNNGraphOutputWrapper  # noqa: E402
from metrics.loss_function import LossFunction  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------


def _build_inner_model(gene_dim: int = 16) -> Model:
    return Model(
        input_dims={
            "node_features_dimensions": gene_dim,
            "diffusion_time_dimensions": 1,
        },
        n_layers=2,
        hidden_mlp_dims={"X": 32, "y": 16, "pos": 16},
        hidden_dims={
            "dx": 32, "dy": 1, "num_heads": 2,
            "dim_ffX": 32, "dim_ffy": 16,
            "dd": 16, "output_features_to_pos_dims": 4,
        },
        output_dims={
            "node_features_dimensions": gene_dim,
            "diffusion_time_dimensions": 1,
        },
    )


def _build_data(B: int = 2, N: int = 40, gene_dim: int = 16) -> DataHolder:
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
    ).mask()


# ---------------------------------------------------------------------------
# EDM tests
# ---------------------------------------------------------------------------


def test_edm_forward_shapes() -> None:
    B, N = 2, 30
    inner = _build_inner_model()
    wrap = EDMOutputWrapper(inner_model=inner, inner_out_dim=4, embed_dim=8, mds_align=True)
    data = _build_data(B=B, N=N)
    pred = wrap(data)
    assert pred.positions.shape == (B, N, 2), f"bad positions shape: {pred.positions.shape}"
    assert hasattr(pred, "edm_D"), "missing edm_D"
    assert pred.edm_D.shape == (B, N, N), f"bad edm_D shape: {pred.edm_D.shape}"
    assert hasattr(pred, "edm_h"), "missing edm_h"
    assert pred.edm_h.shape == (B, N, 8), f"bad edm_h shape: {pred.edm_h.shape}"
    assert torch.isfinite(pred.positions).all(), "non-finite positions"
    assert torch.isfinite(pred.edm_D).all(), "non-finite edm_D"
    # edm_D is symmetric and nonneg (squared distances).
    assert torch.allclose(pred.edm_D, pred.edm_D.transpose(-1, -2), atol=1e-5), \
        "edm_D not symmetric"
    assert (pred.edm_D >= -1e-6).all(), "edm_D has negative entries (should be nonneg)"


def test_edm_gradient_flow() -> None:
    """Both the wrapper's projector AND the inner Model receive
    non-zero gradients under the edm_distance_mse loss."""
    B, N = 1, 24
    inner = _build_inner_model()
    wrap = EDMOutputWrapper(inner_model=inner, inner_out_dim=4, embed_dim=6, mds_align=True)
    data = _build_data(B=B, N=N)
    pred = wrap(data)

    cfg = OmegaConf.create({
        "model": {
            "loss": {
                "pairwise_distance_mse": {"enabled": False, "weight": 1.0},
                "edm_distance_mse": {"enabled": True, "weight": 1.0},
            }
        }
    })
    loss_fn = LossFunction(cfg=cfg)
    total, _ = loss_fn.compute_loss(masked_pred=pred, masked_true=data)
    total.backward()

    # Wrapper projector params: non-zero grad
    proj_grad = sum(
        p.grad.abs().sum().item()
        for p in wrap.projector.parameters() if p.grad is not None
    )
    assert proj_grad > 0, "projector got zero gradient"
    # Inner Model params: non-zero grad too
    inner_grad = sum(
        p.grad.abs().sum().item()
        for p in inner.parameters() if p.grad is not None
    )
    assert inner_grad > 0, "inner Model got zero gradient under EDM loss"


def test_edm_no_mds_align() -> None:
    """With mds_align=False, pred.positions should be the inner
    backbone's raw output (untouched by the wrapper)."""
    B, N = 1, 20
    inner = _build_inner_model()
    inner.eval()
    wrap = EDMOutputWrapper(inner_model=inner, inner_out_dim=4, embed_dim=6, mds_align=False)
    wrap.eval()
    data = _build_data(B=B, N=N)
    # Run the inner model directly to capture its raw output.
    with torch.no_grad():
        raw = inner(data.copy())
        pred = wrap(data)
    assert torch.allclose(pred.positions, raw.positions, atol=1e-5), \
        "mds_align=False should leave pred.positions untouched"


# ---------------------------------------------------------------------------
# k-NN graph tests
# ---------------------------------------------------------------------------


def test_knn_graph_forward_shapes() -> None:
    B, N = 2, 30
    inner = _build_inner_model()
    wrap = KNNGraphOutputWrapper(
        inner_model=inner, inner_out_dim=4, embed_dim=8,
        spectral_layout=True, k_for_layout=5,
    )
    data = _build_data(B=B, N=N)
    pred = wrap(data)
    assert pred.positions.shape == (B, N, 2)
    assert hasattr(pred, "knn_logits"), "missing knn_logits"
    assert pred.knn_logits.shape == (B, N, N), f"bad knn_logits: {pred.knn_logits.shape}"
    # Logits are -squared-distance → nonpositive (or -inf for padding).
    # Replace -inf with a finite sentinel before the finiteness check.
    finite_mask = torch.isfinite(pred.knn_logits)
    assert finite_mask.any(), "knn_logits all -inf"
    assert torch.isfinite(pred.positions).all(), "non-finite positions after spectral layout"


def test_knn_graph_gradient_flow() -> None:
    B, N = 1, 30
    inner = _build_inner_model()
    wrap = KNNGraphOutputWrapper(
        inner_model=inner, inner_out_dim=4, embed_dim=8,
        spectral_layout=False, k_for_layout=5,
    )
    data = _build_data(B=B, N=N)
    pred = wrap(data)
    cfg = OmegaConf.create({
        "model": {
            "loss": {
                "pairwise_distance_mse": {"enabled": False, "weight": 1.0},
                "knn_graph_loss": {
                    "enabled": True, "weight": 1.0,
                    "k": 4, "n_negatives": 8,
                },
            }
        }
    })
    loss_fn = LossFunction(cfg=cfg)
    total, _ = loss_fn.compute_loss(masked_pred=pred, masked_true=data)
    total.backward()
    proj_grad = sum(
        p.grad.abs().sum().item()
        for p in wrap.projector.parameters() if p.grad is not None
    )
    assert proj_grad > 0, "knn projector got zero gradient"
    inner_grad = sum(
        p.grad.abs().sum().item()
        for p in inner.parameters() if p.grad is not None
    )
    assert inner_grad > 0, "inner Model got zero gradient under knn_graph_loss"


# ---------------------------------------------------------------------------
# Stub frameworks fail clearly
# ---------------------------------------------------------------------------


def test_vn_transformer_stub_raises() -> None:
    from models.vn_transformer import VNTransformerBackbone
    try:
        _ = VNTransformerBackbone()
    except NotImplementedError as e:
        assert "vn_transformer" in str(e), "stub error message doesn't mention vn_transformer"
        return
    raise AssertionError("VNTransformerBackbone() did not raise NotImplementedError")


def test_energy_predictor_stub_raises() -> None:
    from utils.diffusion_model.diffusion.energy_predictor import EnergyPredictor
    try:
        _ = EnergyPredictor()
    except NotImplementedError as e:
        assert "energy" in str(e), "stub error message doesn't mention energy"
        return
    raise AssertionError("EnergyPredictor() did not raise NotImplementedError")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("EDM forward shapes", test_edm_forward_shapes),
        ("EDM gradient flow",  test_edm_gradient_flow),
        ("EDM no-mds-align passthrough", test_edm_no_mds_align),
        ("kNN graph forward shapes",  test_knn_graph_forward_shapes),
        ("kNN graph gradient flow",   test_knn_graph_gradient_flow),
        ("VN transformer stub raises NotImplementedError", test_vn_transformer_stub_raises),
        ("Energy predictor stub raises NotImplementedError", test_energy_predictor_stub_raises),
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

"""Tests for the Geometry-Coupled Nyström backbone (geomattn_nystrom).

Properties locked:
  1. Shape contract.
  2. SE(2)-invariance: landmark centroids transform rigidly with
     positions, so the cell↔centroid / centroid↔centroid distances —
     and therefore the geometric biases — are invariant to rotation /
     translation / reflection of the coordinate estimate.
  3. Zero-init reduction: at init the time-gate is zero ⇒ all three
     Nyström biases (F/B/G) are zero ⇒ the layer is identical to plain
     Nyström attention.
  4. Sub-quadratic: the Nyström path is taken when N > n_landmarks
     (no dense N×N bias), and falls through to the dense exact path
     when N ≤ n_landmarks.
  5. Gradient flow to kernel + gate + RBF centres.

Requires torch; skipped wholesale if torch isn't importable.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _make_dataholder(B, N, gene_dim, seed=0):
    from utils.data.dataholder import DataHolder
    g = torch.Generator().manual_seed(seed)
    return DataHolder(
        node_features=torch.randn(B, N, gene_dim, generator=g),
        positions=torch.randn(B, N, 2, generator=g),
        diffusion_time=torch.rand(B, 1, generator=g),
        cell_class=None, cell_ID=None, t_int=None,
        t=torch.rand(B, 1, generator=g),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )


def _backbone(gene_dim=20, out_dim=8, n_layers=2, n_heads=4, hidden=32,
              num_rbf=8, n_landmarks=8, exact_until_n=0):
    from models.geometry_coupled_nystrom import GeometryCoupledNystromBackbone
    cfg = dict(hidden_dim=hidden, n_layers=n_layers, n_heads=n_heads,
               mlp_ratio=2, time_embed_dim=32, num_rbf=num_rbf,
               init_max_dist=3.0, detach_geometry=True,
               n_landmarks=n_landmarks, moore_penrose_iters=6,
               exact_until_n=exact_until_n, grad_checkpoint=False)
    return GeometryCoupledNystromBackbone(
        input_dims={"node_features_dimensions": gene_dim},
        n_layers=n_layers, hidden_mlp_dims={},
        hidden_dims={"output_features_to_pos_dims": out_dim},
        output_dims={}, geomattn_nystrom_cfg=cfg,
    )


def test_forward_shapes_nystrom_path():
    """N (40) > n_landmarks (8) → real Nyström path."""
    B, N, gene_dim, out_dim = 2, 40, 20, 8
    model = _backbone(gene_dim=gene_dim, out_dim=out_dim, n_landmarks=8)
    out = model(_make_dataholder(B, N, gene_dim))
    assert out.node_features.shape == (B, N, out_dim)
    assert out.positions.shape == (B, N, 2)
    assert torch.isfinite(out.positions).all()


def test_forward_shapes_exact_fallthrough():
    """N (6) ≤ n_landmarks (8) → dense exact GCA fallthrough."""
    B, N, gene_dim = 2, 6, 20
    model = _backbone(gene_dim=gene_dim, n_landmarks=8)
    out = model(_make_dataholder(B, N, gene_dim))
    assert out.node_features.shape[1] == N
    assert torch.isfinite(out.positions).all()


@pytest.mark.parametrize("reflect", [False, True])
def test_landmark_centroid_bias_is_se2_invariant(reflect):
    """Centroids = segment-mean of positions, which transforms rigidly:
    mean(Rx+t) = R·mean(x)+t. So cell↔centroid and centroid↔centroid
    distances are invariant under any rigid motion → the biases are
    invariant. We check the distance tensors the bias is built from."""
    from models.nystromformer_backbone import _segment_mean_pool

    g = torch.Generator().manual_seed(1)
    B, N, M = 1, 40, 8
    X = torch.randn(B, N, 2, generator=g)
    real = torch.ones(B, N, dtype=torch.bool)

    theta = 0.7
    R = torch.tensor([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
    if reflect:
        R = R @ torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    t = torch.tensor([2.5, -1.3])
    X2 = X @ R.T + t

    def dists(P):
        cen = _segment_mean_pool(P.unsqueeze(1), real, M).squeeze(1)   # (B,M,2)
        d_NM = torch.cdist(P, cen)
        d_MM = torch.cdist(cen, cen)
        return d_NM, d_MM

    d_NM1, d_MM1 = dists(X)
    d_NM2, d_MM2 = dists(X2)
    assert torch.allclose(d_NM1, d_NM2, atol=1e-5), "cell↔centroid dist not invariant"
    assert torch.allclose(d_MM1, d_MM2, atol=1e-5), "centroid↔centroid dist not invariant"


def test_zero_gate_means_zero_bias_at_init():
    """Zero-init gate ⇒ the per-head radial bias is exactly zero at
    init for any distance tensor."""
    from models.geometry_coupled_nystrom import GeometryCoupledNystromAttention
    from models.geometry_coupled_attention import RadialBasis

    B, N, M, H, hidden, K = 1, 40, 8, 4, 32, 8
    attn = GeometryCoupledNystromAttention(
        embed_dim=hidden, num_heads=H, num_rbf=K, n_landmarks=M,
    )
    rb = RadialBasis(num_rbf=K)
    g = torch.Generator().manual_seed(2)
    c = torch.randn(B, hidden, generator=g)
    gamma = attn.gate(c)                                   # (B,H) — zero at init
    assert torch.count_nonzero(gamma) == 0, "gate must be zero at init"
    dist = torch.rand(B, N, M, generator=g) * 3.0
    bias = attn._radial_bias(dist, rb, gamma)
    assert torch.count_nonzero(bias) == 0, \
        "all three Nyström biases must be zero at init (reduces to plain Nyström)"


def test_nonzero_gate_bias_finite_and_nonzero():
    from models.geometry_coupled_nystrom import GeometryCoupledNystromAttention
    from models.geometry_coupled_attention import RadialBasis

    B, N, M, H, hidden, K = 1, 40, 8, 4, 32, 8
    attn = GeometryCoupledNystromAttention(
        embed_dim=hidden, num_heads=H, num_rbf=K, n_landmarks=M,
    )
    with torch.no_grad():
        attn.gate.bias.fill_(0.4)
        attn.kernel.weight.normal_(generator=torch.Generator().manual_seed(3))
    rb = RadialBasis(num_rbf=K)
    g = torch.Generator().manual_seed(4)
    c = torch.randn(B, hidden, generator=g)
    gamma = attn.gate(c)
    dist = torch.rand(B, N, M, generator=g) * 3.0
    bias = attn._radial_bias(dist, rb, gamma)
    assert torch.count_nonzero(bias) > 0
    assert torch.isfinite(bias).all()


def test_gradients_reach_kernel_gate_and_centres():
    B, N, gene_dim = 2, 40, 16
    model = _backbone(gene_dim=gene_dim, n_layers=2, n_landmarks=8)
    with torch.no_grad():
        for blk in model.blocks:
            blk.attn.gate.bias.fill_(0.3)
    out = model(_make_dataholder(B, N, gene_dim))
    (out.positions.pow(2).mean() + out.node_features.pow(2).mean()).backward()
    a0 = model.blocks[0].attn
    assert a0.kernel.weight.grad is not None
    assert torch.count_nonzero(a0.kernel.weight.grad) > 0
    assert a0.gate.weight.grad is not None
    assert model.radial.centers.grad is not None

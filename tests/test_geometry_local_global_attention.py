"""Tests for Geometry Local-Global Attention (geomattn_localglobal).

The centrepiece is the **exactness validation**: in the K→N limit, both
combine schemes must reduce to dense geometry-biased softmax attention
exactly. This pins the implementation against an independent reference.

  * unified, K=N: per-query landmark count-exclusion drives the
    far-field weight to 0 ⇒ output == dense softmax over all cells.
  * gated, K=N, blend forced to local (g=1): local branch == dense
    softmax over all cells.

Plus: kNN correctness (chunked == naive), SE(2)-invariance, zero-gate
geometry, padding handling, shapes, gradient flow. torch-gated skip.
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


# ----------------------------------------------------------------------
# Reference: dense geometry-biased softmax attention (the ground truth)
# ----------------------------------------------------------------------

def _dense_reference(q, k, v, dist, centers, widths, W, gamma):
    """out = softmax(q·kᵀ·scale + Σ_r W·φ_r(dist)) · v, computed densely.
    Shapes: q,k,v (B,H,N,D); dist (B,N,N). Returns (B,H,N,D)."""
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    s = torch.einsum("bhnd,bhmd->bhnm", q, k) * scale          # (B,H,N,N)
    # geometry bias
    bias = torch.zeros(B, H, N, N)
    for r in range(W.shape[1]):
        phi = torch.exp(-0.5 * ((dist - centers[r]) / widths[r]) ** 2)  # (B,N,N)
        bias = bias + W[:, r].view(1, H, 1, 1) * phi.unsqueeze(1)
    bias = gamma.view(B, H, 1, 1) * bias
    a = (s + bias).softmax(dim=-1)
    return torch.einsum("bhnm,bhmd->bhnd", a, v)


# ----------------------------------------------------------------------
# 1. kNN chunked == naive
# ----------------------------------------------------------------------

def test_knn_chunked_matches_naive():
    from models.geometry_local_global_attention import knn_chunked
    B, N, K = 2, 50, 10
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(B, N, 2, generator=g)
    real = torch.ones(B, N, dtype=torch.bool)

    idx_c, dist_c, valid_c = knn_chunked(pos, real, K, chunk_size=7)  # odd chunk
    # naive
    d = torch.cdist(pos, pos)
    dist_n, idx_n = torch.topk(d, K, dim=-1, largest=False)
    assert torch.allclose(dist_c, dist_n, atol=1e-5)
    # indices: distances match → same neighbours (ties unlikely on random data)
    assert (idx_c == idx_n).all()
    assert valid_c.all()


def test_knn_excludes_padding():
    from models.geometry_local_global_attention import knn_chunked
    B, N, K = 1, 20, 5
    g = torch.Generator().manual_seed(1)
    pos = torch.randn(B, N, 2, generator=g)
    real = torch.ones(B, N, dtype=torch.bool)
    real[0, 10:] = False                                       # last 10 are PAD
    idx, dist, valid = knn_chunked(pos, real, K, chunk_size=8)
    # No neighbour index may point at a padded cell.
    assert (idx[0][valid[0]] < 10).all(), "kNN selected a padding cell"


# ----------------------------------------------------------------------
# 2. EXACTNESS: K=N reduces to dense geometry-biased softmax
# ----------------------------------------------------------------------

def _attn_module(combine, hidden=16, H=2, K=8, M=4, num_rbf=4, n_local=8):
    from models.geometry_local_global_attention import LocalGlobalGeometryAttention
    return LocalGlobalGeometryAttention(
        embed_dim=hidden, num_heads=H, num_rbf=num_rbf,
        n_local=n_local, n_landmarks=M, combine=combine,
    )


def _run_attention_vs_dense(combine, force_local_gate):
    from models.geometry_local_global_attention import knn_chunked
    from models.geometry_coupled_attention import RadialBasis
    from models.nystromformer_backbone import _segment_mean_pool

    torch.manual_seed(0)
    B, N, H, D, M = 1, 12, 2, 8, 4
    hidden = H * D
    K = N  # <-- K = N is the exactness limit
    attn = _attn_module(combine, hidden=hidden, H=H, K=K, M=M, n_local=K)
    rb = RadialBasis(num_rbf=attn.num_rbf)

    # Make the geometry bias ACTIVE (gate nonzero) + random kernel so the
    # test exercises the bias path, not just q·k.
    with torch.no_grad():
        attn.geo_gate.bias.fill_(0.5)
        attn.kernel.weight.normal_()
        if combine == "gated" and force_local_gate:
            # push sigmoid(blend)→1 so out == local branch
            attn.blend_gate.bias.fill_(30.0)

    x = torch.randn(B, N, hidden)
    c = torch.randn(B, hidden)
    pos = torch.randn(B, N, 2)
    real = torch.ones(B, N, dtype=torch.bool)

    # Precompute geometry (mirrors the backbone).
    nbr_idx, nbr_dist, nbr_valid = knn_chunked(pos, real, K, chunk_size=1024)
    centroids = _segment_mean_pool(pos.unsqueeze(1), real, M).squeeze(1)
    n_idx = torch.arange(N)
    seg_id = ((n_idx * M) // N).clamp(max=M - 1)
    base_count = torch.zeros(B, M)
    base_count.scatter_add_(1, seg_id.view(1, N).expand(B, N), real.float())

    out = attn(
        x, c, rb, nbr_idx=nbr_idx, nbr_dist=nbr_dist, nbr_valid=nbr_valid,
        positions=pos, centroids=centroids, seg_id=seg_id,
        base_count=base_count, real_mask=real, key_padding_mask=None,
    )                                                          # (B,N,hidden)

    # Dense reference using the SAME projections + kernel + gate.
    q = attn.q_proj(x).view(B, N, H, D).transpose(1, 2)
    k = attn.k_proj(x).view(B, N, H, D).transpose(1, 2)
    v = attn.v_proj(x).view(B, N, H, D).transpose(1, 2)
    gamma = attn.geo_gate(c)
    widths = torch.exp(rb.log_widths) + 1e-6
    ref = _dense_reference(q, k, v, torch.cdist(pos, pos),
                           rb.centers, widths, attn.kernel.weight, gamma)
    ref = ref.transpose(1, 2).contiguous().view(B, N, hidden)
    ref = attn.out_proj(ref)
    return out, ref


def test_unified_exact_at_K_equals_N():
    out, ref = _run_attention_vs_dense("unified", force_local_gate=False)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_gated_local_exact_at_K_equals_N():
    out, ref = _run_attention_vs_dense("gated", force_local_gate=True)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


# ----------------------------------------------------------------------
# 3. SE(2)-invariance of the geometry inputs (kNN dists + centroids)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("reflect", [False, True])
def test_se2_invariance(reflect):
    from models.geometry_local_global_attention import knn_chunked
    from models.nystromformer_backbone import _segment_mean_pool

    B, N, K, M = 1, 30, 8, 4
    g = torch.Generator().manual_seed(2)
    X = torch.randn(B, N, 2, generator=g)
    real = torch.ones(B, N, dtype=torch.bool)
    theta = 0.6
    R = torch.tensor([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
    if reflect:
        R = R @ torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    X2 = X @ R.T + torch.tensor([1.0, -2.0])

    _, d1, _ = knn_chunked(X, real, K)
    _, d2, _ = knn_chunked(X2, real, K)
    assert torch.allclose(d1.sort(-1).values, d2.sort(-1).values, atol=1e-5)

    c1 = _segment_mean_pool(X.unsqueeze(1), real, M).squeeze(1)
    c2 = _segment_mean_pool(X2.unsqueeze(1), real, M).squeeze(1)
    assert torch.allclose(torch.cdist(c1, c1), torch.cdist(c2, c2), atol=1e-5)


# ----------------------------------------------------------------------
# 4. Backbone: shapes + gradient flow, both combine modes
# ----------------------------------------------------------------------

def _backbone(combine, gene_dim=16, out_dim=8, N_layers=2):
    from models.geometry_local_global_attention import LocalGlobalBackbone
    cfg = dict(hidden_dim=32, n_layers=N_layers, n_heads=4, mlp_ratio=2,
               time_embed_dim=32, num_rbf=8, init_max_dist=3.0,
               detach_geometry=True, n_local=8, n_landmarks=8,
               combine=combine, knn_chunk=1024, grad_checkpoint=False)
    return LocalGlobalBackbone(
        input_dims={"node_features_dimensions": gene_dim}, n_layers=N_layers,
        hidden_mlp_dims={}, hidden_dims={"output_features_to_pos_dims": out_dim},
        output_dims={}, geomattn_localglobal_cfg=cfg,
    )


def _dataholder(B, N, gene_dim, seed=0):
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


@pytest.mark.parametrize("combine", ["gated", "unified"])
def test_backbone_shapes_and_grads(combine):
    B, N, gene_dim, out_dim = 2, 40, 16, 8
    model = _backbone(combine, gene_dim=gene_dim, out_dim=out_dim)
    # activate geometry so kernel grads are non-trivial
    with torch.no_grad():
        for blk in model.blocks:
            blk.attn.geo_gate.bias.fill_(0.3)
    data = _dataholder(B, N, gene_dim)
    out = model(data)
    assert out.node_features.shape == (B, N, out_dim)
    assert out.positions.shape == (B, N, 2)
    assert torch.isfinite(out.positions).all()
    (out.positions.pow(2).mean() + out.node_features.pow(2).mean()).backward()
    a0 = model.blocks[0].attn
    assert a0.kernel.weight.grad is not None
    assert torch.count_nonzero(a0.kernel.weight.grad) > 0
    assert model.radial.centers.grad is not None


@pytest.mark.parametrize("combine", ["gated", "unified"])
def test_backbone_handles_padding(combine):
    B, N, gene_dim = 2, 40, 16
    model = _backbone(combine, gene_dim=gene_dim)
    data = _dataholder(B, N, gene_dim)
    data.node_mask[0, 30:] = False                            # pad last 10 of slice 0
    out = model(data)
    # padded outputs must be exactly zero
    assert torch.count_nonzero(out.node_features[0, 30:]) == 0
    assert torch.count_nonzero(out.positions[0, 30:]) == 0
    assert torch.isfinite(out.positions).all()

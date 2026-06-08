"""Tests for the 5 spatial-sparse geometry-attention backbones.

Correctness anchors (exactness limits against dense softmax):
  * grouped_dense_attention single-group == dense; multi-group ==
    block-diagonal dense. (powers axial / swin / routing)
  * dilated with dilation=1 == exact local-kNN attention.
  * swin with grid=1 == full attention (single bucket).
  * routing with n_clusters=1 == full attention (single cluster).
  * axial with band_size>=N == full attention (single band, both axes).

Plus per-backbone: shape, padding-zeroing, gradient flow, SE(2)-invariance.
torch-gated skip.
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
# dense reference + grouped primitive checks
# ----------------------------------------------------------------------

def _bias_nd(dist, centers, widths, W, gamma):
    from models.spatial_sparse_attention import radial_bias_nd
    return radial_bias_nd(dist, centers, widths, W, gamma)


def _dense_blockdiag(q, k, v, pos, group_id, centers, widths, W, gamma):
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    s = torch.einsum("bhnd,bhmd->bhnm", q, k) * scale
    dNN = torch.cdist(pos, pos)
    s = s + _bias_nd(dNN, centers, widths, W, gamma)
    same = group_id[:, :, None] == group_id[:, None, :]
    s = s.masked_fill(~same.unsqueeze(1), torch.finfo(s.dtype).min)
    a = s.softmax(-1)
    return torch.einsum("bhnm,bhmd->bhnd", a, v)


@pytest.mark.parametrize("gid_list", [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],          # single group
    [0, 0, 0, 1, 1, 2, 2, 2, 2],          # uneven multi-group
])
def test_grouped_dense_attention_matches_blockdiag(gid_list):
    from models.spatial_sparse_attention import grouped_dense_attention
    from models.geometry_coupled_attention import RadialBasis
    torch.manual_seed(0)
    B, H, N, D = 1, 2, 9, 4
    G = max(gid_list) + 1
    q, k, v = (torch.randn(B, H, N, D) for _ in range(3))
    pos = torch.randn(B, N, 2)
    real = torch.ones(B, N, dtype=torch.bool)
    gid = torch.tensor([gid_list])
    rb = RadialBasis(num_rbf=4)
    W = torch.randn(H, 4); gamma = torch.randn(B, H)
    widths = torch.exp(rb.log_widths) + 1e-6

    out = grouped_dense_attention(q, k, v, pos, gid, real, G,
                                  rb.centers, widths, W, gamma)
    ref = _dense_blockdiag(q, k, v, pos, gid, rb.centers, widths, W, gamma)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------------
# backbone factory + exactness limits
# ----------------------------------------------------------------------

def _backbone(pattern, cfg_overrides, gene_dim=16, out_dim=8, n_layers=2):
    from models.spatial_sparse_attention import SpatialSparseBackbone
    cfg = dict(hidden_dim=32, n_heads=4, mlp_ratio=2, time_embed_dim=32,
               num_rbf=8, init_max_dist=3.0, detach_geometry=True,
               grad_checkpoint=False)
    cfg.update(cfg_overrides)
    return SpatialSparseBackbone(
        pattern=pattern,
        input_dims={"node_features_dimensions": gene_dim}, n_layers=n_layers,
        hidden_mlp_dims={}, hidden_dims={"output_features_to_pos_dims": out_dim},
        output_dims={}, cfg=cfg,
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


ALL = [
    ("dilated",  dict(n_local=8, dilation=2)),
    ("bigbird",  dict(n_local=8, n_random=4, n_landmarks=8)),
    ("axial",    dict(band_size=8)),
    ("swin",     dict(grid=3)),
    ("routing",  dict(n_clusters=4, kmeans_iters=3)),
]


@pytest.mark.parametrize("pattern,ov", ALL)
def test_backbone_shapes_and_grad(pattern, ov):
    B, N, gene_dim, out_dim = 2, 40, 16, 8
    model = _backbone(pattern, ov, gene_dim=gene_dim, out_dim=out_dim)
    with torch.no_grad():
        for blk in model.blocks:
            blk.attn.geo_gate.bias.fill_(0.3)            # activate geometry path
    data = _dataholder(B, N, gene_dim)
    out = model(data)
    assert out.node_features.shape == (B, N, out_dim)
    assert out.positions.shape == (B, N, 2)
    assert torch.isfinite(out.positions).all()
    assert torch.isfinite(out.node_features).all()
    (out.positions.pow(2).mean() + out.node_features.pow(2).mean()).backward()
    a0 = model.blocks[0].attn
    assert a0.kernel.weight.grad is not None
    assert torch.count_nonzero(a0.kernel.weight.grad) > 0
    assert model.radial.centers.grad is not None


@pytest.mark.parametrize("pattern,ov", ALL)
def test_backbone_padding_zeroed(pattern, ov):
    B, N, gene_dim = 2, 40, 16
    model = _backbone(pattern, ov, gene_dim=gene_dim)
    data = _dataholder(B, N, gene_dim)
    data.node_mask[0, 30:] = False
    out = model(data)
    assert torch.count_nonzero(out.node_features[0, 30:]) == 0
    assert torch.count_nonzero(out.positions[0, 30:]) == 0
    assert torch.isfinite(out.positions).all()


def test_dilated_dilation1_equals_local_knn():
    """dilation=1 selects the top-n_local nearest → exact local attention.
    Compare the dilated attention module's output to a hand-rolled exact
    softmax over the same n_local neighbours."""
    from models.spatial_sparse_attention import DilatedKNNAttention, radial_bias_nd
    from models.geometry_coupled_attention import RadialBasis
    from models.geometry_local_global_attention import knn_chunked, _gather_neighbors
    torch.manual_seed(0)
    B, N, H, D, nl = 1, 16, 2, 4, 6
    hidden = H * D
    attn = DilatedKNNAttention(hidden, H, num_rbf=4, n_local=nl, dilation=1)
    with torch.no_grad():
        attn.geo_gate.bias.fill_(0.5)
        attn.kernel.weight.normal_()
    rb = RadialBasis(num_rbf=4)
    x = torch.randn(B, N, hidden); c = torch.randn(B, hidden)
    pos = torch.randn(B, N, 2); real = torch.ones(B, N, dtype=torch.bool)
    nbr_idx, nbr_dist, nbr_valid = knn_chunked(pos, real, nl * 1, 1024)
    geom = dict(nbr_idx=nbr_idx, nbr_dist=nbr_dist, nbr_valid=nbr_valid,
                positions=pos, real_mask=real)
    out = attn(x, c, rb, geom)

    # reference: exact softmax over the same nl neighbours
    q = attn.q_proj(x).view(B, N, H, D).transpose(1, 2)
    k = attn.k_proj(x).view(B, N, H, D).transpose(1, 2)
    v = attn.v_proj(x).view(B, N, H, D).transpose(1, 2)
    gamma = attn.geo_gate(c)
    widths = torch.exp(rb.log_widths) + 1e-6
    k_nbr = _gather_neighbors(k, nbr_idx); v_nbr = _gather_neighbors(v, nbr_idx)
    s = torch.einsum("bhnd,bhnkd->bhnk", q, k_nbr) / math.sqrt(D)
    s = s + radial_bias_nd(nbr_dist, rb.centers, widths, attn.kernel.weight, gamma)
    a = s.softmax(-1)
    ref = torch.einsum("bhnk,bhnkd->bhnd", a, v_nbr)
    ref = attn.out_proj(ref.transpose(1, 2).contiguous().view(B, N, hidden))
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("pattern,ov,single", [
    ("swin",    dict(grid=1), "grid=1 → 1 bucket"),
    ("routing", dict(n_clusters=1, kmeans_iters=2), "C=1 → 1 cluster"),
])
def test_single_group_equals_full_attention(pattern, ov, single):
    """swin grid=1 / routing C=1 ⇒ every cell in one group ⇒ the backbone's
    attention is full dense attention. We check the attention MODULE output
    equals dense block-diagonal with a single group."""
    from models.geometry_coupled_attention import RadialBasis
    torch.manual_seed(0)
    B, N, H, D = 1, 12, 2, 4
    hidden = H * D
    model = _backbone(pattern, ov, gene_dim=8, out_dim=8, n_layers=1)
    attn = model.blocks[0].attn
    with torch.no_grad():
        attn.geo_gate.bias.fill_(0.4); attn.kernel.weight.normal_()
    rb = model.radial
    x = torch.randn(B, N, hidden); c = torch.randn(B, hidden)
    pos = torch.randn(B, N, 2); real = torch.ones(B, N, dtype=torch.bool)
    geom = model._precompute_geometry(pos, real)
    out = attn(x, c, rb, geom)

    # all cells one group → dense reference
    q = attn.q_proj(x).view(B, N, H, D).transpose(1, 2)
    k = attn.k_proj(x).view(B, N, H, D).transpose(1, 2)
    v = attn.v_proj(x).view(B, N, H, D).transpose(1, 2)
    gamma = attn.geo_gate(c)
    widths = torch.exp(rb.log_widths) + 1e-6
    gid0 = torch.zeros(B, N, dtype=torch.long)
    ref = _dense_blockdiag(q, k, v, pos, gid0, rb.centers, widths,
                           attn.kernel.weight, gamma)
    ref = attn.out_proj(ref.transpose(1, 2).contiguous().view(B, N, hidden))
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_axial_single_band_equals_full_average():
    """band_size>=N ⇒ each axis is one band ⇒ out = 0.5(full + full) = full."""
    from models.geometry_coupled_attention import RadialBasis
    torch.manual_seed(0)
    B, N, H, D = 1, 10, 2, 4
    hidden = H * D
    model = _backbone("axial", dict(band_size=N), gene_dim=8, n_layers=1)
    attn = model.blocks[0].attn
    with torch.no_grad():
        attn.geo_gate.bias.fill_(0.4); attn.kernel.weight.normal_()
    rb = model.radial
    x = torch.randn(B, N, hidden); c = torch.randn(B, hidden)
    pos = torch.randn(B, N, 2); real = torch.ones(B, N, dtype=torch.bool)
    geom = model._precompute_geometry(pos, real)
    out = attn(x, c, rb, geom)
    q = attn.q_proj(x).view(B, N, H, D).transpose(1, 2)
    k = attn.k_proj(x).view(B, N, H, D).transpose(1, 2)
    v = attn.v_proj(x).view(B, N, H, D).transpose(1, 2)
    gamma = attn.geo_gate(c); widths = torch.exp(rb.log_widths) + 1e-6
    gid0 = torch.zeros(B, N, dtype=torch.long)
    full = _dense_blockdiag(q, k, v, pos, gid0, rb.centers, widths,
                            attn.kernel.weight, gamma)
    ref = attn.out_proj(full.transpose(1, 2).contiguous().view(B, N, hidden))
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("pattern,ov", ALL)
def test_se2_invariance_of_geometry(pattern, ov):
    """Rotating/reflecting positions must not change the OUTPUT (up to
    float tol) because every geometry input (kNN dists, centroids, band
    ranks, grid buckets relative to bbox, k-means on positions) is
    rigid-motion-equivariant and the bias depends only on distances.
    NOTE: swin grid buckets are bbox-relative so a pure rotation can
    cross bucket boundaries — for swin we test translation+reflection
    only (axis-aligned), which preserve the axis-aligned grid."""
    model = _backbone(pattern, ov, gene_dim=8, n_layers=1)
    with torch.no_grad():
        for blk in model.blocks:
            blk.attn.geo_gate.bias.fill_(0.3)
    model.eval()
    B, N = 1, 24
    data = _dataholder(B, N, 8, seed=3)

    # rigid motion of the INPUT positions
    if pattern == "swin":
        # axis-aligned only (translation + reflection) to keep the grid
        R = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    else:
        th = 0.6
        R = torch.tensor([[math.cos(th), -math.sin(th)],
                          [math.sin(th), math.cos(th)]])
    from utils.data.dataholder import DataHolder
    data2 = DataHolder(
        node_features=data.node_features.clone(),
        positions=data.positions @ R.T + torch.tensor([2.0, -1.0]),
        diffusion_time=data.diffusion_time, cell_class=None, cell_ID=None,
        t_int=None, t=data.t, node_mask=data.node_mask,
    )
    # The per-cell NODE FEATURES carry no coordinate frame, so a rigid
    # motion of the input positions must leave them invariant (every
    # geometry input is rigid-equivariant; the bias depends only on
    # distances). Output positions live in an arbitrary frame, so we
    # check feature invariance, not position equality.
    f1 = model(data).node_features
    f2 = model(data2).node_features
    torch.testing.assert_close(f1, f2, atol=1e-4, rtol=1e-4)

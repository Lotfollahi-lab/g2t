"""Tests for the sparse local-neighbourhood distance loss + the EDM
head's ``skip_edm_D_train`` fast path.

The sparse loss (``LossFunction._compute_sparse_local_distance``) is the
O(N*k) sibling of ``locality_weighted_distance``: it reads ``pred.edm_h``
(the (N,k) embedding) and supervises predicted-vs-true Euclidean
distances on each cell's true k-NN (+ random far pairs) only — never an
(N,N) matrix.

Centrepiece = the **full-k exactness check**: with ``local_k = N-1``,
``n_random = 0`` and ``weight_fn="none"``, the directed-pair sparse loss
must equal the undirected ``edm_distance_mse`` (sum and count both
double, so the mean is identical). Same for ``weight_fn`` exp/inverse vs
the dense locality loss. Plus: gradient flow into edm_h, true-kNN cache
consistency, and the head emitting edm_D=None at train time / a full
matrix at eval time. torch-gated skip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from omegaconf import OmegaConf  # noqa: E402

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metrics.loss_function import LossFunction  # noqa: E402
from utils.data.dataholder import DataHolder  # noqa: E402


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _cfg(**sld):
    """Minimal cfg exposing model.loss.sparse_local_distance + the dense
    edm/locality blocks the full-k checks compare against."""
    base = {
        "enabled": True, "weight": 1.0, "local_k": 32, "n_random": 8,
        "global_weight": 0.1, "weight_fn": "none", "sigma": 1.0,
        "knn_chunk": 1024, "cache_true": True, "warmup_steps": 0,
    }
    base.update(sld)
    return OmegaConf.create({
        "model": {"loss": {
            "sparse_local_distance": base,
            "edm_distance_mse": {"enabled": False, "weight": 1.0,
                                 "warmup_steps": 0},
            "locality_weighted_distance": {
                "enabled": False, "weight": 1.0, "weight_fn": "exp",
                "sigma": 1.0, "warmup_steps": 0},
            # keep everything else off
            "pairwise_distance_mse": {"enabled": False, "weight": 1.0},
        }}
    })


def _holders(B=2, N=50, kd=8, seed=0, structured=False):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(B, N, 2, generator=g)
    if structured:
        # embedding linearly related to position -> recoverable geometry
        Wproj = torch.randn(2, kd, generator=g)
        h = pos @ Wproj + 0.01 * torch.randn(B, N, kd, generator=g)
    else:
        h = torch.randn(B, N, kd, generator=g, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool)
    feats = torch.randn(B, N, 4, generator=g)
    pred = DataHolder(positions=pos.clone(), node_features=feats,
                      diffusion_time=0, node_mask=mask)
    true = DataHolder(positions=pos.clone(), node_features=feats,
                      diffusion_time=0, node_mask=mask)
    pred.edm_h = h
    # consistent edm_D = ||h_i - h_j||^2 for the dense comparison
    diff = h.unsqueeze(2) - h.unsqueeze(1)
    pred.edm_D = (diff * diff).sum(-1)
    return pred, true, h


# ----------------------------------------------------------------------
# full-k exactness: sparse == dense
# ----------------------------------------------------------------------

@pytest.mark.parametrize("fn", ["none", "exp", "inverse"])
def test_full_k_matches_dense(fn):
    pred, true, _ = _holders(B=2, N=40, kd=8, seed=1)
    N = pred.positions.shape[1]
    loss = LossFunction(_cfg(local_k=N - 1, n_random=0, weight_fn=fn,
                             sigma=1.0))
    sparse = loss._compute_sparse_local_distance(pred, true)
    if fn == "none":
        dense = loss._compute_edm_distance_mse(pred, true)
    else:
        # dense locality uses the same weight_fn semantics
        loss._locw_fn = fn
        loss._locw_sigma = 1.0
        dense = loss._compute_locality_weighted_distance(pred, true)
    assert torch.allclose(sparse, dense, atol=1e-5), (
        f"full-k sparse ({sparse.item():.6f}) != dense "
        f"({dense.item():.6f}) for weight_fn={fn}"
    )


# ----------------------------------------------------------------------
# basic properties
# ----------------------------------------------------------------------

def test_finite_nonneg_and_grad_flows_to_h():
    pred, true, h = _holders(B=2, N=60, kd=8, seed=2)
    loss = LossFunction(_cfg(local_k=16, n_random=8))
    val = loss._compute_sparse_local_distance(pred, true)
    assert torch.isfinite(val) and val.item() >= 0.0
    val.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert h.grad.abs().sum() > 0, "no gradient reached edm_h"


def test_structured_embedding_beats_random():
    """An embedding linear in true position should incur a much smaller
    local loss than a random embedding."""
    ps, ts, _ = _holders(B=1, N=80, kd=8, seed=3, structured=True)
    pr, tr, _ = _holders(B=1, N=80, kd=8, seed=3, structured=False)
    loss = LossFunction(_cfg(local_k=10, n_random=0))
    l_struct = loss._compute_sparse_local_distance(ps, ts).item()
    l_rand = loss._compute_sparse_local_distance(pr, tr).item()
    assert l_struct < l_rand


def test_missing_edm_h_returns_graph_zero():
    pred, true, _ = _holders(B=1, N=30, seed=4)
    pred.edm_h = None
    loss = LossFunction(_cfg())
    val = loss._compute_sparse_local_distance(pred, true)
    assert torch.isfinite(val) and float(val) == 0.0


# ----------------------------------------------------------------------
# structured global anchor (FPS landmarks)
# ----------------------------------------------------------------------

def test_fps_landmarks_spread_cache_and_shapes():
    loss = LossFunction(_cfg(n_landmarks=32, landmark_mode="fps"))
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(500, 2, generator=g) * torch.tensor([5.0, 1.0])
    idx, d_true = loss._sld_landmarks(pos, 32)
    assert idx.shape == (32,) and d_true.shape == (500, 32)
    assert torch.isfinite(d_true).all() and (d_true >= 0).all()
    # cache hit returns identical
    idx2, d2 = loss._sld_landmarks(pos, 32)
    assert torch.equal(idx, idx2) and torch.allclose(d_true, d2)
    # FPS landmarks are far more spread than a random subset (coverage)
    def spread(ii):
        c = pos[ii]
        return torch.pdist(c).mean()
    rand = torch.randperm(500, generator=torch.Generator().manual_seed(1))[:32]
    assert spread(idx) > spread(rand), "FPS landmarks should out-spread random"


def test_landmark_term_runs_and_grad():
    pred, true, h = _holders(B=2, N=120, kd=8, seed=5)
    loss = LossFunction(_cfg(local_k=16, n_random=0, n_landmarks=32,
                             landmark_weight=0.1))
    val = loss._compute_sparse_local_distance(pred, true)
    assert torch.isfinite(val) and val.item() >= 0.0
    val.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert h.grad.abs().sum() > 0, "no gradient reached edm_h via landmark term"


def test_landmark_term_adds_signal():
    """Enabling landmarks changes the loss (the global term is active)."""
    pred, true, _ = _holders(B=1, N=100, kd=8, seed=6)
    base = LossFunction(_cfg(local_k=16, n_random=0, n_landmarks=0))
    withlm = LossFunction(_cfg(local_k=16, n_random=0, n_landmarks=32,
                               landmark_weight=0.5))
    v0 = base._compute_sparse_local_distance(pred, true).item()
    v1 = withlm._compute_sparse_local_distance(pred, true).item()
    assert v1 > v0, "landmark term should add a positive global penalty"


# ----------------------------------------------------------------------
# local-vs-global balancing (balance="equal")
# ----------------------------------------------------------------------

def test_balance_equal_gives_equal_contribution():
    """With balance='equal' the global term contributes exactly as much
    as local, so total ≈ 2·(local-only)."""
    pred, true, _ = _holders(B=1, N=100, kd=8, seed=7)
    loc_only = LossFunction(_cfg(local_k=16, n_random=0, n_landmarks=0))
    v_loc = loc_only._compute_sparse_local_distance(pred, true).item()
    bal = LossFunction(_cfg(local_k=16, n_random=0, n_landmarks=32,
                            landmark_weight=0.1, balance="equal"))
    v_bal = bal._compute_sparse_local_distance(pred, true).item()
    assert abs(v_bal - 2.0 * v_loc) < 1e-4 * max(1.0, abs(v_loc)), (
        f"equal balance should give total ≈ 2*local; got {v_bal} vs {2*v_loc}"
    )


def test_balance_equal_normalises_away_the_global_weight():
    """In equal mode the absolute global weight is normalised away — only
    the random-vs-landmark split survives. So two very different
    landmark_weights give the same total."""
    pred, true, _ = _holders(B=1, N=100, kd=8, seed=8)
    a = LossFunction(_cfg(local_k=16, n_random=0, n_landmarks=32,
                          landmark_weight=0.1, balance="equal"))
    b = LossFunction(_cfg(local_k=16, n_random=0, n_landmarks=32,
                          landmark_weight=5.0, balance="equal"))
    va = a._compute_sparse_local_distance(pred, true).item()
    vb = b._compute_sparse_local_distance(pred, true).item()
    assert abs(va - vb) < 1e-4 * max(1.0, abs(va)), (
        "equal balance must be invariant to the global weight magnitude"
    )


# ----------------------------------------------------------------------
# true-kNN cache consistency
# ----------------------------------------------------------------------

def test_true_knn_cache_consistent():
    from models.geometry_local_global_attention import knn_chunked
    loss = LossFunction(_cfg(cache_true=True))
    pos = torch.randn(40, 2)
    idx1, dist1 = loss._sld_true_knn(pos, 8, knn_chunked)
    assert len(loss._sld_true_cache) == 1, "cache not populated"
    idx2, dist2 = loss._sld_true_knn(pos, 8, knn_chunked)  # hit
    assert torch.equal(idx1, idx2) and torch.allclose(dist1, dist2)
    # column 0 is self (distance ~0)
    assert torch.allclose(dist1[:, 0], torch.zeros(40), atol=1e-5)


# ----------------------------------------------------------------------
# EDM head skip_edm_D_train fast path
# ----------------------------------------------------------------------

def _tiny_inner(N=20, kd_feat=4):
    """Inner model stub that returns a DataHolder with node_features +
    positions, like a backbone would, so EDMOutputWrapper can project."""
    class _Inner(torch.nn.Module):
        def forward(self, data, **kw):
            return data
    return _Inner()


def test_head_skip_edm_D_train_emits_none_in_train_full_in_eval():
    from models.edm_head import EDMOutputWrapper
    N, fin = 24, 4
    inner = _tiny_inner()
    head = EDMOutputWrapper(
        inner_model=inner, inner_out_dim=fin, embed_dim=8,
        mds_align=True, skip_edm_D_train=True,
    )
    pos = torch.randn(1, N, 2)
    feats = torch.randn(1, N, fin)
    mask = torch.ones(1, N, dtype=torch.bool)
    data = DataHolder(positions=pos, node_features=feats,
                      diffusion_time=0, node_mask=mask)
    # node_features must match inner_out_dim for the projector's concat.
    data.node_features = torch.randn(1, N, fin)

    head.train()
    out = head(data)
    assert out.edm_h is not None and out.edm_h.shape == (1, N, 8)
    assert getattr(out, "edm_D", "missing") is None, (
        "train-mode skip_edm_D_train must emit edm_D=None"
    )

    head.eval()
    data2 = DataHolder(positions=pos, node_features=torch.randn(1, N, fin),
                       diffusion_time=0, node_mask=mask)
    out2 = head(data2)
    assert out2.edm_D is not None and out2.edm_D.shape == (1, N, N), (
        "eval-mode must build the full D_sq for MDS"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

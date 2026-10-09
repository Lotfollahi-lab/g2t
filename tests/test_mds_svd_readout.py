"""The default SVD read-out equals classical MDS of the embedding-induced
squared distances (B = -1/2 J D J = (JH)(JH)^T), up to an orthogonal
transform that the Procrustes step removes."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.edm_head import _classical_mds_2d, _classical_mds_2d_svd  # noqa: E402


def _sq_dists(H):
    n = (H * H).sum(-1)
    return (n[:, None] + n[None, :] - 2.0 * H @ H.T).clamp_min(0.0)


def _orthogonal_fit(a, b):
    """Rotate/reflect ``a`` onto ``b`` (orthogonal Procrustes)."""
    U, _, Vt = torch.linalg.svd(a.T @ b)
    return a @ (U @ Vt)


@pytest.mark.parametrize("k", [2, 8, 16])
def test_svd_readout_matches_classical_mds(k):
    torch.manual_seed(0)
    scales = torch.logspace(1, -1, k, dtype=torch.float64)
    H = torch.randn(400, k, dtype=torch.float64) * scales + 3.0   # off-centre on purpose
    x_svd = _classical_mds_2d_svd(H)
    x_eig = _classical_mds_2d(_sq_dists(H), tikhonov_eps=0.0)
    x_eig = x_eig - x_eig.mean(dim=0, keepdim=True)
    assert x_svd.shape == (400, 2)
    assert torch.allclose(_orthogonal_fit(x_svd, x_eig), x_eig, atol=1e-7)
    # Same pairwise geometry, hence the same distance matrix.
    assert torch.allclose(_sq_dists(x_svd), _sq_dists(x_eig), atol=1e-6)


def test_wrapper_default_solver_is_svd():
    import inspect
    from models.edm_head import EDMOutputWrapper
    sig = inspect.signature(EDMOutputWrapper.__init__)
    assert sig.parameters["mds_solver"].default == "svd"

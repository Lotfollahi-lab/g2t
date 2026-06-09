"""Tests for the gauge-fixed flow-matching variants.

Option 1 (canonicalized coordinate flow) and Option 2 (gauge-fixed
relational / h-space flow) both rest on ``canonicalize_cloud``. The
centrepiece checks pin its two defining properties:

  * **isometry** — pairwise distances are unchanged, so the EDM distance
    loss (and per-cell Spearman metric) is provably unaffected; the
    transform only fixes the global frame;
  * **O(2) gauge invariance** — canonicalize(M·x) == canonicalize(x) for
    ANY rotation OR reflection M, so the flow targets a single
    representative instead of an orbit (the whole point).

Plus determinism, padding/masking, degenerate slices, and apply_noise
smokes for both FM model paths (flag on). torch-gated skip.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from omegaconf import OmegaConf  # noqa: E402

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.data.canonicalize import canonicalize_cloud  # noqa: E402
from utils.data.dataholder import DataHolder  # noqa: E402


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


_REFL = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)


def _pdist(x):
    d = torch.cdist(x, x)
    n = x.shape[0]
    iu = torch.triu_indices(n, n, 1)
    return d[iu[0], iu[1]]


def _chiral_cloud(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g, dtype=torch.float64) * torch.tensor(
        [3.0, 1.0], dtype=torch.float64
    )
    x[:, 0] += 0.1 * x[:, 0] ** 2          # inject skew so chirality is defined
    x[:, 1] += 0.1 * x[:, 1] ** 2
    return x


# ----------------------------------------------------------------------
# canonicalize_cloud: the math
# ----------------------------------------------------------------------

def test_isometry_preserves_distances():
    x = _chiral_cloud()
    mask = torch.ones(1, x.shape[0], dtype=torch.bool)
    cx = canonicalize_cloud(x.unsqueeze(0), mask)[0]
    assert torch.allclose(_pdist(cx), _pdist(x), atol=1e-9), (
        "canonicalization must be an isometry (distances unchanged)"
    )


@pytest.mark.parametrize("reflect", [False, True])
def test_o2_gauge_invariance(reflect):
    x = _chiral_cloud()
    mask = torch.ones(1, x.shape[0], dtype=torch.bool)
    cx = canonicalize_cloud(x.unsqueeze(0), mask)[0]
    worst = 0.0
    g = torch.Generator().manual_seed(7)
    for _ in range(20):
        theta = float(torch.rand(1, generator=g)) * 2 * math.pi
        M = _rot(theta)
        if reflect:
            M = M @ _REFL                  # det -1
        y = x @ M.t()
        cy = canonicalize_cloud(y.unsqueeze(0), mask)[0]
        worst = max(worst, float((cy - cx).abs().max()))
    assert worst < 1e-7, (
        f"canonicalize(M x) must equal canonicalize(x) (reflect={reflect}); "
        f"worst deviation {worst:.2e}"
    )


def test_deterministic():
    x = _chiral_cloud()
    mask = torch.ones(1, x.shape[0], dtype=torch.bool)
    a = canonicalize_cloud(x.unsqueeze(0), mask)
    b = canonicalize_cloud(x.unsqueeze(0), mask)
    assert torch.equal(a, b)


def test_padding_zeroed_and_real_cells_centred():
    # 2 slices, second has 3 padding cells.
    x = _chiral_cloud(n=20).unsqueeze(0).repeat(2, 1, 1).clone()
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[1, -3:] = False
    out = canonicalize_cloud(x, mask)
    # padding rows are exactly zero
    assert torch.count_nonzero(out[1, -3:]) == 0
    # real cells are mean-centred
    real = out[1, :17]
    assert torch.allclose(real.mean(0), torch.zeros(2, dtype=out.dtype), atol=1e-6)


def test_degenerate_slices_no_nan():
    for n in (0, 1, 2, 3):
        x = torch.randn(1, max(n, 1), 2, dtype=torch.float64)
        mask = torch.zeros(1, x.shape[1], dtype=torch.bool)
        mask[0, :n] = True
        out = canonicalize_cloud(x, mask)
        assert torch.isfinite(out).all()


# ----------------------------------------------------------------------
# apply_noise smokes — flag on, both FM paths
# ----------------------------------------------------------------------

def _data(B=2, N=24, F=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataHolder(
        positions=torch.randn(B, N, 2, generator=g),
        node_features=torch.randn(B, N, F, generator=g),
        diffusion_time=0,
        cell_class=torch.zeros(B, N, dtype=torch.long),
        cell_ID=torch.zeros(B, N, dtype=torch.long),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )


def test_coordinate_fm_apply_noise_canonicalized():
    from utils.diffusion_model.diffusion.flow_matching_model import (
        FlowMatchingModel,
    )
    cfg = OmegaConf.create({"model": {"flow_matching": {
        "n_sampling_steps": 10, "eps_t": 1e-3, "sampler": "euler",
        "prediction": "x0", "prior_mode": "gaussian",
        "canonicalize_target": True,
    }}})
    model = FlowMatchingModel(cfg)
    assert model.canonicalize_target is True
    z = model.apply_noise(_data(), train_flag=False)
    assert z.positions.shape == (2, 24, 2)
    assert torch.isfinite(z.positions).all()


def test_relational_hspace_fm_apply_noise_canonicalized():
    from utils.diffusion_model.diffusion.edm_fm_model import (
        EDMFlowMatchingModel,
    )
    cfg = OmegaConf.create({"model": {
        "edm": {"enabled": True, "embed_dim": 8},
        "flow_matching": {
            "n_sampling_steps": 10, "eps_t": 1e-3,
            "canonicalize_target": True,
        },
    }})
    model = EDMFlowMatchingModel(cfg)
    assert model.canonicalize_target is True
    data = _data()
    z = model.apply_noise(data, train_flag=False)
    assert hasattr(z, "_edm_h_0") and z._edm_h_0.shape == (2, 24, 8)
    assert torch.isfinite(z._edm_h_0).all()
    # canon ∘ lift preserves true pairwise distances: d(h_0) == d(true_pos)
    for b in range(2):
        dh = _pdist(z._edm_h_0[b])
        dp = _pdist(data.positions[b])
        assert torch.allclose(dh, dp, atol=1e-4), (
            "lifted canonical h_0 must preserve the true pairwise distances"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""Regression tests for the multi-backend per-cell Spearman dispatch.

There are four backends:
  - scipy (default, baseline)
  - vectorized (batched rankdata + einsum, chunked)
  - numba (JIT'd CPU with average-tie ranking)
  - gpu (chunked torch.cdist + argsort)

Per-cell rho equivalence guarantees:
  - scipy ≡ vectorized ≡ numba   to within ~1e-7  (all use average-tie rankdata)
  - scipy ≈ gpu                   to within ~1e-3  (argsort-of-argsort does
                                                    NOT do average-tie correction;
                                                    only the diagonal is tied so
                                                    drift is small but real)

These tests run conditionally — numba/gpu are skipped gracefully if
the dep isn't installed on the test machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from scgg.evaluation.luna_metrics import (
    DEFAULT_SPEARMAN_BACKEND,
    VALID_SPEARMAN_BACKENDS,
    compute_spearman_correlation,
)


# Tolerances — see the module docstring for the rationale per backend.
RHO_ATOL_BITFAITHFUL = 1e-5    # scipy / vectorized / numba pairwise
RHO_ATOL_GPU         = 5e-3    # gpu (argsort ties)
AGG_ATOL             = 1e-5
AGG_ATOL_GPU         = 5e-3


def _random_coords(n: int, seed: int = 0):
    """Same synthetic-slice generator as the vectorized test file:
    pred = rotated + noisy version of true so we get nontrivial
    correlation (not pure noise that would zero out)."""
    rng = np.random.default_rng(seed)
    coords_true = rng.normal(size=(n, 2)).astype(np.float32)
    theta = 0.6
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    coords_pred = (coords_true @ rot) * 1.3 + rng.normal(
        scale=0.2, size=(n, 2),
    ).astype(np.float32)
    return coords_true, coords_pred


# ----------------------------------------------------------------------
# Backend availability checks
# ----------------------------------------------------------------------

def _have_numba() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


def _have_gpu() -> bool:
    try:
        import torch  # noqa
        return torch.cuda.is_available()
    except ImportError:
        return False


# ----------------------------------------------------------------------
# Basic API contract tests
# ----------------------------------------------------------------------

def test_valid_backends_list_is_canonical():
    """Locks the set of supported backend names so a future addition /
    removal trips this test and forces the author to think about
    docs and CLI flags downstream."""
    assert VALID_SPEARMAN_BACKENDS == ("scipy", "vectorized", "numba", "gpu"), (
        "if you add or remove a backend, update CLI flags in "
        "compute_extended_metrics.py + the LSF wrapper's --backend "
        "choices to match"
    )


def test_default_backend_is_scipy():
    """Default must stay 'scipy' so re-scoring existing timestamps is
    bit-faithful to whatever was scored before this knob existed."""
    assert DEFAULT_SPEARMAN_BACKEND == "scipy"


def test_unknown_backend_raises():
    coords_true, coords_pred = _random_coords(50)
    with pytest.raises(ValueError, match="unknown backend"):
        compute_spearman_correlation(coords_true, coords_pred, backend="bogus")


def test_legacy_vectorized_flag_maps_to_backend():
    """``vectorized=True`` with default backend should dispatch to
    vectorized. Confirms the back-compat shortcut still works."""
    coords_true, coords_pred = _random_coords(100)
    out_legacy = compute_spearman_correlation(
        coords_true, coords_pred, vectorized=True,
    )
    out_explicit = compute_spearman_correlation(
        coords_true, coords_pred, backend="vectorized",
    )
    np.testing.assert_allclose(
        out_legacy["per_cell"], out_explicit["per_cell"],
        atol=RHO_ATOL_BITFAITHFUL, rtol=0,
    )


def test_conflicting_vectorized_and_backend_errors():
    coords_true, coords_pred = _random_coords(50)
    with pytest.raises(ValueError, match="conflicting flags"):
        compute_spearman_correlation(
            coords_true, coords_pred,
            vectorized=True, backend="numba",
        )


# ----------------------------------------------------------------------
# Per-backend equivalence to the scipy baseline
# ----------------------------------------------------------------------

@pytest.mark.parametrize("n", [100, 300])
def test_vectorized_matches_scipy(n: int):
    coords_true, coords_pred = _random_coords(n, seed=n)
    out_scipy = compute_spearman_correlation(coords_true, coords_pred, backend="scipy")
    out_vec   = compute_spearman_correlation(coords_true, coords_pred, backend="vectorized")
    np.testing.assert_allclose(
        out_vec["per_cell"], out_scipy["per_cell"],
        atol=RHO_ATOL_BITFAITHFUL, rtol=0,
    )
    assert out_vec["mean"]   == pytest.approx(out_scipy["mean"],   abs=AGG_ATOL)
    assert out_vec["median"] == pytest.approx(out_scipy["median"], abs=AGG_ATOL)


@pytest.mark.skipif(not _have_numba(), reason="numba not installed")
@pytest.mark.parametrize("n", [100, 300])
def test_numba_matches_scipy(n: int):
    """Numba path uses real average-tie ranking, so it should be
    bit-faithful to scipy within float tolerance — same target as the
    vectorized path."""
    coords_true, coords_pred = _random_coords(n, seed=n + 1000)
    out_scipy = compute_spearman_correlation(coords_true, coords_pred, backend="scipy")
    out_numba = compute_spearman_correlation(coords_true, coords_pred, backend="numba")
    np.testing.assert_allclose(
        out_numba["per_cell"], out_scipy["per_cell"],
        atol=RHO_ATOL_BITFAITHFUL, rtol=0,
        err_msg=(
            "numba backend diverged from scipy — check the "
            "_rankdata_average tie-handling kernel"
        ),
    )
    assert out_numba["mean"]   == pytest.approx(out_scipy["mean"],   abs=AGG_ATOL)
    assert out_numba["median"] == pytest.approx(out_scipy["median"], abs=AGG_ATOL)


@pytest.mark.skipif(not _have_numba(), reason="numba not installed")
def test_numba_handles_degenerate_rows():
    """Both rho-emitting paths return NaN for cells whose distance row
    is constant (zero variance). Confirms the numba kernel's
    norm > 0 guard."""
    n = 50
    coords_true = np.random.default_rng(0).normal(size=(n, 2)).astype(np.float32)
    coords_pred = np.zeros((n, 2), dtype=np.float32)  # all cells at origin → constant rows
    out = compute_spearman_correlation(coords_true, coords_pred, backend="numba")
    assert np.all(np.isnan(out["per_cell"])), \
        "numba backend should NaN-out on degenerate input"


@pytest.mark.skipif(not _have_gpu(), reason="no CUDA GPU available")
@pytest.mark.parametrize("n", [100, 300])
def test_gpu_matches_scipy_approximately(n: int):
    """GPU path uses argsort-of-argsort which doesn't average over tie
    groups (the diagonal d[i,i]=0 is the only systematic tie). Per-cell
    rho should agree within ~1e-3, NOT bit-faithful. Looser tolerance
    here; if this fails it usually means the tie-handling drift got
    worse, not that the GPU kernel is wrong."""
    coords_true, coords_pred = _random_coords(n, seed=n + 2000)
    out_scipy = compute_spearman_correlation(coords_true, coords_pred, backend="scipy")
    out_gpu   = compute_spearman_correlation(coords_true, coords_pred, backend="gpu")
    np.testing.assert_allclose(
        out_gpu["per_cell"], out_scipy["per_cell"],
        atol=RHO_ATOL_GPU, rtol=0,
    )
    assert out_gpu["mean"]   == pytest.approx(out_scipy["mean"],   abs=AGG_ATOL_GPU)
    assert out_gpu["median"] == pytest.approx(out_scipy["median"], abs=AGG_ATOL_GPU)


def test_gpu_backend_raises_without_cuda():
    """If torch isn't installed OR no CUDA device is visible, the
    GPU backend must error LOUDLY (not silently fall back to CPU) so
    users notice and pick a different backend."""
    if _have_gpu():
        pytest.skip("CUDA is available — can't test the no-GPU error path here")
    coords_true, coords_pred = _random_coords(50)
    with pytest.raises((ImportError, RuntimeError)):
        compute_spearman_correlation(coords_true, coords_pred, backend="gpu")


# ----------------------------------------------------------------------
# P-value contract per backend
# ----------------------------------------------------------------------

def test_scipy_returns_real_pvalues():
    coords_true, coords_pred = _random_coords(100)
    out = compute_spearman_correlation(coords_true, coords_pred, backend="scipy")
    assert np.sum(np.isfinite(out["per_cell_p"])) > 50


@pytest.mark.parametrize("backend", ["vectorized"])  # numba/gpu added below conditionally
def test_non_scipy_backends_return_nan_pvalues_basic(backend):
    coords_true, coords_pred = _random_coords(100)
    out = compute_spearman_correlation(coords_true, coords_pred, backend=backend)
    assert np.all(np.isnan(out["per_cell_p"])), (
        f"{backend} backend should NaN-fill p-values"
    )


@pytest.mark.skipif(not _have_numba(), reason="numba not installed")
def test_numba_pvalues_nan():
    coords_true, coords_pred = _random_coords(100)
    out = compute_spearman_correlation(coords_true, coords_pred, backend="numba")
    assert np.all(np.isnan(out["per_cell_p"]))


@pytest.mark.skipif(not _have_gpu(), reason="no CUDA GPU available")
def test_gpu_pvalues_nan():
    coords_true, coords_pred = _random_coords(100)
    out = compute_spearman_correlation(coords_true, coords_pred, backend="gpu")
    assert np.all(np.isnan(out["per_cell_p"]))

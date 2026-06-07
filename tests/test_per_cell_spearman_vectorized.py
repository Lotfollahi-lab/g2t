"""Regression test: vectorized per-cell Spearman matches the loop.

The hot inner loop in ``scgg.evaluation.luna_metrics.compute_spearman_
correlation`` was originally::

    for i in range(n):
        rho[i], _ = scipy.stats.spearmanr(dt[i], dp[i])

which is O(n) Python-level calls × O(n log n) work per call. On CNS-
sized slices (N ~ 100-150k cells) that's literally hours per slice.
The function gained a ``vectorized=True`` path that:

  1. rank-transforms each row of dt / dp in a single
     ``scipy.stats.rankdata(axis=1)`` call (vectorised across rows
     inside scipy's C code), and
  2. reduces to row-wise Pearson correlation of the ranks via
     ``np.einsum``.

That should be 10-50× faster on large N. This test pins the numerical
equivalence so a future change to either path can't silently shift
the per-cell rho values reported in the paper.

We test on a small enough N (200) that the loop version is still fast
(~1 s); the savings only matter on big slices, but the math is the
same.
"""

from __future__ import annotations

import numpy as np
import pytest

from scgg.evaluation.luna_metrics import compute_spearman_correlation


# Tolerances rationale:
# - atol=1e-5 covers float32 distance + float64 rank mean-centring
#   round-trip noise. Empirically the worst-case per-cell diff on
#   random N=200 inputs sits around 1e-7.
# - The MEAN/MEDIAN aggregates are slightly looser because they're
#   sensitive to the order of NaN dropping (degenerate rows).
RHO_ATOL = 1e-5
AGG_ATOL = 1e-5


def _random_coords(n: int, d_true: int = 2, d_pred: int = 2, seed: int = 0):
    """Synthesise (coords_true, coords_pred) for tests. The pred coords
    are a noisy linear transform of true so there's a real correlation
    signal — pure noise would give Spearman ≈ 0 everywhere and miss
    sign-of-correlation bugs."""
    rng = np.random.default_rng(seed)
    coords_true = rng.normal(size=(n, d_true)).astype(np.float32)
    # Pred = rotated + scaled + noisy version of true (still informative).
    theta = 0.6
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    coords_pred = (coords_true @ rot) * 1.3 + rng.normal(
        scale=0.2, size=(n, d_pred),
    ).astype(np.float32)
    return coords_true, coords_pred


@pytest.mark.parametrize("n", [50, 200, 500])
def test_vectorized_matches_loop(n: int):
    """Per-cell rho values must be identical (within atol) between the
    two implementations."""
    coords_true, coords_pred = _random_coords(n, seed=n)  # vary the seed too

    out_loop = compute_spearman_correlation(
        coords_true, coords_pred, vectorized=False,
    )
    out_vec = compute_spearman_correlation(
        coords_true, coords_pred, vectorized=True,
    )

    # Per-cell array must match element-wise.
    np.testing.assert_allclose(
        out_vec["per_cell"], out_loop["per_cell"],
        atol=RHO_ATOL, rtol=0,
        err_msg=f"per-cell rho diverged for n={n}",
    )

    # Aggregates likewise.
    assert out_vec["mean"] == pytest.approx(out_loop["mean"], abs=AGG_ATOL), \
        f"mean diverged for n={n}: {out_vec['mean']} vs {out_loop['mean']}"
    assert out_vec["median"] == pytest.approx(out_loop["median"], abs=AGG_ATOL), \
        f"median diverged for n={n}: {out_vec['median']} vs {out_loop['median']}"
    assert out_vec["n"] == out_loop["n"]


def test_vectorized_returns_nan_pvalues():
    """The vectorised path doesn't compute p-values (the loop path does
    but they're not consumed downstream). We pin this so a future
    refactor doesn't accidentally start emitting bogus zero p-values
    that downstream code might misinterpret as "significant"."""
    coords_true, coords_pred = _random_coords(100)
    out_vec = compute_spearman_correlation(
        coords_true, coords_pred, vectorized=True,
    )
    assert np.all(np.isnan(out_vec["per_cell_p"])), \
        "vectorised path must return NaN p-values"


def test_loop_returns_real_pvalues():
    """Mirror: the loop path SHOULD return real p-values; this test
    locks that contract so callers who need them can still get them
    via vectorized=False."""
    coords_true, coords_pred = _random_coords(100)
    out_loop = compute_spearman_correlation(
        coords_true, coords_pred, vectorized=False,
    )
    # At least some p-values must be real (not all NaN). Some MAY be
    # NaN for degenerate rows but most won't be.
    finite = np.sum(np.isfinite(out_loop["per_cell_p"]))
    assert finite > 50, (
        f"loop path returned only {finite}/100 finite p-values — "
        "scipy.stats.spearmanr behaviour may have changed"
    )


def test_vectorized_default_is_off():
    """The default for ``vectorized`` is False (opt-in). The vectorised
    path is bit-faithful to within 1e-7, but we keep the loop as the
    DEFAULT so that re-running compute_extended_metrics over
    already-scored timestamps (e.g. the MMC sweep) cannot silently
    shift their metric values. Any future flip of this default is a
    deliberate API change that should require updating both this test
    and the docstring."""
    import inspect
    sig = inspect.signature(compute_spearman_correlation)
    assert sig.parameters["vectorized"].default is False, (
        "compute_spearman_correlation should default to vectorized=False "
        "(opt-in) — flipping to True would silently change metric "
        "values for any re-scored timestamp."
    )


@pytest.mark.parametrize("chunk_size", [16, 64, 200, 5000])
def test_chunked_matches_unchunked(chunk_size: int):
    """The chunked vectorised path is memory-bounded — at most
    ``chunk_size`` rows are ranked at a time. Chunking must NOT change
    the per-cell rho values (the chunks are independent rows, no
    cross-chunk coupling). This test sweeps the chunk_size across a
    range that brackets typical use: tiny (16, exercises the boundary
    when chunks don't divide n evenly) → larger-than-n (5000, falls
    back to single-chunk behaviour, equivalent to the un-chunked
    version)."""
    from scgg.evaluation.luna_metrics import (
        _compute_spearman_vectorized,
        compute_distance,
    )

    n = 200  # not a multiple of any chunk_size above except 200 itself
    coords_true, coords_pred = _random_coords(n, seed=1)
    dt = compute_distance(coords_true.astype(np.float32))
    dp = compute_distance(coords_pred.astype(np.float32))

    # Reference: chunk that covers all rows in one pass.
    out_ref = _compute_spearman_vectorized(dt, dp, n, chunk_size=n)
    out_chunked = _compute_spearman_vectorized(dt, dp, n, chunk_size=chunk_size)

    np.testing.assert_allclose(
        out_chunked["per_cell"], out_ref["per_cell"],
        atol=RHO_ATOL, rtol=0,
        err_msg=f"chunked output diverged at chunk_size={chunk_size}",
    )
    assert out_chunked["mean"] == pytest.approx(out_ref["mean"], abs=AGG_ATOL)
    assert out_chunked["median"] == pytest.approx(out_ref["median"], abs=AGG_ATOL)
    assert out_chunked["n"] == out_ref["n"]


def test_chunked_matches_loop():
    """End-to-end: with chunked vectorisation, the public API still
    matches the per-cell scipy.stats.spearmanr loop (the original
    test_vectorized_matches_loop covers the un-chunked case; this
    locks the chunked path on top)."""
    from scgg.evaluation.luna_metrics import (
        _compute_spearman_vectorized,
        compute_distance,
    )

    n = 200
    coords_true, coords_pred = _random_coords(n, seed=2)

    out_loop = compute_spearman_correlation(
        coords_true, coords_pred, vectorized=False,
    )
    dt = compute_distance(coords_true.astype(np.float32))
    dp = compute_distance(coords_pred.astype(np.float32))
    # Tight chunk size to actually exercise chunking
    out_chunked = _compute_spearman_vectorized(dt, dp, n, chunk_size=37)

    np.testing.assert_allclose(
        out_chunked["per_cell"], out_loop["per_cell"],
        atol=RHO_ATOL, rtol=0,
    )


def test_chunked_rejects_invalid_chunk_size():
    """``chunk_size`` ≤ 0 makes no sense (empty slice or infinite loop).
    Validate up front."""
    from scgg.evaluation.luna_metrics import (
        _compute_spearman_vectorized,
        compute_distance,
    )

    n = 50
    coords_true, coords_pred = _random_coords(n)
    dt = compute_distance(coords_true.astype(np.float32))
    dp = compute_distance(coords_pred.astype(np.float32))

    for bad in (0, -1, -1024):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            _compute_spearman_vectorized(dt, dp, n, chunk_size=bad)


def test_handles_degenerate_rows():
    """Both paths should return NaN for a cell whose true OR pred
    distance row is constant (e.g., the cell at exactly the same
    location as every other cell — unrealistic but valid input).
    This tests the NaN-where-denom-is-zero guard in the vectorised
    path."""
    n = 50
    coords_true = np.random.default_rng(0).normal(size=(n, 2)).astype(np.float32)
    # Pred coords: collapse all cells to the same point → all-zero
    # distance matrix → constant rows → spearman undefined → NaN.
    coords_pred = np.zeros((n, 2), dtype=np.float32)

    out_loop = compute_spearman_correlation(
        coords_true, coords_pred, vectorized=False,
    )
    out_vec = compute_spearman_correlation(
        coords_true, coords_pred, vectorized=True,
    )

    # Both should be all-NaN.
    assert np.all(np.isnan(out_loop["per_cell"])), \
        "loop path didn't NaN-out on degenerate input"
    assert np.all(np.isnan(out_vec["per_cell"])), \
        "vectorised path didn't NaN-out on degenerate input"

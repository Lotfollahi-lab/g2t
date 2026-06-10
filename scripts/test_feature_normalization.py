#!/usr/bin/env python
"""Pure-numpy unit tests for the data-prep feature helpers in
``run_scgg_train.py`` (normalization modes, computable batch covariates,
TRAIN-stat z-scoring).

No torch / anndata needed — these exercise the numeric core only. The
anndata-touching wrappers (``_assemble_features`` / ``_compute_feature_stats``)
are integration-gated on the cluster; here we validate the math they call.

Run:  python scgg/scripts/test_feature_normalization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_scgg_train as R  # noqa: E402


def _ok(name: str) -> None:
    print(f"  [PASS] {name}")


def test_gene_base_transform() -> None:
    rng = np.random.default_rng(0)
    X = rng.integers(0, 50, size=(7, 5)).astype(np.float64)

    # none / zscore -> identity at the base (z-score applied later)
    assert np.array_equal(R._gene_base_transform(X, "none"), X)
    assert np.array_equal(R._gene_base_transform(X, "zscore"), X)
    _ok("none/zscore base = identity")

    # log2 / log1p
    assert np.allclose(R._gene_base_transform(X, "log2"), np.log2(X + 1.0))
    assert np.allclose(R._gene_base_transform(X, "log1p"), np.log1p(X))
    _ok("log2 / log1p")

    # lognorm: each nonzero row's pre-log normalized counts sum to target
    out = R._gene_base_transform(X, "lognorm")
    recon = np.expm1(out)
    row_sums = recon.sum(axis=1)
    nz = X.sum(axis=1) > 0
    assert np.allclose(row_sums[nz], R._LOGNORM_TARGET, rtol=1e-5), row_sums
    _ok("lognorm: per-cell library size -> target sum")

    # zero-count row must not produce NaN/inf (div-by-zero guard)
    Xz = X.copy()
    Xz[0, :] = 0.0
    out_z = R._gene_base_transform(Xz, "lognorm")
    assert np.all(np.isfinite(out_z))
    _ok("lognorm: zero-count row stays finite")

    # unknown mode -> fail loud
    try:
        R._gene_base_transform(X, "bogus")
        raise AssertionError("expected ValueError on unknown mode")
    except ValueError:
        _ok("unknown normalize mode raises")


def test_covariates_from_counts() -> None:
    counts = np.array(
        [[10.0, 0.0, 5.0],   # total 15, 2 detected
         [0.0, 0.0, 0.0],    # total 0,  0 detected
         [1.0, 2.0, 3.0]],   # total 6,  3 detected
        dtype=np.float64,
    )
    cov = R._covariates_from_counts(counts)
    assert cov.shape == (3, len(R._COVAR_COLS))

    # col 0: log1p(total)
    assert np.allclose(cov[:, 0], np.log1p([15.0, 0.0, 6.0]))
    # col 1: log1p(#detected)
    assert np.allclose(cov[:, 1], np.log1p([2.0, 0.0, 3.0]))
    # col 2: per-slice median of col0, broadcast (constant within slice)
    assert np.allclose(cov[:, 2], np.median(np.log1p([15.0, 0.0, 6.0])))
    assert len(np.unique(cov[:, 2])) == 1
    # col 3: log1p(n_cells), broadcast
    assert np.allclose(cov[:, 3], np.log1p(3))
    assert np.all(np.isfinite(cov))
    _ok("covariates: depth / complexity / per-slice descriptors")


def test_stats_finalize_and_apply() -> None:
    rng = np.random.default_rng(1)
    A = rng.normal(3.0, 2.0, size=(40, 4))
    B = rng.normal(-1.0, 0.5, size=(25, 4))
    full = np.concatenate([A, B], axis=0)
    cols = ["g0", "g1", "covar_x", "cond_y"]

    # streaming accumulation (mirrors _compute_feature_stats over 2 files)
    sum_ = A.sum(0) + B.sum(0)
    sumsq = (A * A).sum(0) + (B * B).sum(0)
    mean, std = R._finalize_stats(sum_, sumsq, full.shape[0])
    # matches population mean/std (ddof=0) over the concatenation
    assert np.allclose(mean, full.mean(0))
    assert np.allclose(std, full.std(0, ddof=0))
    _ok("streaming (sum,sumsq,n) == population mean/std")

    # only flag a subset of columns (mimics std_mask)
    stats = {"g1": [float(mean[1]), float(std[1])],
             "cond_y": [float(mean[3]), float(std[3])]}
    out = R._apply_feature_stats(full.copy(), cols, stats)
    # flagged columns -> standardized (≈0 mean, ≈1 std over train)
    assert abs(out[:, 1].mean()) < 1e-9 and abs(out[:, 1].std(ddof=0) - 1.0) < 1e-9
    assert abs(out[:, 3].mean()) < 1e-9 and abs(out[:, 3].std(ddof=0) - 1.0) < 1e-9
    # unflagged columns -> untouched
    assert np.array_equal(out[:, 0], full[:, 0])
    assert np.array_equal(out[:, 2], full[:, 2])
    _ok("apply z-scores only the flagged columns, leaves others intact")

    # empty/None stats -> no-op
    assert np.array_equal(R._apply_feature_stats(full.copy(), cols, None), full)
    assert np.array_equal(R._apply_feature_stats(full.copy(), cols, {}), full)
    _ok("empty stats -> identity")


def test_constant_column_std_floor() -> None:
    # a constant column has zero variance; std is floored to 1.0 so
    # standardization maps it to 0 (not inf/nan).
    M = np.ones((10, 1)) * 7.0
    mean, std = R._finalize_stats(M.sum(0), (M * M).sum(0), 10)
    assert std[0] == 1.0  # floored
    out = R._apply_feature_stats(M.copy(), ["c"], {"c": [float(mean[0]), float(std[0])]})
    assert np.allclose(out[:, 0], 0.0)
    assert np.all(np.isfinite(out))
    _ok("constant column: std floor -> standardizes to 0, stays finite")


def test_norm_modes_registered() -> None:
    for m in ("none", "log2", "log1p", "lognorm", "zscore", "lognorm_zscore"):
        assert m in R._NORM_MODES
    _ok("all normalize modes registered")


if __name__ == "__main__":
    print("test_gene_base_transform"); test_gene_base_transform()
    print("test_covariates_from_counts"); test_covariates_from_counts()
    print("test_stats_finalize_and_apply"); test_stats_finalize_and_apply()
    print("test_constant_column_std_floor"); test_constant_column_std_floor()
    print("test_norm_modes_registered"); test_norm_modes_registered()
    print("\nAll feature-normalization unit tests passed.")

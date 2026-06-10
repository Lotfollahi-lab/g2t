"""Tests for the global-structure / anti-collapse metric
(``luna_metrics.compute_global_structure``).

The per-cell Spearman metric is LOCAL and cannot see a 1-D dimensional
collapse (a 'snake' that preserves neighbour order but destroys the 2-D
layout). These tests pin the two detectors:

  * isometry of the truth → anisotropy preserved (error ≈ 0) and global
    distance correlation ≈ 1;
  * collapse to a line → predicted anisotropy ratio → 0 (large error)
    and lower global distance correlation;
  * a space-filling 'snake' (local order kept, global scrambled) → low
    global distance correlation even though it is a worst case for the
    local metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scgg.evaluation.luna_metrics import compute_global_structure  # noqa: E402


def _elongated_blob(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 2)) * np.array([3.0, 1.0])


def _rotate(x, theta=0.7):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return x @ R.T


def _principal_axis_line(x):
    xc = x - x.mean(0)
    vt = np.linalg.svd(xc, full_matrices=False)[2]
    return np.c_[xc @ vt[0], np.zeros(len(x))]


def test_isometry_is_perfect():
    true = _elongated_blob()
    g = compute_global_structure(true, _rotate(true))
    assert abs(g["anisotropy_ratio_error"]) < 1e-6
    assert abs(g["global_distance_spearman"] - 1.0) < 1e-6
    assert abs(g["global_distance_pearson"] - 1.0) < 1e-6


def test_line_collapse_is_flagged():
    true = _elongated_blob()
    c = compute_global_structure(true, _principal_axis_line(true))
    # predicted cloud is rank-1 → anisotropy ratio collapses to ~0
    assert c["anisotropy_ratio_pred"] < 1e-6
    # error ≈ the true ratio (large signal)
    assert c["anisotropy_ratio_error"] > 0.05
    # and global distance correlation is below a perfect isometry's 1.0
    assert c["global_distance_spearman"] < 0.999


def test_snake_collapse_tanks_global_corr():
    """A space-filling snake keeps LOCAL order (would fool per-cell
    Spearman) but scrambles long-range distances → low global corr."""
    true = _elongated_blob(n=2000, seed=1)
    # order cells along a boustrophedon over a coarse grid, lay on a line
    gx = np.floor((true[:, 0] - true[:, 0].min()) /
                  (np.ptp(true[:, 0]) / 20 + 1e-9)).astype(int)
    key = gx * 1e6 + np.where(gx % 2 == 0, true[:, 1], -true[:, 1])
    order = np.argsort(key)
    snake = np.zeros_like(true)
    snake[order, 0] = np.arange(len(true), dtype=float)
    g_iso = compute_global_structure(true, _rotate(true))
    g_snake = compute_global_structure(true, snake)
    assert g_snake["anisotropy_ratio_pred"] < 1e-6
    assert g_snake["global_distance_spearman"] < g_iso["global_distance_spearman"]


def test_degenerate_inputs_no_crash():
    # < 4 cells → distance corr is NaN, anisotropy still defined for >=2
    for n in (1, 2, 3):
        out = compute_global_structure(
            np.random.default_rng(0).standard_normal((n, 2)),
            np.random.default_rng(1).standard_normal((n, 2)),
        )
        assert np.isnan(out["global_distance_spearman"])
        assert set(out) >= {"anisotropy_ratio_pred", "anisotropy_ratio_error"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

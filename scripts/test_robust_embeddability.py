#!/usr/bin/env python
"""Pure-numpy validation of the robust-distance-loss (B) and embeddability /
triangle-inequality (C) math in ``src/metrics/loss_function.py``.

The torch LossFunction can't be imported off-cluster, so we mirror the two
numeric cores here and assert their defining properties — this catches
formula mistakes before the cluster integration gate.

Run:  python scgg/scripts/test_robust_embeddability.py
"""
from __future__ import annotations

import numpy as np


def _ok(name: str) -> None:
    print(f"  [PASS] {name}")


# ---- B: robust reduce (mirror LossFunction._robust_reduce) ----------------
def robust_reduce(resid, mode, c):
    r2 = resid ** 2
    if mode == "none":
        return float(r2.mean())
    c2 = c * c
    if mode == "huber":
        absr = np.abs(resid)
        lin = 2.0 * c * absr - c2
        return float(np.where(absr <= c, r2, lin).mean())
    if mode == "gm":
        return float((c2 * r2 / (r2 + c2)).mean())
    if mode == "tls":
        return float(np.minimum(r2, c2).mean())
    raise ValueError(mode)


def robust_scale(step, c0, gnc_steps):
    if gnc_steps <= 0:
        return c0
    p = min(1.0, step / float(gnc_steps))
    return c0 * (8.0 * (1.0 - p) + p)


def test_robust_reduces_to_mse_near_zero():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1e-3, size=5000)   # tiny residuals
    base = robust_reduce(r, "none", 1.0)
    for mode in ("huber", "gm", "tls"):
        val = robust_reduce(r, mode, 1.0)
        assert abs(val - base) <= 1e-2 * base + 1e-12, (mode, val, base)
    _ok("robust kernels ≈ MSE for small residuals (all pass the quadratic core)")


def test_robust_caps_outlier_influence():
    # 100 inliers + 1 gross outlier. L2 is dominated by the outlier;
    # gm/tls cap its per-term contribution at ~c², so their mean barely moves.
    resid = np.concatenate([np.full(100, 0.01), np.array([1000.0])])
    c = 1.0
    none = robust_reduce(resid, "none", c)
    gm = robust_reduce(resid, "gm", c)
    tls = robust_reduce(resid, "tls", c)
    huber = robust_reduce(resid, "huber", c)
    assert none > 9000, none                       # ~1e6/101 (L2 explodes)
    # gm/tls cap the outlier's per-term contribution at ~c², so the mean is
    # O((inliers + c²)/n) ≈ 0.01, not O(outlier²/n) ≈ 9900.
    assert gm < 0.02, gm
    assert tls < 0.02, tls
    assert huber < none / 100.0, (huber, none)     # linear tail << quadratic
    _ok("robust kernels cap outlier influence (gm/tls bounded, L2 explodes)")


def test_robust_properties():
    # Huber continuity at |r|=c: quad c² == linear 2c·c − c² = c².
    c = 2.0
    assert abs(robust_reduce(np.array([c]), "huber", c)
               - robust_reduce(np.array([c]), "none", c)) < 1e-9
    # GM monotone increasing in |r| and bounded by c².
    rs = np.linspace(0, 100, 50)
    gm_vals = [robust_reduce(np.array([r]), "gm", c) for r in rs]
    assert all(b >= a - 1e-9 for a, b in zip(gm_vals, gm_vals[1:])), "gm not monotone"
    assert gm_vals[-1] <= c * c + 1e-6 and gm_vals[-1] > 0.9 * c * c, gm_vals[-1]
    # TLS caps at c².
    assert abs(robust_reduce(np.array([1e6]), "tls", c) - c * c) < 1e-3
    _ok("robust kernels: huber C¹ at r=c, gm monotone→c², tls capped at c²")


def test_gnc_schedule():
    c0 = 1.0
    assert robust_scale(0, c0, 300) == 8.0 * c0        # starts ~convex (8c0)
    assert abs(robust_scale(300, c0, 300) - c0) < 1e-9  # ends at target c0
    assert robust_scale(10_000, c0, 300) == c0          # clamped at p=1
    assert robust_scale(5, c0, 0) == c0                 # gnc off -> constant
    vals = [robust_scale(s, c0, 300) for s in range(0, 301, 30)]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:])), vals  # non-increasing
    _ok("GNC schedule: 8·c0 → c0 monotone over gnc_steps, clamped, off-switch")


# ---- C: triangle-inequality penalty (mirror _compute_embeddability) -------
def _cdist(x):
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt((diff * diff).sum(-1) + 1e-16)


def tri_penalty(d, triples):
    ii, jj, kk = triples
    viol = np.maximum(0.0, d[ii, jj] - d[ii, kk] - d[kk, jj])
    return float((viol ** 2).mean())


def test_embeddability_zero_for_valid_euclidean():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 2))          # a genuine 2D config
    d = _cdist(X)
    n = X.shape[0]
    tr = (rng.integers(0, n, 6000), rng.integers(0, n, 6000), rng.integers(0, n, 6000))
    pen = tri_penalty(d, tr)
    # Euclidean distances obey the triangle inequality exactly -> ~0.
    assert pen < 1e-9, pen
    _ok("embeddability penalty ~0 for a valid Euclidean (metric) config")


def test_embeddability_positive_for_nonmetric():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 2))
    d = _cdist(X)
    # Inject a gross triangle violation: make one pair's distance huge.
    d[0, 1] = d[1, 0] = 100.0
    n = X.shape[0]
    # triples that include the corrupted (0,1) pair as the long edge.
    ii = np.zeros(500, dtype=int)
    jj = np.ones(500, dtype=int)
    kk = rng.integers(2, n, 500)
    pen = tri_penalty(d, (ii, jj, kk))
    assert pen > 1.0, pen           # d_01 (=100) >> d_0k + d_k1 -> big violation
    _ok("embeddability penalty > 0 for a non-metric (triangle-violating) matrix")


if __name__ == "__main__":
    print("test_robust_reduces_to_mse_near_zero"); test_robust_reduces_to_mse_near_zero()
    print("test_robust_caps_outlier_influence"); test_robust_caps_outlier_influence()
    print("test_robust_properties"); test_robust_properties()
    print("test_gnc_schedule"); test_gnc_schedule()
    print("test_embeddability_zero_for_valid_euclidean"); test_embeddability_zero_for_valid_euclidean()
    print("test_embeddability_positive_for_nonmetric"); test_embeddability_positive_for_nonmetric()
    print("\nAll robust-loss + embeddability math tests passed.")

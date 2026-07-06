#!/usr/bin/env python
"""Pure-numpy validation of the domain-adaptation math in
``src/models/domain_adaptation.py`` (CORAL, RBF-MMD, the DANN λ schedule).

The torch module can't be imported off-cluster, so we re-implement the
formulas here and assert their defining properties — this catches formula
mistakes before the cluster integration gate.

Run:  python scgg/scripts/test_domain_adaptation.py
"""
from __future__ import annotations

import math

import numpy as np


def _ok(name: str) -> None:
    print(f"  [PASS] {name}")


# ---- reference re-implementations (mirror domain_adaptation.py) -----------
def _cov(z, eps=1e-5):
    z = z - z.mean(0, keepdims=True)
    c = (z.T @ z) / max(1, z.shape[0] - 1)
    return c + eps * np.eye(z.shape[1])


def coral(zs, zt):
    if zs.shape[0] < 2 or zt.shape[0] < 2:
        return 0.0
    d = zs.shape[1]
    return float(((_cov(zs) - _cov(zt)) ** 2).sum() / (4.0 * d * d))


def mmd(zs, zt, sigmas=(1.0, 2.0, 4.0, 8.0, 16.0)):
    def k(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        out = np.zeros_like(d2)
        for s in sigmas:
            out += np.exp(-d2 / (2.0 * s * s))
        return out / len(sigmas)
    return float(k(zs, zs).mean() + k(zt, zt).mean() - 2.0 * k(zs, zt).mean())


def grl_lambda(step, warmup, max_lambda=1.0):
    if warmup <= 0:
        return float(max_lambda)
    p = min(1.0, max(0.0, step / warmup))
    return float(max_lambda) * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ---- tests ----------------------------------------------------------------
def test_coral_zero_when_matched() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, size=(4000, 8))
    B = rng.normal(0, 1, size=(4000, 8))  # same distribution
    matched = coral(A, B)
    # shifted COVARIANCE (not just mean — CORAL is 2nd-order, mean-blind)
    C = rng.normal(0, 1, size=(4000, 8)) * np.array([4.0] + [1.0] * 7)
    shifted = coral(A, C)
    assert matched < 0.05, matched
    assert shifted > 10 * matched + 0.1, (matched, shifted)
    _ok("CORAL ~0 for matched covariance, large for shifted")


def test_coral_is_mean_invariant() -> None:
    # CORAL aligns covariance only -> a pure mean shift must not change it.
    rng = np.random.default_rng(1)
    A = rng.normal(0, 1, size=(3000, 6))
    B = A + 10.0  # identical covariance, shifted mean
    assert coral(A, B) < 1e-6, coral(A, B)
    _ok("CORAL is mean-shift invariant (2nd-order only)")


def test_mmd_zero_when_matched() -> None:
    rng = np.random.default_rng(2)
    A = rng.normal(0, 1, size=(800, 5))
    B = rng.normal(0, 1, size=(800, 5))
    C = rng.normal(3.0, 1, size=(800, 5))  # mean-shifted -> MMD sees it
    matched = mmd(A, B)
    shifted = mmd(A, C)
    assert abs(matched) < 0.02, matched
    assert shifted > matched + 0.1, (matched, shifted)
    _ok("MMD ~0 for matched, large for shifted (mean-sensitive, unlike CORAL)")


def _domain_invariance(slices, mode="coral"):
    """Mirror of LossFunction._compute_domain_invariance: mean pairwise
    CORAL/MMD over the per-slice feature sets in a batch."""
    sets = [s for s in slices if s.shape[0] >= 2]
    if len(sets) < 2:
        return 0.0
    fn = coral if mode == "coral" else mmd
    terms = [fn(sets[i], sets[j])
             for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return float(np.mean(terms)) if terms else 0.0


def test_domain_invariance_across_slices() -> None:
    rng = np.random.default_rng(7)
    D = 8
    # 4 slices from the SAME distribution -> mean pairwise CORAL ~0.
    same = [rng.normal(0, 1, size=(2000, D)) for _ in range(4)]
    inv_same = _domain_invariance(same, "coral")
    # 4 slices each with a DIFFERENT per-dim covariance scale -> large.
    scales = [np.array([1.0] * D), np.array([5.0] + [1.0] * (D - 1)),
              np.array([1.0, 6.0] + [1.0] * (D - 2)), np.array([3.0] * D)]
    diff = [rng.normal(0, 1, size=(2000, D)) * s for s in scales]
    inv_diff = _domain_invariance(diff, "coral")
    assert inv_same < 0.05, inv_same
    assert inv_diff > 10 * inv_same + 0.1, (inv_same, inv_diff)
    # < 2 usable slices -> exactly 0 (graph-zero path).
    assert _domain_invariance([same[0]], "coral") == 0.0
    assert _domain_invariance([same[0], same[1][:1]], "coral") == 0.0
    _ok("domain-invariance: mean pairwise CORAL ~0 for matched slices, "
        "large for slice-shifted covariances, 0 when <2 usable slices")


def test_grl_lambda_schedule() -> None:
    assert grl_lambda(0, 300) == 0.0
    assert grl_lambda(-5, 300) == 0.0
    v_mid = grl_lambda(150, 300)
    v_end = grl_lambda(300, 300)
    assert 0.0 < v_mid < v_end <= 1.0, (v_mid, v_end)
    assert v_end > 0.99, v_end                 # ramps to ~max by warmup end
    assert grl_lambda(10_000, 300) == grl_lambda(300, 300)  # clamped at p=1
    assert grl_lambda(5, 0) == 1.0             # warmup<=0 -> constant max
    # monotonic non-decreasing over the ramp
    vals = [grl_lambda(s, 300) for s in range(0, 301, 30)]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])), vals
    _ok("GRL λ schedule: 0 -> max monotone over warmup, clamped, off-switch")


if __name__ == "__main__":
    print("test_coral_zero_when_matched"); test_coral_zero_when_matched()
    print("test_coral_is_mean_invariant"); test_coral_is_mean_invariant()
    print("test_mmd_zero_when_matched"); test_mmd_zero_when_matched()
    print("test_domain_invariance_across_slices"); test_domain_invariance_across_slices()
    print("test_grl_lambda_schedule"); test_grl_lambda_schedule()
    print("\nAll domain-adaptation math tests passed.")

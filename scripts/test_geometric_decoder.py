#!/usr/bin/env python
"""Pure-numpy validation of the differentiable SMACOF decoder math in
``src/models/geometric_decoder.py`` (the torch module can't be imported
off-cluster, so we mirror the Guttman update in numpy and check its
defining properties).

Run:  python scgg/scripts/test_geometric_decoder.py
"""
from __future__ import annotations

import numpy as np


def _cdist(x):
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt((diff * diff).sum(-1) + 1e-24)


def _stress(delta, x):
    d = _cdist(x)
    iu = np.triu_indices(x.shape[0], 1)
    return float(((d - delta)[iu] ** 2).sum())


def smacof(delta, x_init, n_iter=200, eps=1e-8):
    n = x_init.shape[0]
    x = x_init.copy()
    delta = np.clip(delta, 0, None)
    for _ in range(n_iter):
        d = _cdist(x)
        inv = np.where(d > eps, 1.0 / d, 0.0)
        R = delta * inv
        np.fill_diagonal(R, 0.0)
        B = -R
        B[np.diag_indices(n)] = R.sum(1)
        x = (B @ x) / n
        x -= x.mean(0, keepdims=True)
    return x


def classical_mds_2d(delta):
    """Top-2 classical MDS from a distance matrix (for the comparison)."""
    D2 = delta ** 2
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)               # ascending
    idx = np.argsort(w)[::-1][:2]
    L = np.clip(w[idx], 0, None)
    return V[:, idx] * np.sqrt(L)[None, :]


def _ok(name):
    print(f"  [PASS] {name}")


def test_true_config_is_fixed_point():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    X -= X.mean(0)
    delta = _cdist(X)
    out = smacof(delta, X.copy(), n_iter=50)
    # exact config -> should stay (up to recentring); stress ~0 both.
    assert _stress(delta, out) < 1e-6, _stress(delta, out)
    _ok("true configuration is a fixed point (stress stays ~0)")


def test_recovers_2d_from_distances():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    delta = _cdist(X)
    x0 = rng.normal(size=(60, 2)) * 0.1        # bad init
    out = smacof(delta, x0, n_iter=300)
    # recovered pairwise distances match the target (rotation-invariant)
    rel = np.abs(_cdist(out) - delta).sum() / delta.sum()
    assert rel < 1e-3, rel
    _ok("recovers a 2D embedding from its distance matrix (rel err < 1e-3)")


def test_stress_non_increasing():
    rng = np.random.default_rng(2)
    X3 = rng.normal(size=(50, 3))              # 3D -> NOT 2D-embeddable
    delta = _cdist(X3)
    x0 = classical_mds_2d(delta)
    s_prev = _stress(delta, x0)
    x = x0.copy()
    for _ in range(30):                        # check each step is non-increasing
        x = smacof(delta, x, n_iter=1)
        s = _stress(delta, x)
        assert s <= s_prev + 1e-6, (s_prev, s)
        s_prev = s
    _ok("stress is non-increasing across Guttman iterations (majorization)")


def test_beats_classical_mds_on_non2d():
    # On a genuinely non-2D distance matrix, SMACOF (minimising 2D stress)
    # should reach <= the stress of the classical top-2 MDS truncation.
    rng = np.random.default_rng(3)
    X3 = rng.normal(size=(80, 3))
    delta = _cdist(X3)
    x_mds = classical_mds_2d(delta)
    x_smacof = smacof(delta, x_mds.copy(), n_iter=300)
    s_mds = _stress(delta, x_mds)
    s_smacof = _stress(delta, x_smacof)
    assert s_smacof <= s_mds + 1e-6, (s_mds, s_smacof)
    assert s_smacof < 0.999 * s_mds, (s_mds, s_smacof)   # strictly better here
    print(f"    stress: classical MDS={s_mds:.3f} -> SMACOF={s_smacof:.3f}")
    _ok("SMACOF beats classical top-2 MDS on non-2D distances")


def _shape_mse(pp, pt):
    """Mirror of _compute_edm_coord_mse: centre + RMS-normalise both, then
    orthogonal-Procrustes-align pp->pt, then MSE. Scale/rotation/reflection/
    translation invariant."""
    pp = pp - pp.mean(0); pt = pt - pt.mean(0)
    pp = pp / np.sqrt((pp ** 2).sum(-1).mean())
    pt = pt / np.sqrt((pt ** 2).sum(-1).mean())
    U, _, Vt = np.linalg.svd(pp.T @ pt)        # orthogonal Procrustes R=U@Vt
    ppa = pp @ (U @ Vt)
    return float(((ppa - pt) ** 2).mean())


def test_coord_loss_similarity_invariant():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 2))
    # rotate + reflect + scale + translate a copy -> loss ~0 (invariances)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Y = (X @ R) * 3.7 + np.array([5.0, -2.0])
    Y = Y @ np.array([[1, 0], [0, -1]])        # reflection too
    assert _shape_mse(X, Y) < 1e-9, _shape_mse(X, Y)
    # a genuinely different shape -> strictly positive
    Z = rng.normal(size=(50, 2))
    assert _shape_mse(X, Z) > 0.1, _shape_mse(X, Z)
    _ok("coord loss: invariant to rotation+reflection+scale+translation, "
        "positive for different shapes")


if __name__ == "__main__":
    print("test_true_config_is_fixed_point"); test_true_config_is_fixed_point()
    print("test_recovers_2d_from_distances"); test_recovers_2d_from_distances()
    print("test_stress_non_increasing"); test_stress_non_increasing()
    print("test_beats_classical_mds_on_non2d"); test_beats_classical_mds_on_non2d()
    print("test_coord_loss_similarity_invariant"); test_coord_loss_similarity_invariant()
    print("\nAll SMACOF geometric-decoder tests passed.")

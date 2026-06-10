#!/usr/bin/env python
"""Unit tests for the geodesic landmark-distance target
(``src/utils/data/geodesic.py``). numpy + scipy only — no torch.

Run:  python scgg/scripts/test_geodesic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.data.geodesic import geodesic_landmark_distances  # noqa: E402


def _ok(name: str) -> None:
    print(f"  [PASS] {name}")


def test_shape_and_self_zero() -> None:
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 1, size=(120, 2))
    land = np.array([3, 50, 99], dtype=np.int64)
    D = geodesic_landmark_distances(pos, land, k=10)
    assert D.shape == (120, 3)
    assert np.all(np.isfinite(D))
    # a landmark's geodesic distance to itself is 0
    for j, li in enumerate(land):
        assert abs(D[li, j]) < 1e-9, (li, j, D[li, j])
    _ok("shape (n,M), finite, landmark self-distance == 0")


def test_geodesic_geq_euclidean() -> None:
    # Fundamental invariant: a shortest PATH can never be shorter than the
    # straight-line distance. Holds for every (cell, landmark) pair.
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 10, size=(200, 2))
    land = np.array([0, 75, 150, 199], dtype=np.int64)
    D = geodesic_landmark_distances(pos, land, k=12)
    eucl = np.sqrt(
        ((pos[:, None, :] - pos[land][None, :, :]) ** 2).sum(-1)
    )  # (n, M)
    assert np.all(D >= eucl - 1e-6), float((eucl - D).max())
    _ok("geodesic >= euclidean for all (cell, landmark) pairs")


def test_nonconvex_detour() -> None:
    # A "C"/horseshoe: the two free ends are close in the plane but far
    # along the manifold. Geodesic must exceed Euclidean by a clear margin
    # for the end-to-end pair, while Euclidean would shortcut the gap.
    t = np.linspace(0.15 * np.pi, 1.85 * np.pi, 240)   # open arc (a "C")
    arc = np.stack([np.cos(t), np.sin(t)], axis=1)
    pos = arc + 1e-3 * np.random.default_rng(2).normal(size=arc.shape)
    end_a, end_b = 0, len(t) - 1  # the two open ends of the C
    land = np.array([end_b], dtype=np.int64)
    D = geodesic_landmark_distances(pos, land, k=6)
    geo = D[end_a, 0]
    eucl = np.linalg.norm(pos[end_a] - pos[end_b])
    # ends are near in the plane (chord across the C's opening, ~0.9)...
    assert eucl < 1.1, eucl
    # ...but the path around the C is long (~ the arc length ≈ 1.7π·r ≈ 5.3).
    assert geo > 3.0, geo
    assert geo > 3.0 * eucl, (geo, eucl)
    _ok("non-convex 'C': geodesic >> euclidean across the gap")


def test_disconnected_is_capped_finite() -> None:
    # Two far-apart clusters + small k -> the kNN graph is disconnected.
    # Cross-cluster geodesics are unreachable (inf) and must be capped to a
    # finite value, never returned as inf/nan.
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 1, size=(40, 2))
    b = rng.uniform(100, 101, size=(40, 2))
    pos = np.concatenate([a, b], axis=0)
    land = np.array([0], dtype=np.int64)            # a landmark in cluster A
    D = geodesic_landmark_distances(pos, land, k=5)
    assert np.all(np.isfinite(D)), "disconnected pairs must be capped finite"
    # cluster-B points are unreachable -> capped at the max finite (the
    # farthest reachable A point), so they're >= any within-A distance.
    assert D[40:, 0].min() >= D[:40, 0].max() - 1e-9
    _ok("disconnected components capped to finite (no inf/nan)")


def test_degenerate_sizes() -> None:
    # n==1 and empty landmark set must not crash.
    assert geodesic_landmark_distances(np.zeros((1, 2)), np.array([0]), k=5).shape == (1, 1)
    assert geodesic_landmark_distances(np.zeros((0, 2)), np.array([], dtype=np.int64), k=5).shape == (0, 0)
    _ok("degenerate n==1 / empty landmark set handled")


if __name__ == "__main__":
    print("test_shape_and_self_zero"); test_shape_and_self_zero()
    print("test_geodesic_geq_euclidean"); test_geodesic_geq_euclidean()
    print("test_nonconvex_detour"); test_nonconvex_detour()
    print("test_disconnected_is_capped_finite"); test_disconnected_is_capped_finite()
    print("test_degenerate_sizes"); test_degenerate_sizes()
    print("\nAll geodesic unit tests passed.")

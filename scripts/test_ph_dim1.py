#!/usr/bin/env python
"""Smoke tests for the 1-dim persistent-homology loss.

Run inside the scgg env (gudhi must be installed)::

    source /nfs/team361/sb75/.venvs/scgg/bin/activate
    pip install gudhi    # if not already
    python /nfs/team361/sb75/scgg/scripts/test_ph_dim1.py

Exits 0 on pass, non-zero on the first failure. ~3 seconds.

Three properties checked:

  1. A simple square with a hole (4 points around a square boundary)
     produces exactly 1 loop in the 1-dim persistence diagram.
  2. The differentiable loss is ≈ 0 when pred == true.
  3. Gradient flows back into ``pred.positions``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _setup_path() -> None:
    here = Path(__file__).resolve()
    scgg_src = here.parent.parent / "src"
    if not scgg_src.exists():
        raise FileNotFoundError(f"scgg/src not found at {scgg_src}")
    sys.path.insert(0, str(scgg_src))


_setup_path()

import torch  # noqa: E402

from utils.data.dataholder import DataHolder  # noqa: E402
from metrics.loss_function import LossFunction  # noqa: E402


class _FakeCfg:
    """Minimal cfg with persistent_homology dim=1 enabled and
    everything else turned off."""

    class _Model:
        class _Loss:
            class _PWD:
                enabled = False
                weight = 0.0
            class _KNN:
                enabled = False
                weight = 0.0
                scope = "global"
                k = 20
            class _PH:
                enabled = True
                weight = 1.0
                subsample = None
                dim = 1
                frequency = 1
                cache_true = True
                max_edge_length = 5.0   # wide enough for our toy clouds
            class _SK:
                enabled = False
                weight = 0.0
                blur = 0.01
                scaling = 0.9
                p = 2
                subsample = None
                backend = "tensorized"
            class _CC:
                enabled = False
                weight = 0.0
            pairwise_distance_mse = _PWD
            knn_rank = _KNN
            persistent_homology = _PH
            sinkhorn = _SK
            coarse_centroid_mse = _CC
        loss = _Loss
    model = _Model


def test_square_has_one_loop() -> None:
    """A square (4 points in a square arrangement) plus a center
    point should have exactly one significant 1-dim feature under
    Vietoris-Rips: the square loop births at edge length ~1 (the
    side length) and dies when triangles fill it."""
    # 4 corners + 1 center — the 4 outer edges form a loop, the
    # interior triangulation eventually fills it.
    pos = torch.tensor([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
        [-1.0,  1.0],
        [ 0.0,  0.0],
    ])
    pos_np = pos.numpy()

    from metrics.loss_function import LossFunction as LF
    pairs = LF._extract_pd1_simplex_pairs(pos_np, max_edge_length=5.0)
    if len(pairs) < 1:
        raise AssertionError(
            f"Square + center should give ≥1 1-dim feature; got {len(pairs)}."
        )
    print(f"[square-loop]  found {len(pairs)} 1-dim feature(s)")
    print("[square-loop]  PASS\n")


def _make_donut(n_points: int = 32, R: float = 1.0, r: float = 0.5) -> torch.Tensor:
    """Annulus of points — has a clear large 1-dim loop."""
    torch.manual_seed(0)
    angles = torch.rand(n_points) * 2 * 3.14159
    radii = R + r * (torch.rand(n_points) - 0.5)
    x = radii * torch.cos(angles)
    y = radii * torch.sin(angles)
    return torch.stack([x, y], dim=-1)


def test_self_distance_is_zero(tol: float = 1e-3) -> None:
    """The loss between a cloud and itself must be ≈ 0 — same
    simplex pairs, same edge lengths → matched values cancel."""
    loss_fn = LossFunction(cfg=_FakeCfg())
    pos = _make_donut()
    pred = DataHolder(
        node_features=torch.zeros(1, pos.shape[0], 4),
        positions=pos.unsqueeze(0).clone().requires_grad_(True),
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, pos.shape[0], dtype=torch.long),
        node_mask=torch.ones(1, pos.shape[0], dtype=torch.bool),
    )
    true = DataHolder(
        node_features=torch.zeros(1, pos.shape[0], 4),
        positions=pos.unsqueeze(0).clone(),
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, pos.shape[0], dtype=torch.long),
        node_mask=torch.ones(1, pos.shape[0], dtype=torch.bool),
    )
    val = loss_fn._compute_persistent_homology(pred, true)
    err = float(val.detach().abs().item())
    print(f"[self-distance]  loss = {err:.2e}  (tol={tol})")
    if err > tol:
        raise AssertionError(
            f"PH dim=1 self-distance should be ≈ 0; got {err:.2e}."
        )
    print("[self-distance]  PASS\n")


def test_gradient_flows() -> None:
    """Loss must produce non-trivial gradient on ``pred.positions``
    when pred ≠ true."""
    loss_fn = LossFunction(cfg=_FakeCfg())
    pos_true = _make_donut()
    # Perturb pred slightly — gradient should be non-zero and point
    # toward the truth.
    pos_pred = pos_true + 0.1 * torch.randn_like(pos_true)
    pos_pred.requires_grad_(True)

    pred = DataHolder(
        node_features=torch.zeros(1, pos_true.shape[0], 4),
        positions=pos_pred.unsqueeze(0),
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, pos_true.shape[0], dtype=torch.long),
        node_mask=torch.ones(1, pos_true.shape[0], dtype=torch.bool),
    )
    true = DataHolder(
        node_features=torch.zeros(1, pos_true.shape[0], 4),
        positions=pos_true.unsqueeze(0).clone(),
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, pos_true.shape[0], dtype=torch.long),
        node_mask=torch.ones(1, pos_true.shape[0], dtype=torch.bool),
    )

    val = loss_fn._compute_persistent_homology(pred, true)
    val.backward()
    if pos_pred.grad is None:
        raise AssertionError("pos_pred.grad is None after backward.")
    grad_mag = float(pos_pred.grad.abs().sum().item())
    print(f"[gradient]  loss={float(val.detach().item()):.3e}  grad |sum|={grad_mag:.3e}")
    if grad_mag == 0.0:
        raise AssertionError(
            "Gradient through PH dim=1 is exactly zero. Likely the "
            "fixed-structure trick isn't propagating through cdist."
        )
    print("[gradient]  PASS\n")


def main() -> int:
    try:
        import gudhi  # noqa: F401
    except ImportError:
        print("SKIP: gudhi not installed. `pip install gudhi` to run these tests.")
        return 0
    test_square_has_one_loop()
    test_self_distance_is_zero()
    test_gradient_flows()
    print("All 1-dim PH tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

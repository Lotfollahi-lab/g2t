#!/usr/bin/env python
"""Tests for the Sinkhorn loss + its Procrustes-alignment fix.

The diagnosis (see _compute_sinkhorn docstring): without alignment,
Sinkhorn measures the rotation noise produced by EDM's Procrustes-to-
noise step and FM's rotation augmentation, not per-cell placement
error. The fix is a differentiable Kabsch / orthogonal Procrustes
alignment of pred → true before Sinkhorn.

These tests verify:

  (A) ``_procrustes_align_2d`` math
      * Recovers identity on identical clouds.
      * Recovers rotation angle θ on rotated clouds.
      * Recovers reflection on reflected clouds.
      * Gradient flows through SVD backward.

  (B) ``_compute_sinkhorn`` rotation-invariance after the fix
      To run independently of geomloss (which isn't always
      installed locally), we monkey-patch the cached SamplesLoss with
      a *Chamfer-like* stand-in that shares Sinkhorn's
      rotation-NON-invariance property. The stand-in has the same
      signature ``f(pred, true) -> scalar``; we can then verify that:

      * with procrustes_align=False: loss is HIGH on rotated pred.
      * with procrustes_align=True:  loss is LOW on rotated pred.

      This isolates the alignment behavior from geomloss specifics.

  (C) Real geomloss-based smoke test
      * Only runs if geomloss imports cleanly. Confirms the loss
        computes a finite value on a real point cloud, with both
        flags off (legacy behavior) and with the fix on.

Run from the repo root:

    python scripts/test_sinkhorn.py

Exits 0 on pass, non-zero on first failure.
"""

from __future__ import annotations

import math
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
from metrics.loss_function import (  # noqa: E402
    LossFunction,
    _procrustes_align_2d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rot2d(theta: float) -> torch.Tensor:
    """2-D rotation matrix for angle theta (radians)."""
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]])


def _refl_x() -> torch.Tensor:
    """Reflection across the x-axis."""
    return torch.tensor([[1.0, 0.0], [0.0, -1.0]])


def _build_holders(pred_pos: torch.Tensor, true_pos: torch.Tensor):
    """Build matched DataHolders for a single batch element with N cells."""
    N = pred_pos.shape[0]
    mask = torch.ones(1, N, dtype=torch.bool)
    pred = DataHolder(
        node_features=torch.zeros(1, N, 4),
        positions=pred_pos.unsqueeze(0),
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, N, 1, dtype=torch.long),
        cell_ID=torch.arange(N).unsqueeze(0).unsqueeze(-1),
        t_int=torch.zeros(1, 1, dtype=torch.long),
        t=torch.zeros(1, 1),
        node_mask=mask,
    )
    true = DataHolder(
        node_features=torch.zeros(1, N, 4),
        positions=true_pos.unsqueeze(0),
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, N, 1, dtype=torch.long),
        cell_ID=torch.arange(N).unsqueeze(0).unsqueeze(-1),
        t_int=torch.zeros(1, 1, dtype=torch.long),
        t=torch.zeros(1, 1),
        node_mask=mask,
    )
    return pred, true


def _chamfer_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance (sum of squared NN distances both ways).

    Shares Sinkhorn's key property for our test purposes: NOT rotation-
    invariant. So if procrustes_align is bypassed, rotating pred
    should INCREASE this loss; if alignment is on, the rotated pred
    is undone and the loss stays low.
    """
    d = torch.cdist(x, y, p=2.0)
    return d.min(dim=1).values.pow(2).sum() + d.min(dim=0).values.pow(2).sum()


def _make_loss_fn(procrustes_align: bool, scale_invariant: bool = False):
    """Build a LossFunction with sinkhorn ON and a chamfer stand-in
    that bypasses the geomloss dependency. The stand-in goes through
    the same alignment code path as the real Sinkhorn call, so it
    isolates the alignment behavior."""
    cfg = {
        "model": {
            "loss": {
                "pairwise_distance_mse": {"enabled": False},
                "sinkhorn": {
                    "enabled": True, "weight": 1.0,
                    "blur": 0.01, "scaling": 0.9, "p": 2,
                    "backend": "tensorized",
                    "procrustes_align": procrustes_align,
                    "scale_invariant": scale_invariant,
                },
            },
        },
    }
    lf = LossFunction(cfg)
    # Inject the chamfer stand-in BEFORE the lazy-import path triggers.
    lf._sk_loss_fn = _chamfer_sq
    return lf


# ---------------------------------------------------------------------------
# (A) Procrustes math
# ---------------------------------------------------------------------------


def test_procrustes_identity_on_identical_clouds() -> None:
    """Two identical clouds: aligned output equals input (R = I)."""
    torch.manual_seed(0)
    x = torch.randn(64, 2)
    x = x - x.mean(0)
    aligned = _procrustes_align_2d(x, x)
    err = (aligned - x).abs().max().item()
    assert err < 1e-5, f"identity alignment off by {err:.3e}"


def test_procrustes_recovers_rotation() -> None:
    """A rotated cloud should align BACK to the reference."""
    torch.manual_seed(1)
    x = torch.randn(64, 2)
    x = x - x.mean(0)
    for theta in (0.3, 0.7, math.pi / 4, math.pi / 2, math.pi - 0.1):
        R = _rot2d(theta)
        x_rot = x @ R
        aligned = _procrustes_align_2d(x_rot, x)
        err = (aligned - x).abs().max().item()
        assert err < 1e-4, (
            f"rotation θ={theta:.3f} not recovered: max err {err:.3e}"
        )


def test_procrustes_recovers_reflection() -> None:
    """A reflected cloud should align back. Tests that the SVD
    formula correctly handles det(R) = -1 (orthogonal but not SO(2))."""
    torch.manual_seed(2)
    x = torch.randn(64, 2)
    x = x - x.mean(0)
    x_refl = x @ _refl_x()
    aligned = _procrustes_align_2d(x_refl, x)
    err = (aligned - x).abs().max().item()
    assert err < 1e-4, f"reflection not recovered: max err {err:.3e}"


def test_procrustes_gradient_flows() -> None:
    """Gradient must flow through the SVD."""
    torch.manual_seed(3)
    x_src = torch.randn(32, 2, requires_grad=True)
    x_ref = torch.randn(32, 2)
    x_ref = x_ref - x_ref.mean(0)
    aligned = _procrustes_align_2d(x_src - x_src.mean(0), x_ref)
    aligned.pow(2).sum().backward()
    assert x_src.grad is not None, "no grad on x_src"
    assert x_src.grad.abs().sum().item() > 0, (
        "zero grad on x_src — SVD backward didn't propagate"
    )


# ---------------------------------------------------------------------------
# (B) Rotation-invariance of _compute_sinkhorn after the fix
# ---------------------------------------------------------------------------


def test_sinkhorn_without_align_blows_up_on_rotation() -> None:
    """Sanity check on the bug: without procrustes_align, rotating
    pred (with shape unchanged) makes the loss substantially larger.
    This is the failure mode the fix addresses."""
    torch.manual_seed(4)
    true = torch.randn(64, 2)
    true = true - true.mean(0)
    pred_match = true.clone()                           # perfect match
    pred_rot = true @ _rot2d(math.pi / 4)               # 45° rotation

    lf = _make_loss_fn(procrustes_align=False)
    pred_holder_match, true_holder = _build_holders(pred_match, true)
    pred_holder_rot,   _           = _build_holders(pred_rot,   true)

    loss_match = lf._compute_sinkhorn(pred_holder_match, true_holder).item()
    loss_rot   = lf._compute_sinkhorn(pred_holder_rot,   true_holder).item()
    assert loss_rot > 5.0 * loss_match + 1e-3, (
        f"without procrustes_align, rotated pred should have much "
        f"higher Sinkhorn loss; got match={loss_match:.6f} vs "
        f"rotated={loss_rot:.6f}. Either the test is wrong or the "
        f"alignment is sneaking in via another path."
    )


def test_sinkhorn_with_align_recovers_rotation() -> None:
    """The fix: with procrustes_align=True, rotating pred should leave
    the loss essentially unchanged (the Procrustes step undoes the
    rotation before Sinkhorn sees it)."""
    torch.manual_seed(5)
    true = torch.randn(64, 2)
    true = true - true.mean(0)
    pred_match = true.clone()
    pred_rot = true @ _rot2d(math.pi / 4)

    lf = _make_loss_fn(procrustes_align=True)
    pred_holder_match, true_holder = _build_holders(pred_match, true)
    pred_holder_rot,   _           = _build_holders(pred_rot,   true)

    loss_match = lf._compute_sinkhorn(pred_holder_match, true_holder).item()
    loss_rot   = lf._compute_sinkhorn(pred_holder_rot,   true_holder).item()
    # Should be near-zero AND essentially equal.
    assert abs(loss_rot - loss_match) < 1e-3, (
        f"with procrustes_align, rotated and matched pred should give "
        f"the same loss; got match={loss_match:.6f} vs rotated="
        f"{loss_rot:.6f} (Δ={abs(loss_rot - loss_match):.3e})."
    )
    assert loss_match < 1e-3, (
        f"matched cloud loss should be ≈ 0 (perfect match); got {loss_match:.6f}"
    )


def test_sinkhorn_with_align_recovers_reflection() -> None:
    """Same test for reflection — the EDM head can flip sign of one
    axis under MDS, and the FM augmentation can apply random
    reflection. Both should be transparent to the loss."""
    torch.manual_seed(6)
    true = torch.randn(64, 2)
    true = true - true.mean(0)
    pred_refl = true @ _refl_x()

    lf = _make_loss_fn(procrustes_align=True)
    pred_holder, true_holder = _build_holders(pred_refl, true)
    loss = lf._compute_sinkhorn(pred_holder, true_holder).item()
    assert loss < 1e-3, (
        f"reflected pred should give near-zero loss with align on; "
        f"got {loss:.6f}"
    )


def test_sinkhorn_align_still_penalises_real_displacement() -> None:
    """Alignment must NOT make the loss insensitive to actual cell
    misplacement. Adding random noise to pred should produce a
    non-trivial loss even with align on."""
    torch.manual_seed(7)
    true = torch.randn(64, 2)
    true = true - true.mean(0)
    pred_noisy = true + torch.randn_like(true) * 0.5

    lf = _make_loss_fn(procrustes_align=True)
    pred_holder, true_holder = _build_holders(pred_noisy, true)
    loss = lf._compute_sinkhorn(pred_holder, true_holder).item()
    assert loss > 0.5, (
        f"noisy pred should give substantial loss; got {loss:.6f}. "
        f"Alignment may be over-zeroing the loss."
    )


def test_sinkhorn_gradient_through_alignment() -> None:
    """Gradient must flow back to pred through the Procrustes SVD."""
    torch.manual_seed(8)
    true = torch.randn(64, 2)
    true = true - true.mean(0)
    pred = (true @ _rot2d(0.5)).clone().detach().requires_grad_(True)

    lf = _make_loss_fn(procrustes_align=True)
    pred_holder, true_holder = _build_holders(pred, true)
    # We need pred_holder.positions to share storage with `pred` for
    # the gradient to land on `pred`. _build_holders did
    # `.unsqueeze(0)`, which is a view, so this works.
    loss = lf._compute_sinkhorn(pred_holder, true_holder)
    loss.backward()
    assert pred.grad is not None, "no grad on pred"
    assert pred.grad.abs().sum().item() > 0, (
        "zero grad on pred — alignment or chamfer stand-in didn't "
        "propagate gradient."
    )


def test_sinkhorn_scale_invariant_mode_normalises() -> None:
    """With scale_invariant=True, scaling BOTH clouds by 100× should
    leave the loss unchanged (the per-slice RMS rescale undoes it).
    """
    torch.manual_seed(9)
    true = torch.randn(64, 2)
    true = true - true.mean(0)
    pred = true + torch.randn_like(true) * 0.3

    lf_si = _make_loss_fn(procrustes_align=True, scale_invariant=True)
    pred_holder_1, true_holder_1 = _build_holders(pred,        true)
    pred_holder_2, true_holder_2 = _build_holders(pred * 100., true * 100.)
    loss_1 = lf_si._compute_sinkhorn(pred_holder_1, true_holder_1).item()
    loss_2 = lf_si._compute_sinkhorn(pred_holder_2, true_holder_2).item()
    rel_err = abs(loss_1 - loss_2) / (abs(loss_1) + 1e-9)
    assert rel_err < 1e-3, (
        f"scale_invariant=True should be scale-invariant; "
        f"loss_1={loss_1:.6f}, loss_100x={loss_2:.6f} (rel err {rel_err:.3e})."
    )


def test_sinkhorn_default_flag_is_align_on() -> None:
    """Default config value: procrustes_align is ON. Confirms the fix
    is the new default behavior (since the unaligned loss is broken)."""
    cfg = {"model": {"loss": {"sinkhorn": {"enabled": True, "weight": 1.0}}}}
    lf = LossFunction(cfg)
    assert lf._sk_procrustes_align is True, (
        "procrustes_align default should be True (the fix); got False."
    )


# ---------------------------------------------------------------------------
# (C) Optional: real geomloss-based smoke test
# ---------------------------------------------------------------------------


def _geomloss_available() -> bool:
    try:
        import geomloss  # noqa: F401
        return True
    except ImportError:
        return False


def test_real_geomloss_smoke() -> None:
    """End-to-end smoke test using the actual geomloss SamplesLoss.
    Skipped when geomloss isn't installed (it's a cluster dep)."""
    if not _geomloss_available():
        print("    [SKIPPED — geomloss not installed locally]")
        return

    torch.manual_seed(10)
    true = torch.randn(32, 2) * 0.3
    true = true - true.mean(0)
    pred = (true @ _rot2d(0.3)) + torch.randn_like(true) * 0.05

    cfg = {
        "model": {
            "loss": {
                "pairwise_distance_mse": {"enabled": False},
                "sinkhorn": {
                    "enabled": True, "weight": 1.0,
                    "blur": 0.05, "scaling": 0.9, "p": 2,
                    "backend": "tensorized",
                    "procrustes_align": True, "scale_invariant": False,
                },
            },
        },
    }
    lf = LossFunction(cfg)
    pred_holder, true_holder = _build_holders(pred, true)
    loss = lf._compute_sinkhorn(pred_holder, true_holder)
    assert torch.isfinite(loss).item(), f"geomloss returned non-finite: {loss}"
    assert loss.item() >= 0.0, f"Sinkhorn divergence should be >= 0; got {loss.item()}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        # (A) Procrustes
        ("(A) Procrustes: identity on identical clouds",
         test_procrustes_identity_on_identical_clouds),
        ("(A) Procrustes: recovers rotation angles",
         test_procrustes_recovers_rotation),
        ("(A) Procrustes: recovers reflection",
         test_procrustes_recovers_reflection),
        ("(A) Procrustes: gradient flows through SVD",
         test_procrustes_gradient_flows),
        # (B) Sinkhorn rotation-invariance (chamfer stand-in)
        ("(B) Sinkhorn WITHOUT align: rotation blows up loss (bug repro)",
         test_sinkhorn_without_align_blows_up_on_rotation),
        ("(B) Sinkhorn WITH align: rotation recovered",
         test_sinkhorn_with_align_recovers_rotation),
        ("(B) Sinkhorn WITH align: reflection recovered",
         test_sinkhorn_with_align_recovers_reflection),
        ("(B) Sinkhorn WITH align: real displacement still penalised",
         test_sinkhorn_align_still_penalises_real_displacement),
        ("(B) Sinkhorn WITH align: gradient flows through to pred",
         test_sinkhorn_gradient_through_alignment),
        ("(B) Sinkhorn scale_invariant=True is scale-invariant",
         test_sinkhorn_scale_invariant_mode_normalises),
        ("(B) Sinkhorn default flag: procrustes_align ON",
         test_sinkhorn_default_flag_is_align_on),
        # (C) Real geomloss smoke (skipped if not installed)
        ("(C) Real geomloss path returns finite non-negative loss",
         test_real_geomloss_smoke),
    ]
    n_pass = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            n_pass += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    print(f"\n{n_pass}/{len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

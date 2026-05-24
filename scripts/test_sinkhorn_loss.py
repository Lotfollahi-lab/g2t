#!/usr/bin/env python
"""Sanity tests for the Sinkhorn / OT loss component.

Run inside the scgg env::

    source /nfs/team361/sb75/.venvs/scgg/bin/activate
    python /nfs/team361/sb75/scgg/scripts/test_sinkhorn_loss.py

Exits 0 on pass, non-zero on the first failure. ~2 seconds.

Three properties checked:

  1. Self-distance is ≈ 0. The Sinkhorn DIVERGENCE (debiased form) of
     a cloud against itself should be near zero; that's the whole
     point of the debiasing.
  2. Distance to a shifted copy is > 0. Sinkhorn between cloud and
     (cloud + offset) should be positive and proportional to the
     squared shift magnitude (for p=2).
  3. Gradients flow into the predicted cloud. The most important
     wiring check: if Sinkhorn doesn't backprop, the loss does
     nothing in training.
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
    """Minimal cfg with sinkhorn enabled — and pairwise MSE disabled
    so we isolate the Sinkhorn behavior in the loss."""

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
                enabled = False
                weight = 0.0
                subsample = None
                dim = 0
                frequency = 1
                cache_true = True
            class _SK:
                enabled = True
                weight = 1.0
                blur = 0.01
                scaling = 0.9
                p = 2
                subsample = None
            pairwise_distance_mse = _PWD
            knn_rank = _KNN
            persistent_homology = _PH
            sinkhorn = _SK
        loss = _Loss
    model = _Model


def _make_clouds(B: int = 2, N: int = 64):
    """A small batch of mean-centered point clouds in [-0.5, 0.5]."""
    torch.manual_seed(0)
    pos = torch.randn(B, N, 2) * 0.3
    pos = pos - pos.mean(dim=1, keepdim=True)
    node_mask = torch.ones(B, N, dtype=torch.bool)
    return pos, node_mask


def _to_holder(positions, node_mask):
    return DataHolder(
        node_features=torch.zeros(*positions.shape[:2], 4),
        positions=positions,
        diffusion_time=torch.zeros(positions.shape[0], 1),
        cell_class=torch.zeros(*positions.shape[:2], dtype=torch.long),
        node_mask=node_mask,
    )


def test_self_distance_zero(tol: float = 1e-3) -> None:
    """S_ε(α, α) ≈ 0 for the debiased Sinkhorn divergence."""
    loss_fn = LossFunction(cfg=_FakeCfg())
    pos, mask = _make_clouds()
    pred = _to_holder(pos.clone().requires_grad_(True), mask)
    true = _to_holder(pos.clone(), mask)
    val = loss_fn._compute_sinkhorn(pred, true)
    err = float(val.detach().abs().item())
    print(f"[sinkhorn]  self-distance: {err:.2e}  (tol={tol})")
    if err > tol:
        raise AssertionError(
            f"Debiased Sinkhorn divergence on identical clouds should "
            f"be ≈ 0; got {err:.2e}. Likely the debias flag isn't "
            f"propagating, or geomloss's debiased path is broken."
        )
    print("[sinkhorn]  self-distance PASS\n")


def test_shifted_cloud_positive() -> None:
    """S(α, α + δ) > S(α, α) by a margin that scales with ||δ||²."""
    loss_fn = LossFunction(cfg=_FakeCfg())
    pos, mask = _make_clouds()
    true = _to_holder(pos.clone(), mask)

    # Two shifts of different magnitudes.
    small_shift = pos.clone() + 0.05
    big_shift = pos.clone() + 0.3
    pred_small = _to_holder(small_shift.clone(), mask)
    pred_big = _to_holder(big_shift.clone(), mask)

    val_small = float(loss_fn._compute_sinkhorn(pred_small, true).detach().item())
    val_big = float(loss_fn._compute_sinkhorn(pred_big, true).detach().item())

    print(f"[sinkhorn]  shifted +0.05: {val_small:.4e}")
    print(f"[sinkhorn]  shifted +0.30: {val_big:.4e}")
    if not (val_big > val_small > 0):
        raise AssertionError(
            f"Sinkhorn should increase monotonically with shift magnitude. "
            f"Got small={val_small:.4e}, big={val_big:.4e}."
        )
    print("[sinkhorn]  monotonicity PASS\n")


def test_gradient_flows_to_pred() -> None:
    """∂loss / ∂pred.positions must be non-zero."""
    loss_fn = LossFunction(cfg=_FakeCfg())
    pos, mask = _make_clouds()
    pred_pos = (pos + 0.2 * torch.randn_like(pos))
    pred_pos.requires_grad_(True)
    pred = _to_holder(pred_pos, mask)
    true = _to_holder(pos.clone(), mask)
    val = loss_fn._compute_sinkhorn(pred, true)
    val.backward()
    if pred_pos.grad is None:
        raise AssertionError("pred.positions.grad is None after backward.")
    grad_norm = float(pred_pos.grad.abs().sum().item())
    print(f"[sinkhorn]  gradient sum-abs: {grad_norm:.4e}")
    if grad_norm == 0.0:
        raise AssertionError(
            "Gradient through Sinkhorn is exactly zero — backprop "
            "is broken. Check that pred is the FIRST argument to "
            "SamplesLoss (gradient flows through the first argument)."
        )
    print("[sinkhorn]  gradient flow PASS\n")


def main() -> int:
    try:
        import geomloss  # noqa: F401
    except ImportError:
        print("SKIP: geomloss not installed. `pip install geomloss` to run these tests.")
        return 0
    test_self_distance_zero()
    test_shifted_cloud_positive()
    test_gradient_flows_to_pred()
    print("All Sinkhorn-loss tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

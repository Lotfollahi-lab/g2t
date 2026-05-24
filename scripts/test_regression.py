#!/usr/bin/env python
"""Sanity tests for the direct-regression framework option.

Run inside the scgg env::

    source /nfs/team361/sb75/.venvs/scgg/bin/activate
    python /nfs/team361/sb75/scgg/scripts/test_regression.py

Exits 0 on pass, non-zero on the first failure. ~2 seconds.

Four properties checked:

  1. ``apply_noise`` zeros out positions and sets t=1 (the
     contract — backbone must see NO positional info at training
     time).
  2. ``sample_limit_dist`` also produces zeroed positions at t=1
     (matches the training-time input distribution).
  3. ``sample_zs_from_zt_and_pred`` returns ``pred`` unchanged
     (single-step "sampling" — the backbone's forward IS the
     prediction).
  4. ``max_diffusion_steps == 1`` so the outer sampling loop in
     ``sample.iterate_sampling`` iterates exactly once.
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
from utils.diffusion_model.diffusion.regression_predictor import (  # noqa: E402
    RegressionPredictor,
)


class _FakeCfg:
    """Minimal cfg the predictor reads."""
    class _Model:
        framework = "regression"
    model = _Model


def _make_batch(B: int = 2, N: int = 16) -> DataHolder:
    torch.manual_seed(0)
    positions = torch.randn(B, N, 2) * 0.3
    node_features = torch.randn(B, N, 8)
    node_mask = torch.ones(B, N, dtype=torch.bool)
    return DataHolder(
        node_features=node_features,
        positions=positions,
        node_mask=node_mask,
        cell_class=torch.zeros(B, N, dtype=torch.long),
        diffusion_time=torch.zeros(B, 1),
    )


def test_apply_noise_zeros_positions(tol: float = 1e-9) -> None:
    """apply_noise must STRIP positional info: positions → 0, t → 1.
    The training loss compares the backbone's OUTPUT positions
    against the TRUE positions which the loss receives separately
    (see train.training_step_func), so the backbone is forced to
    learn position-from-genes."""
    predictor = RegressionPredictor(_FakeCfg())
    batch = _make_batch()
    z_t = predictor.apply_noise(batch, train_flag=False)

    if z_t.positions.abs().max().item() > tol:
        raise AssertionError(
            f"apply_noise should zero positions; got max abs "
            f"{z_t.positions.abs().max().item():.2e}."
        )
    if not (z_t.t == 1.0).all():
        raise AssertionError("apply_noise should set t=1.")
    if not (z_t.t_int == 1).all():
        raise AssertionError("apply_noise should set t_int=1.")
    if not (z_t.diffusion_time == 1.0).all():
        raise AssertionError("apply_noise should set diffusion_time=1.")
    print("[apply_noise]  PASS\n")


def test_sample_limit_dist_matches_apply_noise(tol: float = 1e-9) -> None:
    """sample_limit_dist (inference start) and apply_noise
    (training input) must produce the SAME distribution — otherwise
    train/inference are mismatched and metrics will be unreliable."""
    predictor = RegressionPredictor(_FakeCfg())
    batch = _make_batch()
    z_t = predictor.sample_limit_dist(
        node_features=batch.node_features,
        node_mask=batch.node_mask,
        cell_ID=torch.zeros(batch.node_features.shape[:2], dtype=torch.long),
        cell_class=batch.cell_class,
    )
    if z_t.positions.abs().max().item() > tol:
        raise AssertionError("sample_limit_dist must zero positions.")
    if not (z_t.t == 1.0).all():
        raise AssertionError("sample_limit_dist must set t=1.")
    print("[sample_limit_dist]  PASS\n")


def test_step_is_identity() -> None:
    """sample_zs_from_zt_and_pred is a single-step identity: it
    returns ``pred`` (mask-applied). The outer sampling loop only
    iterates once (max_diffusion_steps=1), so the backbone's
    forward IS the final prediction."""
    predictor = RegressionPredictor(_FakeCfg())
    batch = _make_batch()
    z_t = predictor.apply_noise(batch)

    # Construct a "prediction" with non-zero positions to verify
    # they survive the passthrough.
    pred = DataHolder(
        node_features=batch.node_features,
        positions=torch.randn_like(batch.positions),
        node_mask=batch.node_mask,
        cell_class=batch.cell_class,
        diffusion_time=z_t.diffusion_time,
        t_int=z_t.t_int,
        t=z_t.t,
    ).mask()
    pred_positions_before = pred.positions.clone()

    s_int = torch.zeros((1, 1), dtype=torch.long)
    z_s = predictor.sample_zs_from_zt_and_pred(z_t, pred, s_int)

    # z_s.positions should equal pred.positions (modulo masking,
    # which is a no-op here since all cells are unmasked).
    err = (z_s.positions - pred_positions_before).abs().max().item()
    if err > 1e-9:
        raise AssertionError(
            f"sample_zs_from_zt_and_pred should return pred unchanged; "
            f"got max-diff {err:.2e}."
        )
    print("[step-identity]  PASS\n")


def test_max_steps_is_one() -> None:
    """max_diffusion_steps must be 1 so the outer reverse loop
    iterates exactly once. Any other value means the network would
    be called multiple times with identical inputs, wasting
    compute."""
    predictor = RegressionPredictor(_FakeCfg())
    if predictor.max_diffusion_steps != 1:
        raise AssertionError(
            f"max_diffusion_steps must be 1 for regression; got "
            f"{predictor.max_diffusion_steps}."
        )
    if predictor.sampler != "euler":
        raise AssertionError(
            f"sampler must be 'euler' for regression (no Heun second "
            f"forward); got {predictor.sampler!r}."
        )
    print("[max-steps]  PASS\n")


def main() -> int:
    test_apply_noise_zeros_positions()
    test_sample_limit_dist_matches_apply_noise()
    test_step_is_identity()
    test_max_steps_is_one()
    print("All regression-predictor tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

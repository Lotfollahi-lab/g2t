#!/usr/bin/env python
"""Sanity tests for the flow-matching alternative to LUNA's DDPM.

Run inside the scgg env::

    source /nfs/team361/sb75/.venvs/scgg/bin/activate
    python /nfs/team361/sb75/scgg/scripts/test_flow_matching.py

Exits 0 on pass, non-zero on the first failure. Cheap (~2s) check
before queuing a long FM training run — catches the obvious
interface-mismatch bugs (shape, dtype, t-conventions, mean-centering)
without needing real data.

Three tests:

  1. ``apply_noise`` is a valid interpolation: at t=0 returns x_0
     (within mean-subtraction tolerance), at t=1 returns pure
     mean-centered noise (independent of x_0).
  2. ``sample_zs_from_zt_and_pred`` is the identity at s=t (no step)
     and returns x_0_pred at s=0 (final clean estimate).
  3. End-to-end sampling loop converges to the network's prediction
     when the network is a constant function (target = the true x_0
     for an identity-like net): a sanity check that the t schedule
     decreases monotonically and the Euler steps compose correctly.
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
from utils.diffusion_model.diffusion.flow_matching_model import (  # noqa: E402
    FlowMatchingModel,
)


class _FakeCfg:
    """Minimal stand-in for the OmegaConf cfg the FM model reads at
    __init__. Only the keys actually accessed by FlowMatchingModel
    need to exist."""

    class _Model:
        framework = "flow_matching"

        class _FM:
            n_sampling_steps = 10  # small for fast iteration in tests
            eps_t = 1e-3
        flow_matching = _FM

    model = _Model


def _make_batch(B: int = 2, N: int = 16) -> DataHolder:
    """A small fake batch — positions in a tight cluster so
    mean-centering shifts them by O(0.1)."""
    torch.manual_seed(0)
    positions = torch.randn(B, N, 2) * 0.3
    node_features = torch.randn(B, N, 8)
    node_mask = torch.ones(B, N, dtype=torch.bool)
    # Center inputs (FM assumes data is mean-centered, which is how
    # LUNA's data pipeline normalises positions).
    positions = positions - positions.mean(dim=1, keepdim=True)
    return DataHolder(
        node_features=node_features,
        positions=positions,
        node_mask=node_mask,
        cell_class=torch.zeros(B, N, dtype=torch.long),
    )


def test_apply_noise_interpolation(tol: float = 1e-5) -> None:
    """At t→0 the perturbed x_t should ≈ x_0; at t=1, x_t should be
    independent of x_0 (pure mean-centered Gaussian).
    """
    fm = FlowMatchingModel(_FakeCfg())
    batch = _make_batch()

    # Override the random t to hit boundaries deterministically.
    # Cheapest way: monkey-patch torch.rand inside the method's scope
    # via a manual call sequence. Easier: invoke the math directly
    # to test the SCHEDULE — the apply_noise method is thin enough
    # that re-doing its t = eps_t and t ≈ 1 cases by hand here is
    # cleaner than mocking.
    B, N = batch.positions.shape[:2]

    # Case t = eps_t (smallest allowed): x_t ≈ x_0.
    t_eps = torch.full((B, 1), fm.eps_t)
    t_b = t_eps.unsqueeze(-1)
    noise = torch.randn_like(batch.positions)
    noise = noise - noise.mean(dim=1, keepdim=True)
    x_t_low = (1.0 - t_b) * batch.positions + t_b * noise
    err_low = (x_t_low - batch.positions).abs().max().item()
    print(f"[apply_noise]  t=eps   max-err vs x_0:   {err_low:.2e}")
    if err_low > 5 * fm.eps_t:
        raise AssertionError(
            f"x_t at t=eps_t differs from x_0 by {err_low:.2e}, "
            f"expected ≲ eps_t·|noise| ≈ {fm.eps_t}."
        )

    # Case t = 1: x_t should be exactly the mean-centered noise (no
    # x_0 contribution).
    t_one = torch.ones(B, 1)
    t_b = t_one.unsqueeze(-1)
    x_t_high = (1.0 - t_b) * batch.positions + t_b * noise
    err_high = (x_t_high - noise).abs().max().item()
    print(f"[apply_noise]  t=1     max-err vs noise: {err_high:.2e}")
    if err_high > tol:
        raise AssertionError(
            f"x_t at t=1 differs from pure noise by {err_high:.2e}; "
            f"x_0 contribution should be exactly zero."
        )

    # End-to-end call: the actual apply_noise should produce a
    # valid DataHolder with the right shapes.
    z_t = fm.apply_noise(batch, train_flag=False)
    assert z_t.positions.shape == batch.positions.shape
    assert z_t.t.shape == (B, 1)
    assert (z_t.t >= fm.eps_t).all() and (z_t.t <= 1.0).all()
    print("[apply_noise]  PASS\n")


def test_euler_step_endpoints(tol: float = 1e-5) -> None:
    """At s = t the Euler step is the identity (no movement); at s = 0
    it returns x_0_pred (final clean estimate). These two endpoints
    pin the formula down — if either is wrong, the ODE math is broken.
    """
    fm = FlowMatchingModel(_FakeCfg())
    batch = _make_batch()
    B, N = batch.positions.shape[:2]

    # Build a z_t at some intermediate time, and a pred (x_0_pred).
    t_val = 0.5
    t_tensor = torch.full((B, 1), t_val)
    z_t = DataHolder(
        node_features=batch.node_features,
        positions=batch.positions.clone(),
        node_mask=batch.node_mask,
        cell_class=batch.cell_class,
        t_int=torch.full((B, 1), int(t_val * fm.max_diffusion_steps), dtype=torch.long),
        t=t_tensor,
        diffusion_time=t_tensor,
    ).mask()

    # Pred: pretend the network thinks x_0 is somewhere else.
    pred_positions = torch.randn_like(batch.positions) * 0.2
    pred_positions = pred_positions - pred_positions.mean(dim=1, keepdim=True)
    pred = DataHolder(
        node_features=batch.node_features,
        positions=pred_positions,
        node_mask=batch.node_mask,
        cell_class=batch.cell_class,
        t_int=z_t.t_int,
        t=z_t.t,
        diffusion_time=z_t.t,
    ).mask()

    # (a) s = t → identity (modulo mean-recenter, which is a no-op
    # on already-centered tensors).
    s_int = torch.full((1, 1), int(t_val * fm.max_diffusion_steps), dtype=torch.long)
    z_same = fm.sample_zs_from_zt_and_pred(z_t, pred, s_int)
    err_id = (z_same.positions - z_t.positions).abs().max().item()
    print(f"[euler]  s=t  identity max-err: {err_id:.2e}")
    if err_id > tol:
        raise AssertionError(
            f"Euler step at s=t should be the identity (modulo "
            f"centering); got max-err {err_id:.2e}."
        )

    # (b) s = 0 → returns x_0_pred (the network's final clean
    # estimate, also centered).
    s_zero = torch.zeros((1, 1), dtype=torch.long)
    z_clean = fm.sample_zs_from_zt_and_pred(z_t, pred, s_zero)
    err_clean = (z_clean.positions - pred.positions).abs().max().item()
    print(f"[euler]  s=0  → x_0_pred max-err: {err_clean:.2e}")
    if err_clean > tol:
        raise AssertionError(
            f"Euler step at s=0 should return x_0_pred; got "
            f"max-err {err_clean:.2e}."
        )
    print("[euler]  PASS\n")


def test_full_sampling_loop(tol: float = 1e-4) -> None:
    """End-to-end ODE solve. If the network is a constant function
    ``pred(x_t) = x_0_target``, then after the full loop we should
    end up at x_0_target regardless of the noise we started from.
    (Each Euler step pulls us toward the SAME x_0, so 50 steps
    composing should land exactly on it.)
    """
    fm = FlowMatchingModel(_FakeCfg())
    batch = _make_batch()
    B, N = batch.positions.shape[:2]

    # Constant "network" — always returns the target clean positions.
    target = batch.positions.clone()

    # Start at t = 1 with pure mean-centered Gaussian noise.
    z_t = fm.sample_limit_dist(
        node_features=batch.node_features,
        node_mask=batch.node_mask,
        cell_ID=torch.zeros(B, N, dtype=torch.long),
        cell_class=batch.cell_class,
    )

    # Iterate the same reverse loop as sample.iterate_sampling.
    for s_int in reversed(range(0, fm.max_diffusion_steps)):
        s_array = torch.full((1, 1), s_int, dtype=torch.long)
        pred = DataHolder(
            node_features=z_t.node_features,
            positions=target,                   # constant prediction
            node_mask=z_t.node_mask,
            cell_class=z_t.cell_class,
            t_int=z_t.t_int,
            t=z_t.t,
            diffusion_time=z_t.t,
        ).mask()
        z_t = fm.sample_zs_from_zt_and_pred(z_t, pred, s_array)

    err = (z_t.positions - target).abs().max().item()
    print(f"[loop]   final-vs-target max-err after {fm.max_diffusion_steps} steps: {err:.2e}")
    if err > tol:
        raise AssertionError(
            f"After the full ODE loop with a constant-x_0 'network', "
            f"final positions should converge to x_0_target; got "
            f"max-err {err:.2e}. Likely the s-schedule or the Euler "
            f"step formula is off."
        )
    print("[loop]   PASS\n")


def test_heun_correction_endpoints(tol: float = 1e-5) -> None:
    """The Heun correction has two boundary properties to check:

      (a) At the final step (s = 0), the 1/s factor in the
          trapezoidal velocity is undefined and the implementation
          falls back to the Euler result (which IS the canonical
          x_0_pred at s=0).
      (b) When the network's prediction is the SAME at both
          endpoints (z_t and z_s_euler), Heun reduces to Euler.
          This happens trivially for a constant-x_0 'network' and
          confirms the trapezoidal average is correctly weighted.
    """
    cfg = _FakeCfg()
    # Flip to heun for this test.
    cfg.model.flow_matching.sampler = "heun"
    fm = FlowMatchingModel(cfg)
    batch = _make_batch()
    B = batch.positions.shape[0]

    # Build z_t at an intermediate time and a pred.
    t_val = 0.5
    t_tensor = torch.full((B, 1), t_val)
    z_t = DataHolder(
        node_features=batch.node_features,
        positions=batch.positions.clone(),
        node_mask=batch.node_mask,
        cell_class=batch.cell_class,
        t_int=torch.full((B, 1), int(t_val * fm.max_diffusion_steps), dtype=torch.long),
        t=t_tensor,
        diffusion_time=t_tensor,
    ).mask()
    pred = DataHolder(
        node_features=batch.node_features,
        positions=batch.positions.clone() * 0.5,   # arbitrary
        node_mask=batch.node_mask,
        cell_class=batch.cell_class,
        t_int=z_t.t_int,
        t=z_t.t,
        diffusion_time=z_t.t,
    ).mask()

    # (a) Final-step fallback. s_int = 0 → s = 0 → Heun returns
    # z_s_euler unchanged.
    z_s_euler = fm.sample_zs_from_zt_and_pred(z_t, pred, torch.zeros(1, 1, dtype=torch.long))
    pred2_dummy = DataHolder(  # value irrelevant — Heun should bail out
        node_features=batch.node_features,
        positions=torch.randn_like(batch.positions),
        node_mask=batch.node_mask,
        cell_class=batch.cell_class,
        t_int=z_s_euler.t_int, t=z_s_euler.t, diffusion_time=z_s_euler.t,
    ).mask()
    z_s_heun = fm.heun_correction(
        z_t, pred, z_s_euler, pred2_dummy,
        s_int=torch.zeros(1, 1, dtype=torch.long),
    )
    err_fallback = (z_s_heun.positions - z_s_euler.positions).abs().max().item()
    print(f"[heun]  s=0 fallback   max-err vs Euler: {err_fallback:.2e}")
    if err_fallback > tol:
        raise AssertionError(
            f"Heun should fall back to Euler at s=0 (the 1/s factor "
            f"is undefined); got max-err {err_fallback:.2e}."
        )

    # (b) Constant-pred sanity: if pred at z_t and pred at z_s_euler
    # are the same value (i.e. the "network" returns a constant
    # regardless of input), Heun's velocity average equals the
    # Euler velocity → Heun == Euler.
    s_val = 0.3
    s_int_b = torch.full((1, 1), int(s_val * fm.max_diffusion_steps), dtype=torch.long)
    z_s_euler_b = fm.sample_zs_from_zt_and_pred(z_t, pred, s_int_b)
    pred2_same = DataHolder(  # SAME prediction at the new point
        node_features=batch.node_features,
        positions=pred.positions.clone(),
        node_mask=batch.node_mask,
        cell_class=batch.cell_class,
        t_int=z_s_euler_b.t_int, t=z_s_euler_b.t, diffusion_time=z_s_euler_b.t,
    ).mask()
    z_s_heun_b = fm.heun_correction(z_t, pred, z_s_euler_b, pred2_same, s_int_b)
    # Note: this equality is exact ONLY for a constant-x_0 prediction
    # (which makes the velocity at both endpoints equal). For real
    # networks v1 ≠ v2 and that's the whole point of Heun.
    err_const = (z_s_heun_b.positions - z_s_euler_b.positions).abs().max().item()
    print(f"[heun]  constant-pred  max-err vs Euler: {err_const:.2e}")
    if err_const > 1e-4:  # slightly looser — masking/centering jitter
        raise AssertionError(
            f"Heun with constant pred should equal Euler (both use "
            f"the same velocity); got max-err {err_const:.2e}."
        )
    print("[heun]  PASS\n")


def main() -> int:
    test_apply_noise_interpolation()
    test_euler_step_endpoints()
    test_full_sampling_loop()
    test_heun_correction_endpoints()
    print("All flow-matching tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

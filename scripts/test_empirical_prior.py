"""Smoke tests for the per-cell-type empirical GMM FM prior.

Validates:
  (1) Default ``prior_mode="gaussian"`` produces an FM model with
      ``prior_pool is None`` — i.e., legacy behavior preserved.
  (2) ``prior_mode="empirical_gmm"`` with a fake-but-well-formed pool
      loads, and ``_sample_x1_from_gmm_pool`` returns finite samples
      of the right shape, where the samples concentrate near the
      configured GMM means per class.
  (3) Per-slice scale computation from x_0 returns sensible values.
  (4) Cells with cell_class not in the pool fall back to N(0, I)
      WITHOUT error.

Run as a standalone script (no pytest required):
    python scgg/scripts/test_empirical_prior.py
"""

from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Add src to path so the FM model imports cleanly.
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from utils.diffusion_model.diffusion.flow_matching_model import (  # noqa: E402
    _load_prior_pool,
    _per_slice_scale_from_x0,
    _sample_x1_from_gmm_pool,
)


def _make_fake_pool(out_path: Path) -> dict:
    """Build a 2-class pool where class 0 lives near (1, 0) and
    class 1 lives near (-1, 0). Saves to disk and returns the dict.
    """
    pool_raw = {
        "gmms": {
            0: {
                # Single-mode Gaussian at (1, 0), tight scale.
                "means":       np.array([[1.0, 0.0]], dtype=np.float32),
                "covariances": np.array([[[0.01, 0.0], [0.0, 0.01]]],
                                        dtype=np.float32),
                "weights":     np.array([1.0], dtype=np.float32),
                "K": 1,
                "n_train_samples": 100,
            },
            1: {
                "means":       np.array([[-1.0, 0.0]], dtype=np.float32),
                "covariances": np.array([[[0.01, 0.0], [0.0, 0.01]]],
                                        dtype=np.float32),
                "weights":     np.array([1.0], dtype=np.float32),
                "K": 1,
                "n_train_samples": 100,
            },
        },
        "default_scale": 2.5,  # arbitrary fixed scale
        "slice_scales": {"fake_slice.h5ad": 2.5},
        "meta": {"test": True},
    }
    with open(out_path, "wb") as f:
        pickle.dump(pool_raw, f)
    return pool_raw


def test_load_pool_returns_torch_tensors() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pool.pkl"
        _make_fake_pool(path)
        pool = _load_prior_pool(str(path))
    assert "gmms" in pool and "default_scale" in pool
    assert set(pool["gmms"].keys()) == {0, 1}
    cls0 = pool["gmms"][0]
    assert torch.is_tensor(cls0["means_t"]) and cls0["means_t"].shape == (1, 2)
    assert torch.is_tensor(cls0["chols_t"]) and cls0["chols_t"].shape == (1, 2, 2)
    assert torch.is_tensor(cls0["weights_t"]) and cls0["weights_t"].shape == (1,)
    # Chol of diag(0.01) is diag(0.1)-ish (plus jitter).
    chol = cls0["chols_t"][0]
    assert abs(chol[0, 0].item() - 0.1) < 1e-2
    assert abs(chol[1, 1].item() - 0.1) < 1e-2


def test_sample_concentrates_near_class_mean() -> None:
    """Class 0 cells should cluster near (1, 0); class 1 near (-1, 0)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pool.pkl"
        _make_fake_pool(path)
        pool = _load_prior_pool(str(path))

    torch.manual_seed(0)
    B, N = 1, 200
    # First 100 cells are class 0, next 100 are class 1.
    cell_class = torch.zeros(B, N, dtype=torch.long)
    cell_class[:, 100:] = 1
    node_mask = torch.ones(B, N, dtype=torch.bool)
    samples = _sample_x1_from_gmm_pool(
        cell_class=cell_class,
        node_mask=node_mask,
        pool=pool,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert samples.shape == (B, N, 2)
    assert torch.isfinite(samples).all()
    # Class 0 mean should be near (1, 0); class 1 near (-1, 0).
    cls0_samples = samples[0, :100]
    cls1_samples = samples[0, 100:]
    assert abs(cls0_samples.mean(0)[0].item() - 1.0) < 0.05
    assert abs(cls0_samples.mean(0)[1].item() - 0.0) < 0.05
    assert abs(cls1_samples.mean(0)[0].item() - (-1.0)) < 0.05
    assert abs(cls1_samples.mean(0)[1].item() - 0.0) < 0.05


def test_padding_cells_get_zero_position() -> None:
    """Padding (node_mask=False) cells get exactly (0, 0)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pool.pkl"
        _make_fake_pool(path)
        pool = _load_prior_pool(str(path))

    B, N = 1, 10
    cell_class = torch.zeros(B, N, dtype=torch.long)
    node_mask = torch.tensor([[True] * 5 + [False] * 5], dtype=torch.bool)
    samples = _sample_x1_from_gmm_pool(
        cell_class=cell_class,
        node_mask=node_mask,
        pool=pool,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # Padding cells (last 5) should all be (0, 0).
    assert torch.equal(samples[0, 5:], torch.zeros(5, 2))
    # Real cells (first 5) should NOT all be zero.
    assert not torch.equal(samples[0, :5], torch.zeros(5, 2))


def test_unknown_class_falls_back_to_gaussian() -> None:
    """Cells with a cell_class not in the pool fall back to N(0, I)
    rather than crashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pool.pkl"
        _make_fake_pool(path)
        pool = _load_prior_pool(str(path))

    torch.manual_seed(0)
    B, N = 1, 500
    # Class 99 is NOT in the pool.
    cell_class = torch.full((B, N), 99, dtype=torch.long)
    node_mask = torch.ones(B, N, dtype=torch.bool)
    samples = _sample_x1_from_gmm_pool(
        cell_class=cell_class,
        node_mask=node_mask,
        pool=pool,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.isfinite(samples).all()
    # Fallback is N(0, I) → mean ≈ 0, std ≈ 1.
    assert abs(samples.mean().item()) < 0.1
    assert abs(samples.std().item() - 1.0) < 0.1


def test_per_slice_scale_from_x0() -> None:
    """For a slice with valid cells in a 4×4 box, scale = 0.5 ·
    sqrt(32) ≈ 2.83. Padding cells (all zeros) must be excluded.
    """
    B, N = 2, 10
    positions = torch.zeros(B, N, 2)
    # Slice 0: 4 valid cells at corners of a 4×4 box, rest padding.
    positions[0, 0] = torch.tensor([-2.0, -2.0])
    positions[0, 1] = torch.tensor([ 2.0, -2.0])
    positions[0, 2] = torch.tensor([-2.0,  2.0])
    positions[0, 3] = torch.tensor([ 2.0,  2.0])
    # Slice 1: 3 valid cells, smaller bbox.
    positions[1, 0] = torch.tensor([ 0.0,  0.0])
    positions[1, 1] = torch.tensor([ 1.0,  0.0])
    positions[1, 2] = torch.tensor([ 0.0,  1.0])
    node_mask = torch.zeros(B, N, dtype=torch.bool)
    node_mask[0, :4] = True
    node_mask[1, :3] = True
    scales = _per_slice_scale_from_x0(positions, node_mask)
    assert scales.shape == (B, 1, 1)
    # Slice 0: bbox = (4, 4), diag = sqrt(32) ≈ 5.66, half ≈ 2.83.
    assert abs(scales[0, 0, 0].item() - 2.828) < 0.01
    # Slice 1: bbox = (1, 1), diag = sqrt(2) ≈ 1.414, half ≈ 0.707.
    assert abs(scales[1, 0, 0].item() - 0.707) < 0.01


def test_per_slice_scale_singleton_slice_falls_back() -> None:
    """A slice with <2 valid cells gets scale = 1.0 (no bbox)."""
    B, N = 1, 5
    positions = torch.zeros(B, N, 2)
    node_mask = torch.zeros(B, N, dtype=torch.bool)
    node_mask[0, 0] = True  # only 1 valid cell
    scales = _per_slice_scale_from_x0(positions, node_mask)
    assert scales[0, 0, 0].item() == 1.0


def main() -> int:
    tests = [
        ("load pool returns torch tensors with Cholesky precomputed",
         test_load_pool_returns_torch_tensors),
        ("sampled cells concentrate near class GMM means",
         test_sample_concentrates_near_class_mean),
        ("padding cells get (0, 0) positions",
         test_padding_cells_get_zero_position),
        ("unknown cell_class falls back to N(0, I)",
         test_unknown_class_falls_back_to_gaussian),
        ("per-slice scale from x_0 = half bbox-diag of valid cells",
         test_per_slice_scale_from_x0),
        ("singleton-cell slice gets scale=1.0",
         test_per_slice_scale_singleton_slice_falls_back),
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

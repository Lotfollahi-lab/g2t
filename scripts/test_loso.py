#!/usr/bin/env python
"""Unit tests for the leave-one-slice-out (LOSO) helpers in
``run_scgg_train.py`` — the pure partition + gap math (no torch/anndata).

Run:  python scgg/scripts/test_loso.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_scgg_train as R  # noqa: E402


def _ok(name: str) -> None:
    print(f"  [PASS] {name}")


def _P(*names):
    return [Path(n) for n in names]


def test_partition_basic() -> None:
    train = _P("a_train.h5ad", "b_train.h5ad", "c_train.h5ad", "d_train.h5ad")
    test = _P("t1_test.h5ad", "t2_test.h5ad")
    kept, extra, unmatched = R._partition_loso_files(
        train, test, holdout_set={"b_train.h5ad"}, eval_train_set={"c_train.h5ad"},
    )
    assert [p.name for p in kept] == ["a_train.h5ad", "c_train.h5ad", "d_train.h5ad"], kept
    # held-out removed from training; eval-train stays in training
    assert "b_train.h5ad" not in {p.name for p in kept}
    assert "c_train.h5ad" in {p.name for p in kept}
    # extra carries (path, split): b -> holdout, c -> train
    extra_map = {p.name: s for p, s in extra}
    assert extra_map == {"b_train.h5ad": "holdout", "c_train.h5ad": "train"}, extra_map
    assert unmatched == []
    _ok("partition: holdout removed from train, both routed to eval w/ tags")


def test_partition_overlap_raises() -> None:
    train = _P("a_train.h5ad", "b_train.h5ad")
    try:
        R._partition_loso_files(train, [], {"a_train.h5ad"}, {"a_train.h5ad"})
        raise AssertionError("expected ValueError on holdout/eval overlap")
    except ValueError:
        _ok("partition: a file in both holdout and eval_train raises")


def test_partition_unmatched_reported() -> None:
    train = _P("a_train.h5ad")
    kept, extra, unmatched = R._partition_loso_files(
        train, [], {"zzz_train.h5ad"}, set(),
    )
    assert [p.name for p in kept] == ["a_train.h5ad"]
    assert extra == []
    assert unmatched == ["zzz_train.h5ad"]
    _ok("partition: unmatched requested names reported, not silently dropped")


def test_gap_math() -> None:
    per_slice = [
        {"section_label": "s1", "split": "train", "spearman_per_cell_median": 0.80,
         "global_distance_spearman": 0.90},
        {"section_label": "s2", "split": "train", "spearman_per_cell_median": 0.70,
         "global_distance_spearman": 0.80},
        {"section_label": "h1", "split": "holdout", "spearman_per_cell_median": 0.50,
         "global_distance_spearman": 0.60},
        {"section_label": "t1", "split": "test", "spearman_per_cell_median": 0.55,
         "global_distance_spearman": 0.65},
    ]
    summary, gaps = R._generalization_gap(per_slice)
    assert summary["train"]["n_slices"] == 2.0
    assert abs(summary["train"]["spearman_per_cell_median_mean"] - 0.75) < 1e-9
    assert abs(summary["holdout"]["spearman_per_cell_median_mean"] - 0.50) < 1e-9
    assert abs(summary["test"]["spearman_per_cell_median_mean"] - 0.55) < 1e-9
    # gap = in-distribution(train) - unseen ; positive = overfitting
    assert abs(gaps["gap_train_minus_holdout__spearman_per_cell_median"] - 0.25) < 1e-9
    assert abs(gaps["gap_train_minus_test__spearman_per_cell_median"] - 0.20) < 1e-9
    assert abs(gaps["gap_train_minus_holdout__global_distance_spearman"] - 0.25) < 1e-9
    _ok("gap: per-split means + train-minus-unseen gaps correct")


def test_gap_no_train_split_no_gaps() -> None:
    # Without a 'train' (in-distribution) reference, gaps are undefined/empty.
    per_slice = [
        {"split": "holdout", "spearman_per_cell_median": 0.5},
        {"split": "test", "spearman_per_cell_median": 0.55},
    ]
    summary, gaps = R._generalization_gap(per_slice)
    assert "train" not in summary
    assert gaps == {}
    _ok("gap: no in-distribution split -> no gap entries (no crash)")


def test_gap_handles_nan() -> None:
    per_slice = [
        {"split": "train", "spearman_per_cell_median": float("nan")},
        {"split": "train", "spearman_per_cell_median": 0.6},
        {"split": "holdout", "spearman_per_cell_median": 0.4},
    ]
    summary, gaps = R._generalization_gap(per_slice)
    # nan dropped from the mean
    assert abs(summary["train"]["spearman_per_cell_median_mean"] - 0.6) < 1e-9
    assert abs(gaps["gap_train_minus_holdout__spearman_per_cell_median"] - 0.2) < 1e-9
    _ok("gap: NaN per-slice values are dropped from the means")


if __name__ == "__main__":
    print("test_partition_basic"); test_partition_basic()
    print("test_partition_overlap_raises"); test_partition_overlap_raises()
    print("test_partition_unmatched_reported"); test_partition_unmatched_reported()
    print("test_gap_math"); test_gap_math()
    print("test_gap_no_train_split_no_gaps"); test_gap_no_train_split_no_gaps()
    print("test_gap_handles_nan"); test_gap_handles_nan()
    print("\nAll LOSO unit tests passed.")

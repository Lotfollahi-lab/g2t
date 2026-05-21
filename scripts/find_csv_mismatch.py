#!/usr/bin/env python
"""
Pinpoint the first row where two CSVs disagree after LUNA-style sorting.

Replicates LUNA's data_module preprocessing up to the point where rows
are committed (sort_values → reset_index) and then walks row-by-row to
find:
  * the first row where cell_id differs (cells in different order)
  * the first row where coords differ
  * the first row where cell_class differs
  * a per-section summary: how many cells differ in each section, and
    what the within-section cell_id orderings look like

Usage:
    python find_csv_mismatch.py --csv_a <bronze> --csv_b <fresh>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv_a", required=True)
    p.add_argument("--csv_b", required=True)
    p.add_argument("--n_genes", type=int, default=254)
    args = p.parse_args()

    print(f"Loading {args.csv_a}")
    a = pd.read_csv(args.csv_a, index_col=0)
    print(f"Loading {args.csv_b}")
    b = pd.read_csv(args.csv_b, index_col=0)

    # Mimic LUNA: sort_values("cell_section"), preserve original index
    a = a.sort_values("cell_section", ignore_index=False)
    b = b.sort_values("cell_section", ignore_index=False)

    if len(a) != len(b):
        print(f"⚠️  Length mismatch: A={len(a)}, B={len(b)}")
        return 1

    # Save index as a column (cell_id) before reset_index
    a_ids = a.index.to_numpy()
    b_ids = b.index.to_numpy()

    a_sec = a["cell_section"].to_numpy()
    b_sec = b["cell_section"].to_numpy()

    # Walk row by row
    n = len(a)
    diff_id = a_ids != b_ids
    n_diff_id = diff_id.sum()
    print(f"\nTotal rows: {n}")
    print(f"Rows where cell_id differs: {n_diff_id}  ({100*n_diff_id/n:.2f}%)")

    if n_diff_id == 0:
        print("✓ All cell_ids match row-by-row. Tensors should be identical.")
        # Even so, check coord+class to be thorough
        coord_diff = (a["coord_X"].to_numpy() != b["coord_X"].to_numpy()).sum()
        print(f"  rows where coord_X differs: {coord_diff}")
        cls_diff = (a["cell_class"].astype(str).to_numpy() != b["cell_class"].astype(str).to_numpy()).sum()
        print(f"  rows where cell_class differs: {cls_diff}")
        return 0

    # Per-section breakdown
    print()
    print("Per-section mismatch breakdown (top 10 offending sections):")
    sections_a = a["cell_section"].unique().tolist()
    rows = []
    for sec in sections_a:
        mask_a = a_sec == sec
        mask_b = b_sec == sec
        if mask_a.sum() != mask_b.sum():
            rows.append((sec, mask_a.sum(), mask_b.sum(), -1, []))
            continue
        a_ids_sec = a_ids[mask_a]
        b_ids_sec = b_ids[mask_b]
        n_diff = (a_ids_sec != b_ids_sec).sum()
        if n_diff > 0:
            # First few mismatching positions
            mismatches = np.where(a_ids_sec != b_ids_sec)[0][:5]
            samples = [
                (int(pos), int(a_ids_sec[pos]), int(b_ids_sec[pos]))
                for pos in mismatches
            ]
            rows.append((sec, len(a_ids_sec), len(b_ids_sec), n_diff, samples))

    rows.sort(key=lambda r: -r[3])  # most-mismatched first
    print(f"{'section':<25s} {'n_a':>6s} {'n_b':>6s} {'n_diff':>7s}  first 5 (pos, a_id, b_id)")
    for sec, n_a, n_b, n_diff, samples in rows[:10]:
        if n_diff < 0:
            print(f"{sec:<25s} {n_a:>6d} {n_b:>6d}   COUNT MISMATCH")
        else:
            print(f"{sec:<25s} {n_a:>6d} {n_b:>6d} {n_diff:>7d}  {samples}")

    # Check if the mismatching sections have differently-ORDERED cell_ids
    print()
    print("Within-section cell_id orderings (first 3 mismatching sections):")
    for sec, n_a, n_b, n_diff, samples in rows[:3]:
        if n_diff <= 0:
            continue
        a_ids_sec = a_ids[a_sec == sec]
        b_ids_sec = b_ids[b_sec == sec]
        a_asc = (a_ids_sec[:-1] <= a_ids_sec[1:]).all()
        b_asc = (b_ids_sec[:-1] <= b_ids_sec[1:]).all()
        # Same set?
        a_set = set(a_ids_sec.tolist())
        b_set = set(b_ids_sec.tolist())
        in_a_only = a_set - b_set
        in_b_only = b_set - a_set
        print(f"  {sec}:")
        print(f"    a: asc={a_asc}, first 15 = {a_ids_sec[:15].tolist()}")
        print(f"    b: asc={b_asc}, first 15 = {b_ids_sec[:15].tolist()}")
        if in_a_only or in_b_only:
            print(f"    ⚠️ CELL SET MISMATCH: in A only n={len(in_a_only)}, in B only n={len(in_b_only)}")
            print(f"       in A only (first 10): {sorted(in_a_only)[:10]}")
            print(f"       in B only (first 10): {sorted(in_b_only)[:10]}")

    # First overall mismatch
    first_diff = int(np.where(diff_id)[0][0])
    print()
    print(f"First mismatching row (overall): {first_diff}")
    print(f"  section: {a_sec[first_diff]} (both)")
    print(f"  A cell_id={a_ids[first_diff]}, B cell_id={b_ids[first_diff]}")
    print(f"  A class={a['cell_class'].iloc[first_diff]}, B class={b['cell_class'].iloc[first_diff]}")
    print(f"  A coord_X={a['coord_X'].iloc[first_diff]}, B coord_X={b['coord_X'].iloc[first_diff]}")

    return 1


if __name__ == "__main__":
    sys.exit(main())

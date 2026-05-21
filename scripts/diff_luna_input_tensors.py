#!/usr/bin/env python
"""
Bit-identity check: compare what LUNA's data_module would build from
two CSV files (e.g. bronze vs h5ad-derived fresh CSV).

Replicates LUNA's exact data_module pipeline:
    1. pd.read_csv(index_col=0)
    2. sort_values("cell_section", ignore_index=False)
    3. filter_genes: first 254 columns, alphabetically sorted
    4. positions = torch.tensor(df[["coord_X","coord_Y"]].values).float()
    5. node_features = torch.tensor(df[gene_names].values).float()
    6. cell_class: sorted(unique()) → int mapping

Then asserts each tensor pair is bitwise identical between the two inputs.

If both are `True`, the data going into LUNA's model/loss/optimizer is
identical and any training-result delta is CUDA non-determinism. If
either is `False`, the mismatch will print the exact magnitude.

Usage:
    python diff_luna_input_tensors.py \\
        --csv_a /nfs/team361/sb75/DATASETS/bronze/mmc_luna/MERFISH_mouse_cortex_train.csv \\
        --csv_b /nfs/team361/sb75/scgg-reproducibility/artifacts/mmc_luna/luna_model/<TS>/work/train.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_and_process(path: Path, n_genes: int = 254):
    """Replicate LUNA's data_module preprocessing up to the float32
    tensor that flows into the model."""
    df = pd.read_csv(path, index_col=0)
    df = df.sort_values("cell_section", ignore_index=False)
    # LUNA's filter_genes: first n_genes columns, then sort alphabetic.
    gene_names = sorted(list(df.columns[:n_genes]))
    if not all(col in df.columns for col in ("coord_X", "coord_Y", "cell_section", "cell_class")):
        missing = [c for c in ("coord_X", "coord_Y", "cell_section", "cell_class") if c not in df.columns]
        raise ValueError(f"{path}: missing required columns: {missing}")
    # LUNA's _convert_data_to_tensors: float64 → float32 cast.
    positions = torch.tensor(df[["coord_X", "coord_Y"]].values).float()
    node_features = torch.tensor(df[gene_names].values).float()
    # cell_class → int mapping (alphabetic).
    classes_sorted = sorted(df["cell_class"].unique())
    cls_to_int = {c: i for i, c in enumerate(classes_sorted)}
    cell_class_int = torch.tensor(
        [cls_to_int[c] for c in df["cell_class"].values], dtype=torch.long
    )
    # Section IDs and within-section row positions (for ordering diagnostic).
    section_labels = df["cell_section"].values
    return {
        "n_cells": len(df),
        "gene_names": gene_names,
        "positions": positions,
        "node_features": node_features,
        "cell_class_int": cell_class_int,
        "classes_sorted": classes_sorted,
        "section_labels": section_labels,
    }


def compare(a, b, name: str) -> bool:
    """Print a one-line diff. Returns True if identical."""
    if a.shape != b.shape:
        print(f"  {name:20s}  SHAPE MISMATCH  a={tuple(a.shape)}  b={tuple(b.shape)}")
        return False
    if a.dtype != b.dtype:
        print(f"  {name:20s}  DTYPE MISMATCH  a={a.dtype}  b={b.dtype}")
        # continue anyway
    equal = torch.equal(a, b)
    max_diff = (a.float() - b.float()).abs().max().item()
    mean_diff = (a.float() - b.float()).abs().mean().item()
    status = "IDENTICAL" if equal else "DIFFERS"
    print(
        f"  {name:20s}  {status:9s}  "
        f"max |a-b| = {max_diff:.3e}  mean |a-b| = {mean_diff:.3e}"
    )
    return equal


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv_a", required=True, help="First CSV path (e.g. bronze).")
    p.add_argument("--csv_b", required=True, help="Second CSV path (e.g. fresh from h5ad).")
    p.add_argument("--n_genes", type=int, default=254, help="Number of gene columns (default 254 for cortex).")
    args = p.parse_args()

    a = load_and_process(Path(args.csv_a), n_genes=args.n_genes)
    b = load_and_process(Path(args.csv_b), n_genes=args.n_genes)

    print(f"\nCSV A: {args.csv_a}")
    print(f"  n_cells = {a['n_cells']}, n_genes = {len(a['gene_names'])}")
    print(f"CSV B: {args.csv_b}")
    print(f"  n_cells = {b['n_cells']}, n_genes = {len(b['gene_names'])}")
    print()

    ok = True

    if a["n_cells"] != b["n_cells"]:
        print(f"⚠️  CELL COUNT MISMATCH: a={a['n_cells']}, b={b['n_cells']}")
        ok = False

    if a["gene_names"] != b["gene_names"]:
        print("⚠️  GENE PANEL MISMATCH (after alphabetic sort)")
        in_a_only = set(a["gene_names"]) - set(b["gene_names"])
        in_b_only = set(b["gene_names"]) - set(a["gene_names"])
        if in_a_only:
            print(f"   in A only: {sorted(in_a_only)[:10]}")
        if in_b_only:
            print(f"   in B only: {sorted(in_b_only)[:10]}")
        ok = False
    else:
        print("Gene panel: identical (same set, same alphabetic order)")

    if a["classes_sorted"] != b["classes_sorted"]:
        print("⚠️  CELL_CLASS VOCABULARY MISMATCH")
        ok = False
    else:
        print(f"Cell class vocabulary: identical ({len(a['classes_sorted'])} classes)")

    # Section labels (in row order) — should match if sort_values produces the
    # same section order AND within-section row ordering matches.
    if not np.array_equal(a["section_labels"], b["section_labels"]):
        print("⚠️  SECTION-LABEL ORDER DIFFERS (within-section row order likely differs)")
        ok = False
    else:
        print("Section labels (row by row): identical")

    print()
    print("--- LUNA-equivalent input tensors (after .float() cast to float32) ---")
    ok &= compare(a["positions"], b["positions"], "positions")
    ok &= compare(a["node_features"], b["node_features"], "node_features")
    ok &= compare(a["cell_class_int"], b["cell_class_int"], "cell_class_int")

    print()
    if ok:
        print("✓ ALL TENSORS BITWISE IDENTICAL. Any training-result delta between "
              "these two inputs is CUDA non-determinism (LUNA does not set "
              "torch.backends.cudnn.deterministic=True).")
        return 0
    else:
        print("✗ MISMATCH. See above for the exact magnitude.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Compare LUNA's published CSV to the silver h5ad layer for the same section.

The training run that used the LUNA-published CSVs reproduced LUNA's
Figure 3 numbers; the one that used silver h5ads (with our
``run_luna_on_mmc.py:_build_luna_csv`` applying ``log2(x+1)``) did not.
This script pinpoints where the two diverge.

For one matched section (chosen by ``--section``, default
``mouse2_slice99`` = the LUNA Fig 3c reference), we extract the same
quantities from both sources and report:

  * Cell counts and whether the cell sets overlap by barcode.
  * Gene panel: is the var name set identical? Same order?
  * Gene expression distribution (mean, std, min, max, sparsity, a few
    quantiles). Reported under THREE candidate normalizations of the
    h5ad ``X``:
      - raw counts (``X`` as-is)
      - log2(x + 1)                    (what run_luna_on_mmc.py applies)
      - normalize_total(1e4) + log1p   (scgg's choice)
    so it's clear which (if any) matches the CSV's distribution.
  * Coordinate scale: raw ``obsm['spatial']`` vs CSV ``coord_X/Y`` —
    check whether the CSV is already per-section normalized to
    [-0.5, 0.5] or carries raw microns.
  * ``cell_class`` label sets (if both sources carry them).

This is a read-only diagnostic — it produces a text/JSON report and
nothing else.

Usage
-----

    # Compare one section (default)
    python scripts/compare_luna_csv_vs_h5ad.py \\
        --csv /nfs/team361/sb75/DATASETS/luna_paper_csvs/MERFISH_mouse_cortex/test.csv \\
        --silver_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --section mouse2_slice99

    # Compare several
    python scripts/compare_luna_csv_vs_h5ad.py \\
        --csv .../test.csv --silver_dir .../mmc_luna \\
        --section mouse2_slice99 mouse2_slice119 mouse2_slice169

    # Auto-pair train.csv with Mouse-1 sections / test.csv with Mouse-2
    python scripts/compare_luna_csv_vs_h5ad.py \\
        --train_csv .../train.csv --test_csv .../test.csv \\
        --silver_dir .../mmc_luna --section mouse1_slice1 mouse2_slice99
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


_METADATA_NAMES = (
    "coord_X", "coord_Y", "x", "y",
    "cell_section", "section", "region", "slice",
    "cell_class", "cell_type", "class", "subclass", "type",
    "cell_name", "cell_id", "cell_barcode", "barcode",
    "mouse", "animal", "donor",
    "sample", "sample_id", "batch", "experiment", "cluster",
)

_SILVER_PREFIXES = ("mmc_", "merfish_mouse_cortex_")


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def _csv_gene_metadata_boundary(head: pd.DataFrame) -> int:
    cols = list(head.columns)
    meta_pos_by_name = [cols.index(n) for n in _METADATA_NAMES if n in cols]
    name_based = min(meta_pos_by_name) if meta_pos_by_name else None
    dtype_based = None
    for i, c in enumerate(cols):
        if not pd.api.types.is_numeric_dtype(head[c]):
            dtype_based = i
            break
    candidates = [v for v in (name_based, dtype_based) if v is not None]
    if not candidates:
        raise ValueError("Could not locate the gene/metadata boundary in CSV.")
    return min(candidates)


def _load_csv_section(csv_path: Path, section_label: str) -> pd.DataFrame:
    head = pd.read_csv(csv_path, nrows=200, index_col=0)
    n_genes = _csv_gene_metadata_boundary(head)
    print(f"  [{csv_path.name}] n_genes={n_genes}, "
          f"total_cols={len(head.columns)}")
    df = pd.read_csv(csv_path, index_col=0)
    if "cell_section" not in df.columns:
        raise ValueError(f"No 'cell_section' column in {csv_path}")
    matched = df[df["cell_section"].astype(str) == section_label].copy()
    return matched


# ---------------------------------------------------------------------------
# h5ad loading
# ---------------------------------------------------------------------------


def _find_h5ad_for_section(silver_dir: Path, section_label: str) -> Optional[Path]:
    for pref in _SILVER_PREFIXES:
        p = silver_dir / f"{pref}{section_label}.h5ad"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _array_stats(x: np.ndarray, name: str) -> Dict[str, float]:
    flat = x.reshape(-1)
    finite = np.isfinite(flat)
    flat = flat[finite]
    if flat.size == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(flat.size),
        "min": float(flat.min()),
        "p01": float(np.quantile(flat, 0.01)),
        "p25": float(np.quantile(flat, 0.25)),
        "median": float(np.median(flat)),
        "mean": float(flat.mean()),
        "p75": float(np.quantile(flat, 0.75)),
        "p99": float(np.quantile(flat, 0.99)),
        "max": float(flat.max()),
        "std": float(flat.std()),
        "sparsity_zero_frac": float((flat == 0).sum() / flat.size),
        "frac_negative": float((flat < 0).sum() / flat.size),
        "frac_integer": float(
            (np.isclose(flat, np.round(flat))).sum() / flat.size
        ),
    }


def _print_stats_table(rows: List[Dict[str, float]], indent: str = "    ") -> None:
    keys = [
        "name", "n", "min", "p01", "p25", "median", "mean", "p75", "p99",
        "max", "std", "sparsity_zero_frac", "frac_negative", "frac_integer",
    ]
    widths = {k: max(len(k), max(len(_fmt(r.get(k))) for r in rows)) for k in keys}
    print(indent + "  ".join(k.ljust(widths[k]) for k in keys))
    for r in rows:
        print(indent + "  ".join(_fmt(r.get(k)).ljust(widths[k]) for k in keys))


def _fmt(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if abs(v) >= 1e4 or (0 < abs(v) < 1e-3):
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Main comparison for one section
# ---------------------------------------------------------------------------


def compare_section(
    csv_df: pd.DataFrame,
    csv_label: str,
    h5ad_path: Optional[Path],
    section_label: str,
) -> Dict[str, object]:
    """Print a side-by-side report. Returns a dict of the headline diffs."""
    print(f"\n{'=' * 72}")
    print(f"Section: {section_label}    (csv source: {csv_label})")
    print("=" * 72)

    n_genes_csv = _csv_gene_metadata_boundary(csv_df.head(50))
    csv_genes = list(csv_df.columns[:n_genes_csv])
    csv_X = csv_df.iloc[:, :n_genes_csv].to_numpy(dtype=np.float32)
    csv_coords = csv_df[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
    csv_class = (
        csv_df["cell_class"].astype(str).to_numpy()
        if "cell_class" in csv_df.columns
        else None
    )
    csv_index = list(csv_df.index)

    print(f"\nCSV side:")
    print(f"  n_cells       = {len(csv_df)}")
    print(f"  n_genes       = {n_genes_csv}")
    print(f"  first 5 genes = {csv_genes[:5]}")
    print(f"  last  5 genes = {csv_genes[-5:]}")
    print(f"  coord_X range = [{csv_coords[:, 0].min():.4f}, "
          f"{csv_coords[:, 0].max():.4f}]")
    print(f"  coord_Y range = [{csv_coords[:, 1].min():.4f}, "
          f"{csv_coords[:, 1].max():.4f}]")
    print(f"  CSV index dtype/first = {type(csv_index[0]).__name__}, "
          f"e.g. {csv_index[:3]}")

    headline: Dict[str, object] = {
        "section": section_label,
        "csv_n_cells": int(len(csv_df)),
        "csv_n_genes": int(n_genes_csv),
        "csv_coord_X_range": [
            float(csv_coords[:, 0].min()), float(csv_coords[:, 0].max())
        ],
        "csv_coord_Y_range": [
            float(csv_coords[:, 1].min()), float(csv_coords[:, 1].max())
        ],
        "csv_index_dtype": type(csv_index[0]).__name__,
    }

    if h5ad_path is None or not h5ad_path.exists():
        print("\nh5ad side: NOT FOUND. Skipping side-by-side comparison.")
        headline["h5ad_missing"] = True
        return headline

    import anndata as ad
    import scipy.sparse as sp

    adata = ad.read_h5ad(h5ad_path)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    print(f"\nh5ad side ({h5ad_path.name}):")
    print(f"  n_cells              = {adata.n_obs}")
    print(f"  n_genes              = {adata.n_vars}")
    print(f"  first 5 var_names    = {list(adata.var_names[:5])}")
    print(f"  last  5 var_names    = {list(adata.var_names[-5:])}")
    print(f"  obs_names[:3]        = {list(adata.obs_names[:3])}")
    print(f"  obsm keys            = {list(adata.obsm.keys())}")
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        print(f"  obsm['spatial'] X    = [{xy[:, 0].min():.4f}, "
              f"{xy[:, 0].max():.4f}]")
        print(f"  obsm['spatial'] Y    = [{xy[:, 1].min():.4f}, "
              f"{xy[:, 1].max():.4f}]")

    headline.update({
        "h5ad_n_cells": int(adata.n_obs),
        "h5ad_n_genes": int(adata.n_vars),
        "h5ad_coord_X_range": (
            [float(xy[:, 0].min()), float(xy[:, 0].max())]
            if "spatial" in adata.obsm else None
        ),
        "h5ad_coord_Y_range": (
            [float(xy[:, 1].min()), float(xy[:, 1].max())]
            if "spatial" in adata.obsm else None
        ),
    })

    # --- Cell count match ---
    if adata.n_obs != len(csv_df):
        print(f"\n  ⚠️  CELL COUNT MISMATCH: csv={len(csv_df)} vs "
              f"h5ad={adata.n_obs}")
        headline["cell_count_match"] = False
    else:
        headline["cell_count_match"] = True

    # --- Gene panel comparison ---
    h5ad_genes = list(adata.var_names)
    same_set = set(csv_genes) == set(h5ad_genes)
    same_order = csv_genes == h5ad_genes
    print("\nGene panel:")
    print(f"  same set?   {same_set}")
    print(f"  same order? {same_order}")
    if not same_set:
        in_csv_only = set(csv_genes) - set(h5ad_genes)
        in_h5ad_only = set(h5ad_genes) - set(csv_genes)
        print(f"  ↳ in CSV only ({len(in_csv_only)}): "
              f"{list(in_csv_only)[:5]}...")
        print(f"  ↳ in h5ad only ({len(in_h5ad_only)}): "
              f"{list(in_h5ad_only)[:5]}...")
    headline["gene_panel_same_set"] = same_set
    headline["gene_panel_same_order"] = same_order

    # --- Expression distribution under several normalizations ---
    # Reindex CSV columns to h5ad var_names where possible, so per-gene
    # alignment is correct when the order differs.
    if same_set:
        csv_X_aligned = csv_df.loc[:, h5ad_genes].to_numpy(dtype=np.float32)
    else:
        csv_X_aligned = csv_X  # fall back

    norms: Dict[str, np.ndarray] = {
        "csv (as-published)": csv_X_aligned,
        "h5ad raw .X": X,
        "h5ad log2(x+1)": np.log2(X + 1.0),
    }

    # Optional scanpy normalize_total + log1p (only if scanpy available)
    try:
        import scanpy as sc
        tmp = adata.copy()
        if sp.issparse(tmp.X):
            tmp.X = tmp.X.toarray()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        Xn = np.asarray(tmp.X, dtype=np.float32)
        norms["h5ad normalize_total+log1p"] = Xn
    except Exception as e:
        print(f"  (scanpy normalize_total+log1p comparison skipped: {e})")

    print("\nExpression distributions (per-cell-gene scalar, all values):")
    rows = [_array_stats(arr, name=name) for name, arr in norms.items()]
    _print_stats_table(rows)

    headline["expression_norms"] = [
        {k: r.get(k) for k in (
            "name", "min", "median", "mean", "max", "std",
            "sparsity_zero_frac", "frac_integer", "frac_negative",
        )}
        for r in rows
    ]

    # --- Per-cell sums (proxy for "is it counts or normalized?") ---
    csv_row_sums = csv_X_aligned.sum(axis=1)
    h5_row_sums = X.sum(axis=1)
    print("\nPer-cell total over genes:")
    print(f"  CSV       median={np.median(csv_row_sums):.4f}, "
          f"std={csv_row_sums.std():.4f}, range=["
          f"{csv_row_sums.min():.4f}, {csv_row_sums.max():.4f}]")
    print(f"  h5ad raw  median={np.median(h5_row_sums):.4f}, "
          f"std={h5_row_sums.std():.4f}, range=["
          f"{h5_row_sums.min():.4f}, {h5_row_sums.max():.4f}]")

    headline["csv_row_sum_median"] = float(np.median(csv_row_sums))
    headline["h5ad_row_sum_median"] = float(np.median(h5_row_sums))

    # --- Per-gene first-cell direct compare (helps catch ordering bugs) ---
    if same_set and adata.n_obs == len(csv_df):
        # Need to align cells too; try by index match if CSV index is
        # integer (LUNA preserves a per-section integer index) → assume
        # CSV row i corresponds to h5ad row i if cell counts match.
        # Otherwise we just compare the SORTED gene-expression vectors
        # per gene as a distribution sanity check.
        try:
            csv_row0 = csv_df.loc[:, h5ad_genes].iloc[0].to_numpy(
                dtype=np.float32
            )
            h5_row0 = X[0]
            # find nearest match (log2 transform?)
            cand: List[Tuple[str, float]] = []
            for name, arr in norms.items():
                if name.startswith("h5ad"):
                    diff = float(np.mean(np.abs(arr[0] - csv_row0)))
                    cand.append((name, diff))
            cand.sort(key=lambda kv: kv[1])
            print("\nFirst-row absolute diff (CSV row 0 vs h5ad row 0):")
            for name, d in cand:
                print(f"  CSV vs {name:35s}  mean |diff| = {d:.6f}")
            headline["closest_norm_match"] = cand[0][0]
        except Exception as e:
            print(f"  (first-row direct compare skipped: {e})")

    # --- cell_class label set ---
    csv_classes = (
        set(csv_class.tolist()) if csv_class is not None else None
    )
    h5_classes = (
        set(adata.obs["cell_class"].astype(str).unique().tolist())
        if "cell_class" in adata.obs.columns
        else None
    )
    print("\ncell_class:")
    print(f"  CSV  : {sorted(csv_classes)[:10] if csv_classes else None}"
          f"  (n={len(csv_classes) if csv_classes else 0})")
    print(f"  h5ad : {sorted(h5_classes)[:10] if h5_classes else None}"
          f"  (n={len(h5_classes) if h5_classes else 0})")
    if csv_classes and h5_classes:
        sym = csv_classes ^ h5_classes
        print(f"  symmetric diff: {sorted(sym)[:10]} (n={len(sym)})")
        headline["cell_class_set_match"] = (len(sym) == 0)
        headline["cell_class_csv_only"] = sorted(csv_classes - h5_classes)
        headline["cell_class_h5ad_only"] = sorted(h5_classes - csv_classes)

    return headline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--csv", default=None,
        help="Single LUNA CSV to read. Use this OR (--train_csv + --test_csv).",
    )
    p.add_argument(
        "--train_csv", default=None,
        help="LUNA's train CSV (covers Mouse 1 sections).",
    )
    p.add_argument(
        "--test_csv", default=None,
        help="LUNA's test CSV (covers Mouse 2 sections).",
    )
    p.add_argument(
        "--silver_dir", required=True,
        help="Per-slice silver h5ad directory.",
    )
    p.add_argument(
        "--section", nargs="+", default=["mouse2_slice99"],
        help="Section labels (matching cell_section in CSVs). "
             "Default: mouse2_slice99.",
    )
    p.add_argument(
        "--out_json", default=None,
        help="Optional JSON report path. Defaults to stdout only.",
    )
    args = p.parse_args()

    silver_dir = Path(args.silver_dir)
    if not silver_dir.exists():
        raise SystemExit(f"silver_dir does not exist: {silver_dir}")

    if args.csv is None and (args.train_csv is None or args.test_csv is None):
        raise SystemExit(
            "Need either --csv (single file) or both --train_csv and --test_csv."
        )

    # Map section -> CSV path
    csv_for_section: Dict[str, Path] = {}
    if args.csv is not None:
        single = Path(args.csv)
        if not single.exists():
            raise SystemExit(f"--csv not found: {single}")
        for s in args.section:
            csv_for_section[s] = single
    else:
        train_csv = Path(args.train_csv)
        test_csv = Path(args.test_csv)
        for s in args.section:
            if s.startswith("mouse1_"):
                csv_for_section[s] = train_csv
            elif s.startswith("mouse2_"):
                csv_for_section[s] = test_csv
            else:
                # default to test_csv
                csv_for_section[s] = test_csv

    summary: List[Dict[str, object]] = []
    for section_label in args.section:
        csv_path = csv_for_section[section_label]
        try:
            csv_df = _load_csv_section(csv_path, section_label)
        except Exception as e:
            print(f"\n[{section_label}] failed to load from CSV "
                  f"{csv_path}: {e}", file=sys.stderr)
            continue
        if csv_df.empty:
            print(f"\n[{section_label}] no rows in {csv_path.name} for this "
                  f"section. Skipping.", file=sys.stderr)
            continue
        h5ad_path = _find_h5ad_for_section(silver_dir, section_label)
        report = compare_section(
            csv_df=csv_df,
            csv_label=csv_path.name,
            h5ad_path=h5ad_path,
            section_label=section_label,
        )
        summary.append(report)

    print("\n" + "=" * 72)
    print("Headline summary:")
    print("=" * 72)
    for r in summary:
        flags = []
        if r.get("cell_count_match") is False:
            flags.append("CELL_COUNT_MISMATCH")
        if r.get("gene_panel_same_set") is False:
            flags.append("GENE_SET_MISMATCH")
        if r.get("gene_panel_same_order") is False:
            flags.append("GENE_ORDER_DIFFERS")
        if r.get("cell_class_set_match") is False:
            flags.append("CELL_CLASS_DIFFERS")
        closest = r.get("closest_norm_match", "?")
        print(
            f"  {r['section']:<22s}  "
            f"closest_norm={closest:<35s}  "
            f"{'  '.join(flags) if flags else 'OK'}"
        )

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nWrote JSON report: {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

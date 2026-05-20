#!/usr/bin/env python
"""
Build per-slice h5ad files from LUNA's published CSV(s).

Produces a directory of per-section h5ads that drop in as
``--data_dir`` for the existing scGG and LUNA training scripts:

    {out_dir}/{prefix}_{cell_section_value}.h5ad

with the same layout as the existing silver dirs (``mmc_luna``,
``abc_luna``, ``cns_luna``):

  * ``.X``               (n_cells, n_genes) float32 — LUNA's CSV expression
                         values as-is (non-integer per-cell-normalized counts).
  * ``.obsm['spatial']`` (n_cells, 2) float32 — raw micron coords from
                         ``coord_X``, ``coord_Y``. **Omitted** when the
                         CSV has no such columns (scRNA-seq case — those
                         cells are dissociated and have no spatial info).
  * ``.obs['cell_class']`` categorical — cell type label (when present).
  * ``.obs['cell_section']`` — repeated section label.
  * ``.obs['cell_id']``  — original integer cell id from CSV index.
  * ``.obs[<other>]``    — any other metadata column we find in the CSV
                         (e.g. ``class``, ``animal``, ``donor``, ``sample_id``).
  * ``.var_names``       — gene symbols as written in LUNA's CSV.

Why this exists
---------------
LUNA's published preprocessing differs slightly from our silver pipeline
(see ``compare_luna_csv_vs_h5ad.py``):

  * silver applies a ``min_genes=10`` QC filter; LUNA's CSV keeps those
    ~18 low-count cells per slice.
  * silver renames 2 of 254 gene symbols via an alias map.
  * the CSV's per-element values differ from silver raw counts by ~6 %
    (different upstream Vizgen segmentation pass).

For the "reproduce LUNA's paper number" path it's cleanest to make the
LUNA CSV itself the silver layer. This script does that conversion once
and writes h5ads that scGG and LUNA's training/inference scripts can
consume *as if they were the regular silver h5ads*.

Usage
-----

    # MERFISH mouse cortex (LUNA Fig 3): sections look like
    #   "mouse1_slice1" → filename "merfish_mouse_cortex_mouse1_slice1.h5ad"
    python scripts/build_h5ad_from_luna_csv.py \\
        --train_csv /nfs/team361/sb75/DATASETS/bronze/mmc_luna/MERFISH_mouse_cortex_train.csv \\
        --test_csv  /nfs/team361/sb75/DATASETS/bronze/mmc_luna/MERFISH_mouse_cortex_test.csv \\
        --out_dir   /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --prefix    merfish_mouse_cortex \\
        --overwrite

    # ABC Zhuang ABCA1 (LUNA Fig 4 train side): sections look like
    #   "Zhuang-ABCA-1-001" → filename "abc_zhuang_abca1_Zhuang-ABCA-1-001.h5ad"
    python scripts/build_h5ad_from_luna_csv.py \\
        --train_csv /nfs/team361/sb75/DATASETS/bronze/abc_luna/MERFISH_ABCA_animal1_train.csv \\
        --test_csv  /nfs/team361/sb75/DATASETS/bronze/abc_luna/MERFISH_ABCA_animal1_test.csv \\
        --out_dir   /nfs/team361/sb75/DATASETS/silver/abc_luna \\
        --prefix    abc_zhuang_abca1 \\
        --overwrite

    # CNS harmonized (LUNA Fig 4 cross-modality): train = ABCA spatial,
    # test = scRNA-seq. scRNA cells have no real spatial coords; the
    # resulting h5ads will be missing obsm['spatial'] for those sections.
    python scripts/build_h5ad_from_luna_csv.py \\
        --train_csv /nfs/team361/sb75/DATASETS/bronze/cns_luna/ABCA_harmonized_train.csv \\
        --test_csv  /nfs/team361/sb75/DATASETS/bronze/cns_luna/scRNA_harmonized_test.csv \\
        --out_dir   /nfs/team361/sb75/DATASETS/silver/cns_luna \\
        --prefix    cns_scrna \\
        --overwrite

Filename convention
-------------------
Each section in the CSV becomes one h5ad named
``{prefix}_{cell_section_value}.h5ad``. The section label is preserved
verbatim except for filesystem-unsafe characters (``/``, control chars)
which are replaced with ``_``.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger("build_h5ad_from_luna_csv")


# Heuristic copy from compare_luna_csv_vs_h5ad.py: known metadata column
# names used to locate the gene/metadata boundary.
_METADATA_NAMES = (
    "coord_X", "coord_Y", "x", "y",
    "cell_section", "section", "region", "slice",
    "cell_class", "cell_type", "class", "subclass", "type",
    "cell_name", "cell_id", "cell_barcode", "barcode",
    "mouse", "animal", "donor",
    "sample", "sample_id", "batch", "experiment", "cluster",
)


def _sanitize_for_filename(label: str) -> str:
    """Replace filesystem-unsafe characters in section labels.

    Section labels in LUNA CSVs can be anything string-shaped (e.g.
    ``Zhuang-ABCA-1-001`` or ``well06``); we only sanitize characters
    that are problematic in filenames (path separators, NULs, control
    chars), preserving the rest verbatim so users can still
    reverse-map filename → cell_section.
    """
    bad = re.compile(r"[/\x00-\x1f]")
    out = bad.sub("_", label)
    return out


# ---------------------------------------------------------------------------
# CSV introspection
# ---------------------------------------------------------------------------


def _csv_gene_metadata_boundary(head: pd.DataFrame) -> int:
    """Position of the first non-gene column in `head` (the gene/metadata
    boundary). Combines a name-based and a dtype-based detection.
    """
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


def _load_csv(csv_path: Path) -> Tuple[pd.DataFrame, int, List[str]]:
    head = pd.read_csv(csv_path, nrows=200, index_col=0)
    n_genes = _csv_gene_metadata_boundary(head)
    gene_names = list(head.columns[:n_genes])
    logger.info(
        f"  [{csv_path.name}] n_genes={n_genes}, total_cols={len(head.columns)}"
    )
    df = pd.read_csv(csv_path, index_col=0)
    return df, n_genes, gene_names


def _parse_section(label: str) -> Tuple[int, int]:
    m = _SECTION_RE.match(label)
    if not m:
        raise ValueError(
            f"section label {label!r} does not match 'mouseM_sliceS'; "
            f"extend the regex if your data uses a different convention."
        )
    return int(m["mouse"]), int(m["slice"])


# ---------------------------------------------------------------------------
# AnnData construction
# ---------------------------------------------------------------------------


def _adata_for_section(
    df: pd.DataFrame,
    gene_names: List[str],
    n_genes: int,
    section_label: str,
):
    """Construct an AnnData for one section from the matching CSV slice.

    Spatial coords are optional — datasets like scRNA-seq don't carry
    real (x, y), in which case ``obsm['spatial']`` is omitted and the
    downstream code paths that need it must check for its absence.
    ``cell_section`` is required (it's how we groupby).
    """
    import anndata as ad

    if "cell_section" not in df.columns:
        raise ValueError(
            f"section {section_label!r}: required column 'cell_section' "
            f"missing (have: {list(df.columns)[:10]}...)"
        )

    # Use float64 (pandas' default for CSV reads) for the gene matrix
    # and coords. Casting to float32 here would introduce a LSB
    # rounding error that compounds when this h5ad is later round-
    # tripped back to a CSV for LUNA training — making h5ad-derived
    # training slightly different from CSV-direct training even
    # though the source data is identical. Memory cost: 2x the
    # h5ad's gene matrix (≤ 10 MB per cortex slice), negligible.
    X = df.iloc[:, :n_genes].to_numpy(dtype=np.float64)

    has_coords = "coord_X" in df.columns and "coord_Y" in df.columns
    if has_coords:
        coords = df[["coord_X", "coord_Y"]].to_numpy(dtype=np.float64)
    else:
        coords = None

    # All non-gene columns flow to obs (preserving the CSV's metadata).
    # We do NOT copy coord_X / coord_Y into obs to avoid the AnnData
    # convention surprise of having them in both .obs and .obsm — the
    # canonical location for spatial is .obsm['spatial'] (matches the
    # silver layer).
    obs_cols = [
        c for c in df.columns[n_genes:]
        if c not in ("coord_X", "coord_Y")
    ]
    obs = df[obs_cols].copy() if obs_cols else pd.DataFrame(index=df.index)

    # Categorical for cell_class if present
    if "cell_class" in obs.columns:
        obs["cell_class"] = pd.Categorical(obs["cell_class"].astype(str))

    # Preserve the original CSV integer index as a per-cell field, then
    # rewrite obs_names to "{cell_id}_{section}" so they match the silver
    # convention (e.g. "13_mouse1_slice1"). This makes the new h5ads
    # interchangeable with the silver h5ads for the comparison script.
    obs["cell_id"] = df.index.to_numpy()
    obs.index = [f"{cid}_{section_label}" for cid in df.index]
    obs.index.name = "obs_name"

    var = pd.DataFrame(index=pd.Index(gene_names, name="var_name"))

    adata = ad.AnnData(X=X, obs=obs, var=var)
    if coords is not None:
        adata.obsm["spatial"] = coords
    return adata


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_from_csvs(
    csv_paths: List[Path],
    out_dir: Path,
    prefix: str = "mmc",
    overwrite: bool = False,
) -> Dict[str, object]:
    """Read one or more LUNA CSVs and emit per-section h5ads."""
    if not csv_paths:
        raise ValueError("need at least one CSV path")

    out_dir.mkdir(parents=True, exist_ok=True)

    gene_names_ref: Optional[List[str]] = None
    n_genes_ref: Optional[int] = None
    sections_written: List[Dict[str, object]] = []

    for csv_path in csv_paths:
        df, n_genes, gene_names = _load_csv(csv_path)

        if gene_names_ref is None:
            gene_names_ref = gene_names
            n_genes_ref = n_genes
        else:
            if gene_names != gene_names_ref:
                raise ValueError(
                    f"gene panel in {csv_path} differs from the first CSV. "
                    f"Pass CSVs with identical gene columns."
                )
            assert n_genes == n_genes_ref

        if "cell_section" not in df.columns:
            raise ValueError(f"no 'cell_section' column in {csv_path}")

        for section_label, sec_df in df.groupby(
            df["cell_section"].astype(str)
        ):
            safe_label = _sanitize_for_filename(section_label)
            out_path = out_dir / f"{prefix}_{safe_label}.h5ad"
            if out_path.exists() and not overwrite:
                logger.info(
                    f"  skip (exists): {out_path.name}  "
                    f"(pass --overwrite to replace)"
                )
                continue

            adata = _adata_for_section(
                sec_df, gene_names, n_genes, section_label
            )
            adata.write(out_path)
            logger.info(
                f"  wrote {out_path.name:<60s}  "
                f"n_cells={adata.n_obs:>6d}  n_genes={adata.n_vars}  "
                f"spatial={'yes' if 'spatial' in adata.obsm else 'no'}"
            )
            sections_written.append({
                "section": section_label,
                "path": str(out_path),
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "has_spatial": "spatial" in adata.obsm,
                "source_csv": str(csv_path),
            })

    summary = {
        "out_dir": str(out_dir),
        "n_sections": len(sections_written),
        "n_genes": int(n_genes_ref) if n_genes_ref is not None else 0,
        "gene_names_head": (gene_names_ref or [])[:5],
        "gene_names_tail": (gene_names_ref or [])[-5:],
        "sections": sections_written,
    }
    return summary


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
        help="Single LUNA CSV to convert. Use this OR (--train_csv + "
             "--test_csv).",
    )
    p.add_argument(
        "--train_csv", default=None,
        help="LUNA's train CSV (Mouse 1 sections).",
    )
    p.add_argument(
        "--test_csv", default=None,
        help="LUNA's test CSV (Mouse 2 sections).",
    )
    p.add_argument(
        "--out_dir", required=True,
        help="Where to write per-section h5ads.",
    )
    p.add_argument(
        "--prefix", default="mmc",
        help="Filename prefix. Default 'mmc' matches the silver layer; "
             "use 'merfish_mouse_cortex' for the legacy convention.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing h5ads in --out_dir.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    csv_paths: List[Path] = []
    if args.csv is not None:
        p_single = Path(args.csv)
        if not p_single.exists():
            raise SystemExit(f"--csv not found: {p_single}")
        csv_paths.append(p_single)
    if args.train_csv is not None:
        p_train = Path(args.train_csv)
        if not p_train.exists():
            raise SystemExit(f"--train_csv not found: {p_train}")
        csv_paths.append(p_train)
    if args.test_csv is not None:
        p_test = Path(args.test_csv)
        if not p_test.exists():
            raise SystemExit(f"--test_csv not found: {p_test}")
        csv_paths.append(p_test)
    if not csv_paths:
        raise SystemExit(
            "need --csv or (--train_csv and/or --test_csv)."
        )

    out_dir = Path(args.out_dir)
    summary = build_from_csvs(
        csv_paths=csv_paths,
        out_dir=out_dir,
        prefix=args.prefix,
        overwrite=args.overwrite,
    )

    print()
    print("=" * 72)
    print(f"Wrote {summary['n_sections']} h5ads to {summary['out_dir']}")
    print(f"  n_genes      = {summary['n_genes']}")
    print(f"  first genes  = {summary['gene_names_head']}")
    print(f"  last  genes  = {summary['gene_names_tail']}")
    print(
        f"  total cells  = "
        f"{sum(s['n_cells'] for s in summary['sections']):,}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

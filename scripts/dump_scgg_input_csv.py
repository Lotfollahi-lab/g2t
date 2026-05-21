#!/usr/bin/env python
"""
Dump scGG's training-input data to a CSV in LUNA's bronze CSV format.

Purpose: verify that scGG and LUNA see *the same cells with the same
values* from a given silver layer. The produced CSV has the same
schema as LUNA's bronze CSV:

    column 0          : cell_id (the original bronze CSV index)
    columns 1..n_genes: gene expression values (alphabetic gene order)
    coord_X           : raw micron X coordinate
    coord_Y           : raw micron Y coordinate
    cell_section      : "mouse{M}_slice{S}" label
    cell_class        : cell-type label (string)

Two ordering modes:

  * ``--match_bronze_order`` (default): cells are sorted by
    ``_bronze_row_pos`` ascending, replaying the EXACT row order that
    LUNA's bronze CSV has. The resulting CSV is bitwise identical to
    bronze (modulo extra metadata columns bronze carries but scGG
    drops). Direct diffable.

  * ``--scgg_natural_order``: cells are written in scGG's natural
    processing order (sections in filename-alphabetic order, within
    each section in the h5ad's obs order = bronze per-section row
    order). Useful for understanding what scGG's data pipeline
    actually feeds to the model batch by batch.

What this script does NOT apply (kept consistent with bronze CSV):
  * scGG's optional ``normalize_total + log1p`` (off by default).
  * scGG's optional ``scale`` (off by default).
  * scGG's coord normalization (``per_section_minmax``) — coords are
    written in raw microns matching bronze.

Usage:
    python dump_scgg_input_csv.py \\
        --silver_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --split train \\
        --out_csv /tmp/scgg_train_dump.csv

    # Then diff against bronze:
    python /nfs/team361/sb75/scgg/scripts/diff_luna_input_tensors.py \\
        --csv_a /nfs/team361/sb75/DATASETS/bronze/mmc_luna/MERFISH_mouse_cortex_train.csv \\
        --csv_b /tmp/scgg_train_dump.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger("dump_scgg_input_csv")


# Silver h5ad filename pattern, matching the silver convention.
_SLICE_RE = re.compile(
    r"^(?:mmc|merfish_mouse_cortex)_mouse(?P<mouse>\d+)_slice(?P<slice>\d+)\.h5ad$"
)


def _enumerate_silver_files(
    silver_dir: Path, mouse_ids: Tuple[int, ...]
) -> List[Tuple[int, int, Path]]:
    """Sorted by filename (matches scGG's `_enumerate_slice_files`)."""
    out: List[Tuple[int, int, Path]] = []
    for p in sorted(silver_dir.iterdir()):
        m = _SLICE_RE.match(p.name)
        if not m:
            continue
        mouse = int(m["mouse"])
        if mouse in mouse_ids:
            out.append((mouse, int(m["slice"]), p))
    return out


def _build_section_df(
    path: Path,
    mouse: int,
    slice_id: int,
    gene_names_ref: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Read one silver h5ad and return a DataFrame in bronze CSV format.

    Columns: gene_1..gene_N, coord_X, coord_Y, cell_section, cell_class,
    _bronze_row_pos (carried through for optional sorting).
    Index: cell_id (the original bronze CSV index).
    """
    import anndata as ad
    import scipy.sparse as sp

    adata = ad.read_h5ad(path)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    gene_names = list(adata.var_names)
    if gene_names_ref is not None and gene_names != gene_names_ref:
        raise ValueError(
            f"Gene panel mismatch in {path.name}: expected "
            f"{len(gene_names_ref)} genes, got {len(gene_names)}"
        )

    section_label = f"mouse{mouse}_slice{slice_id}"

    # Coords from obsm['spatial'] (raw microns) — fall back to obs columns.
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2]
    else:
        xy = np.column_stack([
            adata.obs["coord_X"].to_numpy(dtype=np.float64),
            adata.obs["coord_Y"].to_numpy(dtype=np.float64),
        ])

    cell_class = (
        adata.obs["cell_class"].astype(str).to_numpy()
        if "cell_class" in adata.obs.columns
        else np.full(adata.n_obs, "unknown")
    )

    # Original bronze cell_id (preserved by build_h5ad_from_luna_csv).
    if "cell_id" in adata.obs.columns:
        cell_ids = adata.obs["cell_id"].to_numpy()
    else:
        cell_ids = np.arange(adata.n_obs)
        logger.warning(
            f"  {path.name}: no `cell_id` in obs; using per-slice 0..N-1"
        )

    # Bronze row position (preserved by the latest build_h5ad_from_luna_csv).
    if "_bronze_row_pos" in adata.obs.columns:
        bronze_pos = adata.obs["_bronze_row_pos"].to_numpy()
    else:
        bronze_pos = None  # caller will warn and skip the match_bronze_order sort

    df = pd.DataFrame(X, columns=gene_names)
    df["coord_X"] = xy[:, 0]
    df["coord_Y"] = xy[:, 1]
    df["cell_section"] = section_label
    df["cell_class"] = cell_class
    df.index = cell_ids
    df.index.name = "cell_id"
    if bronze_pos is not None:
        df["_bronze_row_pos"] = bronze_pos

    return df, gene_names


def dump_split(
    silver_dir: Path,
    out_csv: Path,
    mouse_ids: Tuple[int, ...],
    match_bronze_order: bool,
) -> int:
    files = _enumerate_silver_files(silver_dir, mouse_ids)
    if not files:
        raise FileNotFoundError(
            f"No silver h5ads with mouse_ids={mouse_ids} found under {silver_dir}"
        )
    logger.info(f"Discovered {len(files)} h5ads in {silver_dir}")

    gene_names_ref: Optional[List[str]] = None
    section_dfs: List[pd.DataFrame] = []
    for mouse, slice_id, path in files:
        df, gene_names = _build_section_df(path, mouse, slice_id, gene_names_ref)
        if gene_names_ref is None:
            gene_names_ref = gene_names
        logger.info(
            f"  {path.name:<60s}  n_cells={len(df):>6d}  n_genes={len(gene_names)}"
        )
        section_dfs.append(df)

    big = pd.concat(section_dfs, axis=0)

    if match_bronze_order:
        if "_bronze_row_pos" not in big.columns:
            logger.error(
                "No _bronze_row_pos in any h5ad — cannot match bronze row "
                "order. Rebuild silver with the latest "
                "`build_h5ad_from_luna_csv.py` (it stamps _bronze_row_pos "
                "by default). Or re-run with --scgg_natural_order to skip "
                "the bronze-order sort."
            )
            return 2
        big = big.sort_values("_bronze_row_pos", kind="stable")
        logger.info("Sorted by _bronze_row_pos (matches bronze CSV row order exactly)")
    else:
        logger.info(
            "Writing in scGG's natural order: sections in filename-alphabetic "
            "order, within each section in h5ad obs order (= bronze per-section "
            "row order). This is the order scGG's dataset iterates."
        )

    # Drop the helper column before writing.
    if "_bronze_row_pos" in big.columns:
        big = big.drop(columns=["_bronze_row_pos"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # `float_format="%.17g"` gives lossless float64 round-trip in the
    # CSV text — same setting `_build_luna_csv` uses.
    big.to_csv(out_csv, header=True, float_format="%.17g")
    logger.info(
        f"Wrote {len(big):,} cells × {len(gene_names_ref)} genes → {out_csv}"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--silver_dir", required=True,
        help="Directory with per-section silver h5ads "
             "(e.g. /nfs/team361/sb75/DATASETS/silver/mmc_luna).",
    )
    p.add_argument(
        "--out_csv", required=True,
        help="Output CSV path.",
    )
    p.add_argument(
        "--split", choices=("train", "test", "all"), default="train",
        help="Which split to dump. train=Mouse 1, test=Mouse 2, "
             "all=both. Default train.",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--match_bronze_order", action="store_true", default=True,
        help="Sort cells by _bronze_row_pos to replay bronze CSV row "
             "order exactly. Produces a bit-identical comparison "
             "target for bronze. Default ON.",
    )
    group.add_argument(
        "--scgg_natural_order", action="store_true",
        help="Write cells in scGG's natural processing order (sections "
             "filename-alphabetic, within-section h5ad obs order). "
             "Useful for inspecting what scGG's dataset iterates.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    # Resolve mouse ids from split.
    if args.split == "train":
        mouse_ids = (1,)
    elif args.split == "test":
        mouse_ids = (2,)
    else:
        mouse_ids = (1, 2)

    # When --scgg_natural_order is set, override the default.
    match_bronze_order = not args.scgg_natural_order

    return dump_split(
        silver_dir=Path(args.silver_dir),
        out_csv=Path(args.out_csv),
        mouse_ids=mouse_ids,
        match_bronze_order=match_bronze_order,
    )


if __name__ == "__main__":
    sys.exit(main())

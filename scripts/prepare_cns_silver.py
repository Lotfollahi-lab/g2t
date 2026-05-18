#!/usr/bin/env python
"""
Assemble the Shi 2023 / STARmap-integrated mouse CNS scRNA-seq atlas:
join per-well bronze h5ads with the shared metadata.csv and write to silver.

Input  : /nfs/team361/sb75/DATASETS/bronze/cns_luna
            imputation_well03.h5ad ... imputation_well11.h5ad
            imputation_well1_5.h5ad ... imputation_well10_5.h5ad
            metadata.csv          (Single Cell Portal format: row 1 is the
                                   TYPE annotation row [group/numeric])

Output : /nfs/team361/sb75/DATASETS/silver/cns_luna
            cns_scrna_well03.h5ad
            cns_scrna_well1_5.h5ad
            ...

Each silver h5ad contains:
  X                       : whatever expression was in the bronze file
                            (typically imputed counts to the STARmap panel)
  obsm['spatial']         : imputed XY coordinates — this is the
                            *pseudo-ground-truth* derived from the STARmap
                            integration that LUNA validates against
  obs['cell_class']       : Main_molecular_cell_type
                            (matches LUNA's 27-class label space)
  obs['subclass']         : Sub_molecular_cell_type
  obs['tissue_region']    : Main_molecular_tissue_region
  obs['tissue_region_sub']: Sub_molecular_tissue_region
  obs['cell_class_region']: Molecular_spatial_cell_type
                            (joint class+region label, useful for diagnostics)
  obs['well_id']          : derived from filename (e.g. '03', '1_5')
  obs['meta_<col>']       : everything else from metadata.csv, prefixed
  uns['well_id']          : same as obs['well_id'] (single value)
  uns['source']           : 'Shi_2023_STARmap_imputed_scRNA'

Cells in the bronze h5ad that don't appear in metadata.csv are dropped with
a count; the script also reports cells in metadata.csv that don't appear in
any bronze h5ad. Both should be near-zero on a clean download.

Usage:
    python scripts/prepare_cns_silver.py \\
        --bronze_dir /nfs/team361/sb75/DATASETS/bronze/cns_luna \\
        --silver_dir /nfs/team361/sb75/DATASETS/silver/cns_luna
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("scgg.prepare_cns_silver")


# Filename pattern: imputation_well03.h5ad, imputation_well1_5.h5ad, ...
WELL_RE = re.compile(r"^imputation_well(?P<well>[\w]+)\.h5ad$")


# Mapping from metadata.csv columns -> standardized obs column names used
# throughout the scgg codebase (so the same downstream loaders work).
LABEL_RENAME = {
    "Main_molecular_cell_type": "cell_class",
    "Sub_molecular_cell_type": "subclass",
    "Main_molecular_tissue_region": "tissue_region",
    "Sub_molecular_tissue_region": "tissue_region_sub",
    "Molecular_spatial_cell_type": "cell_class_region",
}


def _load_metadata(path: Path) -> pd.DataFrame:
    """Read a Single Cell Portal–style metadata.csv.

    Row 0 is the header; row 1 is the TYPE row (group/numeric) which we
    must skip. The first column is `NAME` (cell barcode) and becomes the
    index.
    """
    logger.info(f"Loading metadata: {path}")
    # Read everything as object first; we'll coerce numeric columns afterwards.
    raw = pd.read_csv(path, dtype=object)
    if "NAME" not in raw.columns:
        raise ValueError(
            f"metadata.csv must have a NAME column. Got: {list(raw.columns)}"
        )

    # The first non-header row is the TYPE row in SCP format.
    if str(raw.iloc[0]["NAME"]).strip().upper() == "TYPE":
        type_row = raw.iloc[0].to_dict()
        meta = raw.iloc[1:].copy()
    else:
        # If the TYPE row isn't there (e.g., user pre-stripped it), just
        # proceed without it.
        type_row = {}
        meta = raw

    meta = meta.set_index("NAME")

    # Coerce the columns flagged "numeric" in the TYPE row.
    for col, dtype_marker in type_row.items():
        if col == "NAME":
            continue
        if dtype_marker == "numeric" and col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")

    logger.info(f"  cells in metadata: {len(meta):,}")
    logger.info(f"  columns: {list(meta.columns)}")
    return meta


def _parse_well_id(filename: str) -> Optional[str]:
    m = WELL_RE.match(filename)
    return m["well"] if m else None


def _attach_spatial(adata, well_id: str) -> bool:
    """Find spatial coordinates in the AnnData and put them in obsm['spatial'].

    Returns True if spatial coords were attached, False if none were found.
    """
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]
        adata.obsm["spatial"] = xy
        return True
    # Common obs-column variants
    for x_col, y_col in [
        ("coord_X", "coord_Y"),
        ("x", "y"),
        ("spatial_x", "spatial_y"),
        ("X", "Y"),
    ]:
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            adata.obsm["spatial"] = np.column_stack([
                adata.obs[x_col].to_numpy(dtype=np.float32),
                adata.obs[y_col].to_numpy(dtype=np.float32),
            ])
            logger.info(
                f"    built obsm['spatial'] from obs columns {x_col!r}/{y_col!r}"
            )
            return True
    logger.warning(f"    well {well_id}: NO spatial coordinates found")
    return False


def _attach_metadata(adata, meta_sub: pd.DataFrame) -> Dict[str, int]:
    """Copy metadata columns onto adata.obs with the standard rename map."""
    n_renamed = 0
    n_preserved = 0
    for src, dst in LABEL_RENAME.items():
        if src in meta_sub.columns:
            adata.obs[dst] = meta_sub[src].astype(str).values
            n_renamed += 1
    # Anything else gets a `meta_` prefix so we don't clobber obs.
    for col in meta_sub.columns:
        if col in LABEL_RENAME:
            continue
        target = f"meta_{col}" if col in adata.obs.columns else col
        if target not in adata.obs.columns:
            adata.obs[target] = meta_sub[col].values
            n_preserved += 1
    return {"n_renamed": n_renamed, "n_preserved": n_preserved}


def assemble(
    bronze_dir: Path,
    silver_dir: Path,
    overwrite: bool,
    min_cells_per_well: int,
) -> None:
    import anndata as ad

    silver_dir.mkdir(parents=True, exist_ok=True)
    metadata = _load_metadata(bronze_dir / "metadata.csv")

    bronze_h5ads = sorted(bronze_dir.glob("imputation_well*.h5ad"))
    if not bronze_h5ads:
        raise FileNotFoundError(
            f"No imputation_well*.h5ad files under {bronze_dir}. "
            "Did the download finish?"
        )
    logger.info(f"Found {len(bronze_h5ads)} bronze h5ad files")

    seen_cells: set = set()
    summary = []
    for h5ad_path in bronze_h5ads:
        well_id = _parse_well_id(h5ad_path.name)
        if well_id is None:
            logger.warning(f"  unrecognized filename, skipping: {h5ad_path.name}")
            continue
        out_path = silver_dir / f"cns_scrna_well{well_id}.h5ad"
        if out_path.exists() and not overwrite:
            logger.info(f"  exists, skipping: {out_path.name}")
            # Still count it toward the summary
            seen_cells.update([])  # no incremental update on skip; print warning
            continue

        logger.info(f"[well {well_id}] loading {h5ad_path.name}")
        adata = ad.read_h5ad(h5ad_path)
        logger.info(f"  bronze AnnData: {adata.shape}")

        # Join with metadata
        adata_cells = pd.Index(adata.obs_names)
        common = adata_cells.intersection(metadata.index)
        n_common = len(common)
        n_dropped = adata.n_obs - n_common
        logger.info(
            f"  metadata-matched cells: {n_common:,} / {adata.n_obs:,}"
            + (f"  ({n_dropped:,} dropped)" if n_dropped > 0 else "")
        )
        if n_common == 0:
            logger.warning(
                f"  no cells match metadata for well {well_id}; skipping output"
            )
            continue
        if n_common < min_cells_per_well:
            logger.warning(
                f"  only {n_common} matched cells in well {well_id} "
                f"(< {min_cells_per_well}); skipping output"
            )
            continue

        adata = adata[common.values].copy()
        meta_sub = metadata.loc[common]

        attach_stats = _attach_metadata(adata, meta_sub)
        logger.info(
            f"  attached {attach_stats['n_renamed']} renamed + "
            f"{attach_stats['n_preserved']} preserved metadata columns"
        )

        has_spatial = _attach_spatial(adata, well_id)

        adata.obs["well_id"] = str(well_id)
        adata.uns["well_id"] = str(well_id)
        adata.uns["source"] = "Shi_2023_STARmap_imputed_scRNA"

        adata.write(out_path)
        logger.info(f"  wrote {adata.n_obs:,} cells -> {out_path.name}")

        seen_cells.update(common.tolist())
        summary.append({
            "well_id": well_id,
            "n_cells": adata.n_obs,
            "n_dropped": int(n_dropped),
            "has_spatial": has_spatial,
        })

    # Final summary
    logger.info("=" * 60)
    logger.info("Silver build complete")
    total_cells = sum(s["n_cells"] for s in summary)
    logger.info(f"  wells written: {len(summary)}")
    logger.info(f"  cells written: {total_cells:,}")
    n_meta_unused = len(metadata.index.difference(pd.Index(list(seen_cells))))
    if n_meta_unused > 0:
        logger.warning(
            f"  {n_meta_unused:,} cells in metadata.csv had no matching bronze h5ad "
            "(this can be normal if the metadata includes unused samples)"
        )
    if any(not s["has_spatial"] for s in summary):
        bad = [s["well_id"] for s in summary if not s["has_spatial"]]
        logger.warning(f"  wells with NO spatial coords: {bad}")
    logger.info("Per-well counts:")
    for s in summary:
        logger.info(f"  {s['well_id']:>6}: {s['n_cells']:>8,} cells   "
                     f"(spatial: {'YES' if s['has_spatial'] else 'NO'})")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bronze_dir",
        default="/nfs/team361/sb75/DATASETS/bronze/cns_luna",
    )
    p.add_argument(
        "--silver_dir",
        default="/nfs/team361/sb75/DATASETS/silver/cns_luna",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-write per-well h5ads even if they already exist.",
    )
    p.add_argument(
        "--min_cells_per_well", type=int, default=100,
        help="Skip wells with fewer matched cells (default 100).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    assemble(
        bronze_dir=Path(args.bronze_dir),
        silver_dir=Path(args.silver_dir),
        overwrite=args.overwrite,
        min_cells_per_well=args.min_cells_per_well,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

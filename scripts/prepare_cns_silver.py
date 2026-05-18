#!/usr/bin/env python
"""
Assemble the Shi 2023 / STARmap-integrated mouse CNS scRNA-seq atlas:
join each per-well bronze h5ad with its matching `well<id>_spatial.csv`
(the imputed coordinates + cell labels) and optionally with the global
`metadata.csv` (sample-level info), then write a silver h5ad per well.

Input  : /nfs/team361/sb75/DATASETS/bronze/cns_luna/
            imputation_well03.h5ad ... imputation_well11.h5ad
            imputation_well1_5.h5ad ... imputation_well10_5.h5ad
            well03_spatial.csv   ... well11_spatial.csv
            well1_5_spatial.csv  ... well10_5_spatial.csv
            metadata.csv          (sample-level metadata; OPTIONAL)

Output : /nfs/team361/sb75/DATASETS/silver/cns_luna/
            cns_scrna_well03.h5ad ... cns_scrna_well1_5.h5ad

Each silver h5ad contains:
  X                          : gene expression from the bronze h5ad
  obsm['spatial']            : (cells, 2) imputed XY coordinates from
                               well<id>_spatial.csv — the pseudo-GT for
                               LUNA's Section 3.3 evaluation
  obs['coord_Z']             : Z coordinate (per-section depth)
  obs['cell_class']          : Main_molecular_cell_type
  obs['subclass']            : Sub_molecular_cell_type
  obs['tissue_region']       : Main_molecular_tissue_region
  obs['tissue_region_sub']   : Sub_molecular_tissue_region
  obs['cell_class_region']   : Molecular_spatial_cell_type
  obs['well_id']             : derived from filename
  obs['meta_<col>']          : sample-level columns from metadata.csv if
                               its NAMEs overlap with this well's cells
                               (best-effort — skipped silently if not)
  uns['well_id']             : single value (same as obs['well_id'])
  uns['source']              : 'Shi_2023_STARmap_imputed_scRNA'

Format notes:
  * metadata.csv is comma-separated; spatial CSVs are tab-separated.
  * Both follow the Single Cell Portal convention: row 1 is a TYPE
    annotation row (group/numeric) — we skip it and coerce numeric
    columns accordingly.
  * The per-well spatial CSV's NAME column (e.g. 'well03_0') is the
    canonical join key against the h5ad's obs_names.

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


# Map metadata column names (in spatial CSV / master CSV) -> standardized
# obs column names that the rest of scgg already understands.
LABEL_RENAME = {
    "Main_molecular_cell_type": "cell_class",
    "Sub_molecular_cell_type": "subclass",
    "Main_molecular_tissue_region": "tissue_region",
    "Sub_molecular_tissue_region": "tissue_region_sub",
    "Molecular_spatial_cell_type": "cell_class_region",
}


# ---------------------------------------------------------------------------
# CSV readers (SCP format with a TYPE annotation row)
# ---------------------------------------------------------------------------


def _read_scp_csv(path: Path, sep: str) -> pd.DataFrame:
    """Read a Single Cell Portal–style CSV/TSV:
      row 0 = header
      row 1 = TYPE annotation row (group / numeric)
      rows 2+ = data
    Returns a DataFrame indexed by NAME, with numeric columns coerced.
    """
    raw = pd.read_csv(path, sep=sep, dtype=object)
    if "NAME" not in raw.columns:
        raise ValueError(
            f"{path.name}: missing required NAME column. Got: {list(raw.columns)}"
        )

    if str(raw.iloc[0]["NAME"]).strip().upper() == "TYPE":
        type_row = raw.iloc[0].to_dict()
        df = raw.iloc[1:].copy()
    else:
        type_row = {}
        df = raw.copy()

    df = df.set_index("NAME")

    # Coerce numeric columns based on the TYPE row.
    for col, marker in type_row.items():
        if col == "NAME":
            continue
        if marker == "numeric" and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _load_master_metadata(path: Path) -> Optional[pd.DataFrame]:
    """Load the optional global metadata.csv (comma-separated, SCP format)."""
    if not path.exists():
        logger.info("  master metadata.csv not present; skipping")
        return None
    logger.info(f"  loading master metadata: {path}")
    meta = _read_scp_csv(path, sep=",")
    logger.info(
        f"    {len(meta):,} rows, columns: {list(meta.columns)[:8]}"
        f"{' ...' if len(meta.columns) > 8 else ''}"
    )
    return meta


def _load_well_spatial(path: Path) -> pd.DataFrame:
    """Load a per-well spatial CSV (tab-separated, SCP format)."""
    return _read_scp_csv(path, sep="\t")


# ---------------------------------------------------------------------------
# Per-well processing
# ---------------------------------------------------------------------------


def _parse_well_id(filename: str) -> Optional[str]:
    m = WELL_RE.match(filename)
    return m["well"] if m else None


def _attach_spatial(adata, spatial_sub: pd.DataFrame, well_id: str) -> bool:
    """Set obsm['spatial'] = (N, 2) float32 XY, obs['coord_Z'] if available."""
    if "X" not in spatial_sub.columns or "Y" not in spatial_sub.columns:
        logger.warning(
            f"  well {well_id}: spatial CSV missing X/Y columns "
            f"(present: {list(spatial_sub.columns)}); cannot attach coords"
        )
        return False

    xy = np.column_stack([
        spatial_sub["X"].astype(np.float32).to_numpy(),
        spatial_sub["Y"].astype(np.float32).to_numpy(),
    ])
    finite = np.isfinite(xy).all(axis=1)
    if (~finite).any():
        logger.warning(
            f"  well {well_id}: {(~finite).sum():,} cells have non-finite coords"
        )
    adata.obsm["spatial"] = xy
    adata.obs["coord_X"] = xy[:, 0]
    adata.obs["coord_Y"] = xy[:, 1]
    if "Z" in spatial_sub.columns:
        adata.obs["coord_Z"] = spatial_sub["Z"].astype(np.float32).to_numpy()
    return True


def _attach_labels(adata, spatial_sub: pd.DataFrame) -> Dict[str, int]:
    """Rename cell-label columns from the per-well CSV onto adata.obs."""
    n_renamed = 0
    n_preserved = 0
    for src, dst in LABEL_RENAME.items():
        if src in spatial_sub.columns:
            adata.obs[dst] = spatial_sub[src].astype(str).to_numpy()
            n_renamed += 1
    # Preserve any extra columns under a `meta_` prefix to avoid clobbering.
    for col in spatial_sub.columns:
        if col in LABEL_RENAME or col in ("X", "Y", "Z"):
            continue
        target = col if col not in adata.obs.columns else f"meta_{col}"
        if target not in adata.obs.columns:
            adata.obs[target] = spatial_sub[col].to_numpy()
            n_preserved += 1
    return {"n_renamed": n_renamed, "n_preserved": n_preserved}


def _maybe_attach_master_metadata(
    adata,
    master_meta: Optional[pd.DataFrame],
    well_id: str,
) -> int:
    """Best-effort join with global metadata.csv. Returns 0 if no overlap."""
    if master_meta is None:
        return 0
    overlap = adata.obs_names.intersection(master_meta.index)
    if len(overlap) == 0:
        logger.info(
            f"  well {well_id}: master metadata.csv NAMEs do not overlap "
            "with this well's cells; sample-level columns not attached"
        )
        return 0
    logger.info(
        f"  well {well_id}: master metadata.csv overlaps with "
        f"{len(overlap):,} / {adata.n_obs:,} cells; attaching meta_* columns"
    )
    # Map per-cell via obs_names (handles partial overlap with NaN).
    cells = adata.obs_names.to_series()
    n_attached = 0
    for col in master_meta.columns:
        # Skip cell-level label columns we already filled from the spatial CSV.
        if col in LABEL_RENAME:
            continue
        target = f"meta_{col}" if col in adata.obs.columns else col
        if target in adata.obs.columns:
            continue
        adata.obs[target] = cells.map(master_meta[col]).to_numpy()
        n_attached += 1
    return n_attached


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------


def assemble(
    bronze_dir: Path,
    silver_dir: Path,
    overwrite: bool,
    min_cells_per_well: int,
) -> None:
    import anndata as ad

    silver_dir.mkdir(parents=True, exist_ok=True)

    master_meta = _load_master_metadata(bronze_dir / "metadata.csv")

    bronze_h5ads = sorted(bronze_dir.glob("imputation_well*.h5ad"))
    if not bronze_h5ads:
        raise FileNotFoundError(
            f"No imputation_well*.h5ad files under {bronze_dir}."
        )
    logger.info(f"Found {len(bronze_h5ads)} bronze h5ad files")

    summary = []
    for h5ad_path in bronze_h5ads:
        well_id = _parse_well_id(h5ad_path.name)
        if well_id is None:
            logger.warning(f"  unrecognized filename, skipping: {h5ad_path.name}")
            continue

        spatial_csv_path = bronze_dir / f"well{well_id}_spatial.csv"
        out_path = silver_dir / f"cns_scrna_well{well_id}.h5ad"

        logger.info(f"[well {well_id}] {h5ad_path.name}")

        if out_path.exists() and not overwrite:
            logger.info(f"  exists, skipping: {out_path.name}")
            continue
        if not spatial_csv_path.exists():
            logger.warning(
                f"  no matching spatial CSV ({spatial_csv_path.name}); skipping"
            )
            continue

        # Load h5ad + spatial CSV
        adata = ad.read_h5ad(h5ad_path)
        logger.info(f"  bronze AnnData: {adata.shape}")
        spatial = _load_well_spatial(spatial_csv_path)
        logger.info(
            f"  spatial CSV: {len(spatial):,} rows, columns: {list(spatial.columns)}"
        )

        # Join on NAME
        common = adata.obs_names.intersection(spatial.index)
        n_common = len(common)
        n_dropped = adata.n_obs - n_common
        logger.info(
            f"  matched cells: {n_common:,} / {adata.n_obs:,}"
            + (f" ({n_dropped:,} dropped)" if n_dropped > 0 else "")
        )
        if n_common < min_cells_per_well:
            logger.warning(
                f"  only {n_common} matched cells (< {min_cells_per_well}); skipping"
            )
            continue

        adata = adata[common.values].copy()
        spatial_sub = spatial.loc[common]

        has_spatial = _attach_spatial(adata, spatial_sub, well_id)
        attach_stats = _attach_labels(adata, spatial_sub)
        n_master_cols = _maybe_attach_master_metadata(adata, master_meta, well_id)

        adata.obs["well_id"] = str(well_id)
        adata.uns["well_id"] = str(well_id)
        adata.uns["source"] = "Shi_2023_STARmap_imputed_scRNA"

        adata.write(out_path)
        logger.info(
            f"  wrote {adata.n_obs:,} cells -> {out_path.name}  "
            f"(spatial={'YES' if has_spatial else 'NO'}, "
            f"labels={attach_stats['n_renamed']}+{attach_stats['n_preserved']}, "
            f"master_cols={n_master_cols})"
        )

        summary.append({
            "well_id": well_id,
            "n_cells": adata.n_obs,
            "n_dropped": int(n_dropped),
            "has_spatial": has_spatial,
            "master_cols": n_master_cols,
        })

    # Final summary
    logger.info("=" * 60)
    logger.info("Silver build complete")
    total_cells = sum(s["n_cells"] for s in summary)
    logger.info(f"  wells written : {len(summary)}")
    logger.info(f"  cells written : {total_cells:,}")
    if any(not s["has_spatial"] for s in summary):
        bad = [s["well_id"] for s in summary if not s["has_spatial"]]
        logger.warning(f"  wells WITHOUT spatial coords: {bad}")
    logger.info("Per-well counts:")
    for s in summary:
        logger.info(
            f"  {s['well_id']:>6}: {s['n_cells']:>8,} cells   "
            f"(spatial: {'YES' if s['has_spatial'] else 'NO'}, "
            f"master_cols: {s['master_cols']})"
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bronze_dir", default="/nfs/team361/sb75/DATASETS/bronze/cns_luna",
    )
    p.add_argument(
        "--silver_dir", default="/nfs/team361/sb75/DATASETS/silver/cns_luna",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--min_cells_per_well", type=int, default=100)
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

#!/usr/bin/env python
"""Download + process the STARmap-PLUS mouse CNS atlas (Shi et al. 2023,
Zenodo 10.5281/zenodo.8327576) RAW counts into silver h5ads — the TARGET
domain for the cross-platform (MERFISH ABC -> STARmap) domain-adaptation
benchmark.

One script, two steps:

  STEP 1 — download (-> <bronze>):
    Per region, pull the three small CSVs from Zenodo:
      {region}raw_expression_pd.csv   GENES x CELLS, raw integer counts
      {region}_spatial.csv            SCP-style: NAME + X/Y[/Z] coords
      {region}_spot_meta.csv          SCP-style: NAME + per-cell labels
    plus the shared metadata.csv + cluster.csv. The big
    processed_expression_pd.csv files (normalised, ~1-4 GB each) are
    intentionally skipped — we only need raw counts.

  STEP 2 — prepare (-> <silver>):
    Per region, read the CSVs -> AnnData and write <region>_test.h5ad:
      X               raw counts (cells x genes), CSR int32 + layers['counts']
      var_names       gene SYMBOLS (STARmap ~1022-gene panel)
      obsm['spatial'] (N, 2) float32 XY (real coords; eval reference)
      obs['coord_Z']  if a Z column is present
      obs[...]        every _spot_meta column, plus obs['cell_class']
      obs['cell_section'] = region ; uns['source'] = 'Shi_2023_STARmapPLUS_raw'

Defaults write to:
      bronze = /nfs/team361/sb75/DATASETS/bronze/cns_luna_raw
      silver = /nfs/team361/sb75/DATASETS/silver/cns_luna_raw

This is the TARGET (STARmap) side. The SOURCE (MERFISH ABC) is a separate
S3 download:
      python scripts/download_abc_zhuang_abca1.py --dest .../bronze/abc_luna_raw
      python scripts/prepare_abc_silver.py \
          --bronze_dir .../bronze/abc_luna_raw \
          --silver_dir /nfs/team361/sb75/DATASETS/silver/cns_luna_raw
  Pointing prepare_abc_silver at the same silver dir colocates the source
  *_train h5ads with these target *_test h5ads, mirroring the harmony
  cns_luna layout.

Usage:
    python scripts/download_and_prepare_cns_raw.py            # download + prepare
    python scripts/download_and_prepare_cns_raw.py --skip_download   # prepare only
    python scripts/download_and_prepare_cns_raw.py --regions well03,sagittal1
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger("download_and_prepare_cns_raw")

ZENODO_RECORD = "8327576"
BRONZE_DEFAULT = "/nfs/team361/sb75/DATASETS/bronze/cns_luna_raw"
SILVER_DEFAULT = "/nfs/team361/sb75/DATASETS/silver/cns_luna_raw"

# Regions in the deposit (== the harmony cns_luna target test sections).
DEFAULT_REGIONS = [
    "well01OB", "well01brain", "well03", "well04", "well05", "well06",
    "well07", "well08", "well09", "well10", "well10_5", "well11",
    "well1_5", "well2_5", "well3_5", "well7_5",
    "sagittal1", "sagittal2", "sagittal3", "spinalcord",
]
SHARED_FILES = ["metadata.csv", "cluster.csv"]

_CELL_CLASS_CANDIDATES = [
    "cell_class", "Main_molecular_cell_type", "Molecular_cell_type",
    "cell_type", "CellType", "subclass", "Sub_molecular_cell_type",
]


def _region_files(region: str) -> List[str]:
    # NB: raw file has NO underscore before 'raw'; spatial/meta DO.
    return [
        f"{region}raw_expression_pd.csv",
        f"{region}_spatial.csv",
        f"{region}_spot_meta.csv",
    ]


# ---------------------------------------------------------------------------
# Step 1 — download
# ---------------------------------------------------------------------------
def _download_file(filename: str, bronze: Path, overwrite: bool,
                   retries: int = 3) -> bool:
    """Stream one Zenodo file to <bronze>/<filename>. Skips if already
    present (non-empty) unless ``overwrite``. Atomic via a .part temp."""
    dest = bronze / filename
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        logger.info(f"    skip (exists): {filename}")
        return True
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{filename}/content"
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "scgg-dl"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
                total = int(resp.headers.get("Content-Length", 0) or 0)
                done = 0
                next_log = 0.10
                while True:
                    buf = resp.read(1 << 20)  # 1 MB
                    if not buf:
                        break
                    fh.write(buf)
                    done += len(buf)
                    if total and done / total >= next_log:
                        logger.info(f"      {filename}: "
                                    f"{done/1e6:.0f}/{total/1e6:.0f} MB")
                        next_log += 0.25
            tmp.replace(dest)
            logger.info(f"    downloaded: {filename} ({dest.stat().st_size/1e6:.1f} MB)")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, ConnectionError) as e:
            logger.warning(f"    attempt {attempt}/{retries} failed for "
                           f"{filename}: {e}")
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                time.sleep(3 * attempt)
    logger.error(f"    GAVE UP on {filename} after {retries} attempts")
    return False


def download_all(bronze: Path, regions: List[str], overwrite: bool) -> List[str]:
    """Download every region's 3 CSVs + the shared files. Returns the list
    of filenames that FAILED (empty == all good)."""
    bronze.mkdir(parents=True, exist_ok=True)
    failed: List[str] = []
    for fn in SHARED_FILES:
        if not _download_file(fn, bronze, overwrite):
            failed.append(fn)
    for region in regions:
        logger.info(f"  downloading region {region} ...")
        for fn in _region_files(region):
            if not _download_file(fn, bronze, overwrite):
                failed.append(fn)
    return failed


# ---------------------------------------------------------------------------
# Step 2 — prepare (CSV -> h5ad)
# ---------------------------------------------------------------------------
def _sniff_separator(path: Path) -> str:
    with open(path, "r") as fh:
        header = fh.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def _read_scp_csv(path: Path) -> pd.DataFrame:
    """Single-Cell-Portal CSV/TSV: header, optional TYPE row, NAME index."""
    sep = _sniff_separator(path)
    raw = pd.read_csv(path, sep=sep, dtype=object)
    if "NAME" not in raw.columns:
        raise ValueError(
            f"{path.name}: missing NAME column (got {list(raw.columns)[:8]})."
        )
    if str(raw.iloc[0]["NAME"]).strip().upper() == "TYPE":
        type_row = raw.iloc[0].to_dict()
        df = raw.iloc[1:].copy()
    else:
        type_row = {}
        df = raw.copy()
    df = df.set_index("NAME")
    for col, marker in type_row.items():
        if col != "NAME" and marker == "numeric" and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _pick_xy(df: pd.DataFrame) -> tuple:
    """Find (X, Y[, Z]) coordinate columns, tolerating naming variants."""
    lower = {str(c).lower(): c for c in df.columns}
    for xx, yy in (("x", "y"), ("coord_x", "coord_y"),
                   ("spatial_x", "spatial_y"), ("px", "py"),
                   ("x_coord", "y_coord")):
        if xx in lower and yy in lower:
            zc = next((lower[z] for z in ("z", "coord_z", "spatial_z", "z_coord")
                       if z in lower), None)
            return lower[xx], lower[yy], zc
    num = df.apply(pd.to_numeric, errors="coerce")
    numcols = [c for c in df.columns if num[c].notna().mean() > 0.9]
    if len(numcols) >= 2:
        logger.warning(
            "    spatial CSV: no X/Y header match; using first two numeric "
            f"columns {numcols[:2]}"
        )
        return numcols[0], numcols[1], (numcols[2] if len(numcols) > 2 else None)
    raise ValueError(f"no usable XY columns (cols={list(df.columns)[:8]})")


def _build_region(bronze: Path, region: str, min_cells: int):
    """Assemble one region's AnnData; None if missing/too small."""
    import anndata as ad
    import scipy.sparse as sp

    expr_p = bronze / f"{region}raw_expression_pd.csv"
    spat_p = bronze / f"{region}_spatial.csv"
    meta_p = bronze / f"{region}_spot_meta.csv"
    if not expr_p.exists():
        logger.warning(f"  {region}: missing {expr_p.name}; skipping")
        return None

    expr = pd.read_csv(expr_p, index_col=0)               # GENES x CELLS
    genes = [str(g) for g in expr.index]
    cells = [str(c) for c in expr.columns]
    X = expr.to_numpy(dtype=np.float32).T                 # CELLS x GENES
    del expr
    if X.shape[0] < min_cells:
        logger.warning(f"  {region}: {X.shape[0]} cells < {min_cells}; skipping")
        return None

    adata = ad.AnnData(
        X=sp.csr_matrix(X.astype(np.int32)),
        obs=pd.DataFrame(index=pd.Index(cells, name="cell_id")),
        var=pd.DataFrame(index=pd.Index(genes, name="gene_symbol")),
    )
    adata.layers["counts"] = adata.X.copy()

    if spat_p.exists():
        spat = _read_scp_csv(spat_p).reindex([str(i) for i in adata.obs_names])
        try:
            xc, yc, zc = _pick_xy(spat)
            xy = np.column_stack([
                pd.to_numeric(spat[xc], errors="coerce").to_numpy(np.float32),
                pd.to_numeric(spat[yc], errors="coerce").to_numpy(np.float32),
            ])
            adata.obsm["spatial"] = xy
            if zc is not None:
                adata.obs["coord_Z"] = pd.to_numeric(
                    spat[zc], errors="coerce").to_numpy(np.float32)
            n_bad = int((~np.isfinite(xy).all(axis=1)).sum())
            if n_bad:
                logger.warning(f"  {region}: {n_bad} cells non-finite coords")
        except ValueError as e:
            logger.warning(f"  {region}: coord parse failed ({e}); no spatial")
    else:
        logger.warning(f"  {region}: missing {spat_p.name}; no coords")

    if meta_p.exists():
        meta = _read_scp_csv(meta_p).reindex([str(i) for i in adata.obs_names])
        for col in meta.columns:
            adata.obs[str(col)] = meta[col].astype(str).to_numpy()
        for cand in _CELL_CLASS_CANDIDATES:
            if cand in meta.columns:
                adata.obs["cell_class"] = meta[cand].astype(str).to_numpy()
                break
    if "cell_class" not in adata.obs.columns:
        adata.obs["cell_class"] = "unknown"

    adata.obs["cell_section"] = region
    adata.uns["source"] = "Shi_2023_STARmapPLUS_raw"
    adata.uns["region"] = region
    return adata


def prepare_all(bronze: Path, silver: Path, regions: List[str],
                min_cells: int, overwrite: bool) -> int:
    silver.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for region in regions:
        out = silver / f"{region}_test.h5ad"
        if out.exists() and not overwrite:
            logger.info(f"{region}: {out.name} exists; skipping (use --overwrite)")
            n_done += 1
            continue
        logger.info(f"{region}: building h5ad ...")
        adata = _build_region(bronze, region, min_cells)
        if adata is None:
            continue
        adata.write_h5ad(out)
        logger.info(
            f"  -> {out} ({adata.n_obs:,} cells x {adata.n_vars} genes, "
            f"spatial={'yes' if 'spatial' in adata.obsm else 'NO'})"
        )
        n_done += 1
    return n_done


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bronze_dir", default=BRONZE_DEFAULT)
    p.add_argument("--silver_dir", default=SILVER_DEFAULT)
    p.add_argument("--regions", default=None,
                   help="Comma-separated subset; default = all 20 regions.")
    p.add_argument("--skip_download", action="store_true",
                   help="Use already-downloaded CSVs in --bronze_dir.")
    p.add_argument("--skip_prepare", action="store_true",
                   help="Download only; don't build h5ads.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download / rebuild even if outputs exist.")
    p.add_argument("--min_cells", type=int, default=100)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    bronze = Path(args.bronze_dir)
    silver = Path(args.silver_dir)
    regions = (
        [r.strip() for r in args.regions.split(",") if r.strip()]
        if args.regions else DEFAULT_REGIONS
    )
    logger.info(f"Regions ({len(regions)}): {regions}")
    logger.info(f"bronze={bronze}  silver={silver}")

    if not args.skip_download:
        logger.info("== STEP 1: download from Zenodo ==")
        failed = download_all(bronze, regions, args.overwrite)
        if failed:
            logger.error(f"{len(failed)} file(s) failed to download: {failed}")
            logger.error("Re-run (resumes by skipping completed files), or "
                         "check connectivity / the Zenodo record.")
            # Continue to prepare whatever DID download, unless nothing did.
    else:
        logger.info("== STEP 1 skipped (--skip_download) ==")

    if args.skip_prepare:
        logger.info("== STEP 2 skipped (--skip_prepare) ==")
        return 0

    logger.info("== STEP 2: build h5ads ==")
    n_done = prepare_all(bronze, silver, regions, args.min_cells, args.overwrite)
    logger.info(f"Done. {n_done}/{len(regions)} region h5ads in {silver}")
    return 0 if n_done else 1


if __name__ == "__main__":
    sys.exit(main())

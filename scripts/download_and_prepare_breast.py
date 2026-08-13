#!/usr/bin/env python
r"""Download + process the 10x Genomics integrated FFPE human breast cancer
trio (Janesick et al. 2023, Nat Commun, doi:10.1038/s41467-023-43458-x) into
scgg/LUNA silver h5ads for the **adjacent-serial-section** benchmark.

WHY this dataset: one FFPE tissue block (Sample #1) was serially sectioned and
profiled by Xenium (in situ, single-cell, 313-gene panel) on two adjacent 5 um
sections (Rep1, Rep2), and the SAME biopsy's dissociated tumour cells were run
as real single cells via Chromium Fixed RNA Profiling ("scFFPE-seq" / Flex).
This is the only public design that gives a genuinely ADJACENT serial section
of spatial ground truth PLUS a genuinely DISSOCIATED (not imputed) single-cell
dataset of the same tissue. Perfect for testing G2T's real deployment case.

Benchmark built here (all three files share ONE identical gene panel = the
Xenium 313 Gene-Expression genes intersected with the scFFPE transcriptome, so
run_benchmark's identical-column requirement is met with no extra harmonise):

  Tier A (quantitative, real coords both sides):
    xenium_rep1_train.h5ad  ->  xenium_rep2_test.h5ad
    Train Xenium Rep1, test the ADJACENT Xenium Rep2 (coords held out) -> full
    per-cell Spearman / RSSD metric (CeLEry got pairwise-dist Pearson 0.74 here).

  Tier B (the real dissociated test — YOUR core ask):
    xenium_rep1_train.h5ad  ->  scffpe_test.h5ad
    Train on spatial Xenium Rep1, apply to real dissociated scFFPE-seq cells.
    scFFPE has NO ground-truth coordinates, so obsm['spatial'] is a documented
    PLACEHOLDER (uniform scatter) purely so the loader runs; the built-in
    coordinate metric on it is meaningless. Evaluate Tier B EXTERNALLY from the
    predicted coords (run_benchmark writes metadata_pred.csv per section) via
    cell-type -> spatial-domain concordance against the adjacent Xenium/Visium.

One run over the silver dir trains on Xenium Rep1 and evaluates BOTH test files
(Rep2 = the number; scFFPE = predictions for the external Tier-B eval).

Per-file contract (matches download_and_prepare_cns_raw.py / run_benchmark):
    X               raw counts (cells x genes), CSR int32
    layers['counts']  copy of X
    var_names       gene SYMBOLS, IDENTICAL + same order across all files
    obsm['spatial'] (N, 2) float32 XY  (Xenium: real centroids; scFFPE: PLACEHOLDER)
    obs['cell_class']  cell-type annotation (10x GSE243275 xlsx); 'unknown' if unlabelled
    obs['cell_section']  = xenium_rep1 | xenium_rep2 | scffpe
    obs['has_true_coords']  True for Xenium, False for scFFPE
    uns['source']   = 'Janesick2023_10x_breast'

Data sources (all public, no auth):
    Xenium Rep1/Rep2 matrices + centroids : 10x CDN (cf.10xgenomics.com)
    scFFPE-seq (Flex) raw matrix          : GEO GSM7782698
    cell-type annotations (all modalities): GEO GSE243275 xlsx

Usage:
    python scripts/download_and_prepare_breast.py                 # full benchmark
    python scripts/download_and_prepare_breast.py --tier A        # Xenium only
    python scripts/download_and_prepare_breast.py --skip_download  # prepare only

After building, ALWAYS:
    python scripts/validate_scgg_h5ad.py --silver_dir <silver>
"""
from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("download_and_prepare_breast")

BRONZE_DEFAULT = "/nfs/team361/sb75/DATASETS/bronze/breast_janesick"
SILVER_DEFAULT = "/nfs/team361/sb75/DATASETS/silver/breast_janesick"

_XCDN = "https://cf.10xgenomics.com/samples/xenium/1.0.1"
_XENIUM = {
    "rep1": f"{_XCDN}/Xenium_FFPE_Human_Breast_Cancer_Rep1",
    "rep2": f"{_XCDN}/Xenium_FFPE_Human_Breast_Cancer_Rep2",
}
_GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
_SCFFPE_H5 = (f"{_GEO}/samples/GSM7782nnn/GSM7782698/suppl/"
              "GSM7782698_count_raw_feature_bc_matrix.h5")
_XLSX = (f"{_GEO}/series/GSE243nnn/GSE243275/suppl/"
         "GSE243275_Barcode_Cell_Type_Matrices.xlsx")

# xlsx sheet -> (barcode col, label col) per modality.
_SHEET = {
    "xenium_rep1": ("Xenium R1 Fig1-5 (supervised)", "Barcode", "Cluster"),
    "xenium_rep2": ("Xenium R2 Fig1-5 (supervised)", "Barcode", "Cluster"),
    "scffpe":      ("scFFPE-Seq", "Barcode", "Annotation"),
}


# ---------------------------------------------------------------------------
# Step 1 — download
# ---------------------------------------------------------------------------
def _get(url: str, dest: Path, overwrite: bool, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        logger.info(f"    skip (exists): {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "scgg-dl"})
            with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as fh:
                total = int(resp.headers.get("Content-Length", 0) or 0)
                done = 0
                next_log = 0.25
                while True:
                    buf = resp.read(1 << 20)
                    if not buf:
                        break
                    fh.write(buf)
                    done += len(buf)
                    if total and done / total >= next_log:
                        logger.info(f"      {dest.name}: "
                                    f"{done/1e6:.0f}/{total/1e6:.0f} MB")
                        next_log += 0.25
            tmp.replace(dest)
            logger.info(f"    downloaded: {dest.name} "
                        f"({dest.stat().st_size/1e6:.1f} MB)")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, ConnectionError) as e:
            logger.warning(f"    attempt {attempt}/{retries} failed for "
                           f"{dest.name}: {e}")
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                time.sleep(3 * attempt)
    logger.error(f"    GAVE UP on {dest.name}")
    return False


def _is_gzip(path: Path) -> bool:
    """True iff the file starts with the gzip magic (1f 8b). Needed because
    some HTTP layers transparently decompress a Content-Encoding: gzip body,
    leaving a .gz/.tar.gz-named file that is actually plain on disk."""
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _untar(tar_gz: Path, dest_dir: Path):
    """Extract cell_feature_matrix/ from a Xenium tar(.gz) into dest_dir.
    Mode 'r:*' auto-detects gzip vs plain tar (see _is_gzip)."""
    if (dest_dir / "cell_feature_matrix").exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_gz, "r:*") as t:
        t.extractall(dest_dir)


def download_all(bronze: Path, need_scffpe: bool, need_rep2: bool,
                 overwrite: bool) -> List[str]:
    failed = []
    reps = ["rep1"] + (["rep2"] if need_rep2 else [])
    for rep in reps:
        base = _XENIUM[rep]
        stem = base.rsplit("/", 1)[1]
        cfm = bronze / f"{rep}_cell_feature_matrix.tar.gz"
        cells = bronze / f"{rep}_cells.csv.gz"
        if not _get(f"{base}/{stem}_cell_feature_matrix.tar.gz", cfm, overwrite):
            failed.append(f"{rep}_cfm")
        else:
            _untar(cfm, bronze / rep)
        if not _get(f"{base}/{stem}_cells.csv.gz", cells, overwrite):
            failed.append(f"{rep}_cells")
    if not _get(_XLSX, bronze / "cell_type_matrices.xlsx", overwrite):
        failed.append("xlsx")
    if need_scffpe:
        if not _get(_SCFFPE_H5, bronze / "scffpe_raw.h5", overwrite):
            failed.append("scffpe")
    return failed


# ---------------------------------------------------------------------------
# Step 2 — prepare
# ---------------------------------------------------------------------------
def _to_int_csr(X):
    """Return X as a CSR matrix with integer-rounded int32 data, regardless of
    whether it came in dense / CSC / CSR (read_10x_* may return any)."""
    import scipy.sparse as sp
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr().copy()
    X.data = np.rint(X.data).astype(np.int32)
    X.eliminate_zeros()
    return X


def _read_labels(xlsx: Path, section: str) -> Dict[str, str]:
    """barcode(str) -> cell-type label, from the xlsx sheet for `section`."""
    import pandas as pd
    sheet, bcol, lcol = _SHEET[section]
    try:
        df = pd.read_excel(xlsx, sheet_name=sheet)
    except Exception as e:
        logger.warning(f"  {section}: cannot read xlsx sheet {sheet!r} ({e}); "
                       "cell_class will be 'unknown'")
        return {}
    df = df[[bcol, lcol]].dropna()
    return {str(b): str(l) for b, l in zip(df[bcol], df[lcol])}


def _load_xenium(bronze: Path, rep: str, xlsx: Path):
    """Xenium replicate -> AnnData (cells x 313 GEX genes, raw counts, real
    centroid coords, cell types)."""
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp

    mtx_dir = bronze / rep / "cell_feature_matrix"
    a = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", gex_only=True)
    a.var_names_make_unique()
    a.X = _to_int_csr(a.X)

    # coords from cells.csv (join by cell_id == barcode). Auto-detect whether
    # the on-disk .gz is really gzip (some HTTP layers pre-decompress it).
    cells_path = bronze / f"{rep}_cells.csv.gz"
    cells = pd.read_csv(cells_path,
                        compression=("gzip" if _is_gzip(cells_path) else None))
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells = cells.set_index("cell_id").reindex([str(b) for b in a.obs_names])
    xy = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    a.obsm["spatial"] = xy

    labels = _read_labels(xlsx, f"xenium_{rep}")
    a.obs["cell_class"] = [labels.get(str(b), "unknown") for b in a.obs_names]
    a.obs["cell_section"] = f"xenium_{rep}"
    a.obs["has_true_coords"] = True
    return a


def _load_scffpe(bronze: Path, xlsx: Path):
    """scFFPE-seq (Flex) raw matrix -> AnnData restricted to the ANNOTATED
    (called) cells, raw counts, cell types, NO real coords."""
    import scanpy as sc
    import scipy.sparse as sp

    a = sc.read_10x_h5(bronze / "scffpe_raw.h5")
    a.var_names_make_unique()
    labels = _read_labels(xlsx, "scffpe")
    if not labels:
        raise SystemExit("[breast] scFFPE annotations empty; cannot define cells.")
    bc = [str(b) for b in a.obs_names]
    keep = np.array([b in labels for b in bc])
    if keep.sum() < 0.2 * len(bc):   # barcode-suffix mismatch fallback
        strip = {b.split("-")[0]: b for b in bc}
        lab2 = {}
        for k, v in labels.items():
            kk = k.split("-")[0]
            if kk in strip:
                lab2[strip[kk]] = v
        labels = lab2
        keep = np.array([b in labels for b in bc])
    logger.info(f"  scffpe: {int(keep.sum())}/{len(bc)} raw barcodes are "
                "annotated (called) cells")
    a = a[keep].copy()
    a.X = _to_int_csr(a.X)
    a.obs["cell_class"] = [labels.get(str(b), "unknown") for b in a.obs_names]
    a.obs["cell_section"] = "scffpe"
    a.obs["has_true_coords"] = False
    return a


def _finalize(a, panel: List[str], role_name: str, seed: int):
    """Subset+reorder to the canonical panel, add counts layer + placeholder
    coords for coord-less sets, return a clean contract AnnData."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    idx = {g: i for i, g in enumerate(map(str, a.var_names))}
    cols = [idx[g] for g in panel]
    X = a.X.tocsr()[:, cols]
    var = pd.DataFrame(index=pd.Index(panel, name="gene_symbol"))
    obs = a.obs.copy()
    obs.index = pd.Index([str(i) for i in a.obs_names], name="cell_id")
    out = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    out.layers["counts"] = out.X.copy()
    if "spatial" in a.obsm and bool(a.obs["has_true_coords"].iloc[0]):
        out.obsm["spatial"] = np.asarray(a.obsm["spatial"], dtype=np.float32)[:, :2]
    else:
        # PLACEHOLDER coords: deterministic uniform scatter so the loader's
        # per-section coord normalisation is non-degenerate. NOT ground truth
        # — Tier B is scored externally from metadata_pred.csv.
        rng = np.random.default_rng(seed)
        out.obsm["spatial"] = rng.random((out.n_obs, 2), dtype=np.float32)
    out.uns["source"] = "Janesick2023_10x_breast"
    out.uns["section"] = role_name
    return out


def prepare_all(bronze: Path, silver: Path, tier: str, overwrite: bool,
                seed: int) -> int:
    silver.mkdir(parents=True, exist_ok=True)
    xlsx = bronze / "cell_type_matrices.xlsx"
    need_rep2 = tier in ("A", "all")
    need_scffpe = tier in ("B", "all")

    logger.info("  loading Xenium Rep1 (shared trainer) ...")
    rep1 = _load_xenium(bronze, "rep1", xlsx)
    xen_panel = [str(g) for g in rep1.var_names]      # 313 GEX genes
    logger.info(f"  Xenium panel: {len(xen_panel)} genes")

    # Canonical panel = Xenium genes present in every built modality.
    panel = list(xen_panel)
    scffpe = None
    if need_scffpe:
        logger.info("  loading scFFPE-seq (Flex) ...")
        scffpe = _load_scffpe(bronze, xlsx)
        flex_genes = set(map(str, scffpe.var_names))
        panel = [g for g in xen_panel if g in flex_genes]
        dropped = len(xen_panel) - len(panel)
        if dropped:
            logger.info(f"  {dropped} Xenium gene(s) absent from scFFPE; "
                        f"canonical panel = {len(panel)} shared genes")

    built = []
    # train: Xenium Rep1
    built.append(("xenium_rep1_train.h5ad",
                  _finalize(rep1, panel, "xenium_rep1", seed)))
    if need_rep2:
        logger.info("  loading Xenium Rep2 (adjacent section, Tier A test) ...")
        rep2 = _load_xenium(bronze, "rep2", xlsx)
        built.append(("xenium_rep2_test.h5ad",
                      _finalize(rep2, panel, "xenium_rep2", seed)))
    if need_scffpe:
        built.append(("scffpe_test.h5ad",
                      _finalize(scffpe, panel, "scffpe", seed)))

    n = 0
    for fname, adata in built:
        out_path = silver / fname
        if out_path.exists() and not overwrite:
            logger.info(f"  {fname} exists; skipping (use --overwrite)")
            n += 1
            continue
        adata.write_h5ad(out_path)
        logger.info(f"  -> {fname} ({adata.n_obs:,} cells x {adata.n_vars} genes; "
                    f"coords={'real' if bool(adata.obs['has_true_coords'].iloc[0]) else 'PLACEHOLDER'})")
        n += 1
    return n


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bronze_dir", default=BRONZE_DEFAULT)
    p.add_argument("--silver_dir", default=SILVER_DEFAULT)
    p.add_argument("--tier", choices=["A", "B", "all"], default="all",
                   help="A = Xenium Rep1->Rep2 only; B = Xenium Rep1->scFFPE "
                        "only; all = both (default).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the scFFPE placeholder-coordinate scatter.")
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--skip_prepare", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    bronze, silver = Path(args.bronze_dir), Path(args.silver_dir)
    logger.info(f"tier={args.tier}  bronze={bronze}  silver={silver}")

    if not args.skip_download:
        logger.info("== STEP 1: download ==")
        failed = download_all(bronze, need_scffpe=args.tier in ("B", "all"),
                              need_rep2=args.tier in ("A", "all"),
                              overwrite=args.overwrite)
        if failed:
            logger.error(f"downloads failed: {failed}")
            if not args.skip_prepare:
                return 1
    else:
        logger.info("== STEP 1 skipped ==")

    if args.skip_prepare:
        return 0
    logger.info("== STEP 2: build h5ads ==")
    n = prepare_all(bronze, silver, args.tier, args.overwrite, args.seed)
    logger.info(f"Done. {n} h5ad(s) in {silver}")
    logger.info(f"Next: python scripts/validate_scgg_h5ad.py --silver_dir {silver}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())

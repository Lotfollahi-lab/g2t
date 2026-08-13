#!/usr/bin/env python
r"""Download + process the human DLPFC 10x Visium atlas (Maynard et al. 2021,
Nat Neurosci, doi:10.1038/s41593-020-00787-0) into scgg/LUNA silver h5ads.

This builds the **spatial (Visium) side** of the DLPFC benchmark: train on
spatial section(s) and test on a *directly adjacent serial section* whose
ground-truth coordinates are known -> the full per-cell Spearman / RSSD
metric applies (Tier A, quantitative). The real dissociated snRNA-seq side
(Tier B, the domain-gap test) is built by a separate script
(``download_and_prepare_dlpfc_snrna.py``) and then gene-harmonised against
these Visium files.

WHY this source: the 12-sample DLPFC Visium data with the manually annotated
cortical layers (Layer1-6 + WM) is distributed as ready AnnData .h5ad files
on Figshare (article 22004273, "Visium DLPFC preprocessed", derived from
LieberInstitute/spatialLIBD). Each per-sample file already carries RAW UMI
counts in .X, the layer annotation in obs['sce.layer_guess'], and pixel
coordinates in obsm['spatial'] -- so no R / Bioconductor is needed.

The 12 samples are 3 donors x 4 sections. Within a donor the sections are two
directly-adjacent (10 um) serial pairs, the two pairs ~300 um apart:

    donor1: 151507 151508 | 151509 151510
    donor2: 151669 151670 | 151671 151672
    donor3: 151673 151674 | 151675 151676
            \____pair____/   \____pair____/   (each pair = 10 um serial)

Output per section -> ``<sid>_{train,test}.h5ad`` matching the scgg contract
(see ``download_and_prepare_cns_raw.py`` / ``run_scgg_train.run_benchmark``):

    X               raw counts (cells x genes), CSR int32
    layers['counts']  copy of X (seurat_v3 HVG needs raw)
    var_names       gene SYMBOLS (33,538-gene whole-transcriptome panel;
                    IDENTICAL across all 12 samples -> valid train/test panel
                    with no harmonisation needed for Visium-only runs)
    obsm['spatial'] (N, 2) float32 pixel XY (ground-truth; eval reference)
    obs['cell_class']  cortical layer label (Layer1..6 / WM / 'NA'); the
                    per-cell aux label. Unannotated spots -> 'NA'.
    obs['layer']    same as cell_class (explicit name for the layer eval)
    obs['cell_section'] = <sid> ; obs['donor'] = donor id
    uns['source']   = 'Maynard2021_DLPFC_Visium'

Presets (``--preset``) fill --train / --test; any explicit --train/--test
override the preset:

    donor3_holdout  (default) train 151673,151674,151675  test 151676
                    -> test section is the 10 um-adjacent serial cut of a
                       training section (151675); 3 train sections (~11k spots).
                       Matches CeLEry's DLPFC "3-section, same-brain" scenario.
    adjacent_min    train 151673                          test 151674
                    -> minimal directly-adjacent (10 um) pair, one train section.
    cross_donor     train 151673,151674,151675,151676     test 151507
                    -> hardest: test section is a DIFFERENT donor.

Usage:
    python scripts/download_and_prepare_dlpfc.py            # default preset
    python scripts/download_and_prepare_dlpfc.py --preset adjacent_min
    python scripts/download_and_prepare_dlpfc.py --train 151673,151674 --test 151675
    python scripts/download_and_prepare_dlpfc.py --skip_download   # prepare only

After building, ALWAYS validate:
    python scripts/validate_scgg_h5ad.py --silver_dir <silver>
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("download_and_prepare_dlpfc")

BRONZE_DEFAULT = "/nfs/team361/sb75/DATASETS/bronze/dlpfc_visium"
SILVER_DEFAULT = "/nfs/team361/sb75/DATASETS/silver/dlpfc_visium"

# Figshare article 22004273 ("Visium DLPFC preprocessed"): sample_id -> file id.
# Direct download URL = https://ndownloader.figshare.com/files/<file_id>
FIGSHARE: Dict[str, str] = {
    "151507": "39055556", "151508": "39055589",
    "151509": "39055586", "151510": "39055583",
    "151669": "39055580", "151670": "39055577",
    "151671": "39055574", "151672": "39055571",
    "151673": "39055568", "151674": "39055565",
    "151675": "39055562", "151676": "39055559",
}
DONOR: Dict[str, str] = {
    **{s: "donor1" for s in ("151507", "151508", "151509", "151510")},
    **{s: "donor2" for s in ("151669", "151670", "151671", "151672")},
    **{s: "donor3" for s in ("151673", "151674", "151675", "151676")},
}
PRESETS: Dict[str, Tuple[List[str], List[str]]] = {
    "donor3_holdout": (["151673", "151674", "151675"], ["151676"]),
    "adjacent_min":   (["151673"], ["151674"]),
    "cross_donor":    (["151673", "151674", "151675", "151676"], ["151507"]),
}

_LAYER_ORDER = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"]


# ---------------------------------------------------------------------------
# Step 1 — download
# ---------------------------------------------------------------------------
def _download_file(sid: str, bronze: Path, overwrite: bool,
                   retries: int = 3) -> bool:
    """Stream one sample's .h5ad from Figshare to <bronze>/<sid>.h5ad."""
    dest = bronze / f"{sid}.h5ad"
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        logger.info(f"    skip (exists): {dest.name}")
        return True
    url = f"https://ndownloader.figshare.com/files/{FIGSHARE[sid]}"
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "scgg-dl"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as fh:
                total = int(resp.headers.get("Content-Length", 0) or 0)
                done = 0
                next_log = 0.25
                while True:
                    buf = resp.read(1 << 20)  # 1 MB
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
    logger.error(f"    GAVE UP on {dest.name} after {retries} attempts")
    return False


def download_samples(bronze: Path, sids: List[str], overwrite: bool) -> List[str]:
    bronze.mkdir(parents=True, exist_ok=True)
    failed = []
    for sid in sids:
        logger.info(f"  downloading {sid} ...")
        if not _download_file(sid, bronze, overwrite):
            failed.append(sid)
    return failed


# ---------------------------------------------------------------------------
# Step 2 — prepare (Figshare h5ad -> contract h5ad)
# ---------------------------------------------------------------------------
def _select_hvgs(bronze: Path, train: List[str], n_hvg: int,
                 min_cells: int) -> Optional[List[str]]:
    """Top-n_hvg highly variable genes (seurat_v3 on raw counts) computed on
    the TRAIN sections ONLY (no test leakage). Returns a sorted gene-symbol
    list, or None to keep all genes. seurat_v3 needs raw counts, which .X is.
    """
    if not n_hvg or n_hvg <= 0:
        return None
    import anndata as ad
    try:
        import scanpy as sc
    except ImportError:
        logger.error("--n_hvg requires scanpy (pip install scanpy). Use "
                     "--n_hvg 0 to keep all genes instead.")
        raise
    parts = []
    for sid in train:
        src = bronze / f"{sid}.h5ad"
        if not src.exists():
            continue
        a = ad.read_h5ad(src)
        if a.n_obs >= min_cells:
            parts.append(a)
    if not parts:
        logger.warning("HVG: no train sections readable; keeping all genes")
        return None
    big = parts[0] if len(parts) == 1 else ad.concat(parts, axis=0, join="inner")
    n = min(n_hvg, big.n_vars)
    sc.pp.highly_variable_genes(big, n_top_genes=n, flavor="seurat_v3")
    genes = sorted(str(g) for g in big.var_names[big.var["highly_variable"].to_numpy()])
    logger.info(f"HVG: selected {len(genes)} genes (seurat_v3, top {n}) "
                f"from {len(parts)} train section(s)")
    return genes


def _build_section(bronze: Path, sid: str, role: str, min_cells: int,
                   gene_subset: Optional[List[str]] = None):
    """Read <sid>.h5ad -> contract-compliant AnnData. None if missing/small.

    If ``gene_subset`` is given, subset+reorder to exactly those genes (same
    canonical order for every file -> identical train/test panel)."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    src = bronze / f"{sid}.h5ad"
    if not src.exists():
        logger.warning(f"  {sid}: missing {src.name}; skipping")
        return None
    a = ad.read_h5ad(src)
    if a.n_obs < min_cells:
        logger.warning(f"  {sid}: {a.n_obs} cells < {min_cells}; skipping")
        return None
    if gene_subset is not None:
        present = [g for g in gene_subset if g in set(map(str, a.var_names))]
        if len(present) != len(gene_subset):
            logger.warning(f"  {sid}: {len(gene_subset)-len(present)} HVG(s) "
                           "absent here; using intersection")
        a = a[:, present].copy()

    # --- X: raw counts, sparse int32 ------------------------------------
    X = a.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr()
    # These are integer-valued UMI counts stored as float32; round -> int32.
    if not np.allclose(X.data, np.round(X.data), atol=1e-3):
        logger.warning(f"  {sid}: .X not integer-valued; storing as-is (float32). "
                       "Check the source is raw counts.")
        Xstore = X.astype(np.float32)
    else:
        Xstore = sp.csr_matrix(
            (np.rint(X.data).astype(np.int32), X.indices.copy(), X.indptr.copy()),
            shape=X.shape,
        )

    # --- var: gene SYMBOLS as index (identical across all 12 samples) ----
    var = pd.DataFrame(index=pd.Index([str(g) for g in a.var_names],
                                      name="gene_symbol"))
    if "gene_ids" in a.var.columns:      # keep Ensembl for reference / harmonise
        var["gene_ids"] = a.var["gene_ids"].astype(str).to_numpy()

    # --- obs: layer label, section, donor -------------------------------
    layer = a.obs["sce.layer_guess"].astype(str).to_numpy()
    layer = np.where(np.isin(layer, ["nan", "NaN", "None", ""]), "NA", layer)
    obs = pd.DataFrame(index=pd.Index([str(i) for i in a.obs_names], name="cell_id"))
    obs["cell_class"] = layer          # aux label used by the model
    obs["layer"] = layer               # explicit name for the layer eval
    obs["cell_section"] = sid
    obs["donor"] = DONOR[sid]

    out = ad.AnnData(X=Xstore, obs=obs, var=var)
    out.layers["counts"] = out.X.copy()

    # --- obsm['spatial']: (N, 2) float32 pixel coords (ground truth) ------
    xy = np.asarray(a.obsm["spatial"], dtype=np.float32)[:, :2]
    n_bad = int((~np.isfinite(xy).all(axis=1)).sum())
    if n_bad:
        logger.warning(f"  {sid}: {n_bad} cells with non-finite coords")
    out.obsm["spatial"] = xy

    out.uns["source"] = "Maynard2021_DLPFC_Visium"
    out.uns["sample"] = sid
    out.uns["donor"] = DONOR[sid]
    out.uns["role"] = role
    return out


def prepare_all(bronze: Path, silver: Path, train: List[str], test: List[str],
                min_cells: int, overwrite: bool, n_hvg: int) -> int:
    import collections
    silver.mkdir(parents=True, exist_ok=True)
    hvgs = _select_hvgs(bronze, train, n_hvg, min_cells)
    n_done = 0
    for role, sids in (("train", train), ("test", test)):
        for sid in sids:
            out_path = silver / f"{sid}_{role}.h5ad"
            if out_path.exists() and not overwrite:
                logger.info(f"{sid} [{role}]: {out_path.name} exists; skipping")
                n_done += 1
                continue
            logger.info(f"{sid} [{role}]: building h5ad ...")
            adata = _build_section(bronze, sid, role, min_cells, gene_subset=hvgs)
            if adata is None:
                continue
            adata.write_h5ad(out_path)
            lc = collections.Counter(adata.obs["layer"].tolist())
            layers_str = " ".join(f"{k}:{lc[k]}" for k in _LAYER_ORDER + ["NA"]
                                  if k in lc)
            logger.info(
                f"  -> {out_path.name} ({adata.n_obs:,} spots x {adata.n_vars} "
                f"genes; layers: {layers_str})"
            )
            n_done += 1
    return n_done


# ---------------------------------------------------------------------------
def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bronze_dir", default=BRONZE_DEFAULT)
    p.add_argument("--silver_dir", default=SILVER_DEFAULT)
    p.add_argument("--preset", choices=sorted(PRESETS), default="donor3_holdout",
                   help="Fills --train / --test (default: donor3_holdout).")
    p.add_argument("--train", default=None,
                   help="Comma-separated sample ids for TRAIN (overrides preset).")
    p.add_argument("--test", default=None,
                   help="Comma-separated sample ids for TEST (overrides preset).")
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--skip_prepare", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--min_cells", type=int, default=100)
    p.add_argument("--n_hvg", type=int, default=2000,
                   help="Select this many highly variable genes (seurat_v3 on "
                        "TRAIN sections) and subset all files to that identical "
                        "panel. 0 = keep the whole 33,538-gene transcriptome "
                        "(large node-feature dim). Default 2000.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train, test = PRESETS[args.preset]
    if args.train:
        train = _parse_list(args.train)
    if args.test:
        test = _parse_list(args.test)

    bad = [s for s in train + test if s not in FIGSHARE]
    if bad:
        logger.error(f"Unknown sample id(s) {bad}. Valid: {sorted(FIGSHARE)}")
        return 1
    overlap = set(train) & set(test)
    if overlap:
        logger.error(f"Samples in BOTH train and test: {sorted(overlap)} "
                     "(would leak). Fix --train/--test.")
        return 1

    bronze, silver = Path(args.bronze_dir), Path(args.silver_dir)
    logger.info(f"preset={args.preset}  train={train}  test={test}")
    logger.info(f"bronze={bronze}  silver={silver}")

    if not args.skip_download:
        logger.info("== STEP 1: download from Figshare ==")
        failed = download_samples(bronze, sorted(set(train + test)), args.overwrite)
        if failed:
            logger.error(f"{len(failed)} sample(s) failed to download: {failed}")
            if not args.skip_prepare:
                logger.error("Aborting prepare (re-run to resume downloads).")
                return 1
    else:
        logger.info("== STEP 1 skipped (--skip_download) ==")

    if args.skip_prepare:
        logger.info("== STEP 2 skipped (--skip_prepare) ==")
        return 0

    logger.info("== STEP 2: build h5ads ==")
    n = prepare_all(bronze, silver, train, test, args.min_cells, args.overwrite,
                    args.n_hvg)
    logger.info(f"Done. {n} h5ad(s) in {silver}")
    logger.info("Next: python scripts/validate_scgg_h5ad.py "
                f"--silver_dir {silver}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())

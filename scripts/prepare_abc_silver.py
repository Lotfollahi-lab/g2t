#!/usr/bin/env python
"""
Assemble the ABC Zhuang-ABCA-1 (Animal 1) bronze download into per-section
AnnData files that scgg can consume directly.

Input  : /nfs/team361/sb75/DATASETS/bronze/abc_luna
         (output of scripts/download_abc_zhuang_abca1.py)
Output : /nfs/team361/sb75/DATASETS/silver/abc_luna
         one h5ad per slice, named
             abc_zhuang_abca1_<section_label>.h5ad

Each silver h5ad contains:
  X                       : (cells × genes) raw counts (sparse int32)
  layers['counts']        : same as X (preserved for scanpy HVG selection)
  obsm['spatial']         : (cells × 2) reconstructed XY coordinates
  obs['cell_class']       : broad cell-class label (one of ~31 classes)
  obs['subclass']         : finer subclass label (one of 338 in Animal 1)
  obs['supertype']        : finer still (cluster-level)
  obs['cluster']          : finest, integer cluster id
  obs['brain_section_label']: slice identifier, also encoded in filename
  obs['neurotransmitter'] : neurotransmitter identity (Glut / GABA / etc.)
  var.index               : gene symbols
  uns['section_label']    : slice identifier (e.g., 'Zhuang-ABCA-1-001')

The script auto-discovers the release version from the bronze cache layout.
If multiple versions are present, the highest (lex-sorted) is used; pass
--release_version to override.

Memory: the raw expression matrix is sparse and the largest section has
~30 k cells, so peak memory should stay under a few GB. The full atlas
takes a single pass through the in-memory AnnData.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("scgg.prepare_abc_silver")


# ---------------------------------------------------------------------------
# Bronze layout discovery
# ---------------------------------------------------------------------------


def _find_path(bronze_dir: Path, glob_pattern: str) -> Optional[Path]:
    """Return the first match for the glob, or None."""
    matches = sorted(bronze_dir.glob(glob_pattern))
    if not matches:
        return None
    if len(matches) > 1:
        logger.info(
            f"  multiple matches for {glob_pattern!r}, using newest by lex order: "
            f"{matches[-1].name}"
        )
    return matches[-1]


def _discover_bronze_files(
    bronze_dir: Path,
    release_version: Optional[str] = None,
) -> Dict[str, Path]:
    """Find the bronze files we need, robust to either abc_atlas_access or
    HTTPS-fallback directory layouts.

    Returns a dict with keys:
        cell_metadata, ccf_coords, taxonomy_cluster, taxonomy_term,
        taxonomy_membership, expression
    """
    out: Dict[str, Path] = {}

    if release_version:
        ver = release_version
    else:
        # Auto-discover version: look for any cell_metadata.csv path.
        # In the real ABC layout `cell_metadata.csv` lives directly in the
        # version dir (not under views/), so look there.
        candidates = list(bronze_dir.rglob("metadata/Zhuang-ABCA-1/*/cell_metadata.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No metadata/Zhuang-ABCA-1/*/cell_metadata.csv under "
                f"{bronze_dir}. Did the download finish?"
            )
        versions = sorted({p.parent.name for p in candidates})
        ver = versions[-1]
        logger.info(f"Auto-discovered Zhuang-ABCA-1 release version: {ver}")

    # Independent release versions per dataset (Allen rolls them out of sync).
    def _discover(dataset_subpath: str, default: str) -> str:
        cands = list(bronze_dir.glob(f"{dataset_subpath}/*/"))
        if not cands:
            return default
        return sorted([c.name for c in cands])[-1]

    ccf_ver = _discover("metadata/Zhuang-ABCA-1-CCF", ver)
    tax_ver = _discover("metadata/WMB-taxonomy", ver)
    expr_ver = _discover("expression_matrices/Zhuang-ABCA-1", ver)
    logger.info(
        f"Per-dataset versions: Zhuang-ABCA-1={ver} CCF={ccf_ver} "
        f"WMB-taxonomy={tax_ver} expression={expr_ver}"
    )

    # Prefer the pre-joined `cell_metadata_with_cluster_annotation.csv` view
    # (cell_metadata + class/subclass/supertype labels merged by Allen) over
    # the bare cell_metadata.csv. Falls back to the bare file + WMB-taxonomy
    # join if the pre-joined view isn't on disk.
    out["cell_metadata_joined"] = _find_path(
        bronze_dir,
        f"metadata/Zhuang-ABCA-1/{ver}/views/"
        f"cell_metadata_with_cluster_annotation.csv",
    )
    out["cell_metadata"] = _find_path(
        bronze_dir, f"metadata/Zhuang-ABCA-1/{ver}/cell_metadata.csv"
    )
    out["gene_metadata"] = _find_path(
        bronze_dir, f"metadata/Zhuang-ABCA-1/{ver}/gene.csv"
    )
    out["ccf_coords"] = _find_path(
        bronze_dir, f"metadata/Zhuang-ABCA-1-CCF/{ccf_ver}/ccf_coordinates.csv"
    )
    out["taxonomy_cluster"] = _find_path(
        bronze_dir, f"metadata/WMB-taxonomy/{tax_ver}/cluster.csv"
    )
    out["taxonomy_term"] = _find_path(
        bronze_dir, f"metadata/WMB-taxonomy/{tax_ver}/cluster_annotation_term.csv"
    )
    out["taxonomy_membership"] = _find_path(
        bronze_dir,
        f"metadata/WMB-taxonomy/{tax_ver}/"
        f"cluster_to_cluster_annotation_membership.csv",
    )
    out["expression"] = _find_path(
        bronze_dir,
        f"expression_matrices/Zhuang-ABCA-1/{expr_ver}/Zhuang-ABCA-1-raw.h5ad",
    )

    missing = [k for k, v in out.items() if v is None]
    if missing:
        # Some files (taxonomy / CCF) may be optional — only fail on expression.
        if "expression" in missing or "cell_metadata" in missing:
            raise FileNotFoundError(
                f"Required bronze files missing: {missing}. "
                f"Look under {bronze_dir} and re-run the download script."
            )
        for m in missing:
            logger.warning(f"  optional bronze file missing: {m}")

    out["release_version"] = ver  # type: ignore[assignment]
    return out


# ---------------------------------------------------------------------------
# Cell-class label resolution from WMB taxonomy
# ---------------------------------------------------------------------------


def _build_cell_label_lookup(
    membership_path: Path,
    term_path: Path,
) -> pd.DataFrame:
    """Return a DataFrame indexed by `cluster_alias` (integer cluster id) with
    columns: cell_class, subclass, supertype, neurotransmitter.

    The ABC taxonomy has a many-to-many membership table; we pivot it to one
    column per annotation set.
    """
    logger.info("Building cluster -> class/subclass/supertype lookup...")
    mem = pd.read_csv(membership_path)
    term = pd.read_csv(term_path)

    # Term sets we care about
    wanted_sets = {
        "class": "cell_class",
        "subclass": "subclass",
        "supertype": "supertype",
        "neurotransmitter": "neurotransmitter",
    }

    # Join membership -> term so each row has the human-readable label
    if "cluster_annotation_term_label" in mem.columns:
        mem_label_col = "cluster_annotation_term_label"
    elif "term_label" in mem.columns:
        mem_label_col = "term_label"
    else:
        # Fall back: join via cluster_annotation_term_label_id
        if "cluster_annotation_term_label_id" in mem.columns:
            mem = mem.merge(
                term[["label", "name", "cluster_annotation_term_set_name"]].rename(
                    columns={"label": "cluster_annotation_term_label_id", "name": "label"}
                ),
                on="cluster_annotation_term_label_id",
                how="left",
            )
            mem_label_col = "label"
        else:
            raise ValueError(
                "Could not locate label column in cluster_to_cluster_annotation_membership.csv. "
                f"Columns: {list(mem.columns)}"
            )

    term_set_col = next(
        (c for c in mem.columns if "cluster_annotation_term_set" in c and "name" in c),
        None,
    )
    if term_set_col is None:
        # Older schemas use 'cluster_annotation_term_set'
        term_set_col = "cluster_annotation_term_set"
    if term_set_col not in mem.columns:
        raise ValueError(
            f"Could not locate term-set column in membership table. "
            f"Columns: {list(mem.columns)}"
        )

    # Cluster id column (varies)
    cluster_col = next(
        (c for c in ("cluster_alias", "cluster_id", "cluster") if c in mem.columns),
        None,
    )
    if cluster_col is None:
        raise ValueError(
            f"Could not locate cluster id column in membership table. "
            f"Columns: {list(mem.columns)}"
        )

    pieces: List[pd.DataFrame] = []
    for set_name, col_out in wanted_sets.items():
        sub = mem[mem[term_set_col] == set_name][[cluster_col, mem_label_col]].copy()
        sub = sub.rename(columns={cluster_col: "cluster_alias", mem_label_col: col_out})
        sub = sub.drop_duplicates("cluster_alias").set_index("cluster_alias")
        pieces.append(sub)

    out = pd.concat(pieces, axis=1)
    logger.info(f"  taxonomy lookup: {len(out)} clusters, columns={list(out.columns)}")
    return out


# ---------------------------------------------------------------------------
# Main per-section split
# ---------------------------------------------------------------------------


def assemble(
    bronze_dir: Path,
    silver_dir: Path,
    release_version: Optional[str],
    overwrite: bool,
    section_filter: Optional[List[str]],
    min_cells_per_section: int,
) -> None:
    import anndata as ad
    import scipy.sparse as sp

    paths = _discover_bronze_files(bronze_dir, release_version)
    silver_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load cell metadata -----------------------------------------
    # Prefer the pre-joined view (cell_metadata + class/subclass/supertype
    # already merged by Allen). If unavailable, load the bare file and rely
    # on the optional WMB-taxonomy join below.
    if paths.get("cell_metadata_joined") is not None:
        logger.info(f"Loading pre-joined metadata: {paths['cell_metadata_joined']}")
        meta = pd.read_csv(paths["cell_metadata_joined"])
        is_prejoined = True
    else:
        logger.info(f"Loading bare cell metadata: {paths['cell_metadata']}")
        meta = pd.read_csv(paths["cell_metadata"])
        is_prejoined = False
    logger.info(f"  cells: {len(meta):,}")
    logger.info(f"  columns: {list(meta.columns)}")

    # Identify the cell-id join key (used later to align metadata with the
    # expression AnnData's obs.index).
    join_key = next(
        (k for k in ("cell_label", "cell_id", "barcode") if k in meta.columns),
        "cell_label",
    )
    logger.info(f"  cell-id join key: {join_key!r}")

    # Look for spatial coordinates. With the pre-joined view, x/y are often
    # already in `meta`. If not, join the CCF coordinates table.
    has_xy_in_meta = (
        any(c in meta.columns for c in ("x_reconstructed", "x_section", "x_ccf", "x"))
        and any(c in meta.columns for c in ("y_reconstructed", "y_section", "y_ccf", "y"))
    )
    if not has_xy_in_meta and paths.get("ccf_coords") is not None:
        logger.info(f"Joining CCF coordinates: {paths['ccf_coords']}")
        coords = pd.read_csv(paths["ccf_coords"])
        if join_key not in coords.columns:
            logger.warning(
                f"  CCF coords table missing {join_key!r}; cannot merge. "
                f"Will rely on coord columns inside the main metadata."
            )
        else:
            keep_coord_cols = [c for c in coords.columns
                                if c == join_key
                                or c in (
                                    "x_reconstructed", "y_reconstructed",
                                    "z_reconstructed",
                                    "x", "y", "z",
                                    "x_ccf", "y_ccf", "z_ccf",
                                    "parcellation_substructure",
                                    "parcellation_structure",
                                    "parcellation_division",
                                )]
            meta = meta.merge(coords[keep_coord_cols], on=join_key, how="left")
    elif has_xy_in_meta:
        logger.info(
            "  spatial coords already in metadata; skipping CCF coordinate merge"
        )

    # ---- 2. Resolve cell labels via WMB taxonomy -----------------------
    # If we're using the pre-joined view, class/subclass/supertype columns
    # are already present. Otherwise, join the WMB-taxonomy tables manually.
    cluster_alias_col = next(
        (c for c in ("cluster_alias", "cluster_id_label", "cluster_label", "cluster")
         if c in meta.columns),
        None,
    )
    label_cols_present = [
        c for c in ("class", "subclass", "supertype", "neurotransmitter")
        if c in meta.columns
    ]
    if is_prejoined and label_cols_present:
        logger.info(
            f"  pre-joined view already contains label columns: {label_cols_present}; "
            "skipping WMB-taxonomy join"
        )
        # Rename `class` -> `cell_class` for downstream consistency.
        if "class" in meta.columns and "cell_class" not in meta.columns:
            meta = meta.rename(columns={"class": "cell_class"})
    elif (cluster_alias_col is not None
            and paths.get("taxonomy_membership") is not None
            and paths.get("taxonomy_term") is not None):
        try:
            lookup = _build_cell_label_lookup(
                paths["taxonomy_membership"], paths["taxonomy_term"]
            )
            meta = meta.merge(
                lookup, left_on=cluster_alias_col, right_index=True, how="left"
            )
            logger.info(
                "  joined WMB-taxonomy labels manually: "
                f"{meta[['cell_class','subclass','supertype']].notna().sum().to_dict()}"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  taxonomy join failed ({e}); continuing without labels.")
    else:
        logger.warning(
            "  skipping taxonomy join — either no cluster id column or "
            "taxonomy tables missing, and view has no pre-joined labels."
        )

    # ---- 3. Pick the X / Y coordinate columns -------------------------
    coord_x_col = next(
        (c for c in ("x_reconstructed", "x", "x_ccf") if c in meta.columns), None,
    )
    coord_y_col = next(
        (c for c in ("y_reconstructed", "y", "y_ccf") if c in meta.columns), None,
    )
    if coord_x_col is None or coord_y_col is None:
        raise ValueError(
            f"No usable XY coords found in metadata. Available columns: {list(meta.columns)}"
        )
    logger.info(f"Using coords: ({coord_x_col}, {coord_y_col})")

    # Pull the section label column
    section_col = next(
        (c for c in ("brain_section_label", "section_label", "z_section", "z_index")
         if c in meta.columns),
        None,
    )
    if section_col is None:
        raise ValueError(
            f"No section-label column found in metadata. Columns: {list(meta.columns)}"
        )
    logger.info(f"Using section column: {section_col}")

    # ---- 4. Load the expression matrix (lazy: keep on disk if possible)
    logger.info(f"Loading expression matrix: {paths['expression']}")
    adata = ad.read_h5ad(paths["expression"])
    logger.info(f"  AnnData: {adata.shape} (cells × genes)")
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    # Align metadata to the obs of the expression matrix
    if join_key not in adata.obs.columns:
        if adata.obs.index.name == join_key or adata.obs.index.equals(meta[join_key]):
            adata.obs[join_key] = adata.obs.index.values
        else:
            # Fall back to whatever the expression matrix uses for cell ids
            adata.obs[join_key] = adata.obs.index.values
            logger.warning(
                f"Expression matrix's obs.index used as {join_key} (best-effort)."
            )

    # Reindex meta to match adata's cell order (drops any cells without metadata)
    meta_aligned = meta.set_index(join_key).reindex(adata.obs[join_key].values)
    n_with_meta = meta_aligned.dropna(subset=[section_col]).shape[0]
    logger.info(
        f"  cells with usable metadata: {n_with_meta:,} / {adata.n_obs:,}"
    )
    keep_mask = ~meta_aligned[section_col].isna()
    adata = adata[keep_mask.values].copy()
    meta_aligned = meta_aligned.loc[keep_mask.values]

    # Attach selected columns to adata.obs
    attached = {}
    for col, dst in (
        (section_col, "brain_section_label"),
        (coord_x_col, "coord_X"),
        (coord_y_col, "coord_Y"),
        ("cell_class", "cell_class"),
        ("subclass", "subclass"),
        ("supertype", "supertype"),
        ("neurotransmitter", "neurotransmitter"),
        (cluster_alias_col, "cluster"),
        ("parcellation_substructure", "parcellation_substructure"),
        ("parcellation_structure", "parcellation_structure"),
    ):
        if col is None:
            continue
        if col in meta_aligned.columns:
            adata.obs[dst] = meta_aligned[col].values
            attached[dst] = col
    logger.info(f"  attached obs columns: {attached}")

    # XY coords -> obsm
    xy = np.column_stack([
        adata.obs["coord_X"].astype(np.float32).values,
        adata.obs["coord_Y"].astype(np.float32).values,
    ])
    adata.obsm["spatial"] = xy

    # Drop cells with non-finite coordinates
    finite = np.isfinite(xy).all(axis=1)
    if (~finite).any():
        logger.info(f"  dropping {(~finite).sum():,} cells with non-finite coords")
        adata = adata[finite].copy()

    # ---- 5. Persist counts layer (raw) --------------------------------
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # ---- 6. Iterate sections and write one h5ad per slice ------------
    sections = adata.obs["brain_section_label"].astype(str).unique().tolist()
    sections.sort()
    if section_filter is not None:
        sections = [s for s in sections if s in section_filter]
    logger.info(f"Writing {len(sections)} sections to {silver_dir}")

    written = 0
    skipped_small = 0
    for sec in sections:
        out_path = silver_dir / f"abc_zhuang_abca1_{sec}.h5ad"
        if out_path.exists() and not overwrite:
            logger.info(f"  exists, skipping: {out_path.name}")
            continue
        sub = adata[adata.obs["brain_section_label"].astype(str) == sec].copy()
        if sub.n_obs < min_cells_per_section:
            logger.info(
                f"  {sec}: only {sub.n_obs} cells (< {min_cells_per_section}); skipping"
            )
            skipped_small += 1
            continue
        sub.uns["section_label"] = sec
        sub.uns["assay"] = "MERFISH"
        sub.uns["tissue"] = "whole_brain"
        sub.uns["species"] = "mus_musculus"
        sub.uns["source"] = "ABC_Zhuang-ABCA-1"
        sub.write(out_path)
        logger.info(f"  wrote {sub.n_obs:,} cells -> {out_path.name}")
        written += 1

    logger.info(
        f"Done. wrote={written} skipped_small={skipped_small} total_sections={len(sections)}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bronze_dir",
        default="/nfs/team361/sb75/DATASETS/bronze/abc_luna",
    )
    p.add_argument(
        "--silver_dir",
        default="/nfs/team361/sb75/DATASETS/silver/abc_luna",
    )
    p.add_argument(
        "--release_version", default=None,
        help="Override auto-detected release version.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-write per-section h5ads even if they already exist.",
    )
    p.add_argument(
        "--sections", default=None,
        help="Optional comma-separated list of section labels to process "
             "(default: all).",
    )
    p.add_argument(
        "--min_cells_per_section", type=int, default=100,
        help="Drop sections with fewer cells.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    section_filter = (
        [s.strip() for s in args.sections.split(",")] if args.sections else None
    )

    assemble(
        bronze_dir=Path(args.bronze_dir),
        silver_dir=Path(args.silver_dir),
        release_version=args.release_version,
        overwrite=args.overwrite,
        section_filter=section_filter,
        min_cells_per_section=args.min_cells_per_section,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

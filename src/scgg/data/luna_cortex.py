"""
Loader for the MERFISH mouse primary motor cortex dataset used in LUNA Figure 3.

Expected directory layout (one h5ad per slice):

    {root}/mmc_mouse1_slice1.h5ad
    {root}/mmc_mouse1_slice10.h5ad
    ...
    {root}/mmc_mouse2_slice99.h5ad

The legacy `merfish_mouse_cortex_mouse{M}_slice{S}.h5ad` naming is also
recognised for backwards compatibility — both prefixes may coexist in
the same directory during a rename.

This corresponds to the LUNA Figure 3 split:
  * Mouse 1 = TRAIN (33 slices, 158,379 cells)
  * Mouse 2 = TEST  (31 slices, 118,036 cells)
  * 254 genes (MERFISH panel from Zhang et al. 2021).

Each h5ad is expected to contain:
  * .X — gene-by-cell expression (sparse int counts or float)
  * .obsm["spatial"] — (N, 2) ground-truth XY coordinates
  * .obs columns including "cell_class" (cell type label) and either
    ("coord_X", "coord_Y") or those values copied into .obsm["spatial"].

The dataset name and animal split are inferred from the filename pattern
`mmc_mouse{M}_slice{S}.h5ad` so adding/removing slices does not require
editing this file.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional, Dict, Iterable, List, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


try:
    import anndata as ad
    import scanpy as sc

    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    logger.warning("scanpy/anndata not available. Install with: pip install scanpy")


# Filename pattern. Group 1 = mouse id (int); group 2 = slice id (int).
# Accepts both naming conventions:
#   * mmc_mouse{M}_slice{S}.h5ad                  (preferred / new)
#   * merfish_mouse_cortex_mouse{M}_slice{S}.h5ad (legacy)
_SLICE_RE = re.compile(
    r"^(?:mmc|merfish_mouse_cortex)_mouse(?P<mouse>\d+)_slice(?P<slice>\d+)\.h5ad$"
)


def _enumerate_slice_files(root: Path) -> List[Tuple[int, int, Path]]:
    """Find per-slice h5ad files under ``root`` and parse (mouse, slice) from name."""
    out: List[Tuple[int, int, Path]] = []
    for p in sorted(root.iterdir()):
        m = _SLICE_RE.match(p.name)
        if not m:
            continue
        out.append((int(m["mouse"]), int(m["slice"]), p))
    return out


def _extract_coords(adata) -> np.ndarray:
    """Pull a (N, 2) float32 coord array out of one of several possible places."""
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        return coords[:, :2]
    if "X_spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["X_spatial"], dtype=np.float32)
        return coords[:, :2]
    if {"coord_X", "coord_Y"} <= set(adata.obs.columns):
        return np.column_stack(
            [
                adata.obs["coord_X"].to_numpy(dtype=np.float32),
                adata.obs["coord_Y"].to_numpy(dtype=np.float32),
            ]
        )
    raise ValueError(
        "No spatial coordinates found in h5ad. Tried .obsm['spatial'], "
        ".obsm['X_spatial'], and .obs['coord_X','coord_Y']."
    )


def _extract_cell_class(adata) -> Optional[np.ndarray]:
    """Try a small set of plausible cell-class column names."""
    for col in ("cell_class", "cell_type", "cell_types", "subclass", "class"):
        if col in adata.obs.columns:
            return adata.obs[col].astype(str).to_numpy()
    return None


def load_luna_cortex(
    data_dir: str | Path,
    mouse_ids_train: Iterable[int] = (1,),
    mouse_ids_test: Iterable[int] = (2,),
    gene_list: Optional[List[str]] = None,
    n_top_hvg: Optional[int] = None,
    normalize: bool = True,
    scale: bool = True,
    min_cells_per_section: int = 100,
) -> Dict[str, Any]:
    """Load the LUNA Figure 3 cortex split from a directory of per-slice h5ad files.

    Args:
        data_dir: Directory containing files like
            ``mmc_mouse1_slice1.h5ad`` (or the legacy
            ``merfish_mouse_cortex_mouse1_slice1.h5ad``).
        mouse_ids_train: Mouse ids treated as TRAIN (default: (1,)).
        mouse_ids_test:  Mouse ids treated as TEST  (default: (2,)).
        gene_list: If given, subset to these genes (intersection with available).
        n_top_hvg: Optionally select top-N highly variable genes (seurat_v3 on counts).
        normalize: Apply normalize_total + log1p.
        scale: z-score per gene (scanpy.pp.scale).
        min_cells_per_section: Drop sections with fewer cells.

    Returns:
        Dict with both train and test arrays:
            gene_expr_train, coords_train, section_ids_train, cell_class_train,
            gene_expr_test,  coords_test,  section_ids_test,  cell_class_test,
            gene_names,
            section_map_train, section_map_test   (id -> "mouseM_sliceS")

        Section ids are CONTIGUOUS integers within each split so the existing
        SpatialTranscriptomicsDataset can be used unchanged. Across-split ids
        do NOT overlap with each other in semantics but may collide in value;
        always use the (split, id) pair if you need uniqueness.
    """
    assert SCANPY_AVAILABLE, "scanpy / anndata required for LUNA cortex loader"

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    files = _enumerate_slice_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No matching slice h5ad files found in {data_dir} "
            f"(expected names like mmc_mouse1_slice1.h5ad or the legacy "
            f"merfish_mouse_cortex_mouse1_slice1.h5ad)"
        )

    train_set, test_set = set(mouse_ids_train), set(mouse_ids_test)
    files_train = [(m, s, p) for (m, s, p) in files if m in train_set]
    files_test = [(m, s, p) for (m, s, p) in files if m in test_set]

    logger.info(
        f"LUNA cortex: discovered {len(files)} slice files in {data_dir} "
        f"({len(files_train)} train, {len(files_test)} test)"
    )

    train = _load_split(
        files_train,
        gene_list=gene_list,
        n_top_hvg=n_top_hvg,
        normalize=normalize,
        scale=scale,
        min_cells_per_section=min_cells_per_section,
        split_name="train",
    )
    test = _load_split(
        files_test,
        gene_list=gene_list,
        n_top_hvg=None,  # do not redo HVG selection on test
        normalize=normalize,
        scale=scale,
        min_cells_per_section=min_cells_per_section,
        split_name="test",
        gene_subset=train["gene_names"],
        class_names=train.get("class_names"),
    )

    return {
        "gene_expr_train": train["gene_expr"],
        "coords_train": train["coords"],
        "section_ids_train": train["section_ids"],
        "cell_class_train": train["cell_class"],          # str array
        "cell_class_id_train": train["cell_class_id"],    # int array (or None)
        "section_map_train": train["section_map"],
        "gene_expr_test": test["gene_expr"],
        "coords_test": test["coords"],
        "section_ids_test": test["section_ids"],
        "cell_class_test": test["cell_class"],
        "cell_class_id_test": test["cell_class_id"],
        "section_map_test": test["section_map"],
        "gene_names": train["gene_names"],
        "class_names": train.get("class_names"),
        "n_classes": len(train["class_names"]) if train.get("class_names") else 0,
    }


def _load_split(
    files: List[Tuple[int, int, Path]],
    gene_list: Optional[List[str]],
    n_top_hvg: Optional[int],
    normalize: bool,
    scale: bool,
    min_cells_per_section: int,
    split_name: str,
    gene_subset: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load one split (train or test), concatenate sections, harmonize genes."""
    import scipy.sparse as sp

    if not files:
        raise ValueError(f"No files found for split={split_name!r}")

    adatas = []
    section_labels: List[str] = []
    section_counts: List[int] = []
    for mouse, slice_id, path in files:
        adata = ad.read_h5ad(path)
        if adata.n_obs < min_cells_per_section:
            logger.info(
                f"  skipping {path.name}: {adata.n_obs} cells < "
                f"min_cells_per_section={min_cells_per_section}"
            )
            continue
        # CRITICAL for LUNA-vs-scGG comparability: when the silver was
        # built from a LUNA bronze CSV via build_h5ad_from_luna_csv,
        # each h5ad carries `_bronze_row_pos` — the cell's original
        # row index in the bronze CSV. LUNA's data_module produces a
        # specific within-section cell ordering after its
        # `sort_values("cell_section")` pass; that ordering is uniquely
        # determined by the bronze row order. By sorting each h5ad's
        # cells by `_bronze_row_pos` here we make scGG's per-section
        # cell ordering match what LUNA would produce on the bronze
        # subset for the same section. (Stable sort so the order is
        # deterministic across runs.)
        if "_bronze_row_pos" in adata.obs.columns:
            order = np.argsort(
                adata.obs["_bronze_row_pos"].to_numpy(), kind="stable"
            )
            adata = adata[order].copy()
        adatas.append(adata)
        section_labels.append(f"mouse{mouse}_slice{slice_id}")
        section_counts.append(adata.n_obs)
    if not adatas:
        raise ValueError(f"All sections in {split_name} fell below min_cells_per_section")

    # Concat along obs; harmonize gene panel by intersection.
    # AnnData.concat with join='inner' takes the gene intersection.
    big = ad.concat(adatas, axis=0, join="inner", label="_split_section",
                    keys=section_labels, merge="same")
    # Add the section labels as a column (one per cell).
    big.obs["_split_section"] = big.obs["_split_section"].astype(str)

    # Optional gene-list subset (for cross-split panel alignment).
    if gene_subset is not None:
        keep = [g for g in gene_subset if g in big.var_names]
        if len(keep) != len(gene_subset):
            missing = set(gene_subset) - set(keep)
            logger.info(
                f"  [{split_name}] {len(missing)} train-panel genes missing in this "
                f"split; using intersection of {len(keep)}"
            )
        big = big[:, keep].copy()
    elif gene_list is not None:
        keep = [g for g in gene_list if g in big.var_names]
        big = big[:, keep].copy()

    # Preserve raw counts (scanpy.pp.highly_variable_genes(seurat_v3) needs them).
    if "counts" not in big.layers:
        big.layers["counts"] = big.X.copy()

    if normalize:
        sc.pp.normalize_total(big, target_sum=1e4)
        sc.pp.log1p(big)

    if n_top_hvg is not None and big.n_vars > n_top_hvg:
        sc.pp.highly_variable_genes(
            big, n_top_genes=n_top_hvg, flavor="seurat_v3", layer="counts"
        )
        big = big[:, big.var.highly_variable].copy()

    if scale:
        sc.pp.scale(big, max_value=10)

    # Coordinates
    if "spatial" in big.obsm:
        coords = np.asarray(big.obsm["spatial"], dtype=np.float32)[:, :2]
    else:
        coords = _extract_coords(big)

    # Cell class — strings + integer encoding with consistent vocabulary
    # across train and test (test maps unseen classes to -1).
    cell_class = _extract_cell_class(big)
    if cell_class is not None:
        if class_names is None:
            class_names = sorted(np.unique(cell_class).tolist())
        cls_to_id = {c: i for i, c in enumerate(class_names)}
        cell_class_id = np.array(
            [cls_to_id.get(c, -1) for c in cell_class], dtype=np.int64
        )
        n_unknown = int((cell_class_id < 0).sum())
        if n_unknown > 0:
            logger.info(
                f"  [{split_name}] {n_unknown} cells have a class not present in "
                f"train vocabulary; marked as -1 (aux loss will skip them)."
            )
    else:
        cell_class_id = None
        class_names = None

    # Dense gene expression
    if sp.issparse(big.X):
        X = np.asarray(big.X.toarray(), dtype=np.float32)
    else:
        X = np.asarray(big.X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Section ids: contiguous integers in order of discovery
    label_to_id = {lab: i for i, lab in enumerate(section_labels)}
    section_ids = np.array(
        [label_to_id[s] for s in big.obs["_split_section"]], dtype=np.int64
    )
    section_map = {i: lab for lab, i in label_to_id.items()}

    logger.info(
        f"  [{split_name}] cells={X.shape[0]} genes={X.shape[1]} "
        f"sections={len(label_to_id)}"
    )

    return {
        "gene_expr": X,
        "coords": coords,
        "section_ids": section_ids,
        "cell_class": cell_class,         # str array (or None)
        "cell_class_id": cell_class_id,   # int array (or None)
        "class_names": class_names,
        "section_map": section_map,
        "gene_names": list(big.var_names),
    }

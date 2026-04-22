"""
MERFISH ABC Atlas data loading.

Handles loading and preprocessing of the Allen Brain Cell Atlas MERFISH
dataset (or compatible formats). Expects AnnData objects stored as .h5ad
files with spatial coordinates in .obsm['spatial'] and section labels
in .obs.

Also supports loading generic spatial transcriptomics data from AnnData
with minimal assumptions.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

try:
    import anndata as ad
    import scanpy as sc

    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    logger.warning("scanpy/anndata not available. Install with: pip install scanpy")


def _validate_raw_counts(adata) -> bool:
    """Check whether .X plausibly contains raw counts."""
    import scipy.sparse as sp

    sample = adata.X[:1000] if adata.n_obs > 1000 else adata.X
    if sp.issparse(sample):
        sample_dense = sample.toarray()
    else:
        sample_dense = np.asarray(sample)

    max_val = sample_dense.max()
    # Raw counts are non-negative integers; check a sample
    is_integer = np.allclose(sample_dense, np.round(sample_dense), atol=1e-3)
    is_nonneg = sample_dense.min() >= -1e-6

    if is_integer and is_nonneg:
        return True
    if max_val > 50 and is_nonneg:
        # Large non-negative values are likely counts even if not perfectly integer
        # (some platforms store float counts)
        return True
    return False


def load_merfish_data(
    data_path: str,
    section_col: str = "brain_section_label",
    coord_key: str = "spatial",
    cell_type_col: str = "cell_type",
    gene_list: Optional[List[str]] = None,
    n_top_hvg: Optional[int] = None,
    normalize: bool = True,
    scale: bool = True,
    min_cells_per_section: int = 100,
    subsample_sections: Optional[int] = None,
) -> Dict[str, Any]:
    """Load and preprocess MERFISH data.

    Assumes .X contains raw counts and applies normalization from scratch.
    If .X does not look like raw counts (e.g., contains negative values or
    non-integer floats with small max), logs a warning and proceeds
    cautiously.

    Args:
        data_path: Path to .h5ad file.
        section_col: Column in .obs containing section labels.
        coord_key: Key in .obsm containing spatial coordinates.
        cell_type_col: Column in .obs containing cell type annotations.
        gene_list: Specific genes to use. If None, uses all (or HVGs).
        n_top_hvg: Number of highly variable genes to select.
        normalize: Whether to apply normalize_total + log1p.
        scale: Whether to z-score normalize per gene.
        min_cells_per_section: Minimum cells to keep a section.
        subsample_sections: If set, randomly select this many sections.

    Returns:
        Dict containing:
            - 'gene_expr': np.ndarray (n_cells, n_genes), preprocessed expression.
            - 'coords': np.ndarray (n_cells, 2 or 3), spatial coordinates.
            - 'section_ids': np.ndarray (n_cells,), integer section labels.
            - 'cell_types': np.ndarray (n_cells,), integer cell type labels (or None).
            - 'cell_type_names': List[str], cell type name mapping (or None).
            - 'gene_names': List[str], gene names.
            - 'section_map': Dict mapping section_id int -> original label.
            - 'adata': The processed AnnData object.
    """
    assert SCANPY_AVAILABLE, "scanpy required for data loading"
    import scipy.sparse as sp

    logger.info(f"Loading data from {data_path}")
    adata = ad.read_h5ad(data_path)
    logger.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")

    # Subset to specific genes if requested
    if gene_list is not None:
        available = [g for g in gene_list if g in adata.var_names]
        logger.info(f"Using {len(available)}/{len(gene_list)} requested genes")
        adata = adata[:, available].copy()

    # Validate and store raw counts
    has_raw_counts = _validate_raw_counts(adata)
    if has_raw_counts:
        logger.info("Validated .X as raw counts")
        adata.layers["counts"] = adata.X.copy()
    else:
        logger.warning(
            ".X does not appear to contain raw counts (non-integer or negative values detected). "
            "Proceeding, but HVG selection (seurat_v3) and normalization may behave unexpectedly. "
            "Consider providing data with raw counts in .X or a 'counts' layer."
        )
        if "counts" in adata.layers:
            logger.info("Found existing 'counts' layer, will use it for HVG selection")
        else:
            adata.layers["counts"] = adata.X.copy()

    # Normalization
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        logger.info("Applied normalize_total + log1p")

    # HVG selection (always use seurat_v3 on counts layer)
    if n_top_hvg is not None and adata.n_vars > n_top_hvg:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top_hvg, flavor="seurat_v3", layer="counts"
        )
        adata = adata[:, adata.var.highly_variable].copy()
        logger.info(f"Selected {adata.n_vars} highly variable genes")

    if scale:
        sc.pp.scale(adata, max_value=10)

    # Extract coordinates
    if coord_key in adata.obsm:
        coords = np.array(adata.obsm[coord_key], dtype=np.float32)
    elif "X_spatial" in adata.obsm:
        coords = np.array(adata.obsm["X_spatial"], dtype=np.float32)
    else:
        # Try x/y columns in .obs
        assert "x" in adata.obs.columns and "y" in adata.obs.columns, (
            f"No spatial coordinates found. Tried .obsm['{coord_key}'], "
            ".obsm['X_spatial'], and .obs['x'/'y']."
        )
        coords = np.column_stack([
            adata.obs["x"].values,
            adata.obs["y"].values,
        ]).astype(np.float32)

    # Take only 2D coordinates if 3D
    if coords.shape[1] > 2:
        logger.info(f"Coordinates have {coords.shape[1]}D, using first 2 dimensions")
        coords = coords[:, :2]

    # Cell types
    cell_types = None
    cell_type_names = None
    if cell_type_col in adata.obs.columns:
        ct_cat = adata.obs[cell_type_col].astype("category")
        cell_types = ct_cat.cat.codes.values.astype(np.int64)
        cell_type_names = list(ct_cat.cat.categories)
        logger.info(f"Found {len(cell_type_names)} cell types in '{cell_type_col}'")
    else:
        logger.warning(f"Cell type column '{cell_type_col}' not found in .obs")

    # Section labels
    if section_col in adata.obs.columns:
        section_labels = adata.obs[section_col].values
    else:
        logger.warning(f"Section column '{section_col}' not found. Treating as single section.")
        section_labels = np.zeros(adata.n_obs, dtype=int)

    # Convert section labels to integer IDs
    unique_sections = np.unique(section_labels)
    section_map = {i: str(s) for i, s in enumerate(unique_sections)}
    label_to_id = {str(s): i for i, s in enumerate(unique_sections)}
    section_ids = np.array([label_to_id[str(s)] for s in section_labels], dtype=np.int64)

    logger.info(f"Found {len(unique_sections)} section(s)")

    # Filter sections by minimum cell count
    section_counts = np.bincount(section_ids)
    valid_sections = np.where(section_counts >= min_cells_per_section)[0]
    if len(valid_sections) < len(unique_sections):
        n_filtered = len(unique_sections) - len(valid_sections)
        logger.info(f"Filtered {n_filtered} sections with <{min_cells_per_section} cells")
        mask = np.isin(section_ids, valid_sections)
        adata = adata[mask].copy()
        coords = coords[mask]
        section_ids = section_ids[mask]
        if cell_types is not None:
            cell_types = cell_types[mask]

    # Optionally subsample sections
    if subsample_sections is not None and len(np.unique(section_ids)) > subsample_sections:
        selected = np.random.choice(
            np.unique(section_ids), subsample_sections, replace=False
        )
        mask = np.isin(section_ids, selected)
        adata = adata[mask].copy()
        coords = coords[mask]
        section_ids = section_ids[mask]
        if cell_types is not None:
            cell_types = cell_types[mask]
        logger.info(f"Subsampled to {subsample_sections} sections")

    # Re-index section IDs to be contiguous
    unique_remaining = np.unique(section_ids)
    remap = {old: new for new, old in enumerate(unique_remaining)}
    section_ids = np.array([remap[s] for s in section_ids], dtype=np.int64)
    section_map = {remap.get(k, k): v for k, v in section_map.items() if k in remap}

    # Extract dense expression matrix
    if sp.issparse(adata.X):
        gene_expr = np.array(adata.X.toarray(), dtype=np.float32)
    else:
        gene_expr = np.array(adata.X, dtype=np.float32)

    # Replace NaN/inf from scaling with 0
    gene_expr = np.nan_to_num(gene_expr, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        f"Preprocessed: {gene_expr.shape[0]} cells, {gene_expr.shape[1]} genes, "
        f"{len(np.unique(section_ids))} sections"
    )

    return {
        "gene_expr": gene_expr,
        "coords": coords,
        "section_ids": section_ids,
        "cell_types": cell_types,
        "cell_type_names": cell_type_names,
        "gene_names": list(adata.var_names),
        "section_map": section_map,
        "adata": adata,
    }


def load_generic_spatial_data(
    data_path: str,
    coord_key: str = "spatial",
    section_col: Optional[str] = None,
    cell_type_col: Optional[str] = None,
    normalize: bool = True,
    scale: bool = True,
) -> Dict[str, Any]:
    """Load generic spatial transcriptomics data from AnnData.

    Minimal preprocessing with fewer assumptions than the MERFISH loader.
    Works with Visium, Slide-seq, Xenium, MERFISH, seqFISH, etc.
    """
    assert SCANPY_AVAILABLE, "scanpy required"
    import scipy.sparse as sp

    adata = ad.read_h5ad(data_path)

    # Assume .X is raw counts; validate and warn if not
    has_raw_counts = _validate_raw_counts(adata)
    if not has_raw_counts:
        logger.warning(
            ".X does not appear to contain raw counts. "
            "Normalization may behave unexpectedly."
        )
    adata.layers["counts"] = adata.X.copy()

    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10)

    # Coordinates
    coords = np.array(adata.obsm[coord_key], dtype=np.float32)[:, :2]

    # Sections
    if section_col and section_col in adata.obs.columns:
        labels = adata.obs[section_col].astype("category").cat.codes.values
        section_ids = np.array(labels, dtype=np.int64)
    else:
        section_ids = np.zeros(adata.n_obs, dtype=np.int64)

    # Cell types
    cell_types = None
    cell_type_names = None
    if cell_type_col and cell_type_col in adata.obs.columns:
        ct_cat = adata.obs[cell_type_col].astype("category")
        cell_types = ct_cat.cat.codes.values.astype(np.int64)
        cell_type_names = list(ct_cat.cat.categories)

    if sp.issparse(adata.X):
        gene_expr = np.array(adata.X.toarray(), dtype=np.float32)
    else:
        gene_expr = np.array(adata.X, dtype=np.float32)

    gene_expr = np.nan_to_num(gene_expr, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "gene_expr": gene_expr,
        "coords": coords,
        "section_ids": section_ids,
        "cell_types": cell_types,
        "cell_type_names": cell_type_names,
        "gene_names": list(adata.var_names),
        "section_map": {i: str(i) for i in np.unique(section_ids)},
        "adata": adata,
    }

"""
Loader for the ABC MERFISH mouse brain atlas (Zhuang-ABCA-1, Animal 1).

Expected directory layout (one h5ad per slice, produced by
scripts/prepare_abc_silver.py):

    {root}/abc_zhuang_abca1_<section_label>.h5ad

The section labels follow the ABC convention (e.g. `Zhuang-ABCA-1-001`).
The number of sections is ~147 for Animal 1.

Each h5ad is expected to contain:
  .X                       : sparse raw counts
  .obsm['spatial']         : (N, 2) reconstructed XY coordinates
  .obs['cell_class']       : broad cell-class label (~31 classes for Animal 1)
  .obs['subclass']         : subclass label (~338 in Animal 1)
  .obs['supertype']        : supertype label (~1,201 in Animal 1)
  .obs['neurotransmitter'] : neurotransmitter identity (Glut/GABA/etc.)
  .obs['cluster']          : integer cluster id

This is the dataset LUNA uses to train its model for the scRNA-seq de novo
reconstruction task (Section 3.3 of the paper).
"""

from __future__ import annotations

import logging
import re
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


_SLICE_RE = re.compile(r"^abc_zhuang_abca1_(?P<section>.+)\.h5ad$")


def _enumerate_slice_files(root: Path) -> List[Tuple[str, Path]]:
    """Find per-slice h5ad files under `root` and parse section label."""
    out: List[Tuple[str, Path]] = []
    for p in sorted(root.iterdir()):
        m = _SLICE_RE.match(p.name)
        if not m:
            continue
        out.append((m["section"], p))
    return out


def load_abc_animal1(
    data_dir: str | Path,
    n_top_hvg: Optional[int] = None,
    normalize: bool = True,
    scale: bool = True,
    min_cells_per_section: int = 100,
    section_filter: Optional[Iterable[str]] = None,
    sections_for_class_vocab: Optional[Iterable[str]] = None,
    label_column: str = "cell_class",
) -> Dict[str, Any]:
    """Load all sections of ABC Animal 1 into one big training set.

    Returns a single split dict (unlike the LUNA cortex loader which returns
    train + test): LUNA's Section 3.3 trains on the full Animal 1, so there's
    no internal held-out split here. Use `--val_fraction` in the trainer to
    carve a validation set out of sections if desired.

    Args:
        data_dir: Per-slice h5ad directory.
        n_top_hvg: Optional HVG selection (seurat_v3 on counts).
        normalize: Apply normalize_total + log1p.
        scale: z-score per gene (scanpy.pp.scale).
        min_cells_per_section: Drop tiny sections.
        section_filter: Optional iterable of section labels to keep
            (e.g. for fast smoke runs).
        sections_for_class_vocab: Sections used to build the cell-class
            integer vocabulary (default: same as section_filter or all).
            Cells whose class is not in this vocab get -1.
        label_column: Which obs column to use as the class label
            (default 'cell_class'; alternatives: 'subclass', 'supertype').

    Returns:
        Dict with:
            gene_expr, coords, section_ids, cell_class (str), cell_class_id,
            section_map, gene_names, class_names, n_classes
    """
    assert SCANPY_AVAILABLE, "scanpy/anndata required for ABC loader"

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    files = _enumerate_slice_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No abc_zhuang_abca1_*.h5ad files under {data_dir}. "
            f"Run scripts/prepare_abc_silver.py first."
        )

    if section_filter is not None:
        section_filter = set(section_filter)
        files = [(s, p) for (s, p) in files if s in section_filter]

    logger.info(f"ABC Animal 1: discovered {len(files)} slice files in {data_dir}")

    return _load_concat(
        files,
        n_top_hvg=n_top_hvg,
        normalize=normalize,
        scale=scale,
        min_cells_per_section=min_cells_per_section,
        sections_for_class_vocab=(
            set(sections_for_class_vocab) if sections_for_class_vocab else None
        ),
        label_column=label_column,
    )


def _load_concat(
    files: List[Tuple[str, Path]],
    n_top_hvg: Optional[int],
    normalize: bool,
    scale: bool,
    min_cells_per_section: int,
    sections_for_class_vocab: Optional[set],
    label_column: str,
) -> Dict[str, Any]:
    import scipy.sparse as sp

    if not files:
        raise ValueError("No section files to load.")

    adatas = []
    section_labels: List[str] = []
    for label, path in files:
        a = ad.read_h5ad(path)
        if a.n_obs < min_cells_per_section:
            logger.info(f"  skipping {label}: {a.n_obs} < min_cells_per_section")
            continue
        adatas.append(a)
        section_labels.append(label)
    if not adatas:
        raise ValueError("All sections fell below min_cells_per_section.")

    logger.info(f"  concatenating {len(adatas)} sections...")
    big = ad.concat(
        adatas, axis=0, join="inner",
        label="_split_section", keys=section_labels, merge="same",
    )
    big.obs["_split_section"] = big.obs["_split_section"].astype(str)
    logger.info(f"  combined: {big.n_obs:,} cells × {big.n_vars:,} genes")

    # Preserve counts so HVG selection (seurat_v3) works correctly.
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
    coords = (
        np.asarray(big.obsm["spatial"], dtype=np.float32)[:, :2]
        if "spatial" in big.obsm
        else np.column_stack([
            big.obs["coord_X"].to_numpy(dtype=np.float32),
            big.obs["coord_Y"].to_numpy(dtype=np.float32),
        ])
    )

    # Cell class label
    class_str = (
        big.obs[label_column].astype(str).to_numpy()
        if label_column in big.obs.columns
        else None
    )
    if class_str is None:
        logger.warning(
            f"  label column {label_column!r} not present in obs; "
            f"cell_class_id will be None."
        )
        class_id = None
        class_names = None
    else:
        # Build class vocabulary
        if sections_for_class_vocab is None:
            vocab_mask = np.ones(big.n_obs, dtype=bool)
        else:
            vocab_mask = np.isin(
                big.obs["_split_section"].astype(str).to_numpy(),
                list(sections_for_class_vocab),
            )
        observed = sorted({str(s) for s in class_str[vocab_mask]
                            if s is not None and s != "nan"})
        cls_to_id = {c: i for i, c in enumerate(observed)}
        class_id = np.array(
            [cls_to_id.get(str(c), -1) for c in class_str], dtype=np.int64
        )
        class_names = observed
        n_unk = int((class_id < 0).sum())
        logger.info(
            f"  class vocabulary ({label_column}): {len(class_names)} classes; "
            f"{n_unk:,} unknowns (set to -1)."
        )

    # Dense gene-expression matrix
    if sp.issparse(big.X):
        X = np.asarray(big.X.toarray(), dtype=np.float32)
    else:
        X = np.asarray(big.X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Contiguous section ids
    label_to_id = {lab: i for i, lab in enumerate(section_labels)}
    section_ids = np.array(
        [label_to_id[s] for s in big.obs["_split_section"]], dtype=np.int64
    )
    section_map = {i: lab for lab, i in label_to_id.items()}

    logger.info(
        f"  final: cells={X.shape[0]:,} genes={X.shape[1]:,} sections={len(label_to_id)}"
    )

    return {
        "gene_expr": X,
        "coords": coords,
        "section_ids": section_ids,
        "cell_class": class_str,
        "cell_class_id": class_id,
        "section_map": section_map,
        "gene_names": list(big.var_names),
        "class_names": class_names,
        "n_classes": len(class_names) if class_names else 0,
    }

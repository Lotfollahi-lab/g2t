#!/usr/bin/env python
"""Precompute a per-cell-type Gaussian Mixture Model prior for the FM
``x_1`` (noise endpoint), as an alternative to the default ``N(0, I)``.

The output is a pickled dict with three things:

  * ``gmms``: ``{cell_class_int → {means, covariances, weights, K}}``
    — per-cell-type sklearn-compatible GMM parameters fit on per-slice-
    normalized 2D positions.
  * ``default_scale``: median per-slice bounding-box scale across all
    training slices. Used at inference time (when we don't have a
    specific test slice's positions to derive its own scale).
  * ``slice_scales``: ``{slice_filename → scale}`` for diagnostics —
    not required at training/inference but useful to inspect the
    spread of slice sizes.

Rotation-invariance design
--------------------------
We DELIBERATELY do NOT rotate slices into a common canonical frame
before fitting. Each slice contributes positions in its own native
orientation; the GMM averages over orientations, so the resulting
distribution is rotation-invariant (more precisely: the GMM has no
preferred direction — under any global rotation R applied to the
input data, the fitted GMM would be unchanged in shape, only
rotated correspondingly).

This means:
  * The prior captures rotation-INVARIANT structure of each cell
    type's spatial layout (radial profile, anisotropy magnitude).
  * The prior does NOT capture rotation-SPECIFIC structure
    (e.g., "L1 always at the top of the cortex").
  * Compatible with ``train.augment_rotation=true`` — that knob
    rotates x_0 per step, the prior happily samples in any frame.
  * Compatible with the EDM head's internal Procrustes alignment.

What we DO normalize
--------------------
  * Center: per-slice centroid subtracted before fitting.
  * Scale: divide by half the bounding-box diagonal so each slice's
    positions live roughly in ``[-1, 1]²``. This is critical because
    different slices have different physical extents; without
    per-slice scaling, the GMM would have to span the entire union
    of scales and would be dominated by the largest slices.

The per-slice scale is saved (median across slices = default_scale)
so that at training/inference we can rescale GMM samples back to the
position units the rest of the model expects.

Usage
-----
::

    python scgg/scripts/precompute_prior_pool.py \\
        --silver_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --out_path   /nfs/team361/sb75/DATASETS/silver/mmc_luna/prior_pool_k8.pkl \\
        --gmm_k 8

Then at training time::

    bash submit_pipeline.sh --method scgg --wandb_run_name scgg_mmc_gmmprior \\
        --override "model.flow_matching.prior_mode=empirical_gmm \\
                    model.flow_matching.prior_pool_path=/nfs/team361/sb75/DATASETS/silver/mmc_luna/prior_pool_k8.pkl"
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


logger = logging.getLogger("precompute_prior_pool")


def _silver_files(silver_dir: Path) -> List[Path]:
    """Discover ``*_train.h5ad`` files; fall back to all ``*.h5ad`` if
    none are found (matches precompute_embeddings.py convention)."""
    train = sorted(silver_dir.glob("*_train.h5ad"))
    if train:
        return train
    return sorted(silver_dir.glob("*.h5ad"))


def _extract_positions(adata) -> np.ndarray:
    """Pull 2D positions from an AnnData. Tries, in order:

      1. ``adata.obsm['X_spatial']`` — standard scanpy convention.
      2. ``adata.obsm['spatial']`` — also common.
      3. ``adata.obs[['x', 'y']]`` — LUNA's CSV-derived layout.

    Returns ``(n_obs, 2)`` float array.
    """
    if "X_spatial" in adata.obsm:
        pos = np.asarray(adata.obsm["X_spatial"], dtype=np.float64)
    elif "spatial" in adata.obsm:
        pos = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    elif {"x", "y"}.issubset(set(adata.obs.columns)):
        pos = adata.obs[["x", "y"]].to_numpy(dtype=np.float64)
    else:
        raise RuntimeError(
            "Cannot find 2D positions on AnnData. Tried obsm['X_spatial'], "
            "obsm['spatial'], obs[['x','y']]. Available obsm keys: "
            f"{list(adata.obsm.keys())}; obs columns: "
            f"{list(adata.obs.columns)}."
        )
    if pos.shape[1] != 2:
        raise RuntimeError(
            f"Expected (n, 2) positions; got shape {pos.shape}. "
            "Only 2D positions are supported."
        )
    return pos


def _extract_cell_classes(adata, field: str) -> np.ndarray:
    """Pull integer cell-class labels from an AnnData. The training
    pipeline's DataHolder uses integer indices in ``cell_class``;
    we match that by reading ``adata.obs[field]`` (categorical or
    numeric) and converting to integer indices via the global ordering.

    Returns ``(n_obs,)`` int array — these are the GMM keys.
    """
    if field not in adata.obs.columns:
        raise RuntimeError(
            f"Cell-class field {field!r} not in adata.obs. "
            f"Available: {list(adata.obs.columns)}."
        )
    series = adata.obs[field]
    # If already numeric / integer-valued, use as-is. Otherwise
    # convert via the pandas Categorical codes (which match the
    # ordering scgg's data pipeline uses).
    if series.dtype.kind in "iu":
        return series.to_numpy(dtype=np.int64)
    cat = series.astype("category")
    return cat.cat.codes.to_numpy(dtype=np.int64)


def _per_slice_normalize(positions: np.ndarray) -> tuple:
    """Center + scale a slice's positions. Returns (positions_norm, scale).

    scale = half the bounding-box diagonal — robust to outliers vs.
    using max-distance from centroid. After dividing by ``scale``,
    positions are roughly in ``[-1, 1]²``.
    """
    centered = positions - positions.mean(axis=0, keepdims=True)
    bbox_diag = float(
        np.linalg.norm(centered.max(axis=0) - centered.min(axis=0))
    )
    scale = max(0.5 * bbox_diag, 1e-8)  # guard against zero-extent slices
    return centered / scale, scale


def _fit_gmm_per_class(
    positions_normalized: np.ndarray,
    cell_classes: np.ndarray,
    k: int,
    min_samples_for_full_k: int,
    seed: int,
) -> Dict[int, dict]:
    """Fit one GMM per unique cell class. If a class has too few samples
    to support the requested K (heuristic: ``5*K`` samples), fall back
    to K=1 (single Gaussian) for that class — better than failing.
    """
    from sklearn.mixture import GaussianMixture

    out: Dict[int, dict] = {}
    for cls in sorted(np.unique(cell_classes).tolist()):
        mask = cell_classes == cls
        pos_cls = positions_normalized[mask]
        if pos_cls.shape[0] < 4:
            logger.warning(
                f"  cell_class={cls}: only {pos_cls.shape[0]} samples, "
                "skipping (need ≥4 for GMM). Cells of this class will "
                "fall back to N(0, I) at sample time."
            )
            continue
        k_eff = k if pos_cls.shape[0] >= min_samples_for_full_k else 1
        if k_eff != k:
            logger.info(
                f"  cell_class={cls}: only {pos_cls.shape[0]} samples "
                f"(<{min_samples_for_full_k}); using K=1 instead of K={k}."
            )
        gmm = GaussianMixture(
            n_components=k_eff,
            covariance_type="full",
            random_state=seed,
            reg_covar=1e-6,
        )
        gmm.fit(pos_cls)
        out[int(cls)] = {
            "means":       gmm.means_.astype(np.float32),         # (K, 2)
            "covariances": gmm.covariances_.astype(np.float32),   # (K, 2, 2)
            "weights":     gmm.weights_.astype(np.float32),       # (K,)
            "K":           int(k_eff),
            "n_train_samples": int(pos_cls.shape[0]),
        }
        logger.info(
            f"  cell_class={cls}: K={k_eff}, fit on {pos_cls.shape[0]} "
            f"normalized samples. Means range "
            f"[{gmm.means_.min():.2f}, {gmm.means_.max():.2f}]."
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--silver_dir", required=True, help="Directory with training h5ads.")
    p.add_argument("--out_path",   required=True, help="Output .pkl path.")
    p.add_argument("--gmm_k", type=int, default=8,
                   help="Number of GMM components per cell class (default 8).")
    p.add_argument("--cell_class_field", default="cell_class",
                   help="adata.obs column for cell-class labels.")
    p.add_argument("--min_samples_for_full_k", type=int, default=None,
                   help="Minimum samples required to fit with --gmm_k "
                        "components; otherwise fall back to K=1. Default: "
                        "5 * gmm_k.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.min_samples_for_full_k is None:
        args.min_samples_for_full_k = 5 * args.gmm_k

    try:
        import anndata as ad
    except ImportError as e:
        raise SystemExit(
            f"anndata is required. Install via `pip install anndata`. "
            f"Underlying error: {e}"
        )

    silver = Path(args.silver_dir).resolve()
    files = _silver_files(silver)
    if not files:
        logger.error(f"No h5ad files found in {silver}")
        return 1
    logger.info(f"Found {len(files)} h5ad files in {silver}")

    # Per-slice normalize + concatenate.
    all_pos: List[np.ndarray] = []
    all_cls: List[np.ndarray] = []
    slice_scales: Dict[str, float] = {}
    for path in files:
        adata = ad.read_h5ad(path)
        pos = _extract_positions(adata)
        cls = _extract_cell_classes(adata, args.cell_class_field)
        if pos.shape[0] != cls.shape[0]:
            raise RuntimeError(
                f"{path.name}: position count {pos.shape[0]} != "
                f"cell_class count {cls.shape[0]}."
            )
        pos_norm, scale = _per_slice_normalize(pos)
        all_pos.append(pos_norm)
        all_cls.append(cls)
        slice_scales[path.name] = scale
        logger.info(
            f"  {path.name}: {pos.shape[0]} cells, scale={scale:.3g}"
        )
    positions_normalized = np.concatenate(all_pos, axis=0)
    cell_classes = np.concatenate(all_cls, axis=0)
    logger.info(
        f"Total: {positions_normalized.shape[0]} cells across "
        f"{len(files)} slices."
    )

    # Fit per-cell-class GMM.
    logger.info(f"Fitting per-class GMM with K={args.gmm_k} ...")
    gmms = _fit_gmm_per_class(
        positions_normalized=positions_normalized,
        cell_classes=cell_classes,
        k=args.gmm_k,
        min_samples_for_full_k=args.min_samples_for_full_k,
        seed=args.seed,
    )
    logger.info(f"Fit GMMs for {len(gmms)} unique cell classes.")

    # Aggregate stats.
    default_scale = float(np.median(list(slice_scales.values())))
    logger.info(f"default_scale (median slice scale) = {default_scale:.3g}")

    # Write the pool.
    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pool = {
        "gmms":          gmms,
        "default_scale": default_scale,
        "slice_scales":  slice_scales,
        "meta": {
            "gmm_k_requested": args.gmm_k,
            "min_samples_for_full_k": args.min_samples_for_full_k,
            "n_slices": len(files),
            "n_cells_total": int(positions_normalized.shape[0]),
            "n_classes_fit": len(gmms),
            "cell_class_field": args.cell_class_field,
            "normalize": "per-slice center + scale-by-half-bbox-diagonal",
            "rotation_alignment": (
                "NONE — slices contribute positions in their native "
                "orientations, so the resulting GMM is rotation-averaged "
                "and the prior is rotation-invariant in distribution."
            ),
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(pool, f)
    logger.info(f"Wrote prior pool to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

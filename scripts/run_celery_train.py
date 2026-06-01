"""run_celery_train.py — train the CeLEry baseline for the LUNA benchmark.

CeLEry (Zhang et al. 2023, Nat Commun) is the supervised coordinate-
regression baseline LUNA's Fig 3 benchmarks against on the MMC cortex
dataset. Per the **LUNA paper Supplementary Note 2** ("Baselines"), the
benchmark protocol for CeLEry specifically is:

  * "Notably, CeLEry is the only method capable of processing multiple
    slices simultaneously. Consequently, we trained CeLEry using all
    33 slices from the first animal and tested it on all 31 slices
    from the second animal."

So unlike novosparc / tangram / cytospace (which are trained per-
reference-slice), CeLEry gets **ONE global training run on the
concatenated training set**. This script implements that protocol.

Slice concatenation gotcha
==========================
Each silver h5ad has its own coordinate frame (the slice's own
micron-scale (x, y) bounding box). CeLEry's MLP learns to map gene
expression → 2D coords; if we concatenated training slices without
normalising, the model would see contradictory targets (cells of the
same cell type at very different (x,y) values across slices because
each slice's frame is anchored to a different origin).

Fix: **per-slice min-max normalise each training slice's coords to
[0, 1] before concatenating**. CeLEry's sigmoid output natively lives
in [0, 1], so this also matches the model's output assumption. The
per-slice bounding boxes are stashed in ``manifest.json`` so inference
can inverse-transform predictions to the test slice's own coord scale
(needed for RSSD on the original-scale truth; Spearman is rank-based
so the inverse is cosmetic for that metric).

Caveat — what LUNA actually used
=================================
Supp Note 2 doesn't spell out how LUNA normalised coords across the 33
training slices before concatenating. Per-slice [0,1] is the natural
choice given CeLEry's sigmoid output range, but if LUNA used a
different convention (e.g. shared affine alignment) the numbers will
still differ.

Outputs (the artifact contract every pipeline in this repo satisfies):
    metrics.csv                       — single-row training summary
    per_slice_metrics.csv             — one row per test slice (cell counts only)
    aggregate_metrics.json            — JSON summary
    runtime.csv                       — phase-by-phase timing
    compute_requirements.csv          — single-row resource summary
    config.yaml                       — invocation snapshot
    model.obj                         — CeLEry pickle (single global model)
    manifest.json                     — train slice list + per-slice bboxes
    best_model.ckpt                   — symlink → model.obj (pipeline convention)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Reuse helpers from run_luna_train.py — see the docstring there.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
import run_luna_train as _luna  # noqa: E402

_RuntimeTracker = _luna._RuntimeTracker
_discover_split_files = _luna._discover_split_files
_section_label_from_filename = _luna._section_label_from_filename
_per_cell_spearman_median = _luna._per_cell_spearman_median
_plot_pred_vs_truth = _luna._plot_pred_vs_truth


logger = logging.getLogger("celery_train")


# ---------------------------------------------------------------------------
# Constants matching run_luna_train.py's conventions
# ---------------------------------------------------------------------------

ENGINE_OUTPUT_SUBDIR = "celery_model"

_DEFAULT_ARTIFACTS_ROOT = Path(
    "/nfs/team361/sb75/scgg-reproducibility/artifacts"
)
_ARTIFACTS_ROOT = Path(
    os.environ.get(
        "CELERY_ARTIFACTS_ROOT",
        os.environ.get(
            "SCGG_ARTIFACTS_ROOT",
            os.environ.get(
                "LUNA_ARTIFACTS_ROOT",
                str(_DEFAULT_ARTIFACTS_ROOT),
            ),
        ),
    )
)


# ---------------------------------------------------------------------------
# Per-slice loading + coord normalization
# ---------------------------------------------------------------------------

def _load_h5ad_for_celery(
    h5ad_path: Path,
    coord_keys: Tuple[str, str] = ("coord_X", "coord_Y"),
):
    """Read a silver h5ad and arrange it for CeLEry, returning ORIGINAL coords.

    Silver h5ads in this repo store spatial coordinates in
    ``adata.obsm['spatial']`` (the canonical scanpy layout) by
    convention; some older or method-specific h5ads instead carry
    them as ``obs['coord_X'] / obs['coord_Y']`` columns. We prefer
    obsm['spatial'] and fall back to obs columns.

    CeLEry's ``Fit_cord`` / ``Predict_cord`` reads coordinates from
    the first two columns of ``data_train.obs`` POSITIONALLY (it
    ignores column names — see CeLEry/datasetgenemap.py) AND it casts
    the ENTIRE obs DataFrame to float32. So obs MUST contain ONLY the
    two coord columns; ``cell_class`` (which has string values like
    'Other') is stashed in ``adata.uns['_celery_cell_class']`` for
    later recovery in metadata_true.csv writing.

    Returns:
        (adata, x_min, x_max, y_min, y_max)
        — adata.obs has been reduced to [coord_X, coord_Y]. The four
        scalars are the bounding box of the ORIGINAL (pre-normalisation)
        coordinate cloud. Caller decides whether to normalise.
    """
    import scanpy as sc

    adata = sc.read(h5ad_path)
    obs = adata.obs.copy()

    if "spatial" in adata.obsm_keys():
        spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        if spatial.ndim != 2 or spatial.shape[1] < 2:
            raise ValueError(
                f"{h5ad_path}: obsm['spatial'] has unexpected shape "
                f"{spatial.shape}; expected (N, 2) or (N, >=2)."
            )
        x = spatial[:, 0]
        y = spatial[:, 1]
    elif coord_keys[0] in obs.columns and coord_keys[1] in obs.columns:
        x = obs[coord_keys[0]].to_numpy(dtype=np.float64)
        y = obs[coord_keys[1]].to_numpy(dtype=np.float64)
    else:
        raise KeyError(
            f"{h5ad_path}: no spatial coordinates found. Expected "
            f"adata.obsm['spatial'] OR adata.obs[{coord_keys[0]!r}/{coord_keys[1]!r}]. "
            f"Have obs columns: {list(obs.columns)}, "
            f"obsm keys: {list(adata.obsm_keys())}"
        )

    # Stash cell_class (string) before stripping obs.
    if "cell_class" in obs.columns:
        cell_class_array = obs["cell_class"].astype(str).to_numpy()
    else:
        cell_class_array = None

    # Replace obs with just the two numeric coord cols (CeLEry casts all
    # of obs to float32; non-numeric cols would raise ValueError).
    obs_new = pd.DataFrame(
        {coord_keys[0]: x, coord_keys[1]: y},
        index=obs.index,
    )
    adata.obs = obs_new
    if cell_class_array is not None:
        adata.uns["_celery_cell_class"] = cell_class_array

    return adata, float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _normalize_coords_to_unit(
    adata,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    coord_keys: Tuple[str, str] = ("coord_X", "coord_Y"),
):
    """In-place [0,1] min-max normalise the coord columns of adata.obs.

    Uses the supplied bounding box (not recomputed from adata.obs) so
    the caller has a single source of truth that gets saved in the
    manifest. CeLEry's sigmoid output lives in [0,1] so post-fit
    predictions are in the same normalised frame.

    Edge case: a degenerate axis (x_min == x_max) would divide by
    zero. Handled by setting that axis to 0.5 (centroid) — CeLEry
    can still train, the affected coordinate just provides no signal.
    """
    obs = adata.obs.copy()
    dx = x_max - x_min
    dy = y_max - y_min
    obs[coord_keys[0]] = (
        (obs[coord_keys[0]].to_numpy() - x_min) / dx if dx > 0 else 0.5
    )
    obs[coord_keys[1]] = (
        (obs[coord_keys[1]].to_numpy() - y_min) / dy if dy > 0 else 0.5
    )
    adata.obs = obs


def _invert_celery_normalisation(
    pred_normed: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> np.ndarray:
    """Map CeLEry's [0,1] sigmoid output back to a target coord scale.

    For multi-slice CeLEry, predictions come out in the "shared
    normalised [0,1] frame" used during training (since every train
    slice was normalised to [0,1] before concatenation). At inference
    we want pred and true on the SAME scale so RSSD's Kabsch
    alignment can do its job — so we inverse-transform using the
    TEST slice's own bounding box. Spearman is rank-based so the
    inverse is cosmetic for that metric.
    """
    pred = np.asarray(pred_normed, dtype=np.float64)
    out = np.empty_like(pred)
    out[:, 0] = pred[:, 0] * (x_max - x_min) + x_min
    out[:, 1] = pred[:, 1] * (y_max - y_min) + y_min
    return out


# ---------------------------------------------------------------------------
# Multi-slice gene alignment + concatenation
# ---------------------------------------------------------------------------

def _common_genes_across_slices(adatas: List) -> List[str]:
    """Intersection of var_names across all slices, lex-sorted for determinism."""
    gene_sets = [set(a.var_names) for a in adatas]
    common = sorted(set.intersection(*gene_sets))
    if not common:
        raise ValueError(
            "no overlapping genes across the training slices — "
            "CeLEry needs a shared gene panel for multi-slice training."
        )
    return common


def _concat_normalized_slices(
    train_files: List[Path],
):
    """Load every train slice, normalise coords to [0,1], **per-slice
    z-score gene expression**, then concatenate.

    BUG FIX (2026-05-31): previously we concatenated raw counts and
    z-scored GLOBALLY across the 165k+ cells of all training slices
    pooled. That diverged from inference behaviour — where each test
    slice is z-scored INDEPENDENTLY with its own ~5k cells' mean/std
    via ``cel.get_zscore``. Different z-score scopes give train and
    test inputs from different distributions, which is why
    multi-slice CeLEry was producing WORSE results than the old
    per-reference protocol.

    Fix: z-score each train slice independently BEFORE concat. The
    concatenated AnnData then has uniformly-z-scored features per
    slice — same distributional shape every test slice will see at
    inference. Batch effects across slices are partially absorbed
    (each slice anchored to its own mean=0, std=1), matching how
    each test slice is also anchored to its own mean=0, std=1.

    Returns:
        (adata_concat, slice_bboxes)
        — adata_concat: AnnData with z-scored X stacked across slices,
          obs containing only normalised coord_X / coord_Y as cols 0/1.
        — slice_bboxes: list of dicts {slice_name, path, x_min, x_max,
          y_min, y_max, n_cells} for the manifest.
    """
    import anndata as ad
    import CeLEry as cel  # type: ignore

    adatas = []
    bboxes = []
    for p in train_files:
        slice_label = _section_label_from_filename(p)
        logger.info(f"  loading {slice_label}: {p.name}")
        a, x_min, x_max, y_min, y_max = _load_h5ad_for_celery(p)
        _normalize_coords_to_unit(a, x_min, x_max, y_min, y_max)
        # PER-SLICE z-score — see the docstring above for why this is
        # critical for train↔test distributional parity.
        cel.get_zscore(a)
        adatas.append(a)
        bboxes.append({
            "slice_name": slice_label,
            "path": str(p),
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "n_cells": int(a.n_obs),
        })

    # Restrict to common genes BEFORE concat — anndata's concat would
    # require this anyway (default join="inner"), but doing it
    # explicitly gives us a clean log line + deterministic gene order.
    common = _common_genes_across_slices(adatas)
    logger.info(
        f"  common genes across {len(adatas)} train slice(s): {len(common)}"
    )
    adatas = [a[:, common].copy() for a in adatas]

    # Concatenate. The X matrices are already z-scored at this point,
    # so the concat is just a vertical stack.
    adata_concat = ad.concat(adatas, axis=0, join="inner", merge="first")
    adata_concat = adata_concat[:, common].copy()
    logger.info(
        f"  concatenated: {adata_concat.n_obs} cells × "
        f"{adata_concat.n_vars} genes (per-slice z-scored)"
    )
    return adata_concat, bboxes


# ---------------------------------------------------------------------------
# Global CeLEry training
# ---------------------------------------------------------------------------

def _train_global_model(
    train_files: List[Path],
    out_dir: Path,
    *,
    num_epochs_max: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: List[int],
    num_workers: int,
    seed: int,
) -> Dict[str, object]:
    """Train ONE CeLEry model on all training slices concatenated.

    Mirrors the LUNA-paper Supp Note 2 protocol for CeLEry. Returns a
    dict with training stats; checkpoint is written to
    ``out_dir/model.obj`` and ``out_dir/manifest.json``.
    """
    import CeLEry as cel  # type: ignore

    # ---- 1. Build the concatenated training AnnData ----
    # _concat_normalized_slices already runs cel.get_zscore PER SLICE
    # before concatenating, so adata_concat.X is already z-scored.
    # We deliberately DON'T call get_zscore again on the concat —
    # see the comment in _concat_normalized_slices for why per-slice
    # scoping matches inference behaviour.
    adata_concat, slice_bboxes = _concat_normalized_slices(train_files)

    # ---- 3. Save manifest BEFORE training so a fit-crash still leaves
    # the bbox metadata on disk for debugging.
    manifest = {
        "training_mode": "multi_slice",
        "n_train_slices": len(train_files),
        "n_train_cells_total": int(adata_concat.n_obs),
        "n_genes": int(adata_concat.n_vars),
        "var_names": list(adata_concat.var_names),
        "slice_bboxes": slice_bboxes,
        "celery_hparams": {
            "num_epochs_max": num_epochs_max,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_dims": list(hidden_dims),
            "num_workers": num_workers,
            "seednum": seed,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- 4. Fit ----
    logger.info("=" * 60)
    logger.info(f"Training CeLEry on {adata_concat.n_obs} cells × "
                f"{adata_concat.n_vars} genes")
    logger.info(f"  hparams: lr={learning_rate}, batch={batch_size}, "
                f"epochs={num_epochs_max}, hidden={hidden_dims}")
    logger.info("=" * 60)
    t0 = time.perf_counter()
    cel.Fit_cord(
        data_train=adata_concat,
        hidden_dims=list(hidden_dims),
        num_epochs_max=int(num_epochs_max),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        initial_learning_rate=float(learning_rate),
        path=str(out_dir),
        filename="model",  # → <out_dir>/model.obj
        seednum=int(seed),
    )
    dt = time.perf_counter() - t0
    logger.info(f"  Trained in {dt:.1f}s")

    ckpt = out_dir / "model.obj"
    if not ckpt.exists():
        raise RuntimeError(
            f"CeLEry training completed but no checkpoint at {ckpt} — "
            f"this usually means Fit_cord raised silently. Check stderr."
        )

    return {
        "train_seconds": dt,
        "n_train_slices": len(train_files),
        "n_train_cells_total": int(adata_concat.n_obs),
        "n_genes": int(adata_concat.n_vars),
    }


# ---------------------------------------------------------------------------
# Per-reference training mode (the LUNA main-text protocol's secondary
# mode — for novosparc / tangram / cytospace; LUNA used multi_slice
# for CeLEry specifically, but the per-reference mode is also useful
# for comparison and was our original implementation).
# ---------------------------------------------------------------------------

@dataclass
class _RefAssignment:
    """One row of the per-test-slice reference assignment table.

    Captured in ``celery_models/<test_slice>/manifest.json`` so reviewers
    can audit exactly which (ref, test) pair each per-reference CeLEry
    model was trained on. Reproducible from ``--seed``.
    """
    test_slice: str
    test_path: str
    ref_slice: str
    ref_path: str


def _assign_references(
    train_files: List[Path],
    test_files: List[Path],
    seed: int,
) -> List[_RefAssignment]:
    """Pick a training slice for each test slice (seeded with-replacement).

    Mirrors LUNA's "randomly selected a single slice from the training
    mouse to serve as the reference, repeating the procedure for each
    slice of the testing mouse" (paper §4). With-replacement sampling
    handles the case where the train mouse has fewer slices than the
    test mouse, though for MMC both have 30+ slices so it's
    effectively without-replacement most of the time.
    """
    if not train_files:
        raise ValueError("no training slices — cannot pick a CeLEry reference")
    if not test_files:
        raise ValueError("no test slices — nothing to evaluate")
    rng = np.random.default_rng(seed)
    train_indices = rng.integers(0, len(train_files), size=len(test_files))
    return [
        _RefAssignment(
            test_slice=_section_label_from_filename(t),
            test_path=str(t),
            ref_slice=_section_label_from_filename(train_files[int(train_indices[i])]),
            ref_path=str(train_files[int(train_indices[i])]),
        )
        for i, t in enumerate(test_files)
    ]


def _train_one_slice_per_reference(
    assignment: _RefAssignment,
    ckpt_dir: Path,
    *,
    num_epochs_max: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: List[int],
    num_workers: int,
    seed: int,
) -> Dict[str, object]:
    """Train ONE CeLEry model on the reference→test pair for one test slice.

    Saves ``model.obj`` + ``manifest.json`` under ``ckpt_dir/<test_slice>/``.
    """
    import CeLEry as cel  # type: ignore

    slice_dir = ckpt_dir / assignment.test_slice
    slice_dir.mkdir(parents=True, exist_ok=True)

    ref_path = Path(assignment.ref_path)
    qry_path = Path(assignment.test_path)

    logger.info(f"    loading ref  : {ref_path.name}")
    adata_ref, x_min, x_max, y_min, y_max = _load_h5ad_for_celery(ref_path)
    logger.info(f"    loading qry  : {qry_path.name}")
    adata_qry, _, _, _, _ = _load_h5ad_for_celery(qry_path)

    # Restrict to common genes (matches multi-slice's deterministic
    # sorted-intersection approach).
    common = _common_genes_across_slices([adata_ref, adata_qry])
    adata_ref = adata_ref[:, common].copy()
    adata_qry = adata_qry[:, common].copy()
    logger.info(
        f"    ref={adata_ref.n_obs} cells, qry={adata_qry.n_obs} cells, "
        f"common genes={len(common)}"
    )

    # Per-slice z-score (matches what test-time inference does — same
    # bug fix as for multi_slice mode).
    cel.get_zscore(adata_ref)
    cel.get_zscore(adata_qry)

    # Save manifest first so a fit-crash still leaves bbox metadata
    # on disk for debugging.
    manifest = {
        **asdict(assignment),
        "training_mode": "per_reference",
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "n_ref_cells": int(adata_ref.n_obs),
        "n_qry_cells": int(adata_qry.n_obs),
        "n_genes_common": int(adata_ref.n_vars),
        "var_names": list(adata_ref.var_names),
        "celery_hparams": {
            "num_epochs_max": num_epochs_max,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_dims": list(hidden_dims),
            "num_workers": num_workers,
            "seednum": seed,
        },
    }
    (slice_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    t0 = time.perf_counter()
    cel.Fit_cord(
        data_train=adata_ref,
        hidden_dims=list(hidden_dims),
        num_epochs_max=int(num_epochs_max),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        initial_learning_rate=float(learning_rate),
        path=str(slice_dir),
        filename="model",
        seednum=int(seed),
    )
    dt = time.perf_counter() - t0
    logger.info(f"    trained in {dt:.1f}s")

    ckpt = slice_dir / "model.obj"
    if not ckpt.exists():
        raise RuntimeError(
            f"CeLEry training completed but no checkpoint at {ckpt}. "
            f"Check the CeLEry stderr above."
        )

    return {
        "test_slice": assignment.test_slice,
        "ref_slice": assignment.ref_slice,
        "train_seconds": dt,
        "n_ref_cells": int(adata_ref.n_obs),
        "n_qry_cells": int(adata_qry.n_obs),
        "n_genes_common": int(adata_ref.n_vars),
    }


def _train_per_reference(
    train_files: List[Path],
    test_files: List[Path],
    out_dir: Path,
    *,
    num_epochs_max: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: List[int],
    num_workers: int,
    seed: int,
) -> Dict[str, object]:
    """Train N CeLEry models (one per test slice with a random reference).

    The original implementation, restored as the ``per_reference``
    branch of the ``--training_mode`` flag. Each per-test-slice
    model lives at ``out_dir/celery_models/<test_slice>/model.obj``.
    A top-level ``manifest.json`` records the assignments + which
    training_mode this dir was built with (so inference can dispatch
    correctly).
    """
    assignments = _assign_references(train_files, test_files, seed=seed)
    ckpt_dir = out_dir / "celery_models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Per-slice assignment table for reproducibility audits.
    assn_table = pd.DataFrame([asdict(a) for a in assignments])
    assn_table.to_csv(out_dir / "ref_assignments.csv", index=False)
    logger.info(f"  wrote {out_dir / 'ref_assignments.csv'}")

    train_info_rows: List[Dict[str, float]] = []
    for idx, asn in enumerate(assignments, start=1):
        logger.info(
            f"[{idx}/{len(assignments)}] test={asn.test_slice}  "
            f"ref={asn.ref_slice}"
        )
        info = _train_one_slice_per_reference(
            asn,
            ckpt_dir=ckpt_dir,
            num_epochs_max=num_epochs_max,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dims=list(hidden_dims),
            num_workers=num_workers,
            seed=seed,
        )
        train_info_rows.append(info)

    # Top-level manifest — lets inference detect mode without
    # walking the celery_models/ tree.
    (out_dir / "manifest.json").write_text(json.dumps({
        "training_mode": "per_reference",
        "n_train_slices_pool": len(train_files),
        "n_models_trained": len(train_info_rows),
        "n_test_slices": len(test_files),
        "celery_hparams": {
            "num_epochs_max": num_epochs_max,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_dims": list(hidden_dims),
            "num_workers": num_workers,
            "seednum": seed,
        },
    }, indent=2))

    train_info_df = pd.DataFrame(train_info_rows)
    train_info_df.to_csv(out_dir / "celery_train_info.csv", index=False)

    total_train_seconds = float(
        sum(r["train_seconds"] for r in train_info_rows)
    )
    total_train_cells = int(sum(r["n_ref_cells"] for r in train_info_rows))

    return {
        "train_seconds": total_train_seconds,
        "n_train_slices": len(train_files),
        "n_train_cells_total": total_train_cells,
        "n_models_trained": len(train_info_rows),
    }


# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------

def run_benchmark(
    data_dir: Optional[str],
    output_dir: Optional[str] = None,
    seed: int = 0,
    num_epochs_max: int = 500,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    hidden_dims: Tuple[int, ...] = (30, 25, 15),
    num_workers: int = 0,
    training_mode: str = "multi_slice",
    wandb_mode: str = "disabled",
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    run_name: Optional[str] = None,
    exclude_test_files: Optional[List[str]] = None,
    output_subdir: Optional[str] = None,
    run_timestamp: Optional[str] = None,
) -> Dict[str, float]:
    """Train ONE global CeLEry model on all training slices concatenated.

    See module docstring for the protocol justification (LUNA Supp
    Note 2). ``exclude_test_files`` is accepted for cross-pipeline
    parity (forwarded to the inference subprocess); CeLEry's training
    phase always uses ALL training slices regardless.
    """
    if data_dir is None:
        raise ValueError(
            "--data_dir is required: silver h5ad directory with "
            "*_train.h5ad and *_test.h5ad files."
        )
    if len(hidden_dims) != 3:
        raise ValueError(
            f"--hidden_dims must have EXACTLY 3 widths (CeLEry's "
            f"DNN.__init__ assumes a 3-layer MLP); got {len(hidden_dims)}: "
            f"{list(hidden_dims)}"
        )
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"--data_dir not found: {data_path}")

    # ---- TS + output dir resolution (same convention as scgg/luna) ----
    if run_timestamp is not None:
        if not re.fullmatch(r"\d{8}_\d{6}(?:_[A-Za-z0-9]+)?", run_timestamp):
            raise ValueError(
                f"run_timestamp must be YYYYMMDD_HHMMSS or "
                f"YYYYMMDD_HHMMSS_<suffix>; got {run_timestamp!r}"
            )
        run_ts = run_timestamp
    else:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_name = data_path.name
    subdir = output_subdir or ENGINE_OUTPUT_SUBDIR
    out = (
        Path(output_dir).resolve()
        if output_dir
        else _ARTIFACTS_ROOT / dataset_name / subdir / run_ts
    )
    out.mkdir(parents=True, exist_ok=True)

    log_path = out / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w"),
        ],
        force=True,
    )

    logger.info("=" * 60)
    logger.info("CeLEry training-phase (multi-slice global, LUNA protocol)")
    logger.info("=" * 60)
    logger.info(f"  data_dir       : {data_path}")
    logger.info(f"  output_dir     : {out}")
    logger.info(f"  run_ts         : {run_ts}")
    logger.info(f"  seed           : {seed}")
    logger.info(f"  num_epochs_max : {num_epochs_max}")
    logger.info(f"  batch_size     : {batch_size}")
    logger.info(f"  learning_rate  : {learning_rate}")
    logger.info(f"  hidden_dims    : {hidden_dims}")
    logger.info(f"  num_workers    : {num_workers}")
    logger.info("=" * 60)

    tracker = _RuntimeTracker()

    # ---- wandb init ----
    use_wandb = wandb_mode != "disabled"
    wandb_run_obj = None
    if use_wandb:
        try:
            import wandb  # type: ignore
            wandb_run_obj = wandb.init(
                project=wandb_project or "celery",
                name=wandb_run_name or run_name,
                mode=wandb_mode,
                config={
                    "method": "celery",
                    "phase": "training",
                    "training_mode": training_mode,
                    "data_dir": str(data_path),
                    "seed": seed,
                    "num_epochs_max": num_epochs_max,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "hidden_dims": list(hidden_dims),
                    "num_workers": num_workers,
                    "run_timestamp": run_ts,
                    "output_dir": str(out),
                    "tags": ["celery", "training", dataset_name, "multi_slice"],
                },
            )
            logger.info(f"wandb initialised: {wandb_run_obj.url}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"wandb init failed: {e}; continuing without wandb")
            use_wandb = False

    # ---- Discover train + test slices ----
    with tracker.phase("discover_h5ads", flush_to=out / "runtime.csv"):
        train_files = _discover_split_files(data_path, "train")
        test_files = _discover_split_files(data_path, "test")
        # exclude_test_files applies to the INFERENCE phase, not
        # training — but we record the count here for the manifest /
        # metrics.csv.
        if exclude_test_files:
            excluded_set = set(exclude_test_files)
            test_files_kept = [p for p in test_files if p.name not in excluded_set]
            logger.info(
                f"  (note: --exclude_test_files {sorted(excluded_set)} will "
                f"drop {len(test_files) - len(test_files_kept)} test slice(s) "
                f"at inference time)"
            )
        else:
            test_files_kept = test_files
        logger.info(f"  found {len(train_files)} train slice(s)")
        logger.info(f"  found {len(test_files_kept)} test slice(s) "
                    f"(after exclude_test_files)")
        if not train_files:
            raise RuntimeError("no train slices to train on")

    # ---- Validate training mode ----
    training_mode = str(training_mode).lower()
    if training_mode not in ("multi_slice", "per_reference"):
        raise ValueError(
            f"training_mode must be 'multi_slice' or 'per_reference'; "
            f"got {training_mode!r}"
        )
    logger.info(f"  training_mode  : {training_mode}")

    # ---- Train ----
    # ``multi_slice`` (LUNA Supp Note 2 protocol): one global model on
    # all train slices concatenated; best_model.ckpt → model.obj.
    # ``per_reference``: N models, one per test slice, each trained
    # on a randomly-selected single training slice as reference;
    # best_model.ckpt → celery_models/.
    with tracker.phase("training", flush_to=out / "runtime.csv"):
        if training_mode == "multi_slice":
            train_info = _train_global_model(
                train_files=train_files,
                out_dir=out,
                num_epochs_max=num_epochs_max,
                batch_size=batch_size,
                learning_rate=learning_rate,
                hidden_dims=list(hidden_dims),
                num_workers=num_workers,
                seed=seed,
            )
        else:
            train_info = _train_per_reference(
                train_files=train_files,
                test_files=test_files_kept,
                out_dir=out,
                num_epochs_max=num_epochs_max,
                batch_size=batch_size,
                learning_rate=learning_rate,
                hidden_dims=list(hidden_dims),
                num_workers=num_workers,
                seed=seed,
            )

    # ---- best_model.ckpt symlink ----
    # multi_slice → model.obj (single file)
    # per_reference → celery_models/ (directory containing per-test-slice subdirs)
    best_ptr = out / "best_model.ckpt"
    if best_ptr.exists() or best_ptr.is_symlink():
        best_ptr.unlink()
    if training_mode == "multi_slice":
        os.symlink("model.obj", best_ptr)
        logger.info("  pinned best_model.ckpt → model.obj")
    else:
        os.symlink("celery_models", best_ptr)
        logger.info("  pinned best_model.ckpt → celery_models/")

    # ---- Summary CSVs ----
    (out / "config.yaml").write_text(yaml.safe_dump({
        "method": "celery",
        "training_mode": training_mode,
        "data_dir": str(data_path),
        "output_dir": str(out),
        "run_timestamp": run_ts,
        "seed": seed,
        "num_epochs_max": num_epochs_max,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dims": list(hidden_dims),
        "num_workers": num_workers,
        "wandb_mode": wandb_mode,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "exclude_test_files": exclude_test_files or [],
        "n_train_slices": len(train_files),
        "n_test_slices": len(test_files_kept),
    }, default_flow_style=False))

    placeholder_metrics = {
        "method": "celery",
        "training_mode": training_mode,
        "n_train_slices": len(train_files),
        "n_test_slices": len(test_files_kept),
        "n_train_cells_total": int(train_info["n_train_cells_total"]),
        "train_seconds": float(train_info["train_seconds"]),
        "spearman_mean_of_medians": float("nan"),  # filled in by inference
    }
    pd.DataFrame([placeholder_metrics]).to_csv(out / "metrics.csv", index=False)
    # Per-slice CSV is empty-ish at train time; populated by inference.
    pd.DataFrame([
        {"section_label": _section_label_from_filename(p), "n_cells": 0}
        for p in test_files_kept
    ]).to_csv(out / "per_slice_metrics.csv", index=False)
    (out / "aggregate_metrics.json").write_text(json.dumps(
        placeholder_metrics, indent=2, default=str
    ))

    # ---- Compute requirements ----
    peak = tracker.peak_summary()
    tracker.write_compute_requirements_csv(
        out / "compute_requirements.csv",
        method="CeLEry",
        run_timestamp=run_ts,
        extra={
            "training_mode": training_mode,
            "n_train_slices": len(train_files),
            "n_test_slices": len(test_files_kept),
            "n_train_cells_total": int(train_info["n_train_cells_total"]),
            "num_epochs_max": num_epochs_max,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
    )

    logger.info("=" * 60)
    logger.info(f"Training complete. Wrote artifacts to {out}")
    logger.info(f"  peak GPU : {peak.get('peak_gpu_mib', float('nan'))} MiB")
    logger.info(f"  peak RSS : {peak.get('peak_rss_mib', float('nan'))} MiB")
    logger.info("=" * 60)

    # ---- wandb finish ----
    if use_wandb:
        try:
            import wandb  # type: ignore
            summary = {
                "n_train_slices": len(train_files),
                "n_test_slices": len(test_files_kept),
                "n_train_cells_total": int(train_info["n_train_cells_total"]),
                "train_seconds": float(train_info["train_seconds"]),
                "peak_gpu_mib": peak.get("peak_gpu_mib"),
                "peak_rss_mib": peak.get("peak_rss_mib"),
            }
            wandb.log(summary)
            for k, v in summary.items():
                try:
                    wandb.run.summary[k] = v
                except Exception:  # noqa: BLE001
                    pass
            wandb.finish()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"wandb finish failed: {e}")

    return placeholder_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--epochs", type=int, default=500,
        help="num_epochs_max for CeLEry's Fit_cord. CeLEry source default 500.",
    )
    p.add_argument(
        "--batch_size", type=int, default=4,
        help="CeLEry Fit_cord batch_size. CeLEry source default 4. "
             "With multi-slice training and ~165k cells (33 MMC slices "
             "× ~5000), larger batches (32-128) are typically faster "
             "without quality loss — worth tuning.",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="initial_learning_rate. CeLEry source default 1e-3. "
             "LUNA's Supp Note 2 hyperparameter sweep tested "
             "5e-5..1.0 across 9 values; the best for MMC is in "
             "Supp Note 2 (we use the source default here).",
    )
    p.add_argument(
        "--hidden_dims", type=int, nargs="+", default=[30, 25, 15],
        help="3 hidden-layer widths. CeLEry's DNN.__init__ hardcodes "
             "exactly 3 layers; longer lists raise IndexError. "
             "LUNA's Supp Note 2 swept latent dim ∈ {32, 64, 128, 256} "
             "and n_layers ∈ {3, 4, 5, 7, 8, 9, 10} — but >3 layers "
             "requires patching CeLEry's DNN class.",
    )
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--training_mode", default="multi_slice",
        choices=("multi_slice", "per_reference"),
        help="multi_slice: one global model trained on the concat of "
             "all training slices (LUNA Supp Note 2 protocol for "
             "CeLEry). per_reference: N models, one per test slice "
             "with a randomly-selected single training slice as "
             "reference (the protocol LUNA's paper uses for "
             "novosparc/tangram/cytospace — and our original "
             "CeLEry implementation). Default: multi_slice.",
    )
    p.add_argument(
        "--wandb_run_name", "--run_name",
        dest="wandb_run_name", default=None,
    )
    p.add_argument(
        "--wandb_mode", default="disabled",
        choices=("disabled", "online", "offline", "dryrun"),
    )
    p.add_argument("--wandb_project", default=None)
    p.add_argument(
        "--exclude_test_files", default=None,
        help="Comma-separated *_test.h5ad basenames; affects inference only "
             "(CeLEry trains on ALL train slices regardless).",
    )
    p.add_argument("--run_timestamp", default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    exclude_test_files = (
        [s.strip() for s in args.exclude_test_files.split(",") if s.strip()]
        if args.exclude_test_files else None
    )

    try:
        run_benchmark(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            num_epochs_max=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            hidden_dims=tuple(args.hidden_dims),
            num_workers=args.num_workers,
            training_mode=args.training_mode,
            wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project or "celery",
            wandb_run_name=args.wandb_run_name,
            run_name=args.wandb_run_name,
            exclude_test_files=exclude_test_files,
            run_timestamp=args.run_timestamp,
        )
    except Exception:
        logger.exception("CeLEry training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""run_celery_train.py — train the CeLEry baseline for the LUNA benchmark.

CeLEry (Zhang et al. 2023, Nat Commun) is the supervised coordinate-
regression baseline LUNA's Fig 3 benchmarks against on the MMC cortex
dataset. Per the LUNA paper, the benchmark protocol is:

  * Train and test mice are held out (one mouse = train set, the
    other = test set), the same cross-mouse split the LUNA model
    itself uses.
  * CeLEry can only train on a SINGLE reference slice at a time, so
    LUNA repeats the procedure per test slice: for each test slice,
    randomly draw one slice from the training mouse, train CeLEry on
    that one slice, predict positions on the test slice.
  * "All methods used the same seed for random slice selection." We
    honour that here with the standard ``--seed`` flag.

So this script does N independent CeLEry trainings, one per test
slice, with the reference picked from the training pool by a seeded
np.random.Generator. Checkpoints are saved in a tree:

    <output_dir>/celery_models/<test_slice_label>/model.obj
    <output_dir>/celery_models/<test_slice_label>/manifest.json

The manifest captures which reference slice was used + the
normalisation parameters needed to invert CeLEry's internal min-max
scaling at inference time (CeLEry's sigmoid output lives in [0,1],
but Spearman is rank-based so this doesn't actually affect the
headline metric — we save them anyway because RSSD and contact F1
need original-scale coordinates).

Inference (run_celery_inference.py) loads each per-slice checkpoint,
predicts, inverse-transforms, scores, and writes metadata_pred.csv +
metadata_true.csv per slice under

    <output_dir>/luna_run/test_results/<run_name>/celery/<slice_label>/

The directory naming mirrors run_luna_train.py's layout so the
existing compute_extended_metrics.py + plot scripts work unchanged.

Outputs (the contract every pipeline in this repo satisfies):
    metrics.csv                       — single-row summary
    per_slice_metrics.csv             — one row per test slice
    aggregate_metrics.json            — JSON summary
    runtime.csv                       — phase-by-phase timing
    compute_requirements.csv          — single-row resource summary
    config.yaml                       — invocation snapshot
    celery_models/<slice>/model.obj   — per-slice CeLEry pickle
    celery_models/<slice>/manifest.json
    best_model.ckpt                   — convenience symlink → newest model.obj
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
# Reuse helpers from run_luna_train.py:
#   - _RuntimeTracker (timing + GPU/RSS sampling)
#   - _discover_split_files (silver h5ad enumeration)
#   - _section_label_from_filename (canonical slice naming)
#   - _per_cell_spearman_median (per-cell Spearman aggregation)
#   - _plot_pred_vs_truth (side-by-side GT-vs-pred scatter)
# These have no CeLEry-specific behaviour; they're the shared
# benchmarking primitives so all four methods report comparable
# numbers from identical CSV columns.
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

ENGINE_OUTPUT_SUBDIR = "celery_model"  # <artifacts_root>/<dataset>/<this>/<TS>/

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
# Reference-slice picker
# ---------------------------------------------------------------------------

@dataclass
class _RefAssignment:
    """One row of the per-test-slice reference assignment table.

    Captured as JSON in the run's config.yaml so reviewers can audit
    exactly which (train_slice, test_slice) pair each CeLEry model
    was trained on. Reproducible from --seed.
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
    """Pick a training slice for each test slice (seeded).

    Mirrors LUNA's "randomly selected a single slice from the
    training mouse to serve as the reference, repeating the procedure
    for each slice of the testing mouse" (paper §4). The seed
    controls which reference each test slice gets — matching seed
    across CeLEry runs reproduces the exact reference-slice
    assignment, so that's the right knob for the multi-seed sweep.

    With-replacement sampling: when the train mouse has fewer slices
    than the test mouse (rare but possible), reuse training slices.
    """
    if not train_files:
        raise ValueError("no training slices found — cannot pick a CeLEry reference")
    if not test_files:
        raise ValueError("no test slices found — nothing to evaluate")
    rng = np.random.default_rng(seed)
    train_indices = rng.integers(0, len(train_files), size=len(test_files))
    assignments = []
    for test_idx, train_idx in enumerate(train_indices):
        t = test_files[test_idx]
        r = train_files[int(train_idx)]
        assignments.append(_RefAssignment(
            test_slice=_section_label_from_filename(t),
            test_path=str(t),
            ref_slice=_section_label_from_filename(r),
            ref_path=str(r),
        ))
    return assignments


# ---------------------------------------------------------------------------
# CeLEry IO helpers
# ---------------------------------------------------------------------------

def _load_h5ad_for_celery(
    h5ad_path: Path,
    coord_keys: Tuple[str, str] = ("coord_X", "coord_Y"),
):
    """Read a silver h5ad and arrange it for CeLEry.

    Silver h5ads in this repo store spatial coordinates in
    ``adata.obsm['spatial']`` (the canonical scanpy layout) by
    convention; some older or method-specific h5ads instead carry
    them as ``obs['coord_X'] / obs['coord_Y']`` columns. We prefer
    obsm['spatial'] and fall back to obs columns — same precedence
    as run_novosparc_pipeline.py and run_luna_train.py use.

    CeLEry's ``Fit_cord`` / ``Predict_cord`` reads coordinates from
    the first two columns of ``data_train.obs`` POSITIONALLY (it
    ignores column names — see CeLEry/datasetgenemap.py). So we
    INJECT the coords as the first two obs columns (overwriting
    any existing ``coord_X`` / ``coord_Y`` of the same name, if
    they exist) so CeLEry sees them at obs[:, 0:2].

    Args:
        h5ad_path: path to a silver h5ad. Coords must live at one of:
            - ``adata.obsm['spatial']`` (preferred), or
            - ``adata.obs[coord_keys[0]]`` + ``adata.obs[coord_keys[1]]``.
        coord_keys: (x_col, y_col) fallback names + the names used
            for the injected obs columns CeLEry will read positionally.

    Returns:
        (adata, x_min, x_max, y_min, y_max)
        — adata has obs reordered so cols 0/1 are (x, y); the four
        scalars are the bounding-box of the reference's coordinate
        cloud, needed to invert CeLEry's internal min-max scaling
        when we score predictions on the original scale.
    """
    import scanpy as sc  # local import — keeps non-CeLEry callers free of scanpy

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
            f"adata.obsm['spatial'] (preferred) OR "
            f"adata.obs[{coord_keys[0]!r}] + adata.obs[{coord_keys[1]!r}]. "
            f"Have obs columns: {list(obs.columns)}, "
            f"obsm keys: {list(adata.obsm_keys())}"
        )

    # Inject (and overwrite if present) coord_X / coord_Y as obs cols 0/1
    # so CeLEry's positional indexing picks them up. Preserve all other
    # obs columns behind them.
    other_cols = [c for c in obs.columns if c not in coord_keys]
    obs_new = pd.DataFrame(index=obs.index)
    obs_new[coord_keys[0]] = x
    obs_new[coord_keys[1]] = y
    for c in other_cols:
        obs_new[c] = obs[c].values
    adata.obs = obs_new

    return adata, float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _align_genes(adata_ref, adata_qry):
    """Restrict both AnnDatas to the intersection of var_names.

    CeLEry feeds the genes positionally to its MLP — train and test
    AnnDatas MUST have identical gene order or the model will be
    fitted on (gene_i_in_ref) and predicted on (gene_i_in_qry) which
    are entirely different signals. We take the intersection and
    sort lexicographically so the order is deterministic across
    runs (sets in Python are insertion-ordered but the intersection
    operator returns an arbitrary order).
    """
    common = sorted(set(adata_ref.var_names) & set(adata_qry.var_names))
    if not common:
        raise ValueError(
            "no overlapping genes between reference and query AnnDatas"
        )
    return adata_ref[:, common].copy(), adata_qry[:, common].copy()


def _invert_celery_normalisation(
    pred_normed: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> np.ndarray:
    """Map CeLEry's [0,1] sigmoid output back to the reference's coordinate scale.

    CeLEry's ``datasetgenemap.py`` does:
        cordx_norm = (cordx - xmin) / (xmax - xmin)
    with a ±1 buffer on the min/max bounds. Inverse:
        cordx = cordx_norm * (xmax - xmin) + xmin
    We don't replicate the ±1 buffer because:
      (a) Spearman is rank-based — the buffer is a rank-preserving
          affine, so it cancels out.
      (b) RSSD uses Kabsch alignment which absorbs any global
          translation + scale, so it cancels too.
    The buffer only matters if a downstream consumer cares about
    absolute Euclidean error on the same scale as the reference,
    which none of LUNA's reported metrics do.
    """
    pred = np.asarray(pred_normed, dtype=np.float64)
    out = np.empty_like(pred)
    out[:, 0] = pred[:, 0] * (x_max - x_min) + x_min
    out[:, 1] = pred[:, 1] * (y_max - y_min) + y_min
    return out


# ---------------------------------------------------------------------------
# Per-slice CeLEry training
# ---------------------------------------------------------------------------

def _train_one_slice(
    assignment: _RefAssignment,
    ckpt_dir: Path,
    *,
    num_epochs_max: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: List[int],
    num_workers: int,
    seed: int,
) -> Dict[str, float]:
    """Train ONE CeLEry model on the reference→test pair for one test slice.

    Saves ``model.obj`` (the CeLEry pickle) + ``manifest.json`` (the
    reference-slice metadata + normalisation params) under
    ``ckpt_dir/<test_slice>/``.

    Returns a runtime info dict with: train_seconds, n_ref_cells,
    n_qry_cells, n_genes_common.
    """
    import CeLEry as cel  # type: ignore  # local import — env-gated

    slice_dir = ckpt_dir / assignment.test_slice
    slice_dir.mkdir(parents=True, exist_ok=True)

    ref_path = Path(assignment.ref_path)
    qry_path = Path(assignment.test_path)

    logger.info(f"    loading ref  : {ref_path.name}")
    adata_ref, x_min, x_max, y_min, y_max = _load_h5ad_for_celery(ref_path)
    logger.info(f"    loading qry  : {qry_path.name}")
    adata_qry, _, _, _, _ = _load_h5ad_for_celery(qry_path)

    adata_ref, adata_qry = _align_genes(adata_ref, adata_qry)
    n_genes = adata_ref.n_vars
    n_ref = adata_ref.n_obs
    n_qry = adata_qry.n_obs
    logger.info(
        f"    ref={n_ref} cells, qry={n_qry} cells, common genes={n_genes}"
    )

    # CeLEry expects z-scored features per the paper's preprocessing.
    cel.get_zscore(adata_ref)
    cel.get_zscore(adata_qry)

    # Save normalisation params first — useful even if training
    # subsequently fails (we can re-attempt with a fresh model.obj
    # while keeping the manifest stable).
    manifest = {
        **asdict(assignment),
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "n_ref_cells": n_ref,
        "n_qry_cells": n_qry,
        "n_genes_common": n_genes,
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
        filename="model",  # → <slice_dir>/model.obj
        seednum=int(seed),
    )
    dt = time.perf_counter() - t0
    logger.info(f"    trained in {dt:.1f}s")

    ckpt = slice_dir / "model.obj"
    if not ckpt.exists():
        raise RuntimeError(
            f"CeLEry training completed but no checkpoint at {ckpt} "
            f"— this usually means Fit_cord raised silently. Check "
            f"the CeLEry stderr above."
        )

    return {
        "test_slice": assignment.test_slice,
        "ref_slice": assignment.ref_slice,
        "train_seconds": dt,
        "n_ref_cells": n_ref,
        "n_qry_cells": n_qry,
        "n_genes_common": n_genes,
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
    wandb_mode: str = "disabled",
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    run_name: Optional[str] = None,
    exclude_test_files: Optional[List[str]] = None,
    output_subdir: Optional[str] = None,
    run_timestamp: Optional[str] = None,
) -> Dict[str, float]:
    """Train one CeLEry model per test slice; return the headline metric.

    Mirrors ``run_luna_train.run_benchmark`` in signature where it
    makes sense; arguments specific to LUNA's vendored engine (Hydra
    overrides, n_inference_samples, etc.) are omitted because CeLEry
    doesn't have those concepts.
    """
    if data_dir is None:
        raise ValueError(
            "--data_dir is required for CeLEry training: silver h5ad "
            "directory containing *_train.h5ad and *_test.h5ad files."
        )
    # CeLEry's DNN class hardcodes a 3-layer MLP (it indexes
    # hidden_dims[0], [1], [2] directly in CeLEry/DNN.py). Passing
    # fewer raises IndexError mid-training; validate up-front so
    # the user sees a clear message before N slices are wasted.
    if len(hidden_dims) != 3:
        raise ValueError(
            f"--hidden_dims must have EXACTLY 3 widths (CeLEry's "
            f"DNN.__init__ assumes a 3-layer MLP); got {len(hidden_dims)}: "
            f"{list(hidden_dims)}"
        )
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"--data_dir not found: {data_path}")

    # ---- 1. Resolve TS and output dir ----
    # Same three-source TS logic as the other pipelines.
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

    # Configure logging to the out_dir as well as stdout.
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
    logger.info("CeLEry training-phase")
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
    if exclude_test_files:
        logger.info(f"  exclude_test_files : {exclude_test_files}")
    logger.info("=" * 60)

    tracker = _RuntimeTracker()

    # ---- wandb init ----
    # Mirrors run_novosparc_pipeline.py's pattern: try-import + try-init
    # so a wandb failure (auth, network, etc.) downgrades to "log
    # locally only" instead of taking down the whole training run.
    # The wandb run carries the same TS the on-disk artifacts use, so
    # a wandb run page and the artifacts dir pair up at a glance.
    use_wandb = wandb_mode != "disabled"
    wandb_run_obj = None
    if use_wandb:
        try:
            import wandb  # type: ignore
            wandb_run_obj = wandb.init(
                # Default project = "celery". Distinct from the scgg
                # project so the CeLEry-vs-scgg comparison rows don't
                # share a wandb table. Override at submission time
                # with --wandb_project <other>.
                project=wandb_project or "celery",
                name=wandb_run_name or run_name,
                mode=wandb_mode,
                config={
                    "method": "celery",
                    "phase": "training",
                    "data_dir": str(data_path),
                    "seed": seed,
                    "num_epochs_max": num_epochs_max,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "hidden_dims": list(hidden_dims),
                    "num_workers": num_workers,
                    "exclude_test_files": exclude_test_files or [],
                    "run_timestamp": run_ts,
                    "output_dir": str(out),
                    # Tags useful for filtering in the wandb UI.
                    "tags": ["celery", "training", dataset_name],
                },
            )
            logger.info(f"wandb initialised: {wandb_run_obj.url}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"wandb init failed: {e}; continuing without wandb")
            use_wandb = False

    # ---- 2. Discover train + test slices ----
    with tracker.phase("discover_h5ads", flush_to=out / "runtime.csv"):
        train_files = _discover_split_files(data_path, "train")
        test_files = _discover_split_files(data_path, "test")
        if exclude_test_files:
            excluded_set = set(exclude_test_files)
            kept = [p for p in test_files if p.name not in excluded_set]
            dropped = [p.name for p in test_files if p.name in excluded_set]
            if dropped:
                logger.info(
                    f"--exclude_test_files dropped {len(dropped)} file(s): {dropped}"
                )
            unmatched = excluded_set - set(dropped)
            if unmatched:
                logger.warning(
                    f"--exclude_test_files entries did NOT match any "
                    f"*_test.h5ad: {sorted(unmatched)}"
                )
            test_files = kept

        logger.info(f"  found {len(train_files)} train slice(s)")
        logger.info(f"  found {len(test_files)} test slice(s)")
        if not train_files or not test_files:
            raise RuntimeError("no train or test slices to score against")

    # ---- 3. Assign references ----
    with tracker.phase("assign_references", flush_to=out / "runtime.csv"):
        assignments = _assign_references(train_files, test_files, seed=seed)
        assn_table = pd.DataFrame([asdict(a) for a in assignments])
        assn_path = out / "ref_assignments.csv"
        assn_table.to_csv(assn_path, index=False)
        logger.info(f"  wrote {assn_path}")

    # ---- 4. Train one CeLEry per test slice ----
    ckpt_dir = out / "celery_models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_info_rows: List[Dict[str, float]] = []
    with tracker.phase("training", flush_to=out / "runtime.csv"):
        for idx, asn in enumerate(assignments, start=1):
            logger.info(
                f"[{idx}/{len(assignments)}] test={asn.test_slice}  "
                f"ref={asn.ref_slice}"
            )
            info = _train_one_slice(
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

            # Per-slice wandb log. The ``slice_idx`` step counter lets
            # the wandb UI plot train_seconds as a curve over the N
            # per-slice trainings, so slowdowns / outliers stand out.
            if use_wandb:
                try:
                    import wandb  # type: ignore
                    wandb.log({
                        "slice_idx": idx,
                        "test_slice": info["test_slice"],
                        "ref_slice": info["ref_slice"],
                        f"train_seconds/{info['test_slice']}": info["train_seconds"],
                        f"n_ref_cells/{info['test_slice']}": info["n_ref_cells"],
                        f"n_qry_cells/{info['test_slice']}": info["n_qry_cells"],
                        f"n_genes_common/{info['test_slice']}": info["n_genes_common"],
                        # Scalar series the UI can plot as curves:
                        "train_seconds_running_mean": float(
                            np.mean([r["train_seconds"] for r in train_info_rows])
                        ),
                    })
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"wandb.log failed for {info['test_slice']}: {e}")

    # ---- 5. Write the "best_model.ckpt" convenience symlink ----
    # The pipeline orchestrator's checkpoint resolver expects a
    # ``best_model.ckpt`` in the output_dir root. For CeLEry there
    # isn't a single best model — we point the symlink at the
    # checkpoints DIRECTORY so the orchestrator can pass that path
    # to the inference subprocess, which then resolves per-slice.
    best_ptr = out / "best_model.ckpt"
    if best_ptr.exists() or best_ptr.is_symlink():
        best_ptr.unlink()
    # Use a relative symlink so the file moves cleanly if the
    # artifacts root is moved later.
    os.symlink("celery_models", best_ptr)
    logger.info(f"  pinned best_model.ckpt → celery_models/")

    # ---- 6. Write summary CSVs ----
    # config.yaml snapshot — mirrors run_luna_train.py's invocation snapshot.
    config_path = out / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "method": "celery",
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
        "n_test_slices": len(test_files),
    }, default_flow_style=False))

    # train_info table — one row per test slice with per-slice
    # training stats. Useful for sanity-checking that CeLEry didn't
    # silently skip any slice.
    train_info_df = pd.DataFrame(train_info_rows)
    train_info_df.to_csv(out / "celery_train_info.csv", index=False)

    # Training-side metrics.csv is intentionally a stub here —
    # the per-slice Spearman comes from the SEPARATE inference
    # subprocess (run_celery_inference.py), which uses these
    # checkpoints. Mirrors the LUNA pipeline's mode=train_only
    # convention.
    placeholder_metrics = {
        "method": "celery",
        "n_test_slices": len(assignments),
        "n_models_trained": len(train_info_rows),
        "spearman_mean_of_medians": float("nan"),  # filled in by inference
    }
    pd.DataFrame([placeholder_metrics]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame([
        {"section_label": r["test_slice"], "n_cells": r["n_qry_cells"]}
        for r in train_info_rows
    ]).to_csv(out / "per_slice_metrics.csv", index=False)
    (out / "aggregate_metrics.json").write_text(json.dumps(
        placeholder_metrics, indent=2, default=str
    ))

    # ---- 7. Compute requirements ----
    peak = tracker.peak_summary()
    tracker.write_compute_requirements_csv(
        out / "compute_requirements.csv",
        method="CeLEry",
        run_timestamp=run_ts,
        extra={
            "n_train_slices": len(train_files),
            "n_test_slices": len(test_files),
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
    # Push aggregate counters + resource peaks into wandb.run.summary
    # so the wandb run-overview table shows the comparison-friendly
    # numbers without having to dig into per-step plots.
    if use_wandb:
        try:
            import wandb  # type: ignore
            train_seconds = [r["train_seconds"] for r in train_info_rows]
            summary = {
                "n_test_slices": len(assignments),
                "n_models_trained": len(train_info_rows),
                "train_seconds_total": float(np.sum(train_seconds)) if train_seconds else 0.0,
                "train_seconds_mean": float(np.mean(train_seconds)) if train_seconds else 0.0,
                "train_seconds_median": float(np.median(train_seconds)) if train_seconds else 0.0,
                "train_seconds_max": float(np.max(train_seconds)) if train_seconds else 0.0,
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
    p.add_argument(
        "--data_dir", required=True,
        help="Silver h5ad directory containing *_train.h5ad and *_test.h5ad files.",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Where to write artifacts. Default: "
             "<ARTIFACTS_ROOT>/<data_dir_name>/celery_model/<TS>/",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Seed for both the reference-slice random pick AND "
             "CeLEry's internal seednum. Matches the LUNA paper's "
             "shared-seed protocol.",
    )
    p.add_argument(
        "--epochs", type=int, default=500,
        help="num_epochs_max for CeLEry's Fit_cord. Default 500 "
             "matches CeLEry's source-default. Aliased as --epochs "
             "for cross-pipeline parity (LUNA + scgg use --epochs).",
    )
    p.add_argument(
        "--batch_size", type=int, default=4,
        help="CeLEry Fit_cord batch_size. Default 4 matches CeLEry's "
             "source-default.",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="CeLEry initial_learning_rate. Default 1e-3.",
    )
    p.add_argument(
        "--hidden_dims", type=int, nargs="+", default=[30, 25, 15],
        help="Hidden-layer widths for CeLEry's MLP. Default "
             "[30, 25, 15] matches the CeLEry paper's "
             "2-D coordinate-regression head.",
    )
    p.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader num_workers for CeLEry. Default 0 — keeps "
             "memory usage predictable on LSF and avoids fork-based "
             "RSS multiplication on large data.",
    )
    p.add_argument(
        "--wandb_run_name", "--run_name",
        dest="wandb_run_name", default=None,
        help="Run name (forwarded to wandb if --wandb_mode != disabled).",
    )
    p.add_argument(
        "--wandb_mode", default="disabled",
        choices=("disabled", "online", "offline", "dryrun"),
        help="wandb mode (default 'disabled' — CeLEry doesn't log "
             "training-loss curves natively, so wandb gets only "
             "config + summary metrics).",
    )
    p.add_argument(
        "--wandb_project", default=None,
        help="wandb project. Default: scgg.",
    )
    p.add_argument(
        "--exclude_test_files", default=None,
        help="Comma-separated *_test.h5ad basenames to drop before "
             "training. Mirrors the same flag on scgg/luna; useful "
             "for skipping too-large slices on CNS data.",
    )
    p.add_argument(
        "--run_timestamp", default=None,
        help="Optional YYYYMMDD_HHMMSS timestamp. Pinned by the LSF "
             "submitter so the LSF logs, the artifacts dir, and the "
             "wandb tag all share one TS.",
    )
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
            wandb_mode=args.wandb_mode,
            # CLI-side default = "celery"; mirrors the wandb.init
            # default above. Users who want CeLEry runs in the scgg
            # project pass --wandb_project scgg explicitly.
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

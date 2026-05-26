#!/usr/bin/env python
"""Train + evaluate the novoSpaRc baseline in one invocation.

novoSpaRc (Nitzan et al. 2019; Moriel et al. 2021) maps scRNA-seq
cells to spatial coordinates via optimal transport: it solves a fresh
fused Gromov-Wasserstein problem per query slice, no neural network
training step. So unlike ``run_luna_pipeline.py`` / ``run_scgg_pipeline.py``,
this is **one process** — there's no separate train/inference split,
no checkpoint to load, and no ``--skip_inference`` option. Each
invocation does the full thing for every test slice in the silver dir
and writes outputs in the same shape as LUNA/scgg:

    artifacts/<dataset>/novosparc_inference/<TS>/
        <section_1>/metadata_pred.csv
        <section_1>/metadata_true.csv
        <section_2>/...
        per_slice_metrics.csv
        aggregate_metrics.json
        metrics.csv
        runtime.csv
        config.yaml
        plots/<section_1>.svg
        plots/<section_2>.svg

Input is the **same silver h5ad directory** that LUNA / scgg read
from. The atlas (spatial reference) is the concatenation of all
``*_train.h5ad`` files in the directory; queries are the
``*_test.h5ad`` files, scored independently. For datasets where the
total atlas exceeds ``--max_atlas_cells`` (default 10000) we
randomly subsample to that cap — full-atlas OT memory grows as
N_query × N_atlas, which is the binding constraint on cortex.

Outputs land in the same paired-by-timestamp layout as LUNA/scgg, so
all three methods' results sit side-by-side in the artifacts tree
and can be diffed cell.

Typical use::

    python scgg/scripts/run_novosparc_pipeline.py \\
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --wandb_run_name novosparc_mmc_baseline

To sweep the atlas size / alpha::

    python scgg/scripts/run_novosparc_pipeline.py \\
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --wandb_run_name novosparc_mmc_atlas_5k \\
        --max_atlas_cells 5000 \\
        --alpha_linear 0.7

To run inside the novosparc venv created by
``setup_novosparc_env.sh``::

    source /nfs/team361/sb75/.venvs/novosparc/bin/activate
    python scgg/scripts/run_novosparc_pipeline.py ...
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Reuse helpers from run_luna_train.py (silver discovery, runtime
# tracker, plotting, the per-cell Spearman metric). That module's
# top-level imports are stdlib + numpy + pandas + yaml, so importing
# it does not pull in torch / pyg / pytorch_lightning — it's safe to
# import from inside the novosparc venv. Torch-using paths in
# run_luna_train are gated to functions we never call from here.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
import run_luna_train as _luna  # noqa: E402

# Pull the helpers out by name so any future refactor of
# run_luna_train can't silently break us (NameError beats silently
# wrong behavior).
_RuntimeTracker = _luna._RuntimeTracker
_discover_split_files = _luna._discover_split_files
_section_label_from_filename = _luna._section_label_from_filename
_per_cell_spearman_median = _luna._per_cell_spearman_median
_plot_pred_vs_truth = _luna._plot_pred_vs_truth


logger = logging.getLogger("novosparc_pipeline")

# Mirrors run_luna_train.py's defaults so all three methods write
# under the same artifacts root.
_DEFAULT_ARTIFACTS_ROOT = Path(
    "/nfs/team361/sb75/scgg-reproducibility/artifacts"
)
_OUTPUT_SUBDIR = "novosparc_inference"


# ===========================================================================
# Argument parsing
# ===========================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---------------- Dataset ----------------
    p.add_argument(
        "--data_dir", required=True,
        help="Silver h5ad directory (suffix-based train/test discovery). "
             "Same input format as run_luna_pipeline.py.",
    )

    # ---------------- Output location ----------------
    p.add_argument(
        "--output_dir", default=None,
        help="Where to write per-slice metrics / aggregate metrics / "
             "runtime / plots. Default: "
             "$NOVOSPARC_ARTIFACTS_ROOT/<dataset>/novosparc_inference/<TS>/. "
             "Set $NOVOSPARC_ARTIFACTS_ROOT to redirect the root "
             "(defaults to LUNA/scgg's shared artifacts root).",
    )

    # ---------------- novosparc knobs ----------------
    p.add_argument(
        "--alpha_linear", type=float, default=0.5,
        help="Atlas-aided OT weight. 0 = de novo (gene-expression "
             "geometry alone); 1 = atlas-driven only. 0.5 is the "
             "novosparc tutorial default for atlas-aided runs and a "
             "reasonable starting point.",
    )
    p.add_argument(
        "--epsilon", type=float, default=5e-4,
        help="Entropic regularisation. novosparc default is 5e-4. "
             "Smaller → sharper OT plan but slower convergence; "
             "larger → smoother plan, potentially more stable.",
    )
    p.add_argument(
        "--num_neighbors_s", type=int, default=5,
        help="kNN for source (query) gene-expression graph.",
    )
    p.add_argument(
        "--num_neighbors_t", type=int, default=5,
        help="kNN for target (atlas) location graph.",
    )
    p.add_argument(
        "--n_markers", type=int, default=None,
        help="If set, restrict the atlas matrix to the top-variance "
             "N genes from the atlas. Default: use every shared gene. "
             "Reducing helps wall-clock on dense panels but typically "
             "doesn't change the result much on MERFISH-scale panels "
             "(500 genes already).",
    )
    p.add_argument(
        "--max_atlas_cells", type=int, default=10000,
        help="Cap on the atlas size (concatenation of all training "
             "slices). Random-subsample down to this when the full "
             "atlas exceeds it. Memory of the OT plan grows as "
             "N_query × N_atlas; on cortex (~7k query cells and "
             "~230k total train cells) the full atlas would OOM most "
             "GPUs. 10000 is a memory/quality compromise; bump if "
             "you have the headroom.",
    )

    # ---------------- Reproducibility ----------------
    p.add_argument(
        "--seed", type=int, default=0,
        help="numpy seed for the atlas subsample. Default 0 (matches "
             "LUNA's general.seed default).",
    )

    # ---------------- WandB ----------------
    p.add_argument(
        "--wandb_project", default=None,
        help="wandb project name. Default 'novosparc' when omitted.",
    )
    p.add_argument(
        "--wandb_mode", default="online",
        choices=("disabled", "online", "offline", "dryrun"),
        help="wandb mode. Set 'disabled' to skip logging entirely.",
    )
    p.add_argument(
        "--wandb_run_name", "--run_name", dest="wandb_run_name",
        default="novosparc_mmc_baseline",
        help="wandb run name. Used as the human-readable label in the "
             "wandb UI; doesn't affect output paths (those use the "
             "timestamp).",
    )

    # ---------------- Misc ----------------
    p.add_argument(
        "--no_plots", action="store_true",
        help="Skip per-section ground-truth-vs-prediction plots. "
             "Plots ON by default (parity with run_luna_inference.py).",
    )
    p.add_argument(
        "--run_timestamp", default=None,
        help="Optional ``YYYYMMDD_HHMMSS`` timestamp to use as the run "
             "label. When set, the pipeline does NOT generate a fresh "
             "wall-clock timestamp; it uses this one for the output "
             "subdir AND the wandb config's ``run_timestamp`` field. "
             "Used by the LSF submitter (submit_pipeline.sh) so the "
             "LSF log dir, the artifacts dir, and the wandb run all "
             "share the same TS. Format-checked: ``YYYYMMDD_HHMMSS``.",
    )

    return p


# ===========================================================================
# Atlas construction
# ===========================================================================


def _load_atlas_from_train_files(
    train_files: List[Path],
    max_atlas_cells: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Concatenate all training h5ads into one atlas (X, locations, gene_names).

    Returns
    -------
    atlas_X : (n_atlas, n_genes) float64 expression matrix
    atlas_locs : (n_atlas, 2) float64 spatial coordinates
    gene_names : list[str]    column order of atlas_X

    Subsamples down to ``max_atlas_cells`` (uniformly at random across
    the concatenated pool) when the total exceeds the cap. The
    subsample is seeded by ``seed`` so reruns are bit-identical.
    """
    import anndata as ad
    import scipy.sparse as sp

    Xs: List[np.ndarray] = []
    locs: List[np.ndarray] = []
    gene_names: Optional[List[str]] = None

    for path in train_files:
        adata = ad.read_h5ad(path)
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        # Track gene panel for consistency across slices.
        if gene_names is None:
            gene_names = list(adata.var_names)
        elif list(adata.var_names) != gene_names:
            raise ValueError(
                f"Gene panel mismatch in {path.name}: expected "
                f"{len(gene_names)} genes, got {adata.n_vars}. "
                f"All train h5ads must share the same gene panel."
            )

        # Spatial coords: prefer obsm['spatial'] (the canonical silver
        # layout), fall back to obs['coord_X'] / obs['coord_Y'].
        if "spatial" in adata.obsm:
            xy = np.asarray(
                adata.obsm["spatial"], dtype=np.float64
            )[:, :2]
        else:
            xy = np.column_stack([
                adata.obs["coord_X"].to_numpy(dtype=np.float64),
                adata.obs["coord_Y"].to_numpy(dtype=np.float64),
            ])

        Xs.append(X)
        locs.append(xy)
        logger.info(f"  loaded train slice {path.name}: "
                    f"{X.shape[0]} cells × {X.shape[1]} genes")

    atlas_X = np.concatenate(Xs, axis=0)
    atlas_locs = np.concatenate(locs, axis=0)
    logger.info(f"concatenated atlas: {atlas_X.shape[0]} cells total")

    # Cap. Random subsample if the full atlas exceeds it.
    if atlas_X.shape[0] > max_atlas_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(atlas_X.shape[0], size=max_atlas_cells, replace=False)
        # Sort the indices for determinism in any downstream code that
        # assumes monotonic ordering (and so the same subsample is bit-
        # identical run-to-run for the same seed).
        idx = np.sort(idx)
        atlas_X = atlas_X[idx]
        atlas_locs = atlas_locs[idx]
        logger.info(f"subsampled atlas to {atlas_X.shape[0]} cells "
                    f"(cap={max_atlas_cells}, seed={seed})")

    return atlas_X, atlas_locs, gene_names


# ===========================================================================
# Per-slice novosparc prediction
# ===========================================================================


def _predict_one_slice(
    test_adata,
    atlas_X: np.ndarray,
    atlas_locs: np.ndarray,
    *,
    alpha_linear: float,
    epsilon: float,
    num_neighbors_s: int,
    num_neighbors_t: int,
    n_markers: Optional[int],
) -> np.ndarray:
    """Run novosparc on a single test slice; return predicted (N, 2) coords.

    Coordinates are the **expected location under the OT plan**
    (T-weighted average of atlas locations per query cell), which
    is the standard novosparc point-estimate. argmax would also work
    but produces lumpier predictions (the predicted coord is always
    exactly one of the atlas's locations).
    """
    import novosparc
    import scipy.sparse as sp

    # Test cells' expression matrix in dense float64 (novosparc reads
    # adata.X via numpy; sparse matrices break some downstream paths).
    if sp.issparse(test_adata.X):
        test_adata.X = test_adata.X.toarray()
    test_adata.X = np.asarray(test_adata.X, dtype=np.float64)

    n_genes = atlas_X.shape[1]

    # Marker-gene selection. With dense MERFISH panels (~500 genes)
    # this is usually a no-op. For larger panels (Visium, scRNA-seq)
    # capping to top-variance markers is a wall-clock win.
    if n_markers is not None and n_markers < n_genes:
        gene_var = atlas_X.var(axis=0)
        # Top-N by variance, sorted ascending so the highest-var gene
        # is last — doesn't matter for the result, just makes the
        # marker indices stable across runs with the same atlas.
        markers_to_use = np.argsort(gene_var)[-n_markers:]
        markers_to_use = np.sort(markers_to_use)
    else:
        markers_to_use = np.arange(n_genes)
    atlas_for_setup = atlas_X[:, markers_to_use]

    # Build the Tissue and solve. The Tissue's `dataset` argument is
    # the QUERY (test cells); `locations` is the TARGET geometry
    # (atlas spatial coordinates); `atlas_matrix` carries gene
    # expression at those known locations.
    tissue = novosparc.cm.Tissue(
        dataset=test_adata,
        locations=atlas_locs,
    )
    tissue.setup_reconstruction(
        markers_to_use=markers_to_use,
        atlas_matrix=atlas_for_setup,
        num_neighbors_s=num_neighbors_s,
        num_neighbors_t=num_neighbors_t,
    )
    tissue.reconstruct(
        alpha_linear=alpha_linear,
        epsilon=epsilon,
    )

    gw = tissue.gw  # (n_query, n_atlas) — the OT plan
    # Row-normalise so each query cell's row is a proper probability
    # distribution over atlas locations. novosparc's OT plan is
    # normalised in expectation by construction, but float drift can
    # leave some rows summing to < 1 — explicit re-normalisation
    # avoids the bias that would otherwise creep into the expected
    # coordinates.
    row_sums = gw.sum(axis=1, keepdims=True)
    # Guard against any all-zero rows (shouldn't happen in practice
    # but a 0/0 here would produce NaN coords downstream).
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    gw_norm = gw / row_sums

    predicted_coords = gw_norm @ atlas_locs  # (n_query, 2)
    return np.asarray(predicted_coords, dtype=np.float64)


# ===========================================================================
# Output writers (LUNA-compatible format)
# ===========================================================================


def _write_slice_outputs(
    out_dir: Path,
    test_adata,
    section_label: str,
    coords_pred: np.ndarray,
) -> Tuple[Path, Path]:
    """Write metadata_pred.csv + metadata_true.csv for one slice in the
    same shape LUNA's pipeline writes them, so the downstream eval /
    plotting helpers can read them without modification.

    Returns the two written paths.
    """
    section_dir = out_dir / section_label
    section_dir.mkdir(parents=True, exist_ok=True)

    # True coords from the test h5ad — same priority as the LUNA
    # reader: obsm['spatial'] first, fall back to obs columns.
    if "spatial" in test_adata.obsm:
        coords_true = np.asarray(
            test_adata.obsm["spatial"], dtype=np.float64,
        )[:, :2]
    else:
        coords_true = np.column_stack([
            test_adata.obs["coord_X"].to_numpy(dtype=np.float64),
            test_adata.obs["coord_Y"].to_numpy(dtype=np.float64),
        ])

    cell_class = (
        test_adata.obs["cell_class"].astype(str).to_numpy()
        if "cell_class" in test_adata.obs.columns
        else np.full(test_adata.n_obs, "unknown")
    )

    # Index = cell barcode (obs_names). LUNA's reader uses
    # ``pd.read_csv(..., index_col=0)`` and ``pred.index.equals(true.index)``,
    # so both files must share the same row order.
    pred_df = pd.DataFrame({
        "coord_X": coords_pred[:, 0],
        "coord_Y": coords_pred[:, 1],
        "cell_class": cell_class,
        "cell_section": section_label,
    }, index=test_adata.obs_names)
    true_df = pd.DataFrame({
        "coord_X": coords_true[:, 0],
        "coord_Y": coords_true[:, 1],
        "cell_class": cell_class,
        "cell_section": section_label,
    }, index=test_adata.obs_names)

    pred_path = section_dir / "metadata_pred.csv"
    true_path = section_dir / "metadata_true.csv"
    pred_df.to_csv(pred_path)
    true_df.to_csv(true_path)
    return pred_path, true_path


# ===========================================================================
# Main pipeline
# ===========================================================================


def main() -> int:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        sys.exit(f"--data_dir not found: {data_dir}")

    # ---- Pin output directory ----
    # When --run_timestamp is set (e.g. by the LSF submitter), use it
    # verbatim so LSF logs + on-disk artifacts + wandb run all share
    # the same TS. Otherwise generate a fresh wall-clock timestamp.
    if args.run_timestamp:
        import re as _re
        if not _re.fullmatch(r"\d{8}_\d{6}", args.run_timestamp):
            sys.exit(
                f"--run_timestamp must match YYYYMMDD_HHMMSS; got "
                f"{args.run_timestamp!r}."
            )
        run_ts = args.run_timestamp
    else:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out = Path(args.output_dir).resolve()
    else:
        artifacts_root = Path(os.environ.get(
            "NOVOSPARC_ARTIFACTS_ROOT", _DEFAULT_ARTIFACTS_ROOT,
        ))
        out = artifacts_root / data_dir.name / _OUTPUT_SUBDIR / run_ts
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"novosparc pipeline timestamp: {run_ts}")
    logger.info(f"  data_dir   = {data_dir}")
    logger.info(f"  output_dir = {out}")
    logger.info("=" * 60)

    # ---- Runtime tracking ----
    runtime_csv = out / "runtime.csv"
    tracker = _RuntimeTracker()

    # ---- Discover splits ----
    tracker.start("discovery")
    train_files = _discover_split_files(data_dir, "train")
    test_files = _discover_split_files(data_dir, "test")
    if not train_files:
        sys.exit(f"no *_train.h5ad found under {data_dir}")
    if not test_files:
        sys.exit(f"no *_test.h5ad found under {data_dir}")
    logger.info(
        f"discovered {len(train_files)} train h5ads, {len(test_files)} test h5ads"
    )
    tracker.end("discovery", flush_to=runtime_csv)

    # ---- Build atlas (concatenated train pool, optionally subsampled) ----
    tracker.start("atlas_build")
    atlas_X, atlas_locs, gene_names = _load_atlas_from_train_files(
        train_files,
        max_atlas_cells=args.max_atlas_cells,
        seed=args.seed,
    )
    tracker.end("atlas_build", flush_to=runtime_csv)

    # ---- Init wandb (optional) ----
    use_wandb = args.wandb_mode != "disabled"
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project or "novosparc",
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config={
                    "method": "novosparc",
                    "data_dir": str(data_dir),
                    "alpha_linear": args.alpha_linear,
                    "epsilon": args.epsilon,
                    "num_neighbors_s": args.num_neighbors_s,
                    "num_neighbors_t": args.num_neighbors_t,
                    "n_markers": args.n_markers,
                    "max_atlas_cells": args.max_atlas_cells,
                    "atlas_size_used": int(atlas_X.shape[0]),
                    "seed": args.seed,
                    "run_timestamp": run_ts,
                },
            )
            logger.info(f"wandb initialised: {wandb.run.url}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"wandb init failed: {e}; continuing without wandb")
            use_wandb = False

    # ---- Per-slice predict + write ----
    import anndata as ad

    per_slice: List[Dict[str, object]] = []
    tracker.start("reconstruction_all_slices")
    for test_path in test_files:
        section_label = _section_label_from_filename(test_path)
        logger.info(f"--- {section_label} ({test_path.name}) ---")

        # Per-slice phase timer for runtime.csv granularity.
        per_slice_phase = f"reconstruction:{section_label}"
        tracker.start(per_slice_phase)

        test_adata = ad.read_h5ad(test_path)
        # Gene panel must match the atlas.
        if list(test_adata.var_names) != gene_names:
            logger.warning(
                f"  gene panel mismatch in {test_path.name}; "
                f"taking the intersection (may degrade quality)"
            )
            common = [g for g in test_adata.var_names if g in gene_names]
            if not common:
                logger.error("  no common genes; skipping slice")
                tracker.end(per_slice_phase, flush_to=runtime_csv)
                continue
            test_adata = test_adata[:, common].copy()
            atlas_idx = [gene_names.index(g) for g in common]
            atlas_X_slice = atlas_X[:, atlas_idx]
        else:
            atlas_X_slice = atlas_X

        coords_pred = _predict_one_slice(
            test_adata,
            atlas_X_slice,
            atlas_locs,
            alpha_linear=args.alpha_linear,
            epsilon=args.epsilon,
            num_neighbors_s=args.num_neighbors_s,
            num_neighbors_t=args.num_neighbors_t,
            n_markers=args.n_markers,
        )

        _write_slice_outputs(out, test_adata, section_label, coords_pred)
        tracker.end(per_slice_phase, flush_to=runtime_csv)

        # Per-slice metric. Compute right here (no need to re-load the
        # CSVs we just wrote) — keeps the dependency on
        # _per_cell_spearman_median to one call site.
        if "spatial" in test_adata.obsm:
            coords_true = np.asarray(
                test_adata.obsm["spatial"], dtype=np.float64,
            )[:, :2]
        else:
            coords_true = np.column_stack([
                test_adata.obs["coord_X"].to_numpy(dtype=np.float64),
                test_adata.obs["coord_Y"].to_numpy(dtype=np.float64),
            ])
        med, mean = _per_cell_spearman_median(coords_true, coords_pred)
        n_cells = int(coords_true.shape[0])
        per_slice.append({
            "section_label": section_label,
            "n_cells": n_cells,
            "spearman_per_cell_median": med,
            "spearman_per_cell_mean": mean,
        })
        logger.info(
            f"  {section_label:32s}  n={n_cells:>5d}  "
            f"spr_median={med:.4f}  spr_mean={mean:.4f}"
        )
        if use_wandb:
            wandb.log({
                "section_label": section_label,
                f"spearman_per_cell_median/{section_label}": med,
                f"spearman_per_cell_mean/{section_label}": mean,
                f"n_cells/{section_label}": n_cells,
            })

        # Plot (if not suppressed).
        if not args.no_plots:
            plots_dir = out / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            cell_class = (
                test_adata.obs["cell_class"].astype(str).to_numpy()
                if "cell_class" in test_adata.obs.columns else None
            )
            try:
                _plot_pred_vs_truth(
                    coords_true=coords_true,
                    coords_pred=coords_pred,
                    cell_class=cell_class,
                    out_path=plots_dir / f"{section_label}.svg",
                    title_prefix=f"{section_label}  |  ",
                    method_label="novoSpaRc prediction",
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  plot failed for {section_label}: {e}")
    tracker.end("reconstruction_all_slices", flush_to=runtime_csv)

    # ---- Aggregate metrics + LUNA-format outputs ----
    tracker.start("write_artifacts")
    headline = float("nan")
    if per_slice:
        medians = [
            r["spearman_per_cell_median"] for r in per_slice
            if not np.isnan(r["spearman_per_cell_median"])
        ]
        if medians:
            headline = float(np.mean(medians))

    luna_paper = 0.448
    logger.info("=" * 72)
    logger.info("novoSpaRc — aggregated metrics across test slices")
    logger.info("=" * 72)
    logger.info(
        f"  spearman_mean_of_medians (n={len(per_slice)} slices) = {headline:.4f}"
    )
    logger.info(f"  LUNA paper headline                                   = {luna_paper:.4f}")
    if not np.isnan(headline):
        logger.info(
            f"  Delta vs LUNA paper                                  = "
            f"{(headline - luna_paper) * 100:+.2f} pp"
        )

    # per_slice_metrics.csv — one row per slice.
    if per_slice:
        fieldnames = sorted({k for r in per_slice for k in r.keys()})
        with open(out / "per_slice_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in per_slice:
                w.writerow(r)

    # aggregate_metrics.json + metrics.csv — same mean/median/std/min/
    # max breakdown as run_luna_train.py writes, so downstream
    # notebooks comparing LUNA vs scgg vs novosparc rows can pandas-
    # concat them without per-method branching.
    agg: Dict[str, object] = {
        "spearman_mean_of_medians": headline,
        "n_test_slices": len(per_slice),
    }
    if per_slice:
        numeric_keys = sorted({
            k for r in per_slice for k, v in r.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        })
        for k in numeric_keys:
            vals = [
                float(r[k]) for r in per_slice
                if k in r
                and isinstance(r[k], (int, float))
                and not isinstance(r[k], bool)
                and not (isinstance(r[k], float) and (np.isnan(r[k]) or np.isinf(r[k])))
            ]
            if not vals:
                continue
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_median"] = float(np.median(vals))
            agg[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
            agg[f"{k}_min"] = float(np.min(vals))
            agg[f"{k}_max"] = float(np.max(vals))

    # Resource cost from the tracker (shared with LUNA/scgg via the
    # sys.path-imported _RuntimeTracker). novosparc is CPU-only so
    # peak_gpu_mib will typically be 0 / very small (background
    # processes on the host); peak_rss_mib is the binding constraint
    # for novosparc since the OT plan is N_query × N_atlas dense.
    peak = tracker.peak_summary()
    if peak["peak_gpu_mib"] is not None:
        agg["peak_gpu_mib"] = peak["peak_gpu_mib"]
    if peak["peak_rss_mib"] is not None:
        agg["peak_rss_mib"] = peak["peak_rss_mib"]

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg.keys()))
        w.writeheader()
        w.writerow(agg)
    with open(out / "aggregate_metrics.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)

    # Snapshot of the invocation so notebooks can reconstruct what
    # produced the artifacts. Mirrors run_luna_train.py's config.yaml.
    cfg_snap = {
        "method": "novoSpaRc",
        "run_timestamp": run_ts,
        "data_source": "silver_h5ad",
        "data_dir": str(data_dir),
        "n_train_files": len(train_files),
        "n_test_files": len(test_files),
        "atlas_size_used": int(atlas_X.shape[0]),
        "max_atlas_cells": args.max_atlas_cells,
        "alpha_linear": args.alpha_linear,
        "epsilon": args.epsilon,
        "num_neighbors_s": args.num_neighbors_s,
        "num_neighbors_t": args.num_neighbors_t,
        "n_markers": args.n_markers,
        "seed": args.seed,
        "wandb_run_name": args.wandb_run_name,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
    }
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_snap, f, sort_keys=False)

    # compute_requirements.csv — single-row resource summary that
    # sits next to runtime.csv and metrics.csv. Surfaces method-
    # specific extras (alpha, atlas size) so a paper table comparing
    # novosparc / LUNA / scgg shows the relevant config alongside
    # the peak resource numbers.
    tracker.write_compute_requirements_csv(
        out / "compute_requirements.csv",
        method="novoSpaRc",
        run_timestamp=run_ts,
        extra={
            "alpha_linear": args.alpha_linear,
            "epsilon": args.epsilon,
            "max_atlas_cells": args.max_atlas_cells,
            "atlas_size_used": int(atlas_X.shape[0]),
            "n_test_slices": len(per_slice),
        },
    )
    tracker.end("write_artifacts", flush_to=runtime_csv)

    # ---- Push aggregate to wandb ----
    if use_wandb:
        wandb.log({
            "spearman_mean_of_medians": headline,
            "n_test_slices": len(per_slice),
        })
        # Also surface the full agg dict as a summary so wandb's run
        # overview shows the comparison-friendly numbers directly.
        for k, v in agg.items():
            try:
                wandb.run.summary[k] = v
            except Exception:  # noqa: BLE001
                pass
        wandb.finish()

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info(f"  Artifacts: {out}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

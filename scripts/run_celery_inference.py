"""run_celery_inference.py — score CeLEry checkpoints on the held-out mouse.

Companion to ``run_celery_train.py``. Reads the per-test-slice
checkpoint tree produced by training:

    <ckpt_root>/celery_models/<test_slice>/model.obj
    <ckpt_root>/celery_models/<test_slice>/manifest.json

…loads each CeLEry model with ``cel.Predict_cord``, inverse-transforms
the [0,1] sigmoid output back to the reference's coordinate scale,
writes ``metadata_pred.csv`` + ``metadata_true.csv`` per slice (the
schema scgg/luna pipelines also produce — same column names, same
dir nesting), and computes the LUNA-paper benchmark metrics.

The output dir layout mirrors run_luna_inference.py exactly so the
existing compute_extended_metrics.py + plot scripts find these files
without changes:

    <output_dir>/
        luna_run/test_results/<run>/celery/<slice>/metadata_pred.csv
        luna_run/test_results/<run>/celery/<slice>/metadata_true.csv
        per_slice_metrics.csv
        metrics.csv
        aggregate_metrics.json
        runtime.csv
        compute_requirements.csv
        config.yaml
        plots/<slice>.svg   (if --plots)

Inputs (CLI):
    --data_dir          silver h5ad directory with *_test.h5ad
    --checkpoint        path to the training run's ``best_model.ckpt``
                        symlink (which points at celery_models/)
                        OR directly at the celery_models/ dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
import run_luna_train as _luna  # noqa: E402
import run_celery_train as _celery_train  # noqa: E402

_RuntimeTracker = _luna._RuntimeTracker
_discover_split_files = _luna._discover_split_files
_section_label_from_filename = _luna._section_label_from_filename
_per_cell_spearman_median = _luna._per_cell_spearman_median
_plot_pred_vs_truth = _luna._plot_pred_vs_truth

_load_h5ad_for_celery = _celery_train._load_h5ad_for_celery
_align_genes = _celery_train._align_genes
_invert_celery_normalisation = _celery_train._invert_celery_normalisation


logger = logging.getLogger("celery_inference")


ENGINE_OUTPUT_SUBDIR = "celery_inference"
_DEFAULT_ARTIFACTS_ROOT = Path("/nfs/team361/sb75/scgg-reproducibility/artifacts")
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
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _resolve_celery_models_dir(checkpoint: Path) -> Path:
    """Map ``--checkpoint`` to the celery_models/ root.

    The training script writes ``best_model.ckpt`` as a symlink to
    ``celery_models/`` inside the same output dir. So:
      - If --checkpoint IS that symlink → resolve it.
      - If --checkpoint is the celery_models/ dir directly → use it.
      - If --checkpoint is a specific model.obj → walk up to its
        celery_models/ ancestor.
      - Else, fail loudly.
    """
    p = Path(checkpoint).resolve()
    if p.is_dir() and p.name == "celery_models":
        return p
    if p.is_dir() and (p / "celery_models").is_dir():
        # User pointed at the training run's output dir.
        return p / "celery_models"
    if p.is_file() and p.name == "model.obj":
        # Walk up: <models>/<slice>/model.obj → <models>
        return p.parent.parent
    if p.name == "best_model.ckpt":
        target = Path(os.readlink(p)) if p.is_symlink() else p
        if not target.is_absolute():
            target = (p.parent / target).resolve()
        if target.is_dir() and target.name == "celery_models":
            return target
    raise FileNotFoundError(
        f"--checkpoint={checkpoint} does not resolve to a celery_models/ "
        f"directory. Expected the path to point at a training-run "
        f"output dir (so we can find celery_models/ inside), the "
        f"best_model.ckpt symlink, or the celery_models/ dir directly."
    )


def _load_slice_checkpoint(slice_dir: Path) -> Tuple[Dict, Path]:
    """Read a per-slice checkpoint's manifest + return the model.obj path."""
    manifest_path = slice_dir / "manifest.json"
    model_path = slice_dir / "model.obj"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json missing: {manifest_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model.obj missing: {model_path}")
    manifest = json.loads(manifest_path.read_text())
    return manifest, model_path


# ---------------------------------------------------------------------------
# Per-slice inference
# ---------------------------------------------------------------------------

def _infer_one_slice(
    slice_label: str,
    slice_dir: Path,
    test_h5ad: Path,
    out_slice_dir: Path,
    n_inference_samples: int,
) -> Dict[str, object]:
    """Load model + predict + write metadata_{pred,true}.csv for one slice.

    CeLEry's output is a deterministic point estimate per call, so
    when n_inference_samples > 1 we re-evaluate with re-z-scored
    input each time and average. This matches LUNA + scgg's multi-
    sample ensembling convention so the n_inference_samples knob
    means the same thing across all four methods (though for CeLEry
    the gain is tiny because the model is deterministic).
    """
    import CeLEry as cel  # type: ignore

    manifest, model_path = _load_slice_checkpoint(slice_dir)
    x_min, x_max = manifest["x_min"], manifest["x_max"]
    y_min, y_max = manifest["y_min"], manifest["y_max"]

    # Reload the test slice as CeLEry expects.
    adata_qry, _, _, _, _ = _load_h5ad_for_celery(test_h5ad)

    # Align genes to the reference's gene order. We need the reference
    # AnnData to do this — load only its var_names (lightweight; the
    # silver h5ads carry the full obs/X but we just need the gene
    # names for the intersection).
    import scanpy as sc
    ref_path = Path(manifest["ref_path"])
    adata_ref_genes_only = sc.read(ref_path).copy()
    adata_ref_genes_only, adata_qry = _align_genes(
        adata_ref_genes_only, adata_qry,
    )

    cel.get_zscore(adata_qry)

    # CeLEry's Predict_cord expects (path, filename) without the .obj
    # extension — it appends ".obj" internally. We saved as model.obj
    # so filename="model".
    preds: List[np.ndarray] = []
    for sample_idx in range(max(1, n_inference_samples)):
        pred_normed = cel.Predict_cord(
            data_test=adata_qry,
            path=str(model_path.parent),
            filename="model",
            location_data=None,
        )
        preds.append(np.asarray(pred_normed))

    pred_normed_mean = np.mean(np.stack(preds, axis=0), axis=0)
    pred_orig_scale = _invert_celery_normalisation(
        pred_normed_mean, x_min, x_max, y_min, y_max,
    )

    # Build metadata DFs in the exact schema scgg/luna pipelines write.
    obs = adata_qry.obs
    cell_class_col = "cell_class" if "cell_class" in obs.columns else None
    df_pred = pd.DataFrame({
        "coord_X": pred_orig_scale[:, 0],
        "coord_Y": pred_orig_scale[:, 1],
        **({"cell_class": obs[cell_class_col].to_numpy()} if cell_class_col else {}),
    }, index=obs.index)
    df_pred.index.name = "cell_ID"

    # Ground truth: first two cols of obs (we put them there during loading).
    df_true = pd.DataFrame({
        "coord_X": obs.iloc[:, 0].to_numpy(dtype=np.float64),
        "coord_Y": obs.iloc[:, 1].to_numpy(dtype=np.float64),
        **({"cell_class": obs[cell_class_col].to_numpy()} if cell_class_col else {}),
    }, index=obs.index)
    df_true.index.name = "cell_ID"

    out_slice_dir.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(out_slice_dir / "metadata_pred.csv")
    df_true.to_csv(out_slice_dir / "metadata_true.csv")

    coords_true = df_true[["coord_X", "coord_Y"]].to_numpy()
    coords_pred = df_pred[["coord_X", "coord_Y"]].to_numpy()
    median_rho, mean_rho = _per_cell_spearman_median(coords_true, coords_pred)

    return {
        "section_label": slice_label,
        "n_cells": int(adata_qry.n_obs),
        "spearman_per_cell_median": float(median_rho),
        "spearman_per_cell_mean": float(mean_rho),
        "ref_slice": manifest.get("ref_slice", ""),
    }


# ---------------------------------------------------------------------------
# Inference driver
# ---------------------------------------------------------------------------

def run_inference(
    data_dir: str,
    checkpoint: str,
    output_dir: Optional[str] = None,
    *,
    seed: int = 0,
    wandb_mode: str = "disabled",
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    n_inference_samples: int = 1,
    exclude_test_files: Optional[List[str]] = None,
    make_plots: bool = True,
    run_timestamp: Optional[str] = None,
) -> Dict[str, float]:
    """Score every per-slice CeLEry checkpoint and aggregate metrics."""
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"--data_dir not found: {data_path}")

    ckpt_root = _resolve_celery_models_dir(Path(checkpoint))
    logger.info(f"celery_models root: {ckpt_root}")

    # ---- TS resolution ----
    if run_timestamp is not None:
        if not re.fullmatch(r"\d{8}_\d{6}(?:_[A-Za-z0-9]+)?", run_timestamp):
            raise ValueError(
                f"run_timestamp must be YYYYMMDD_HHMMSS or "
                f"YYYYMMDD_HHMMSS_<suffix>; got {run_timestamp!r}"
            )
        run_ts = run_timestamp
    else:
        # Try to inherit from checkpoint path (so inference pairs
        # with the training run's TS); fall back to fresh clock.
        m = re.search(r"\d{8}_\d{6}(?:_[A-Za-z0-9]+)?", str(checkpoint))
        run_ts = m.group(0) if m else datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Output dir ----
    dataset_name = data_path.name
    out = (
        Path(output_dir).resolve()
        if output_dir
        else _ARTIFACTS_ROOT / dataset_name / ENGINE_OUTPUT_SUBDIR / run_ts
    )
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out / "inference.log", mode="w"),
        ],
        force=True,
    )

    logger.info("=" * 60)
    logger.info("CeLEry inference-phase")
    logger.info("=" * 60)
    logger.info(f"  data_dir            : {data_path}")
    logger.info(f"  celery_models       : {ckpt_root}")
    logger.info(f"  output_dir          : {out}")
    logger.info(f"  run_ts              : {run_ts}")
    logger.info(f"  seed                : {seed}")
    logger.info(f"  n_inference_samples : {n_inference_samples}")
    if exclude_test_files:
        logger.info(f"  exclude_test_files  : {exclude_test_files}")
    logger.info("=" * 60)

    tracker = _RuntimeTracker()

    # ---- wandb init ----
    # Same pattern as run_celery_train.py / run_novosparc_pipeline.py.
    # Inference is a SEPARATE wandb run from training (matches the
    # scgg/luna convention — training and inference each get their
    # own wandb entry, and the pipeline appends ``_inference`` to
    # the run name so they pair up by name in the wandb UI).
    use_wandb = wandb_mode != "disabled"
    wandb_run_obj = None
    if use_wandb:
        try:
            import wandb  # type: ignore
            wandb_run_obj = wandb.init(
                project=wandb_project or "scgg",
                name=wandb_run_name,
                mode=wandb_mode,
                config={
                    "method": "celery",
                    "phase": "inference",
                    "data_dir": str(data_path),
                    "celery_models": str(ckpt_root),
                    "seed": seed,
                    "n_inference_samples": n_inference_samples,
                    "exclude_test_files": exclude_test_files or [],
                    "run_timestamp": run_ts,
                    "output_dir": str(out),
                    "tags": ["celery", "inference", data_path.name],
                },
            )
            logger.info(f"wandb initialised: {wandb_run_obj.url}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"wandb init failed: {e}; continuing without wandb")
            use_wandb = False

    # ---- Discover test slices ----
    with tracker.phase("discover_h5ads", flush_to=out / "runtime.csv"):
        test_files = _discover_split_files(data_path, "test")
        if exclude_test_files:
            excluded_set = set(exclude_test_files)
            kept = [p for p in test_files if p.name not in excluded_set]
            dropped = [p.name for p in test_files if p.name in excluded_set]
            if dropped:
                logger.info(f"  dropped: {dropped}")
            unmatched = excluded_set - set(dropped)
            if unmatched:
                logger.warning(f"  unmatched exclude entries: {sorted(unmatched)}")
            test_files = kept
        logger.info(f"  scoring {len(test_files)} test slice(s)")

    # ---- Per-slice inference ----
    plots_dir = out / "plots" if make_plots else None
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    test_results_dir = (
        out / "luna_run" / "test_results"
        / (wandb_run_name or "celery_inference") / "celery"
    )

    per_slice_rows: List[Dict[str, object]] = []
    n_failed = 0

    with tracker.phase("inference", flush_to=out / "runtime.csv"):
        for idx, test_h5ad in enumerate(test_files, start=1):
            slice_label = _section_label_from_filename(test_h5ad)
            slice_ckpt_dir = ckpt_root / slice_label
            if not slice_ckpt_dir.exists():
                logger.warning(
                    f"[{idx}/{len(test_files)}] {slice_label}: no checkpoint "
                    f"at {slice_ckpt_dir} — skipping (train pass didn't cover "
                    f"this slice). This usually means a different "
                    f"exclude_test_files was used at train time."
                )
                n_failed += 1
                continue
            try:
                logger.info(f"[{idx}/{len(test_files)}] {slice_label}")
                row = _infer_one_slice(
                    slice_label=slice_label,
                    slice_dir=slice_ckpt_dir,
                    test_h5ad=test_h5ad,
                    out_slice_dir=test_results_dir / slice_label,
                    n_inference_samples=n_inference_samples,
                )
                per_slice_rows.append(row)
                logger.info(
                    f"      n={row['n_cells']:5d}  "
                    f"spr_med={row['spearman_per_cell_median']:.4f}  "
                    f"spr_mean={row['spearman_per_cell_mean']:.4f}  "
                    f"ref={row['ref_slice']}"
                )
                # Per-slice wandb log — wandb's UI plots
                # spearman_per_cell_median over slice_idx so a regression
                # on a specific slice is immediately visible.
                if use_wandb:
                    try:
                        import wandb  # type: ignore
                        wandb.log({
                            "slice_idx": idx,
                            "section_label": slice_label,
                            "ref_slice": row["ref_slice"],
                            f"spearman_per_cell_median/{slice_label}": row["spearman_per_cell_median"],
                            f"spearman_per_cell_mean/{slice_label}": row["spearman_per_cell_mean"],
                            f"n_cells/{slice_label}": row["n_cells"],
                            # Running aggregate so the wandb UI shows
                            # the cumulative mean-of-medians curve.
                            "spearman_mean_of_medians_running": float(
                                np.nanmean([r["spearman_per_cell_median"] for r in per_slice_rows])
                            ),
                        })
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"wandb.log failed for {slice_label}: {e}")

                # Optional per-slice GT-vs-pred scatter plot
                if make_plots and plots_dir is not None:
                    try:
                        pred = pd.read_csv(
                            test_results_dir / slice_label / "metadata_pred.csv",
                            index_col=0,
                        )
                        true = pd.read_csv(
                            test_results_dir / slice_label / "metadata_true.csv",
                            index_col=0,
                        )
                        _plot_pred_vs_truth(
                            coords_true=true[["coord_X", "coord_Y"]].to_numpy(),
                            coords_pred=pred[["coord_X", "coord_Y"]].to_numpy(),
                            cell_class=(true["cell_class"].astype(str).to_numpy()
                                        if "cell_class" in true.columns else None),
                            out_path=plots_dir / f"{slice_label}.svg",
                            title_prefix=slice_label,
                            method_label="CeLEry",
                        )
                    except Exception as plot_exc:
                        logger.warning(
                            f"  plot failed for {slice_label}: {plot_exc}"
                        )
            except Exception:
                logger.exception(
                    f"[{idx}/{len(test_files)}] {slice_label}: inference failed"
                )
                n_failed += 1

    if not per_slice_rows:
        raise RuntimeError(
            "No slices were successfully scored. Check the LSF stderr "
            "for traceback details."
        )

    # ---- Aggregate ----
    per_slice_df = pd.DataFrame(per_slice_rows)
    per_slice_df.to_csv(out / "per_slice_metrics.csv", index=False)

    medians = per_slice_df["spearman_per_cell_median"].dropna().to_numpy()
    means = per_slice_df["spearman_per_cell_mean"].dropna().to_numpy()

    aggregate = {
        "method": "celery",
        "spearman_mean_of_medians": float(medians.mean()) if medians.size else float("nan"),
        "spearman_median_of_medians": float(np.median(medians)) if medians.size else float("nan"),
        "spearman_std_of_medians": float(medians.std()) if medians.size else float("nan"),
        "spearman_mean_of_means": float(means.mean()) if means.size else float("nan"),
        "n_test_slices": int(len(per_slice_df)),
        "n_failed": int(n_failed),
        "n_cells_mean": float(per_slice_df["n_cells"].mean()),
        "n_cells_median": float(per_slice_df["n_cells"].median()),
        "n_cells_min": int(per_slice_df["n_cells"].min()),
        "n_cells_max": int(per_slice_df["n_cells"].max()),
    }

    pd.DataFrame([aggregate]).to_csv(out / "metrics.csv", index=False)
    (out / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, default=str)
    )

    # config snapshot
    (out / "config.yaml").write_text(yaml.safe_dump({
        "method": "celery",
        "phase": "inference",
        "data_dir": str(data_path),
        "celery_models": str(ckpt_root),
        "output_dir": str(out),
        "run_timestamp": run_ts,
        "seed": seed,
        "n_inference_samples": n_inference_samples,
        "wandb_mode": wandb_mode,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "exclude_test_files": exclude_test_files or [],
    }, default_flow_style=False))

    # compute requirements
    tracker.write_compute_requirements_csv(
        out / "compute_requirements.csv",
        method="CeLEry",
        run_timestamp=run_ts,
        extra={
            "phase": "inference",
            "n_test_slices": len(test_files),
            "n_inference_samples": n_inference_samples,
        },
    )

    logger.info("=" * 60)
    logger.info("Inference complete.")
    logger.info(f"  spearman_mean_of_medians: "
                f"{aggregate['spearman_mean_of_medians']:.4f}")
    logger.info(f"  per_slice_metrics.csv → {out / 'per_slice_metrics.csv'}")
    logger.info(f"  metrics.csv           → {out / 'metrics.csv'}")
    logger.info("=" * 60)

    # ---- wandb finish ----
    # Aggregate dict goes into wandb.run.summary so the run overview
    # row shows the headline metric AND every breakdown without
    # opening the run page. Mirrors the novosparc pipeline pattern.
    if use_wandb:
        try:
            import wandb  # type: ignore
            peak = tracker.peak_summary()
            wandb.log({
                "spearman_mean_of_medians": aggregate["spearman_mean_of_medians"],
                "spearman_mean_of_means": aggregate.get("spearman_mean_of_means"),
                "n_test_slices": aggregate["n_test_slices"],
                "n_failed": aggregate.get("n_failed", 0),
                "peak_gpu_mib": peak.get("peak_gpu_mib"),
                "peak_rss_mib": peak.get("peak_rss_mib"),
            })
            for k, v in aggregate.items():
                try:
                    wandb.run.summary[k] = v
                except Exception:  # noqa: BLE001
                    pass
            for k, v in (peak.items() if peak else {}.items()):
                try:
                    wandb.run.summary[k] = v
                except Exception:  # noqa: BLE001
                    pass
            wandb.finish()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"wandb finish failed: {e}")

    return aggregate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", required=True)
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to the training run's ``best_model.ckpt`` symlink, "
             "the celery_models/ dir, or a single model.obj. The "
             "script auto-resolves these three forms to the "
             "celery_models/ root.",
    )
    p.add_argument("--output_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n_inference_samples", type=int, default=1,
        help="CeLEry is deterministic so ensembling provides marginal "
             "gain, but the flag exists for cross-pipeline parity.",
    )
    p.add_argument(
        "--wandb_run_name", "--run_name",
        dest="wandb_run_name", default="celery_inference",
    )
    p.add_argument(
        "--wandb_mode", default="disabled",
        choices=("disabled", "online", "offline", "dryrun"),
    )
    p.add_argument("--wandb_project", default=None)
    p.add_argument(
        "--no_plots", action="store_true",
        help="Skip per-slice GT-vs-pred scatter plots (plots ON by default).",
    )
    p.add_argument(
        "--exclude_test_files", default=None,
        help="Comma-separated *_test.h5ad basenames to drop before scoring.",
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
        run_inference(
            data_dir=args.data_dir,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            seed=args.seed,
            wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project or "scgg",
            wandb_run_name=args.wandb_run_name,
            n_inference_samples=args.n_inference_samples,
            exclude_test_files=exclude_test_files,
            make_plots=not args.no_plots,
            run_timestamp=args.run_timestamp,
        )
    except Exception:
        logger.exception("CeLEry inference failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

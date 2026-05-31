"""run_celery_inference.py — score the global CeLEry model on the held-out mouse.

Companion to ``run_celery_train.py``. Loads the single ``model.obj``
(trained on ALL training slices concatenated per LUNA Supp Note 2),
predicts coordinates on each test slice, inverse-transforms back to
that slice's original coord scale, and computes the LUNA-paper
benchmark metrics.

Inputs (CLI):
    --data_dir          silver h5ad directory with *_test.h5ad
    --checkpoint        path to the training run's ``best_model.ckpt``
                        symlink (which points at model.obj) OR
                        directly at a model.obj file.
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
from typing import Dict, List, Optional

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

def _resolve_celery_model_path(checkpoint: Path):
    """Resolve --checkpoint to (model_dir, model_filename_stem).

    The new training script writes ``model.obj`` at the output_dir
    root + a ``best_model.ckpt`` symlink pointing at it. So:
      - --checkpoint = best_model.ckpt symlink → follow it.
      - --checkpoint = model.obj path → use it directly.
      - --checkpoint = directory containing model.obj → use that file.

    Returns:
        (model_dir, filename_stem)
        — model_dir is the path CeLEry's Predict_cord wants for ``path=``.
        — filename_stem is what it wants for ``filename=`` (no .obj suffix).
    """
    p = Path(checkpoint).resolve()
    if p.is_file() and p.name == "model.obj":
        return p.parent, "model"
    if p.is_dir() and (p / "model.obj").is_file():
        return p, "model"
    if p.name == "best_model.ckpt":
        # Resolve the symlink (or read the .ckpt itself if it's a regular file).
        target = Path(os.readlink(p)) if p.is_symlink() else p
        if not target.is_absolute():
            target = (p.parent / target).resolve()
        if target.is_file() and target.name == "model.obj":
            return target.parent, "model"
        if target.is_dir() and (target / "model.obj").is_file():
            return target, "model"
    raise FileNotFoundError(
        f"--checkpoint={checkpoint} does not resolve to a CeLEry model.obj. "
        f"Expected: a path to model.obj, a directory containing model.obj, "
        f"or the best_model.ckpt symlink pointing at one."
    )


def _load_manifest(model_dir: Path) -> Dict:
    """Read the training run's manifest.json (slice bboxes, hparams)."""
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json missing at {manifest_path}. The training run "
            f"either crashed before writing it or used an old per-test-slice "
            f"protocol; re-run training with the current multi-slice script."
        )
    return json.loads(manifest_path.read_text())


# ---------------------------------------------------------------------------
# Per-slice inference
# ---------------------------------------------------------------------------

def _infer_one_slice(
    slice_label: str,
    test_h5ad: Path,
    model_dir: Path,
    filename_stem: str,
    train_var_names: List[str],
    out_slice_dir: Path,
    n_inference_samples: int,
) -> Dict[str, object]:
    """Load test slice, predict with the global CeLEry model, score.

    Train-time gene order (from manifest.var_names) is the source of
    truth — we re-index the test AnnData to match before predicting,
    which is critical because CeLEry's MLP reads features positionally.
    A gene-order mismatch silently produces garbage predictions.
    """
    import CeLEry as cel  # type: ignore

    adata_qry, x_min, x_max, y_min, y_max = _load_h5ad_for_celery(test_h5ad)

    # ---- Restrict to (and reorder by) the training gene panel ----
    # Cells the model was trained on saw genes in a specific order;
    # use that exact order at inference. Genes present in train but
    # absent in test are an error (the model expects them as input);
    # genes present in test but not train are dropped silently.
    train_set = set(train_var_names)
    test_set = set(adata_qry.var_names)
    missing = train_set - test_set
    if missing:
        raise ValueError(
            f"{test_h5ad.name}: {len(missing)} training genes are missing "
            f"from this test slice (e.g. {sorted(missing)[:5]}). The "
            f"CeLEry model can't predict without all input features."
        )
    adata_qry = adata_qry[:, list(train_var_names)].copy()

    # ---- Z-score (matches training preprocessing) ----
    cel.get_zscore(adata_qry)

    # ---- Predict ----
    preds: List[np.ndarray] = []
    for _ in range(max(1, n_inference_samples)):
        pred_normed = cel.Predict_cord(
            data_test=adata_qry,
            path=str(model_dir),
            filename=filename_stem,
            location_data=None,
        )
        preds.append(np.asarray(pred_normed))
    pred_normed_mean = np.mean(np.stack(preds, axis=0), axis=0)

    # ---- Inverse-transform to test slice's coord scale ----
    # The model predicts in [0,1] (sigmoid output, training cells were
    # per-slice normalised to [0,1] before concat). For RSSD on the
    # ORIGINAL-scale truth, we need pred in original scale too. Use
    # the TEST slice's own bounding box. Spearman / contact-F1 are
    # rank- / percentile-based and unaffected by this transform; only
    # RSSD needs scale parity.
    pred_orig_scale = _invert_celery_normalisation(
        pred_normed_mean, x_min, x_max, y_min, y_max,
    )

    # ---- Build metadata DataFrames (same schema scgg/luna write) ----
    obs = adata_qry.obs
    if "_celery_cell_class" in adata_qry.uns:
        cell_class_array = np.asarray(adata_qry.uns["_celery_cell_class"])
    elif "cell_class" in obs.columns:
        cell_class_array = obs["cell_class"].to_numpy()
    else:
        cell_class_array = None

    df_pred = pd.DataFrame({
        "coord_X": pred_orig_scale[:, 0],
        "coord_Y": pred_orig_scale[:, 1],
        **({"cell_class": cell_class_array} if cell_class_array is not None else {}),
    }, index=obs.index)
    df_pred.index.name = "cell_ID"

    df_true = pd.DataFrame({
        "coord_X": obs.iloc[:, 0].to_numpy(dtype=np.float64),
        "coord_Y": obs.iloc[:, 1].to_numpy(dtype=np.float64),
        **({"cell_class": cell_class_array} if cell_class_array is not None else {}),
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
    """Score the global CeLEry model on every test slice + aggregate."""
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"--data_dir not found: {data_path}")

    model_dir, filename_stem = _resolve_celery_model_path(Path(checkpoint))
    logger.info(f"model_dir: {model_dir}, filename: {filename_stem}.obj")
    manifest = _load_manifest(model_dir)
    train_var_names = manifest.get("var_names", [])
    if not train_var_names:
        raise ValueError(
            f"manifest.json at {model_dir}/manifest.json has no 'var_names'. "
            f"This means the training run used an older script; re-run "
            f"training with the current multi-slice script."
        )

    # ---- TS resolution ----
    if run_timestamp is not None:
        if not re.fullmatch(r"\d{8}_\d{6}(?:_[A-Za-z0-9]+)?", run_timestamp):
            raise ValueError(
                f"run_timestamp must be YYYYMMDD_HHMMSS or "
                f"YYYYMMDD_HHMMSS_<suffix>; got {run_timestamp!r}"
            )
        run_ts = run_timestamp
    else:
        m = re.search(r"\d{8}_\d{6}(?:_[A-Za-z0-9]+)?", str(checkpoint))
        run_ts = m.group(0) if m else datetime.now().strftime("%Y%m%d_%H%M%S")

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
    logger.info("CeLEry inference-phase (multi-slice global, LUNA protocol)")
    logger.info("=" * 60)
    logger.info(f"  data_dir            : {data_path}")
    logger.info(f"  model_dir           : {model_dir}")
    logger.info(f"  output_dir          : {out}")
    logger.info(f"  run_ts              : {run_ts}")
    logger.info(f"  n_inference_samples : {n_inference_samples}")
    logger.info(f"  train n_slices      : {manifest.get('n_train_slices', '?')}")
    logger.info(f"  train n_genes       : {len(train_var_names)}")
    logger.info("=" * 60)

    tracker = _RuntimeTracker()

    # ---- wandb init ----
    use_wandb = wandb_mode != "disabled"
    if use_wandb:
        try:
            import wandb  # type: ignore
            wandb.init(
                project=wandb_project or "celery",
                name=wandb_run_name,
                mode=wandb_mode,
                config={
                    "method": "celery",
                    "phase": "inference",
                    "training_mode": "multi_slice_global",
                    "data_dir": str(data_path),
                    "model_dir": str(model_dir),
                    "seed": seed,
                    "n_inference_samples": n_inference_samples,
                    "exclude_test_files": exclude_test_files or [],
                    "run_timestamp": run_ts,
                    "output_dir": str(out),
                    "tags": ["celery", "inference", dataset_name, "multi_slice"],
                },
            )
            logger.info("wandb initialised")
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
            try:
                logger.info(f"[{idx}/{len(test_files)}] {slice_label}")
                row = _infer_one_slice(
                    slice_label=slice_label,
                    test_h5ad=test_h5ad,
                    model_dir=model_dir,
                    filename_stem=filename_stem,
                    train_var_names=train_var_names,
                    out_slice_dir=test_results_dir / slice_label,
                    n_inference_samples=n_inference_samples,
                )
                per_slice_rows.append(row)
                logger.info(
                    f"      n={row['n_cells']:5d}  "
                    f"spr_med={row['spearman_per_cell_median']:.4f}  "
                    f"spr_mean={row['spearman_per_cell_mean']:.4f}"
                )
                if use_wandb:
                    try:
                        import wandb  # type: ignore
                        wandb.log({
                            "slice_idx": idx,
                            "section_label": slice_label,
                            f"spearman_per_cell_median/{slice_label}":
                                row["spearman_per_cell_median"],
                            f"spearman_per_cell_mean/{slice_label}":
                                row["spearman_per_cell_mean"],
                            f"n_cells/{slice_label}": row["n_cells"],
                            "spearman_mean_of_medians_running": float(
                                np.nanmean([r["spearman_per_cell_median"]
                                            for r in per_slice_rows])
                            ),
                        })
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"wandb.log failed for {slice_label}: {e}")

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
            "No slices were successfully scored. Check the LSF stderr."
        )

    # ---- Aggregate ----
    per_slice_df = pd.DataFrame(per_slice_rows)
    per_slice_df.to_csv(out / "per_slice_metrics.csv", index=False)

    medians = per_slice_df["spearman_per_cell_median"].dropna().to_numpy()
    means = per_slice_df["spearman_per_cell_mean"].dropna().to_numpy()

    aggregate = {
        "method": "celery",
        "training_mode": "multi_slice_global",
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

    (out / "config.yaml").write_text(yaml.safe_dump({
        "method": "celery",
        "phase": "inference",
        "training_mode": "multi_slice_global",
        "data_dir": str(data_path),
        "model_dir": str(model_dir),
        "output_dir": str(out),
        "run_timestamp": run_ts,
        "seed": seed,
        "n_inference_samples": n_inference_samples,
        "wandb_mode": wandb_mode,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "exclude_test_files": exclude_test_files or [],
    }, default_flow_style=False))

    tracker.write_compute_requirements_csv(
        out / "compute_requirements.csv",
        method="CeLEry",
        run_timestamp=run_ts,
        extra={
            "phase": "inference",
            "training_mode": "multi_slice_global",
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
        help="Path to the training run's best_model.ckpt symlink, "
             "model.obj file, or the directory containing model.obj.",
    )
    p.add_argument("--output_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n_inference_samples", type=int, default=1,
        help="CeLEry is deterministic so ensembling provides marginal "
             "gain; flag exists for cross-pipeline parity.",
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
            wandb_project=args.wandb_project or "celery",
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

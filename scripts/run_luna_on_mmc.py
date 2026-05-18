#!/usr/bin/env python
"""
Train LUNA on the MERFISH mouse cortex dataset (LUNA paper Figure 3 split).

Mirror of ``scgg/scripts/run_luna_cortex_benchmark.py`` (which trains scGG),
but the trainee is LUNA itself — invoked as a subprocess into the LUNA
Python 3.9 / torch-2.0.1 venv set up by
``scgg-reproducibility/analysis/benchmarking/setup_luna_env.sh``. Use this
for direct A/B comparisons against scGG: same data, same artifact layout,
same per-slice / aggregate metric outputs.

Pipeline
--------
  1. Discover Mouse 1 (train) and Mouse 2 (test) silver h5ad files. Both
     prefix conventions are accepted (``mmc_mouseM_sliceS.h5ad`` and the
     legacy ``merfish_mouse_cortex_mouseM_sliceS.h5ad``).
  2. Convert per-slice h5ads to LUNA's expected CSV layout
     (gene columns first, then coord_X / coord_Y / cell_section / cell_class)
     and write train.csv + test.csv under ``<output_dir>/work/``.
  3. Invoke LUNA's ``main.py`` with ``general.mode=train_and_test``,
     ``hydra.run.dir=<output_dir>/luna_run``, and the appropriate
     Hydra overrides for epochs / batch size / paths.
  4. After training, read LUNA's per-section test outputs
     (``metadata_pred.csv`` / ``metadata_true.csv``) and compute Spearman /
     contact F1 / Kabsch RSSD via ``scgg.evaluation.luna_metrics``.
  5. Write ``per_slice_metrics.csv``, ``aggregate_metrics.json``,
     ``config.yaml`` (frozen snapshot), and ``train.log`` to
     ``<output_dir>`` — exactly the artifacts the scgg trainer produces.

Output layout
-------------
By default ``--output_dir`` resolves to
``/nfs/team361/sb75/scgg-reproducibility/artifacts/<data_dir.name>/luna_model/<YYYYMMDD_HHMMSS>/``
so multiple LUNA training runs coexist and the matching inference
outputs can pin to a specific timestamp under
``.../luna_inference/<YYYYMMDD_HHMMSS>/``.

Usage
-----
    python scripts/run_luna_on_mmc.py \\
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --luna_venv /nfs/team361/sb75/.venvs/luna \\
        --luna_repo /nfs/team361/sb75/code/LUNA \\
        --epochs 1000 \\
        --batch_size 6

(--output_dir defaults to a timestamped subdir; --epochs / --batch_size
defaults match the LUNA paper.)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from scgg.luna_bridge import (
    _ARTIFACTS_ROOT,
    build_luna_csv,
    enumerate_slice_files,
    find_latest_checkpoint,
    fresh_run_timestamp,
    invoke_luna,
    read_luna_predictions,
    split_by_mouse,
)

logger = logging.getLogger("luna_train")


# ---------------------------------------------------------------------------
# Metric evaluation (delegates to scgg.evaluation.luna_metrics)
# ---------------------------------------------------------------------------


def _evaluate_luna_outputs(
    test_save_dir: Path,
    contact_percentile: float,
    rssd: bool,
) -> tuple[List[Dict[str, float]], Dict[str, float]]:
    """Walk LUNA's per-section predictions and compute scgg metrics."""
    from scgg.evaluation.luna_metrics import aggregate_slices, evaluate_slice

    sections = read_luna_predictions(test_save_dir)
    if not sections:
        raise FileNotFoundError(
            f"No LUNA prediction files under {test_save_dir}."
        )
    logger.info(f"Evaluating {len(sections)} LUNA prediction sections")

    per_slice: List[Dict[str, float]] = []
    for section_label, dfs in sections.items():
        pred = dfs["pred"]
        true = dfs["true"]
        coords_pred = pred[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
        coords_true = true[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
        cell_class = (
            true["cell_class"].astype(str).to_numpy()
            if "cell_class" in true.columns
            else None
        )
        if len(coords_true) < 10:
            logger.info(f"  {section_label}: only {len(coords_true)} cells; skipping")
            continue
        row = evaluate_slice(
            coords_true, coords_pred, cell_class,
            contact_percentile=contact_percentile,
            compute_rssd=rssd,
        )
        row["section_label"] = section_label
        per_slice.append(row)
        logger.info(
            f"  {section_label:32s}  "
            f"spr_median={row['spearman_per_cell_median']:.4f}  "
            f"spr_mean={row['spearman_per_cell_mean']:.4f}  "
            f"prec={row['precision']:.4f}  "
            f"rssd={row.get('absolute_rssd', float('nan')):.2f}"
        )
    agg = aggregate_slices(per_slice)
    return per_slice, agg


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def train_luna(
    data_dir: str,
    output_dir: Optional[str] = None,
    epochs: int = 1000,
    batch_size: int = 6,
    lr: Optional[float] = None,
    seed: int = 42,
    luna_venv: str = "/nfs/team361/sb75/.venvs/luna",
    luna_repo: str = "/nfs/team361/sb75/code/LUNA",
    run_name: str = "MERFISH_mouse_cortex",
    log2_normalize: bool = True,
    contact_percentile: float = 0.01,
    compute_rssd: bool = True,
    extra_overrides: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Train LUNA on Mouse 1 and evaluate on Mouse 2.

    Args mirror scgg/scripts/run_luna_cortex_benchmark.py:run_benchmark
    where applicable. LUNA-specific extras (``luna_venv``, ``luna_repo``,
    ``run_name``, ``log2_normalize``, ``extra_overrides``) are added.

    Returns the aggregated metrics dict from
    ``scgg.evaluation.luna_metrics.aggregate_slices``.
    """
    data_path = Path(data_dir)
    run_timestamp = fresh_run_timestamp()
    if output_dir is None:
        out = _ARTIFACTS_ROOT / data_path.name / "luna_model" / run_timestamp
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out / "train.log", mode="w"),
        ],
        force=True,
    )
    logger.info(f"Run timestamp: {run_timestamp}")
    logger.info(f"Output dir:    {out}")

    luna_venv_p = Path(luna_venv)
    luna_repo_p = Path(luna_repo)

    # ---- 1. Discover silver h5ads ---------------------------------------
    files = enumerate_slice_files(data_path)
    train_files = split_by_mouse(files, mouse_id=1)
    test_files = split_by_mouse(files, mouse_id=2)
    logger.info(
        f"Silver dir: {data_path} "
        f"({len(train_files)} train [Mouse 1], {len(test_files)} test [Mouse 2])"
    )
    if not train_files or not test_files:
        raise FileNotFoundError(
            f"Need Mouse 1 (train) AND Mouse 2 (test) slices under "
            f"{data_path}. Found: train={len(train_files)}, "
            f"test={len(test_files)}"
        )

    # ---- 2. Build LUNA CSVs ---------------------------------------------
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    train_csv = work / "train.csv"
    test_csv = work / "test.csv"
    if train_csv.exists() and test_csv.exists():
        logger.info("LUNA CSVs already exist under work/; reusing")
        # Recover n_genes from the existing CSV header
        import pandas as pd
        head = pd.read_csv(train_csv, nrows=1, index_col=0)
        n_genes = len(head.columns) - 4  # subtract coord_X, coord_Y, cell_section, cell_class
    else:
        logger.info(f"Writing train CSV -> {train_csv}")
        train_stats = build_luna_csv(train_files, train_csv, log2_normalize=log2_normalize)
        logger.info(f"  train: {train_stats['n_rows']:,} rows, "
                    f"{train_stats['n_genes']} genes, "
                    f"{train_stats['n_sections']} sections")
        logger.info(f"Writing test CSV  -> {test_csv}")
        test_stats = build_luna_csv(test_files, test_csv, log2_normalize=log2_normalize)
        logger.info(f"  test : {test_stats['n_rows']:,} rows, "
                    f"{test_stats['n_genes']} genes, "
                    f"{test_stats['n_sections']} sections")
        n_genes = int(train_stats["n_genes"])

    # ---- 3. Invoke LUNA training ----------------------------------------
    luna_run_dir = out / "luna_run"
    luna_run_dir.mkdir(parents=True, exist_ok=True)
    test_save_dir = luna_run_dir / "test_results"
    test_save_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"general.name={run_name}",
        "general.mode=train_and_test",
        f"general.seed={seed}",
        f"dataset.train_data_path={train_csv.resolve()}",
        f"dataset.test_data_path={test_csv.resolve()}",
        "dataset.gene_columns_start=0",
        f"dataset.gene_columns_end={n_genes}",
        f"train.batch_size={batch_size}",
        f"train.n_epochs={epochs}",
        f"test.save_dir={test_save_dir.resolve()}",
        f"hydra.run.dir={luna_run_dir.resolve()}",
    ]
    if lr is not None:
        overrides.append(f"train.lr={lr}")
    if extra_overrides:
        overrides.extend(extra_overrides)

    log_path = out / "luna_stdout.log"
    rc = invoke_luna(
        luna_venv=luna_venv_p,
        luna_repo=luna_repo_p,
        overrides=overrides,
        cwd=luna_repo_p,
        log_path=log_path,
    )
    if rc != 0:
        raise RuntimeError(
            f"LUNA training failed (exit {rc}). See {log_path}"
        )

    # ---- 4. Pin a "best_model.pt"-style reference to the final checkpoint
    final_ckpt = find_latest_checkpoint(luna_run_dir)
    if final_ckpt is not None:
        # Symlink so downstream inference scripts can resolve a stable path
        # without knowing LUNA's epoch numbering. Use a relative symlink
        # so the artifact dir remains self-contained.
        stable_link = out / "best_model.ckpt"
        if stable_link.exists() or stable_link.is_symlink():
            stable_link.unlink()
        try:
            stable_link.symlink_to(final_ckpt.relative_to(out))
        except (OSError, ValueError):
            # Fallback: write the path as a tiny pointer file (some
            # filesystems disallow symlinks).
            stable_link = out / "best_model.path"
            stable_link.write_text(str(final_ckpt.resolve()))
        logger.info(f"Best checkpoint: {final_ckpt}  (pinned at {stable_link})")
    else:
        logger.warning(
            "No checkpoint found under luna_run/checkpoints/ — did training "
            "complete?"
        )

    # ---- 5. Evaluate LUNA's test outputs --------------------------------
    per_slice, agg = _evaluate_luna_outputs(
        test_save_dir,
        contact_percentile=contact_percentile,
        rssd=compute_rssd,
    )

    luna_paper = 0.448
    headline = agg.get("spearman_mean_of_medians", float("nan"))
    logger.info("=" * 72)
    logger.info("LUNA (this run) — aggregated metrics across test slices")
    logger.info("=" * 72)
    for k, v in agg.items():
        logger.info(f"  {k:34s} = {v}")
    logger.info("-" * 72)
    logger.info(
        f"Headline (mean-of-per-slice-median Spearman): {headline:.4f}   |   "
        f"LUNA paper: {luna_paper:.4f}"
    )
    if not np.isnan(headline):
        logger.info(f"Delta vs LUNA paper: {(headline - luna_paper) * 100:+.2f} pp")

    # ---- 6. Save artifacts (mirror scgg's run_luna_cortex_benchmark.py) -
    fieldnames = sorted({k for r in per_slice for k in r.keys()})
    with open(out / "per_slice_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_slice:
            w.writerow(r)
    with open(out / "aggregate_metrics.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)
    cfg_snap = {
        "method": "LUNA",
        "run_timestamp": run_timestamp,
        "data_dir": str(data_path),
        "luna_venv": str(luna_venv_p),
        "luna_repo": str(luna_repo_p),
        "run_name": run_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "log2_normalize": log2_normalize,
        "contact_percentile": contact_percentile,
        "compute_rssd": compute_rssd,
        "extra_overrides": extra_overrides or [],
        "luna_run_dir": str(luna_run_dir),
        "test_save_dir": str(test_save_dir),
        "best_checkpoint": str(final_ckpt) if final_ckpt else None,
    }
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_snap, f, sort_keys=False)

    logger.info(f"Wrote LUNA training artifacts to {out}")
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data_dir", required=True,
        help="Per-slice silver h5ad directory (LUNA cortex split). "
             "Mouse 1 slices => train, Mouse 2 slices => test.",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Where to write the trained LUNA checkpoint + per-slice / "
             "aggregate metrics. Default derives from --data_dir's basename "
             "and adds a timestamp: "
             "/nfs/team361/sb75/scgg-reproducibility/artifacts/"
             "<data_dir_name>/luna_model/<YYYYMMDD_HHMMSS>/.",
    )
    p.add_argument("--epochs", type=int, default=1000,
                   help="train.n_epochs override (LUNA paper default: 1000).")
    p.add_argument("--batch_size", type=int, default=6,
                   help="train.batch_size override (LUNA paper default: 6 sections).")
    p.add_argument("--lr", type=float, default=None,
                   help="Optional train.lr override (LUNA default: 5e-4).")
    p.add_argument("--seed", type=int, default=42,
                   help="general.seed override (LUNA default: 0).")
    p.add_argument(
        "--luna_venv", default="/nfs/team361/sb75/.venvs/luna",
        help="Path to the uv venv created by setup_luna_env.sh.",
    )
    p.add_argument(
        "--luna_repo", default="/nfs/team361/sb75/code/LUNA",
        help="Path to the cloned LUNA repository.",
    )
    p.add_argument(
        "--run_name", default="MERFISH_mouse_cortex",
        help="Sets general.name in LUNA's Hydra config.",
    )
    p.add_argument(
        "--no_log2_normalize", action="store_true",
        help="Skip log2(x+1) normalization when writing the LUNA CSVs. "
             "LUNA expects log2-normalized expression; only flip this off "
             "if your silver h5ads are ALREADY log2-normalized.",
    )
    p.add_argument(
        "--contact_percentile", type=float, default=0.01,
        help="Percentile for the contact F1 metric (matches scgg default).",
    )
    p.add_argument("--no_rssd", action="store_true",
                   help="Skip the Kabsch RSSD computation.")
    p.add_argument(
        "--luna_override", action="append", default=[],
        help="Extra Hydra overrides to pass to LUNA, e.g. "
             "'--luna_override train.lr=1e-4'. Repeatable.",
    )
    args = p.parse_args()

    try:
        train_luna(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            luna_venv=args.luna_venv,
            luna_repo=args.luna_repo,
            run_name=args.run_name,
            log2_normalize=not args.no_log2_normalize,
            contact_percentile=args.contact_percentile,
            compute_rssd=not args.no_rssd,
            extra_overrides=args.luna_override,
        )
    except Exception:
        logger.exception("LUNA training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

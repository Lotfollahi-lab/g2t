#!/usr/bin/env python
"""
Train LUNA on the MERFISH mouse cortex dataset (LUNA paper Figure 3 split).

Mirror of ``scgg/scripts/run_luna_cortex_benchmark.py`` (the scGG trainer)
but the trainee is LUNA itself. This script is **self-contained**: no
dependency on the scgg package. It is meant to be run *in the LUNA Python
environment* (Python 3.9, torch 2.0.1, etc. — the one that
``scgg-reproducibility/analysis/benchmarking/setup_luna_env.sh`` creates
and that LUNA itself runs in).

Input is the **same silver h5ad directory** that scGG reads from
(``--data_dir``). We apply LUNA's expected ``log2(x + 1)`` normalization
when building its input CSVs (scGG applies its own
``normalize_total + log1p + scale`` separately). The silver layer itself
is identical for both methods — same h5ads, same coords, same
``cell_class`` labels.

Pipeline
--------
  1. Discover Mouse 1 (train) and Mouse 2 (test) silver h5ad files. Both
     prefix conventions are accepted (``mmc_mouseM_sliceS.h5ad`` and the
     legacy ``merfish_mouse_cortex_mouseM_sliceS.h5ad``).
  2. Convert per-slice h5ads to LUNA's expected CSV layout
     (gene columns first, then ``coord_X`` / ``coord_Y`` /
     ``cell_section`` / ``cell_class``).
  3. Invoke LUNA's ``main.py`` (same Python interpreter, same env) with
     ``general.mode=train_and_test`` and the appropriate Hydra overrides.
  4. After training, pin the latest checkpoint to ``best_model.ckpt``,
     read LUNA's per-section test outputs, and write a per-slice metrics
     CSV — same artifact layout as the scGG trainer.

Output layout
-------------
``--output_dir`` defaults to
``/nfs/team361/sb75/scgg-reproducibility/artifacts/<data_dir.name>/luna_model/<YYYYMMDD_HHMMSS>/``
so multiple LUNA training runs coexist. The matching inference outputs
land under ``.../luna_inference/<YYYYMMDD_HHMMSS>/``.

Usage
-----
    # Activate the LUNA env first
    source /nfs/team361/sb75/.venvs/luna/bin/activate

    python scripts/run_luna_on_mmc.py \\
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --luna_repo /nfs/team361/sb75/scgg-reproducibility/analysis/benchmarking/luna \\
        --epochs 1000 --batch_size 6
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("luna_train")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


_ARTIFACTS_ROOT = Path("/nfs/team361/sb75/scgg-reproducibility/artifacts")
_DEFAULT_LUNA_REPO = Path(
    "/nfs/team361/sb75/scgg-reproducibility/analysis/benchmarking/luna"
)

# Cortex silver-file naming. Accepts both prefixes so a mid-rename dir works.
_SLICE_RE = re.compile(
    r"^(?:mmc|merfish_mouse_cortex)_mouse(?P<mouse>\d+)_slice(?P<slice>\d+)\.h5ad$"
)
_EPOCH_RE = re.compile(r"epoch=(\d+)")


# ---------------------------------------------------------------------------
# Silver-h5ad discovery
# ---------------------------------------------------------------------------


def _enumerate_slice_files(silver_dir: Path) -> List[Tuple[int, int, Path]]:
    """Return (mouse_id, slice_id, path) for each cortex silver h5ad."""
    out: List[Tuple[int, int, Path]] = []
    for p in sorted(silver_dir.iterdir()):
        m = _SLICE_RE.match(p.name)
        if not m:
            continue
        out.append((int(m["mouse"]), int(m["slice"]), p))
    return out


def _split_by_mouse(
    files: List[Tuple[int, int, Path]], mouse_id: int,
) -> List[Tuple[int, int, Path]]:
    return [f for f in files if f[0] == mouse_id]


# ---------------------------------------------------------------------------
# Build LUNA-format CSVs from per-slice h5ads
# ---------------------------------------------------------------------------


def _build_luna_csv(
    files: List[Tuple[int, int, Path]],
    out_csv: Path,
    log2_normalize: bool = True,
) -> Dict[str, object]:
    """Concatenate per-slice h5ads into one CSV in LUNA's input format.

    LUNA expects:
      * gene columns first (positions ``0..n_genes-1``)
      * then ``coord_X``, ``coord_Y``, ``cell_section``, ``cell_class``
      * index = original cell barcode
    """
    import anndata as ad
    import scipy.sparse as sp

    rows_total = 0
    gene_names: Optional[List[str]] = None
    first = True

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    for mouse, slice_id, path in files:
        adata = ad.read_h5ad(path)
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        if log2_normalize:
            X = np.log2(X + 1.0)

        if gene_names is None:
            gene_names = list(adata.var_names)
        elif list(adata.var_names) != gene_names:
            raise ValueError(
                f"Gene panel mismatch in {path.name}: expected "
                f"{len(gene_names)} genes, got {adata.n_vars}"
            )

        section_label = f"mouse{mouse}_slice{slice_id}"
        cell_class = (
            adata.obs["cell_class"].astype(str).values
            if "cell_class" in adata.obs.columns
            else np.full(adata.n_obs, "unknown")
        )
        if "spatial" in adata.obsm:
            xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]
        else:
            xy = np.column_stack([
                adata.obs["coord_X"].to_numpy(dtype=np.float32),
                adata.obs["coord_Y"].to_numpy(dtype=np.float32),
            ])

        df = pd.DataFrame(X, columns=gene_names)
        df["coord_X"] = xy[:, 0]
        df["coord_Y"] = xy[:, 1]
        df["cell_section"] = section_label
        df["cell_class"] = cell_class
        df.index = adata.obs_names
        df.index.name = "cell_id"

        df.to_csv(out_csv, mode="a", header=first)
        rows_total += len(df)
        first = False
        logger.info(f"    wrote {len(df):>6,} cells from {section_label}")

    return {
        "n_rows": rows_total,
        "n_genes": int(len(gene_names)) if gene_names else 0,
        "n_sections": len(files),
    }


# ---------------------------------------------------------------------------
# Invoke LUNA in-process (same env, same Python via sys.executable)
# ---------------------------------------------------------------------------


def _invoke_luna(
    luna_repo: Path,
    overrides: List[str],
    log_path: Path,
) -> int:
    """Run LUNA's main.py with Hydra overrides, in the same Python env.

    Uses ``sys.executable`` so the subprocess inherits whatever env the
    script is running in — that's the LUNA venv if the user activated
    it before launching, exactly as documented.
    """
    main_py = luna_repo / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"LUNA main.py not found: {main_py}")

    cmd = [sys.executable, str(main_py), *overrides]
    logger.info("Invoking LUNA:")
    for arg in cmd:
        logger.info(f"    {arg}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "wb") as f:
        proc = subprocess.run(
            cmd, cwd=str(luna_repo), stdout=f, stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = (time.time() - t0) / 60.0
    logger.info(
        f"LUNA exited with code {proc.returncode} after {elapsed:.1f} min "
        f"(log: {log_path})"
    )
    if proc.returncode != 0 and log_path.exists():
        with open(log_path) as f:
            tail = f.read().splitlines()[-50:]
        for line in tail:
            logger.error(f"  | {line}")
    return proc.returncode


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def _find_latest_checkpoint(luna_run_dir: Path) -> Optional[Path]:
    """Latest-epoch checkpoint under ``{run_dir}/checkpoints/``."""
    ckpt_dir = luna_run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates: List[Tuple[int, Path]] = []
    for p in ckpt_dir.glob("*.ckpt"):
        m = _EPOCH_RE.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


# ---------------------------------------------------------------------------
# Read LUNA test outputs + compute per-slice metrics
# ---------------------------------------------------------------------------


def _read_luna_predictions(
    test_save_dir: Path,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Return {section_label: (pred_df, true_df)} from LUNA outputs."""
    pred_files = list(test_save_dir.rglob("metadata_pred.csv"))
    if not pred_files:
        raise FileNotFoundError(
            f"No metadata_pred.csv under {test_save_dir} — did LUNA finish?"
        )
    out: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for pred_path in sorted(pred_files):
        true_path = pred_path.parent / "metadata_true.csv"
        if not true_path.exists():
            logger.warning(f"  {pred_path.parent.name}: missing metadata_true.csv")
            continue
        pred = pd.read_csv(pred_path, index_col=0)
        true = pd.read_csv(true_path, index_col=0)
        if not pred.index.equals(true.index):
            common = pred.index.intersection(true.index)
            pred = pred.loc[common]
            true = true.loc[common]
        out[pred_path.parent.name] = (pred, true)
    return out


def _per_cell_spearman_median(
    coords_true: np.ndarray, coords_pred: np.ndarray,
) -> Tuple[float, float]:
    """LUNA's headline metric: median & mean of per-cell Spearman of
    pairwise-distance rows. Self-contained — no scgg deps."""
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr

    dt = cdist(coords_true, coords_true)
    dp = cdist(coords_pred, coords_pred)
    rhos: List[float] = []
    n = coords_true.shape[0]
    for i in range(n):
        r, _ = spearmanr(dt[i], dp[i])
        if r is not None and not np.isnan(r):
            rhos.append(float(r))
    if not rhos:
        return float("nan"), float("nan")
    return float(np.median(rhos)), float(np.mean(rhos))


def _evaluate_predictions(
    sections: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
) -> List[Dict[str, float]]:
    """Compute per-section Spearman; return one row per section."""
    rows: List[Dict[str, float]] = []
    for label, (pred, true) in sections.items():
        coords_pred = pred[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
        coords_true = true[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
        if len(coords_true) < 10:
            logger.info(f"  {label}: only {len(coords_true)} cells; skipping")
            continue
        med, mean = _per_cell_spearman_median(coords_true, coords_pred)
        rows.append({
            "section_label": label,
            "n_cells": int(coords_true.shape[0]),
            "spearman_per_cell_median": med,
            "spearman_per_cell_mean": mean,
        })
        logger.info(
            f"  {label:32s}  n={coords_true.shape[0]:>5d}  "
            f"spr_median={med:.4f}  spr_mean={mean:.4f}"
        )
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_benchmark(
    data_dir: str,
    output_dir: Optional[str] = None,
    epochs: int = 1000,
    batch_size: int = 6,
    lr: Optional[float] = None,
    seed: int = 42,
    luna_repo: str = str(_DEFAULT_LUNA_REPO),
    run_name: str = "MERFISH_mouse_cortex",
    log2_normalize: bool = True,
    extra_overrides: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Train LUNA on Mouse 1, evaluate on Mouse 2.

    Args mirror scgg/scripts/run_luna_cortex_benchmark.py:run_benchmark
    where applicable. LUNA-specific extras (``luna_repo``, ``run_name``,
    ``log2_normalize``, ``extra_overrides``) replace the scgg
    loss-shaping knobs that don't apply here.

    Returns a dict with the headline ``spearman_mean_of_medians`` metric.
    """
    data_path = Path(data_dir)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        out = _ARTIFACTS_ROOT / data_path.name / "luna_model" / run_ts
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
    logger.info(f"Run timestamp: {run_ts}")
    logger.info(f"Output dir:    {out}")

    luna_repo_p = Path(luna_repo)
    if not luna_repo_p.exists():
        raise FileNotFoundError(f"LUNA repo not found: {luna_repo_p}")

    # ---- 1. Discover silver h5ads ---------------------------------------
    files = _enumerate_slice_files(data_path)
    train_files = _split_by_mouse(files, 1)
    test_files = _split_by_mouse(files, 2)
    logger.info(
        f"Silver dir: {data_path} "
        f"({len(train_files)} train [Mouse 1], {len(test_files)} test [Mouse 2])"
    )
    if not train_files or not test_files:
        raise FileNotFoundError(
            f"Need Mouse 1 AND Mouse 2 slices under {data_path}. "
            f"Found train={len(train_files)}, test={len(test_files)}"
        )

    # ---- 2. Build LUNA CSVs ---------------------------------------------
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    train_csv = work / "train.csv"
    test_csv = work / "test.csv"
    if train_csv.exists() and test_csv.exists():
        logger.info("LUNA CSVs already exist under work/; reusing")
        head = pd.read_csv(train_csv, nrows=1, index_col=0)
        n_genes = len(head.columns) - 4  # coord_X, coord_Y, cell_section, cell_class
    else:
        logger.info(f"Writing train CSV -> {train_csv}")
        train_stats = _build_luna_csv(train_files, train_csv, log2_normalize=log2_normalize)
        logger.info(
            f"  train: {train_stats['n_rows']:,} rows, "
            f"{train_stats['n_genes']} genes, "
            f"{train_stats['n_sections']} sections"
        )
        logger.info(f"Writing test CSV  -> {test_csv}")
        test_stats = _build_luna_csv(test_files, test_csv, log2_normalize=log2_normalize)
        logger.info(
            f"  test : {test_stats['n_rows']:,} rows, "
            f"{test_stats['n_genes']} genes, "
            f"{test_stats['n_sections']} sections"
        )
        n_genes = int(train_stats["n_genes"])

    # ---- 3. Invoke LUNA train_and_test ---------------------------------
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
    rc = _invoke_luna(luna_repo_p, overrides, log_path)
    if rc != 0:
        raise RuntimeError(f"LUNA training failed (exit {rc}). See {log_path}")

    # ---- 4. Pin a stable "best_model.ckpt" reference -------------------
    final_ckpt = _find_latest_checkpoint(luna_run_dir)
    if final_ckpt is not None:
        stable_link = out / "best_model.ckpt"
        if stable_link.exists() or stable_link.is_symlink():
            stable_link.unlink()
        try:
            stable_link.symlink_to(final_ckpt.relative_to(out))
        except (OSError, ValueError):
            stable_link = out / "best_model.path"
            stable_link.write_text(str(final_ckpt.resolve()))
        logger.info(f"Best checkpoint: {final_ckpt}  (pinned at {stable_link})")
    else:
        logger.warning("No checkpoint found under luna_run/checkpoints/")

    # ---- 5. Evaluate predictions ---------------------------------------
    sections = _read_luna_predictions(test_save_dir)
    per_slice = _evaluate_predictions(sections)

    headline = float("nan")
    if per_slice:
        medians = [r["spearman_per_cell_median"] for r in per_slice
                   if not np.isnan(r["spearman_per_cell_median"])]
        if medians:
            headline = float(np.mean(medians))

    luna_paper = 0.448
    logger.info("=" * 72)
    logger.info("LUNA (this run) — aggregated metrics across test slices")
    logger.info("=" * 72)
    logger.info(f"  spearman_mean_of_medians (n={len(per_slice)} slices) = {headline:.4f}")
    logger.info(f"  LUNA paper headline                                   = {luna_paper:.4f}")
    if not np.isnan(headline):
        logger.info(f"  Delta vs LUNA paper                                  = "
                    f"{(headline - luna_paper) * 100:+.2f} pp")

    # ---- 6. Write artifacts (mirror scgg's training script) ------------
    if per_slice:
        fieldnames = sorted({k for r in per_slice for k in r.keys()})
        with open(out / "per_slice_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in per_slice:
                w.writerow(r)
    agg = {"spearman_mean_of_medians": headline, "n_test_slices": len(per_slice)}
    with open(out / "aggregate_metrics.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)

    cfg_snap = {
        "method": "LUNA",
        "run_timestamp": run_ts,
        "data_dir": str(data_path),
        "luna_repo": str(luna_repo_p),
        "run_name": run_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "log2_normalize": log2_normalize,
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
             "Mouse 1 => train, Mouse 2 => test.",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Where to write LUNA's outputs + per-slice metrics. Default: "
             "/nfs/team361/sb75/scgg-reproducibility/artifacts/"
             "<data_dir_name>/luna_model/<YYYYMMDD_HHMMSS>/.",
    )
    p.add_argument("--epochs", type=int, default=1000,
                   help="train.n_epochs override (LUNA paper default: 1000).")
    p.add_argument("--batch_size", type=int, default=6,
                   help="train.batch_size override (LUNA paper default: 6).")
    p.add_argument("--lr", type=float, default=None,
                   help="Optional train.lr override.")
    p.add_argument("--seed", type=int, default=42,
                   help="general.seed override.")
    p.add_argument(
        "--luna_repo", default=str(_DEFAULT_LUNA_REPO),
        help=f"Path to the LUNA repository. Default: {_DEFAULT_LUNA_REPO}",
    )
    p.add_argument(
        "--run_name", default="MERFISH_mouse_cortex",
        help="Sets general.name in LUNA's Hydra config.",
    )
    p.add_argument(
        "--no_log2_normalize", action="store_true",
        help="Skip log2(x+1) when writing the LUNA CSVs (only if your "
             "silver h5ads are already log2-normalized).",
    )
    p.add_argument(
        "--luna_override", action="append", default=[],
        help="Extra Hydra overrides, e.g. '--luna_override train.lr=1e-4'. "
             "Repeatable.",
    )
    args = p.parse_args()

    try:
        run_benchmark(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            luna_repo=args.luna_repo,
            run_name=args.run_name,
            log2_normalize=not args.no_log2_normalize,
            extra_overrides=args.luna_override,
        )
    except Exception:
        logger.exception("LUNA training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

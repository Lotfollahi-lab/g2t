"""LUNA-on-cortex benchmark, scGG entry point.

This script preserves the historical scGG CLI (``--data_dir``,
``--wandb_run_name``, ``--output_dir``, ...) but the actual model,
training loop, loss, and sampling are now LUNA's, vendored verbatim
under ``scgg/src/{models,utils,metrics,datasets,configs}``.

Flow on a normal invocation::

    python scripts/run_scgg.py \\
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --wandb_run_name scgg_luna_baseline

    1. Discover h5ad files under --data_dir, splitting on filename
       suffix: ``*_train.h5ad`` -> training, ``*_test.h5ad`` -> held
       out for inference. Layout produced by
       ``build_h5ad_from_luna_csv.py``; works for any dataset (cortex,
       ABC, CNS, ...).
    2. Materialise LUNA-format train.csv + test.csv under the run dir.
    3. Invoke the vendored LUNA (``scgg/src/main.py``) as a subprocess
       with Hydra overrides — same Python interpreter, same env.
    4. Read LUNA's ``metadata_pred.csv`` / ``metadata_true.csv`` per slice.
    5. Compute per-slice + aggregate metrics via
       ``scgg.evaluation.luna_metrics``.
    6. Write ``per_slice_metrics.csv``, ``aggregate_metrics.json``,
       ``config.yaml``, ``benchmark.log`` — same filenames as the
       previous scGG run, so downstream tooling and notebooks keep
       working.

Removed CLI flags that referred to deleted scGG abstractions
(``--objective``, ``--metric_embed_dim``, ``--no_coord_regression``,
``--class_stratified_distance``, ``--k_default``, ``--n_top_hvg``,
``--no_normalize``, ``--no_scale``, ``--config``, ``--device``,
``--val_fraction``). Passing any of these will now produce a clear
``unrecognized arguments`` error from argparse — that's the signal that
LUNA is the engine now.
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

logger = logging.getLogger("scgg.luna_cortex_benchmark")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the scgg repo (this file lives in scgg/scripts/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
# LUNA vendored under scgg/src/. `main.py`, `models/`, `utils/`, etc.
# all live here. The subprocess cwd we set below must be this dir so
# Hydra resolves ``configs/`` and LUNA's top-level imports
# (``from models.X import ...``) work.
_VENDORED_LUNA_SRC = _REPO_ROOT / "src"

# Default artifacts root (NFS on the cluster). Overridable via
# --output_dir; if not set and not on NFS we still fall back to this
# path, mirroring the prior behaviour.
_ARTIFACTS_ROOT = Path("/nfs/team361/sb75/scgg-reproducibility/artifacts")

# LUNA's PyTorch-Lightning checkpoint filename pattern: ``epoch=NNN.ckpt``.
_EPOCH_RE = re.compile(r"epoch=(\d+)")


# ---------------------------------------------------------------------------
# h5ad discovery + LUNA-format CSV materialisation
# ---------------------------------------------------------------------------
#
# Silver layout produced by ``build_h5ad_from_luna_csv.py``:
#   <silver_dir>/<section>_train.h5ad     <- training cells
#   <silver_dir>/<section>_test.h5ad      <- held-out test cells
# Discovery here is suffix-based — works for any dataset, no
# cortex-specific filename regex needed.


def _discover_split_files(silver_dir: Path, split: str) -> List[Path]:
    """Return sorted ``*_{split}.h5ad`` paths under ``silver_dir``."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
    return sorted(silver_dir.glob(f"*_{split}.h5ad"))


def _section_label_from_filename(path: Path) -> str:
    """Filename-stem fallback for section label. Strips ``_train`` /
    ``_test``."""
    stem = path.stem
    for suf in ("_train", "_test"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _build_luna_csv(
    files: List[Path],
    out_csv: Path,
    log2_normalize: bool = False,
) -> Dict[str, object]:
    """Concatenate per-section h5ads into one LUNA-format CSV.

    Each file in ``files`` is one section. The section label comes from
    the h5ad's ``obs['cell_section']`` (must be uniform per file); if
    missing or non-uniform, falls back to the filename stem with the
    ``_train`` / ``_test`` suffix stripped.

    LUNA expects: gene columns first (positions ``0..n_genes-1``), then
    ``coord_X``, ``coord_Y``, ``cell_section``, ``cell_class``. Index =
    original cell barcode.

    No log2 normalisation by default — LUNA's published CSVs are
    non-integer per-cell-normalised counts in the same magnitude range
    as raw counts (max ~250). Set ``log2_normalize=True`` only for
    ablations.

    Within-section row order is reconstructed from each cell's
    ``_bronze_row_pos`` (stamped by ``build_h5ad_from_luna_csv.py``)
    so LUNA's unstable ``sort_values("cell_section")`` produces the
    same post-sort ordering as it would on the bronze CSV.
    """
    import anndata as ad
    import scipy.sparse as sp

    gene_names: Optional[List[str]] = None

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    per_section_dfs: List[pd.DataFrame] = []
    has_bronze_pos = True

    for path in files:
        adata = ad.read_h5ad(path)
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)
        if log2_normalize:
            X = np.log2(X + 1.0)

        if gene_names is None:
            gene_names = list(adata.var_names)
        elif list(adata.var_names) != gene_names:
            raise ValueError(
                f"Gene panel mismatch in {path.name}: expected "
                f"{len(gene_names)} genes, got {adata.n_vars}"
            )

        # Section label: prefer obs['cell_section'] (preserved by
        # build_h5ad_from_luna_csv); fall back to filename stem.
        if "cell_section" in adata.obs.columns:
            uniq = adata.obs["cell_section"].astype(str).unique()
            if len(uniq) == 1:
                section_label = str(uniq[0])
            else:
                section_label = _section_label_from_filename(path)
                logger.warning(
                    f"  {path.name}: obs['cell_section'] has "
                    f"{len(uniq)} distinct values; using filename "
                    f"label {section_label!r}"
                )
        else:
            section_label = _section_label_from_filename(path)

        cell_class = (
            adata.obs["cell_class"].astype(str).values
            if "cell_class" in adata.obs.columns
            else np.full(adata.n_obs, "unknown")
        )
        if "spatial" in adata.obsm:
            xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2]
        else:
            xy = np.column_stack([
                adata.obs["coord_X"].to_numpy(dtype=np.float64),
                adata.obs["coord_Y"].to_numpy(dtype=np.float64),
            ])

        if "cell_id" in adata.obs.columns:
            cell_ids = adata.obs["cell_id"].to_numpy()
        else:
            cell_ids = np.arange(len(X))

        if "_bronze_row_pos" in adata.obs.columns:
            bronze_row_pos = adata.obs["_bronze_row_pos"].to_numpy()
        else:
            has_bronze_pos = False
            order = np.argsort(cell_ids, kind="stable")
            X = X[order]
            xy = xy[order]
            cell_class = cell_class[order]
            cell_ids = cell_ids[order]
            bronze_row_pos = cell_ids

        df = pd.DataFrame(X, columns=gene_names)
        df["coord_X"] = xy[:, 0]
        df["coord_Y"] = xy[:, 1]
        df["cell_section"] = section_label
        df["cell_class"] = cell_class
        df.index = cell_ids
        df.index.name = "cell_id"
        df["_bronze_row_pos"] = bronze_row_pos
        per_section_dfs.append(df)
        logger.info(f"    collected {len(df):>6,} cells from {section_label}")

    if not per_section_dfs:
        raise RuntimeError("no sections collected — no h5ads matched the file list")

    big_df = pd.concat(per_section_dfs, axis=0)
    if has_bronze_pos:
        big_df = big_df.sort_values("_bronze_row_pos", kind="stable")
    else:
        logger.warning(
            "  no _bronze_row_pos in any h5ad — rebuild silver with the latest "
            "build_h5ad_from_luna_csv.py to enable bit-identical LUNA-on-h5ad "
            "≡ LUNA-on-bronze."
        )
    big_df = big_df.drop(columns=["_bronze_row_pos"])
    big_df.to_csv(out_csv, header=True, float_format="%.17g")
    rows_total = len(big_df)
    logger.info(f"  wrote {rows_total:,} cells total to {out_csv}")

    return {
        "n_rows": rows_total,
        "n_genes": int(len(gene_names)) if gene_names else 0,
        "n_sections": len(files),
    }


# ---------------------------------------------------------------------------
# LUNA invocation
# ---------------------------------------------------------------------------


def _invoke_luna(overrides: List[str], log_path: Path) -> int:
    """Run vendored LUNA's main.py with Hydra overrides.

    Uses the same Python interpreter (``sys.executable``), so the
    subprocess inherits the active virtualenv (must have torch,
    pytorch-lightning, hydra-core, torch-geometric,
    linear-attention-transformer installed).

    ``cwd`` is set to the vendored LUNA root (``scgg/src/``) so
      * Hydra resolves ``configs/`` relative to ``main.py``;
      * LUNA's top-level absolute imports (``from models.X import ...``)
        find their packages on ``sys.path[0]``.
    """
    main_py = _VENDORED_LUNA_SRC / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(
            f"vendored LUNA entry point missing: {main_py}. "
            f"Did the LUNA copy under scgg/src/ get clobbered?"
        )

    cmd = [sys.executable, str(main_py), *overrides]
    logger.info("Invoking LUNA:")
    for arg in cmd:
        logger.info(f"    {arg}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "wb") as f:
        proc = subprocess.run(
            cmd, cwd=str(_VENDORED_LUNA_SRC), stdout=f, stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = (time.time() - t0) / 60.0
    logger.info(
        f"LUNA exited with code {proc.returncode} after {elapsed:.1f} min "
        f"(log: {log_path})"
    )
    if proc.returncode != 0 and log_path.exists():
        with open(log_path) as f:
            tail = f.read().splitlines()[-60:]
        for line in tail:
            logger.error(f"  | {line}")
    return proc.returncode


# ---------------------------------------------------------------------------
# Read LUNA predictions + run scgg.evaluation.luna_metrics
# ---------------------------------------------------------------------------


def _read_luna_predictions(
    test_save_dir: Path,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Return ``{section_label: (coords_pred, coords_true, cell_class)}``."""
    pred_files = list(test_save_dir.rglob("metadata_pred.csv"))
    if not pred_files:
        raise FileNotFoundError(
            f"No metadata_pred.csv under {test_save_dir} — did LUNA finish?"
        )

    out: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
    for pred_path in sorted(pred_files):
        true_path = pred_path.parent / "metadata_true.csv"
        if not true_path.exists():
            logger.warning(f"  {pred_path.parent.name}: missing metadata_true.csv")
            continue
        pred = pd.read_csv(pred_path, index_col=0)
        true = pd.read_csv(true_path, index_col=0)
        # Align on index — LUNA writes them in matching order, but be defensive.
        if not pred.index.equals(true.index):
            common = pred.index.intersection(true.index)
            pred = pred.loc[common]
            true = true.loc[common]

        coords_pred = pred[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
        coords_true = true[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
        cell_class = None
        if "cell_class" in true.columns:
            cell_class = true["cell_class"].astype(str).to_numpy()
        out[pred_path.parent.name] = (coords_pred, coords_true, cell_class)
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_benchmark(
    data_dir: str,
    output_dir: Optional[str] = None,
    epochs: int = 1000,
    batch_size: int = 6,
    lr: Optional[float] = None,
    seed: int = 0,
    wandb_mode: str = "disabled",
    wandb_run_name: Optional[str] = None,
    contact_percentile: float = 0.01,
    compute_rssd: bool = True,
    skip_training: bool = False,
    load_checkpoint: Optional[str] = None,
    log2_normalize: bool = False,
    extra_overrides: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Train vendored LUNA on ``*_train.h5ad`` files, evaluate on
    ``*_test.h5ad`` files, under a single silver directory.

    Returns the aggregated metrics dict (same shape as
    ``scgg.evaluation.luna_metrics.aggregate_slices``).
    """
    # Defer the scgg import so the script can show ``--help`` even when
    # the scgg env isn't activated.
    from scgg.evaluation.luna_metrics import aggregate_slices, evaluate_slice

    data_path = Path(data_dir)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        out_dir = _ARTIFACTS_ROOT / data_path.name / "model" / run_ts
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "benchmark.log", mode="w"),
        ],
        force=True,
    )
    logger.info(f"Run timestamp: {run_ts}")
    logger.info(f"Output dir:    {out_dir}")
    logger.info(f"Vendored LUNA: {_VENDORED_LUNA_SRC}")

    # ---- 1. Materialise LUNA-format CSVs from the silver h5ads ---------
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    train_csv = work / "train.csv"
    test_csv = work / "test.csv"

    if train_csv.exists() and test_csv.exists() and not skip_training:
        logger.info("LUNA CSVs already exist under work/; reusing")
        head = pd.read_csv(train_csv, nrows=1, index_col=0)
        n_genes = len(head.columns) - 4  # coord_X, coord_Y, cell_section, cell_class
    else:
        # Suffix-based split: any *_train.h5ad is for training,
        # *_test.h5ad is held out for inference. Works for any silver
        # dir produced by build_h5ad_from_luna_csv.
        train_files = _discover_split_files(data_path, "train")
        test_files = _discover_split_files(data_path, "test")
        logger.info(
            f"Silver dir: {data_path} "
            f"({len(train_files)} *_train.h5ad, {len(test_files)} *_test.h5ad)"
        )
        if not train_files or not test_files:
            raise FileNotFoundError(
                f"Need *_train.h5ad AND *_test.h5ad under {data_path}. "
                f"Found train={len(train_files)}, test={len(test_files)}. "
                f"Did you run build_h5ad_from_luna_csv.py to populate "
                f"this silver dir?"
            )
        logger.info(f"Writing train CSV -> {train_csv}")
        train_stats = _build_luna_csv(train_files, train_csv, log2_normalize)
        logger.info(f"Writing test CSV  -> {test_csv}")
        test_stats = _build_luna_csv(test_files, test_csv, log2_normalize)
        n_genes = int(train_stats["n_genes"])

    # ---- 2. Build Hydra overrides + invoke LUNA -------------------------
    luna_run_dir = out_dir / "luna_run"
    luna_run_dir.mkdir(parents=True, exist_ok=True)
    test_save_dir = luna_run_dir / "test_results"
    test_save_dir.mkdir(parents=True, exist_ok=True)

    def _h(v: object) -> str:
        # Hydra single-quote escape — required for paths that contain '='
        # (e.g. LUNA's epoch=N.ckpt filenames).
        return f"'{v}'"

    run_name = wandb_run_name or "scgg_luna_benchmark"
    mode = "test_only" if skip_training else "train_and_test"

    overrides = [
        f"general.name={run_name}",
        f"general.mode={mode}",
        f"general.seed={seed}",
        f"general.wandb={wandb_mode}",
        f"dataset.train_data_path={_h(train_csv.resolve())}",
        f"dataset.test_data_path={_h(test_csv.resolve())}",
        "dataset.gene_columns_start=0",
        f"dataset.gene_columns_end={n_genes}",
        f"train.batch_size={batch_size}",
        f"train.n_epochs={epochs}",
        f"test.save_dir={_h(test_save_dir.resolve())}",
        f"hydra.run.dir={_h(luna_run_dir.resolve())}",
    ]
    if lr is not None:
        overrides.append(f"train.lr={lr}")
    if skip_training:
        if load_checkpoint is None:
            raise ValueError(
                "--skip_training requires --load_checkpoint to point at a "
                "LUNA-format .ckpt path."
            )
        ckpt_path = Path(load_checkpoint).resolve()
        overrides.append(f"test.checkpoints_parent_dir={_h(ckpt_path.parent)}")
        overrides.append(f"test.checkpoints_name_list=[{_h(ckpt_path.name)}]")
    if extra_overrides:
        overrides.extend(extra_overrides)

    log_path = out_dir / "luna_stdout.log"
    rc = _invoke_luna(overrides, log_path)
    if rc != 0:
        raise RuntimeError(f"LUNA training failed (exit {rc}). See {log_path}")

    # ---- 3. Read LUNA predictions, run scgg.evaluation.luna_metrics ----
    logger.info(f"Reading LUNA predictions from {test_save_dir}")
    sections = _read_luna_predictions(test_save_dir)
    logger.info(f"Found predictions for {len(sections)} slices")

    per_slice: List[Dict[str, float]] = []
    for label, (coords_pred, coords_true, cell_class) in sections.items():
        if coords_true.shape[0] < 10:
            logger.info(f"  skipping {label} (n<10)")
            continue
        row = evaluate_slice(
            coords_true, coords_pred, cell_class,
            contact_percentile=contact_percentile,
            compute_rssd=compute_rssd,
            rssd_projection="pca",
        )
        row["section_label"] = label
        per_slice.append(row)
        logger.info(
            f"  {label:30s}  "
            f"spr_median={row['spearman_per_cell_median']:.4f}  "
            f"spr_mean={row['spearman_per_cell_mean']:.4f}  "
            f"prec={row['precision']:.4f}  "
            f"rssd={row.get('absolute_rssd', float('nan')):.2f}"
        )

    # ---- 4. Aggregate + write outputs ----------------------------------
    agg = aggregate_slices(per_slice)
    headline = agg.get("spearman_mean_of_medians", float("nan"))
    luna_reported = 0.448
    logger.info("=" * 72)
    logger.info("LUNA Figure 3 reproduction — aggregated metrics across test slices")
    logger.info("=" * 72)
    for k, v in agg.items():
        logger.info(f"  {k:34s} = {v}")
    logger.info("-" * 72)
    logger.info(
        f"Headline (LUNA-equivalent mean-of-per-slice-median Spearman): "
        f"{headline:.4f}   |   LUNA paper: {luna_reported:.4f}"
    )
    if not np.isnan(headline):
        delta = (headline - luna_reported) * 100
        logger.info(f"Delta vs LUNA: {delta:+.2f} percentage points")

    fieldnames = sorted({k for r in per_slice for k in r.keys()})
    with open(out_dir / "per_slice_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_slice:
            w.writerow(r)
    with open(out_dir / "aggregate_metrics.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)

    logger.info(f"Wrote results to {out_dir}")
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=(
            "Train the vendored LUNA model (under scgg/src/) on the "
            "*_train.h5ad files of any silver directory and evaluate "
            "on the *_test.h5ad files. Defaults reproduce the LUNA "
            "Figure 3 cortex benchmark when --data_dir points at the "
            "mmc_luna silver dir, but any dataset built by "
            "build_h5ad_from_luna_csv works."
        ),
    )
    p.add_argument(
        "--data_dir", required=True,
        help="Path to a per-section h5ad silver directory. The script "
             "discovers *_train.h5ad (for training) and *_test.h5ad "
             "(held out for inference) under it.",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Where to write LUNA checkpoints + per-slice / aggregate "
             "metrics. Default: "
             "/nfs/team361/sb75/scgg-reproducibility/artifacts/<data_dir_name>/"
             "model/<YYYYMMDD_HHMMSS>/.",
    )
    p.add_argument("--epochs", type=int, default=1000,
                   help="LUNA train.n_epochs (default 1000 — matches paper).")
    p.add_argument("--batch_size", type=int, default=6,
                   help="LUNA train.batch_size (number of SECTIONS per "
                        "gradient step; default 6 — matches paper).")
    p.add_argument("--lr", type=float, default=None,
                   help="LUNA train.lr (default unset → LUNA default 5e-4).")
    p.add_argument("--seed", type=int, default=0,
                   help="LUNA general.seed (default 0 — matches paper).")
    p.add_argument("--no_wandb", action="store_true",
                   help="Set LUNA's general.wandb=disabled (default).")
    p.add_argument("--wandb_online", action="store_true",
                   help="Set LUNA's general.wandb=online — requires "
                        "`wandb login` on the host.")
    p.add_argument("--wandb_run_name", default=None,
                   help="Becomes LUNA's general.name (drives the wandb "
                        "run name and the LUNA run-dir basename).")
    p.add_argument("--contact_percentile", type=float, default=0.01,
                   help="Percentile for LUNA's contact F1 metric "
                        "(scgg.evaluation.luna_metrics).")
    p.add_argument("--skip_rssd", action="store_true",
                   help="Skip Kabsch RSSD (faster).")
    p.add_argument("--skip_training", action="store_true",
                   help="Run LUNA in test-only mode (requires --load_checkpoint).")
    p.add_argument("--load_checkpoint", default=None,
                   help="Path to a LUNA .ckpt — only used with --skip_training.")
    p.add_argument(
        "--extra_override", action="append", default=None, metavar="KEY=VALUE",
        help="Extra Hydra override(s) passed straight through to LUNA. "
             "Repeat the flag for multiple. Example: "
             "--extra_override train.lr=1e-4 --extra_override model.layers=12",
    )
    args = p.parse_args()

    if args.wandb_online and args.no_wandb:
        p.error("--wandb_online and --no_wandb are mutually exclusive.")
    wandb_mode = "online" if args.wandb_online else "disabled"

    run_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        wandb_mode=wandb_mode,
        wandb_run_name=args.wandb_run_name,
        contact_percentile=args.contact_percentile,
        compute_rssd=not args.skip_rssd,
        skip_training=args.skip_training,
        load_checkpoint=args.load_checkpoint,
        extra_overrides=args.extra_override,
    )


if __name__ == "__main__":
    sys.exit(main())

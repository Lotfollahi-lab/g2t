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
(``--data_dir``). The h5ad's ``.X`` is written to the CSV **as-is**
(raw integer counts, no transformation by default), because LUNA's
published CSVs are themselves non-integer per-cell-normalized counts
in the [0, ~250] range — *not* log-transformed. Empirically, training
LUNA on log2(x+1) of raw integer counts reproduces ~0% of the paper
Spearman (the input distribution sits in [0, 8] while the model
expects [0, ~250]). The silver layer itself is identical for scGG and
LUNA — same h5ads, same coords, same ``cell_class`` labels.

Optional: pass ``--log2_normalize`` to re-enable the old behavior
(only useful for ablation or if you know your silver h5ads are
already volume-normalized and you want to compress dynamic range).

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
    log2_normalize: bool = False,
) -> Dict[str, object]:
    """Concatenate per-slice h5ads into one CSV in LUNA's input format.

    LUNA expects:
      * gene columns first (positions ``0..n_genes-1``)
      * then ``coord_X``, ``coord_Y``, ``cell_section``, ``cell_class``
      * index = original cell barcode

    Expression normalization: the default is **no transformation** because
    LUNA's published CSVs are non-integer per-cell-normalized counts in
    the same magnitude range as raw counts (max ~250). Applying log2(x+1)
    on top compresses the input to [0, 8] and the model fails to learn —
    we verified this with ``compare_luna_csv_vs_h5ad.py``. Set
    ``log2_normalize=True`` only for ablations.
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
        # Preserve the h5ad's native precision (float64 if the silver
        # was built via build_h5ad_from_luna_csv.py with the float64
        # default). Casting to float32 here would introduce LSB
        # rounding that, combined with the cast inside LUNA's
        # data_module (`.float()`), makes h5ad-derived training
        # diverge slightly from CSV-direct training even when the
        # source data is identical. Both paths end up at float32
        # inside LUNA; we just want THAT cast to be the only one.
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

        section_label = f"mouse{mouse}_slice{slice_id}"
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

        df = pd.DataFrame(X, columns=gene_names)
        df["coord_X"] = xy[:, 0]
        df["coord_Y"] = xy[:, 1]
        df["cell_section"] = section_label
        df["cell_class"] = cell_class
        # LUNA's DataModule does `cell_ID = torch.tensor(input_data.index)`
        # which fails with "too many dimensions 'str'" on string-barcode
        # indices. Use a per-section integer index instead — LUNA preserves
        # it through to per-section ``metadata_pred.csv`` outputs, so row
        # order is recoverable downstream.
        df.index = range(len(df))
        df.index.name = "cell_id"

        # `float_format="%.17g"` writes up to 17 significant digits
        # (shortest unambiguous float64 representation). This is the
        # safest setting across pandas versions — float_format=None
        # uses str() which is lossless in modern pandas (>=1.0) but
        # can be 6-digit truncated in older versions. With %.17g we
        # guarantee the round-trip is lossless:
        #   bronze CSV → h5ad (float64) → fresh CSV (17g) → LUNA's
        #   pandas (float64) → torch.float() (same float32 cast)
        df.to_csv(out_csv, mode="a", header=first, float_format="%.17g")
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
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: int = 1000,
    batch_size: int = 6,
    lr: Optional[float] = None,
    seed: int = 0,
    luna_repo: str = str(_DEFAULT_LUNA_REPO),
    run_name: str = "MERFISH_mouse_cortex",
    log2_normalize: bool = False,
    wandb_mode: str = "disabled",
    extra_overrides: Optional[List[str]] = None,
    train_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    n_genes: Optional[int] = None,
) -> Dict[str, float]:
    """Train LUNA on Mouse 1, evaluate on Mouse 2.

    Args mirror scgg/scripts/run_luna_cortex_benchmark.py:run_benchmark
    where applicable. LUNA-specific extras (``luna_repo``, ``run_name``,
    ``log2_normalize``, ``extra_overrides``) replace the scgg
    loss-shaping knobs that don't apply here.

    Returns a dict with the headline ``spearman_mean_of_medians`` metric.
    """
    # Validate args: either we build CSVs from silver h5ads (data_dir),
    # or the caller supplies pre-built CSVs (train_csv + test_csv).
    use_prebuilt = train_csv is not None and test_csv is not None
    if not use_prebuilt and data_dir is None:
        raise ValueError(
            "Either --data_dir (build CSVs from silver h5ads) or both "
            "--train_csv and --test_csv (use pre-built LUNA CSVs) must "
            "be provided."
        )
    if (train_csv is None) != (test_csv is None):
        raise ValueError("--train_csv and --test_csv must be passed together.")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        if use_prebuilt:
            # Derive a sensible default from the train CSV's parent dir name.
            base = Path(train_csv).resolve().parent.name or "luna_paper_csvs"
        else:
            base = Path(data_dir).name
        out = _ARTIFACTS_ROOT / base / "luna_model" / run_ts
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

    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    if use_prebuilt:
        # ---- Pre-built CSV path: use LUNA's preprocessed files directly.
        # This is the bit-exact paper-reproduction path; no h5ad → CSV
        # conversion. Symlink them into work/ so LUNA's run dir is
        # self-contained, and so a later inference script can resolve
        # train.csv from a deterministic relative path.
        src_train = Path(train_csv).resolve()
        src_test = Path(test_csv).resolve()
        if not src_train.exists():
            raise FileNotFoundError(f"--train_csv not found: {src_train}")
        if not src_test.exists():
            raise FileNotFoundError(f"--test_csv not found: {src_test}")
        train_csv_path = work / "train.csv"
        test_csv_path = work / "test.csv"
        for link, target in [(train_csv_path, src_train), (test_csv_path, src_test)]:
            if link.exists() or link.is_symlink():
                link.unlink()
            try:
                link.symlink_to(target)
            except OSError:
                # Filesystems that disallow symlinks: fall through to direct path.
                pass
        # Use the symlink if it materialized; otherwise the source path.
        train_csv_path = train_csv_path if train_csv_path.exists() else src_train
        test_csv_path = test_csv_path if test_csv_path.exists() else src_test
        logger.info(f"Using pre-built train CSV: {src_train}")
        logger.info(f"Using pre-built test  CSV: {src_test}")

        # n_genes: trust user override if passed; otherwise infer by
        # finding where the metadata block starts. LUNA's CSV convention
        # is "gene columns first, then metadata" — but in practice their
        # preprocessed CSVs include MORE metadata than the standard four
        # (e.g., `cell_name`, `class`, `mouse`, `sample_id` are present
        # in addition to coord_X / coord_Y / cell_class / cell_section).
        # If we include any of those in the gene block, torch fails with
        # "can't convert np.ndarray of type numpy.object_" because some
        # of them carry string values.
        #
        # We use TWO complementary signals to locate the boundary and
        # take whichever appears earlier:
        #   (a) NAME-based: lowest index of any known metadata column
        #       name. Catches numeric-typed metadata (coord_X / coord_Y
        #       are floats — dtype check wouldn't see them).
        #   (b) DTYPE-based: lowest index of a non-numeric column.
        #       Catches metadata columns we didn't anticipate by name
        #       (e.g., `cell_name` with too-big-for-int64 cell IDs that
        #       parse as strings).
        _METADATA_NAMES = (
            # standard positions
            "coord_X", "coord_Y", "x", "y",
            "cell_section", "section", "region", "slice",
            "cell_class", "cell_type", "class", "subclass", "type",
            # additional metadata commonly present in LUNA's CSVs
            "cell_name", "cell_id", "cell_barcode", "barcode",
            "mouse", "animal", "donor",
            "sample", "sample_id", "batch", "experiment", "cluster",
        )
        if n_genes is None:
            head_data = pd.read_csv(src_train, nrows=20, index_col=0)
            cols = list(head_data.columns)

            # (a) name-based
            meta_positions_by_name = [
                cols.index(name) for name in _METADATA_NAMES if name in cols
            ]
            name_based = min(meta_positions_by_name) if meta_positions_by_name else None

            # (b) dtype-based
            dtype_based = None
            for i, c in enumerate(cols):
                if not pd.api.types.is_numeric_dtype(head_data[c]):
                    dtype_based = i
                    break

            candidates = [v for v in (name_based, dtype_based) if v is not None]
            if not candidates:
                raise ValueError(
                    f"Could not locate the gene/metadata boundary in "
                    f"{src_train}. None of {_METADATA_NAMES} found in "
                    f"the header, and all columns parse as numeric. "
                    f"Pass --n_genes explicitly."
                )
            n_genes = min(candidates)
            if n_genes <= 0:
                raise ValueError(
                    f"Inferred n_genes={n_genes} from {src_train} but "
                    f"that means there are no gene columns before the "
                    f"first metadata column ({cols[n_genes]!r}). "
                    f"The CSV layout looks wrong."
                )
            logger.info(
                f"n_genes inferred = {n_genes}  "
                f"(boundary at column {cols[n_genes]!r}; "
                f"name-based={name_based}, dtype-based={dtype_based}; "
                f"CSV has {len(cols)} total columns)"
            )
        else:
            logger.info(f"n_genes (explicit) = {n_genes}")
    else:
        # ---- Silver h5ad path: build CSVs ourselves -----------------------
        data_path = Path(data_dir)
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

        train_csv_path = work / "train.csv"
        test_csv_path = work / "test.csv"
        if train_csv_path.exists() and test_csv_path.exists():
            logger.info("LUNA CSVs already exist under work/; reusing")
            head = pd.read_csv(train_csv_path, nrows=1, index_col=0)
            n_genes = len(head.columns) - 4
        else:
            logger.info(f"Writing train CSV -> {train_csv_path}")
            train_stats = _build_luna_csv(
                train_files, train_csv_path, log2_normalize=log2_normalize,
            )
            logger.info(
                f"  train: {train_stats['n_rows']:,} rows, "
                f"{train_stats['n_genes']} genes, "
                f"{train_stats['n_sections']} sections"
            )
            logger.info(f"Writing test CSV  -> {test_csv_path}")
            test_stats = _build_luna_csv(
                test_files, test_csv_path, log2_normalize=log2_normalize,
            )
            logger.info(
                f"  test : {test_stats['n_rows']:,} rows, "
                f"{test_stats['n_genes']} genes, "
                f"{test_stats['n_sections']} sections"
            )
            n_genes = int(train_stats["n_genes"])

    # Reuse the path-variables for the rest of the function.
    train_csv = train_csv_path  # noqa: F811  (intentional rebinding for downstream f-strings)
    test_csv = test_csv_path

    # ---- 3. Invoke LUNA train_and_test ---------------------------------
    luna_run_dir = out / "luna_run"
    luna_run_dir.mkdir(parents=True, exist_ok=True)
    test_save_dir = luna_run_dir / "test_results"
    test_save_dir.mkdir(parents=True, exist_ok=True)

    # NOTE on hyperparameters: the defaults below are bit-identical to
    # LUNA's published MERFISH cortex config (configs/experiment/
    # MERFISH_mouse_cortex.yaml on the upstream LUNA repo):
    #
    #   train.n_epochs         = 1000   (experiment override)
    #   train.batch_size       = 6      (experiment override)
    #   train.lr               = 5e-4   (LUNA train default, we don't touch)
    #   train.weight_decay     = 1e-12  (LUNA train default, we don't touch)
    #   general.seed           = 0      (LUNA general default, we now match)
    #   general.mode           = train_and_test
    #   validation.if_validate = False  (LUNA experiment default)
    #   validation.save_model_every_n_epochs = 250  (LUNA experiment default)
    #
    # Only deliberate departure: general.wandb defaults to "disabled" here
    # (LUNA defaults to "online", which crashes if the host isn't logged
    # in). Override via --wandb_mode if you want LUNA to log to wandb.
    # Single-quote path values so Hydra's override parser tolerates any
    # `=` (or other special chars) in the path tree. Critical for paths
    # containing LUNA's `epoch=N.ckpt` checkpoint filenames; harmless for
    # the rest.
    def _h(v: object) -> str:
        return f"'{v}'"

    overrides = [
        f"general.name={run_name}",
        "general.mode=train_and_test",
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
        "data_source": "prebuilt_csv" if use_prebuilt else "silver_h5ad",
        "data_dir": (str(data_dir) if not use_prebuilt else None),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "n_genes": n_genes,
        "luna_repo": str(luna_repo_p),
        "run_name": run_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "log2_normalize": log2_normalize if not use_prebuilt else None,
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
        "--data_dir", default=None,
        help="Per-slice silver h5ad directory (LUNA cortex split). "
             "Mouse 1 => train, Mouse 2 => test. Either this OR "
             "(--train_csv + --test_csv) must be provided.",
    )
    p.add_argument(
        "--train_csv", default=None,
        help="Pre-built LUNA-format train CSV (e.g., LUNA's published "
             "MERFISH_mouse_cortex_train.csv from their Google Drive: "
             "https://drive.google.com/drive/folders/1vWxVUSuQzRDF1o9Vw_cnm-wbEYw_e1Gu"
             "). Skips the h5ad → CSV conversion step entirely. "
             "Required (with --test_csv) for bit-exact LUNA-paper "
             "reproduction.",
    )
    p.add_argument(
        "--test_csv", default=None,
        help="Pre-built LUNA-format test CSV. Pair with --train_csv.",
    )
    p.add_argument(
        "--n_genes", type=int, default=None,
        help="Number of gene columns in the pre-built CSVs. Auto-inferred "
             "from the CSV header (n_columns - 4 metadata) when omitted.",
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
                   help="Optional train.lr override. LUNA's published "
                        "default for the cortex experiment is 5e-4 (used "
                        "when this flag is not passed).")
    p.add_argument("--seed", type=int, default=0,
                   help="general.seed override. Default 0 matches LUNA's "
                        "published config (configs/general/default.yaml).")
    p.add_argument(
        "--wandb_mode", default="disabled",
        choices=("disabled", "online", "offline", "dryrun"),
        help="general.wandb override. Default 'disabled' to avoid LUNA "
             "crashing when the host isn't logged into WandB. Pass "
             "'online' to match LUNA's upstream default.",
    )
    p.add_argument(
        "--luna_repo", default=str(_DEFAULT_LUNA_REPO),
        help=f"Path to the LUNA repository. Default: {_DEFAULT_LUNA_REPO}",
    )
    p.add_argument(
        "--run_name", default="MERFISH_mouse_cortex",
        help="Sets general.name in LUNA's Hydra config.",
    )
    # By default we write raw counts (LUNA's published CSVs are
    # non-integer per-cell-normalized values in the same magnitude
    # range as raw counts, NOT log-transformed — verified with
    # `compare_luna_csv_vs_h5ad.py`). `--log2_normalize` opts in to the
    # old behavior for ablation.
    p.add_argument(
        "--log2_normalize", action="store_true",
        help="Apply log2(x+1) when writing the LUNA CSVs. OFF by default "
             "since LUNA's published CSVs are not log-transformed; "
             "training on log-compressed inputs collapses to ~0 Spearman.",
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
            log2_normalize=args.log2_normalize,
            wandb_mode=args.wandb_mode,
            extra_overrides=args.luna_override,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            n_genes=args.n_genes,
        )
    except Exception:
        logger.exception("LUNA training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

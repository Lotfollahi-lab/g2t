#!/usr/bin/env python
"""
Run LUNA inference on held-out silver sections using a trained LUNA checkpoint.

Mirror of ``scgg/scripts/infer_scgg_on_cns.py`` (the scGG inference
script). This script is **self-contained**: no dependency on the scgg
package. Meant to be run *in the LUNA Python environment* — same env as
``run_luna_on_mmc.py``.

Input is the **same silver h5ad directory** that the scGG pipeline reads
from (``--silver_dir``). The two methods differ in downstream
normalization:

  * scGG: ``normalize_total(1e4) + log1p + scale`` (via scanpy in
    ``scgg.data.luna_cortex``).
  * LUNA: **raw counts as-is** (default). LUNA's published CSVs are
    non-integer per-cell-normalized values in the [0, ~250] range, NOT
    log-transformed (verified empirically with
    ``compare_luna_csv_vs_h5ad.py``). Pass ``--log2_normalize`` to
    opt into log2(x+1) for ablation only.

So the silver layer is exactly identical for both; only the normalization
applied downstream differs.

Pipeline
--------
  1. Resolve ``--sections`` to silver h5ad paths (same logic as the scgg
     inference script: ``paper`` => mouse2_slice99, ``all_test`` => every
     mouse2_*, bare IDs, globs).
  2. Build a combined LUNA-format test CSV for just those sections; also
     produce/reuse a train CSV (LUNA's data loader needs both paths even
     in test mode).
  3. Invoke LUNA's ``main.py`` with ``general.mode=test`` and
     ``test.checkpoint_path=<--checkpoint>``.
  4. Read LUNA's per-section ``metadata_pred.csv``, write the predictions
     into copies of the original h5ads as ``obsm['spatial_pred']``.
  5. Produce a side-by-side comparison plot (GT vs prediction) using a
     Umeyama similarity alignment for visual A/B with the scGG plots.
     Optional per-class Spearman diagnostic.

Output layout
-------------
``--output_dir`` defaults to
``/nfs/team361/sb75/scgg-reproducibility/artifacts/<silver_dir.name>/luna_inference/<model_timestamp>/``
where ``<model_timestamp>`` is auto-extracted from the checkpoint path
(the ``YYYYMMDD_HHMMSS`` segment under ``luna_model/`` written by
``run_luna_on_mmc.py``). Falls back to a fresh timestamp if the
checkpoint isn't under that canonical layout.

Usage
-----
    # Activate the LUNA env first
    source /nfs/team361/sb75/.venvs/luna/bin/activate

    python scripts/infer_luna_on_mmc.py \\
        --checkpoint /nfs/.../luna_model/<TS>/best_model.ckpt \\
        --silver_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --sections paper \\
        --include_train_section \\
        --per_class_spearman
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

logger = logging.getLogger("luna_infer")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


_ARTIFACTS_ROOT = Path("/nfs/team361/sb75/scgg-reproducibility/artifacts")
_DEFAULT_LUNA_REPO = Path(
    "/nfs/team361/sb75/scgg-reproducibility/analysis/benchmarking/luna"
)

_SLICE_RE = re.compile(
    r"^(?:mmc|merfish_mouse_cortex)_mouse(?P<mouse>\d+)_slice(?P<slice>\d+)\.h5ad$"
)
_RUN_TS_RE = re.compile(r"^\d{8}_\d{6}$")

_KNOWN_SILVER_PREFIXES = (
    "cns_scrna_",
    "mmc_",
    "merfish_mouse_cortex_",
    "abc_zhuang_abca1_",
)
_CORTEX_MOUSE_GLOBS = ("mmc_mouse{m}_*.h5ad", "merfish_mouse_cortex_mouse{m}_*.h5ad")


# ---------------------------------------------------------------------------
# Section / file resolution (same semantics as infer_scgg_on_cns.py)
# ---------------------------------------------------------------------------


def _strip_known_prefix(stem: str) -> str:
    """Strip the silver-file prefix to recover a bare section id."""
    for pref in _KNOWN_SILVER_PREFIXES:
        if stem.startswith(pref):
            return stem[len(pref):]
    return stem


def _all_silver_files(silver_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(silver_dir.glob("*.h5ad")):
        if any(p.name.startswith(pref) for pref in _KNOWN_SILVER_PREFIXES):
            out.append(p)
    return out


def _resolve_section_files(
    silver_dir: Path, sections_arg: List[str],
) -> List[Path]:
    """Section ids / filenames / globs / preset values -> list of h5ad paths.

    Accepted forms (mirror of the scgg inference script):
      - ``all``           every silver h5ad with a known prefix
      - ``all_test``      every cortex Mouse 2 slice (both prefixes)
      - ``paper``         mouse2_slice99 (LUNA paper Fig 3c reference)
      - bare section id   ``mouse2_slice1``, ``Zhuang-ABCA-1-001``, etc.
      - filename          ``mmc_mouse2_slice1.h5ad``
      - glob              ``mouse2_*``
    """
    if not silver_dir.exists():
        raise FileNotFoundError(f"silver_dir does not exist: {silver_dir}")

    if sections_arg == ["all"]:
        files = _all_silver_files(silver_dir)
        if not files:
            raise FileNotFoundError(
                f"No silver h5ad files under {silver_dir} matching any of "
                f"the known prefixes: {_KNOWN_SILVER_PREFIXES}"
            )
        return files

    if sections_arg == ["all_test"]:
        files: List[Path] = []
        seen: set = set()
        for g in _CORTEX_MOUSE_GLOBS:
            for p in sorted(silver_dir.glob(g.format(m=2))):
                if p not in seen:
                    seen.add(p)
                    files.append(p)
        if not files:
            raise FileNotFoundError(
                f"No held-out test files (mmc_mouse2_*.h5ad or legacy) "
                f"under {silver_dir}"
            )
        return files

    if sections_arg == ["paper"]:
        return _resolve_section_files(silver_dir, ["mouse2_slice99"])

    out: List[Path] = []
    for s in sections_arg:
        if any(ch in s for ch in "*?["):
            matched: List[Path] = []
            patterns = [s, s + ".h5ad"]
            for pref in _KNOWN_SILVER_PREFIXES:
                patterns.append(f"{pref}{s}.h5ad")
                patterns.append(f"{pref}{s}")
            for pat in patterns:
                matched.extend(sorted(silver_dir.glob(pat)))
            seen: set = set()
            uniq: List[Path] = []
            for p in matched:
                if p in seen or p.suffix != ".h5ad":
                    continue
                seen.add(p)
                uniq.append(p)
            if not uniq:
                raise FileNotFoundError(
                    f"Glob {s!r} matched no silver h5ads under {silver_dir}"
                )
            out.extend(uniq)
            continue

        candidates = [silver_dir / s, silver_dir / f"{s}.h5ad"]
        for pref in _KNOWN_SILVER_PREFIXES:
            candidates.append(silver_dir / f"{pref}{s}.h5ad")
        for c in candidates:
            if c.exists() and c.suffix == ".h5ad":
                out.append(c)
                break
        else:
            raise FileNotFoundError(
                f"No silver h5ad matches section spec {s!r}. "
                f"Tried: {[str(c) for c in candidates]}"
            )

    seen2: set = set()
    uniq_out: List[Path] = []
    for p in out:
        if p in seen2:
            continue
        seen2.add(p)
        uniq_out.append(p)
    return uniq_out


def _pick_representative_cortex_slice(
    silver_dir: Path, mouse_id: int,
) -> Optional[Path]:
    """Pick one Mouse-{mouse_id} slice for ``--include_train_section``.

    Prefers slice99 (the LUNA Fig 3c reference); falls back to whatever
    Mouse-{mouse_id} h5ad is first on disk in sorted order.
    """
    for pref in ("mmc", "merfish_mouse_cortex"):
        p = silver_dir / f"{pref}_mouse{mouse_id}_slice99.h5ad"
        if p.exists():
            return p
    candidates: List[Path] = []
    for g in _CORTEX_MOUSE_GLOBS:
        candidates.extend(sorted(silver_dir.glob(g.format(m=mouse_id))))
    return candidates[0] if candidates else None


def _section_files_with_mouse(
    paths: List[Path],
) -> List[Tuple[int, int, Path]]:
    out: List[Tuple[int, int, Path]] = []
    for p in paths:
        m = _SLICE_RE.match(p.name)
        if not m:
            raise ValueError(
                f"Section file does not match cortex naming: {p.name}. "
                "Expected mmc_mouse{M}_slice{S}.h5ad or "
                "merfish_mouse_cortex_mouse{M}_slice{S}.h5ad."
            )
        out.append((int(m["mouse"]), int(m["slice"]), p))
    return out


# ---------------------------------------------------------------------------
# Timestamp extraction (pin inference to model's training run)
# ---------------------------------------------------------------------------


def _timestamp_from_path(p: Path) -> Optional[str]:
    for parent in p.resolve().parents:
        if _RUN_TS_RE.match(parent.name):
            return parent.name
    return None


# ---------------------------------------------------------------------------
# Pre-built CSV mode: filter LUNA's published CSVs to specific sections
# ---------------------------------------------------------------------------


_METADATA_NAMES_FOR_CSV = (
    "coord_X", "coord_Y", "x", "y",
    "cell_section", "section", "region", "slice",
    "cell_class", "cell_type", "class", "subclass", "type",
    "cell_name", "cell_id", "cell_barcode", "barcode",
    "mouse", "animal", "donor",
    "sample", "sample_id", "batch", "experiment", "cluster",
)


def _infer_n_genes_from_csv(csv_path: Path) -> int:
    """Same heuristic as run_luna_on_mmc.py: combine name-based + dtype-
    based detection of the gene/metadata boundary."""
    head = pd.read_csv(csv_path, nrows=20, index_col=0)
    cols = list(head.columns)
    meta_pos_by_name = [
        cols.index(n) for n in _METADATA_NAMES_FOR_CSV if n in cols
    ]
    name_based = min(meta_pos_by_name) if meta_pos_by_name else None
    dtype_based = None
    for i, c in enumerate(cols):
        if not pd.api.types.is_numeric_dtype(head[c]):
            dtype_based = i
            break
    candidates = [v for v in (name_based, dtype_based) if v is not None]
    if not candidates:
        raise ValueError(
            f"Could not locate the gene/metadata boundary in {csv_path}. "
            "Pass --n_genes explicitly."
        )
    n = min(candidates)
    if n <= 0:
        raise ValueError(
            f"Inferred n_genes={n} from {csv_path}; CSV layout looks wrong."
        )
    return n


def _filter_luna_csv_by_section(
    train_csv: Path,
    test_csv: Path,
    sections: List[str],
    out_csv: Path,
) -> Dict[str, object]:
    """Read LUNA's train + test CSVs, concatenate, filter to the requested
    sections, save as a single test CSV for inference.

    LUNA needs the `dataset.test_data_path` to contain ONLY the cells we
    want predictions for. Since `--sections` can mix train-side (Mouse 1)
    and test-side (Mouse 2) slices freely, we scan BOTH source CSVs by
    `cell_section` column value.

    Returns a dict with `n_rows`, `n_sections`, `matched`, `missing` for
    logging.
    """
    parts = []
    for src in (train_csv, test_csv):
        df = pd.read_csv(src, index_col=0)
        if "cell_section" not in df.columns:
            raise ValueError(
                f"No 'cell_section' column in {src} — is this a LUNA-format CSV?"
            )
        parts.append(df)
    combined = pd.concat(parts, axis=0)
    available = set(str(s) for s in combined["cell_section"].unique())
    matched = [s for s in sections if s in available]
    missing = [s for s in sections if s not in available]
    if not matched:
        raise ValueError(
            f"None of --sections {sections} matched any cell_section value "
            f"in the provided CSVs. Available (first 10): "
            f"{sorted(available)[:10]}..."
        )
    if missing:
        logger.warning(
            f"  --sections values not found in CSVs and skipped: {missing}"
        )
    filtered = combined[combined["cell_section"].astype(str).isin(matched)].copy()
    # Preserve the LUNA index dtype (integer cell IDs) — we don't reset.
    filtered.to_csv(out_csv)
    return {
        "n_rows": len(filtered),
        "n_sections": int(filtered["cell_section"].nunique()),
        "matched": matched,
        "missing": missing,
    }


def _adata_from_luna_outputs(pred_df: pd.DataFrame, true_df: pd.DataFrame):
    """Build a minimal AnnData from LUNA's per-section pred + true CSVs
    so the existing `_plot_comparison(adata, ...)` can render directly,
    without needing a silver h5ad on disk.

    The AnnData carries:
      .obsm['spatial']        = GT coords from metadata_true.csv (raw scale)
      .obsm['spatial_pred']   = LUNA's predicted coords (normalized [-0.5, 0.5])
      .obs['cell_class']      = cell class labels (categorical)
      .X                      = empty (0 vars) — plotting only uses .obsm/.obs
    """
    import anndata as ad

    # Align by index if possible; otherwise assume row order matches.
    if not pred_df.index.equals(true_df.index):
        common = pred_df.index.intersection(true_df.index)
        if len(common) < min(len(pred_df), len(true_df)):
            logger.warning(
                f"    pred and true CSVs partially mismatch; using "
                f"{len(common)} common cells."
            )
        pred_df = pred_df.loc[common]
        true_df = true_df.loc[common]

    coords_true = true_df[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
    coords_pred = pred_df[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
    cell_class = (
        true_df["cell_class"].astype(str).to_numpy()
        if "cell_class" in true_df.columns
        else np.full(len(true_df), "unknown")
    )
    n = len(true_df)
    adata = ad.AnnData(X=np.zeros((n, 0), dtype=np.float32))
    adata.obs_names = [str(i) for i in true_df.index]
    adata.obs["cell_class"] = pd.Categorical(cell_class)
    adata.obsm["spatial"] = coords_true
    adata.obsm["spatial_pred"] = coords_pred
    return adata


# ---------------------------------------------------------------------------
# Build LUNA-format CSVs from per-slice h5ads
# ---------------------------------------------------------------------------


def _build_luna_csv(
    files: List[Tuple[int, int, Path]],
    out_csv: Path,
    log2_normalize: bool = False,
) -> Dict[str, object]:
    """Concatenate per-slice h5ads into a LUNA-format CSV.

    Default is no transformation — LUNA's CSVs are not log-transformed
    (see top-of-file docstring).
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
        # LUNA's DataModule does `cell_ID = torch.tensor(input_data.index)`
        # which fails with "too many dimensions 'str'" on string-barcode
        # indices. Use a per-section integer index instead — LUNA
        # preserves it through to per-section metadata_pred.csv outputs.
        df.index = range(len(df))
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
# Invoke LUNA (same Python env via sys.executable)
# ---------------------------------------------------------------------------


def _invoke_luna(
    luna_repo: Path,
    overrides: List[str],
    log_path: Path,
) -> int:
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
# Read LUNA prediction CSVs
# ---------------------------------------------------------------------------


def _read_luna_predictions(
    test_save_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Return {section_label: pred_df} from LUNA's test outputs."""
    pred_files = list(test_save_dir.rglob("metadata_pred.csv"))
    if not pred_files:
        raise FileNotFoundError(
            f"No metadata_pred.csv under {test_save_dir} — did LUNA finish?"
        )
    out: Dict[str, pd.DataFrame] = {}
    for pred_path in sorted(pred_files):
        out[pred_path.parent.name] = pd.read_csv(pred_path, index_col=0)
    return out


# ---------------------------------------------------------------------------
# Umeyama / Kabsch similarity alignment (Procrustes)
# ---------------------------------------------------------------------------


def _umeyama_align(
    src: np.ndarray, dst: np.ndarray, allow_reflection: bool = True,
) -> np.ndarray:
    """Best similarity transform (rotate + scale + translate + optional
    reflection) mapping ``src`` onto ``dst``. Returns transformed src."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    if finite.sum() < 3:
        return src.astype(np.float32)
    a = src[finite]
    b = dst[finite]
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    ac, bc = a - mu_a, b - mu_b
    var_a = (ac ** 2).sum() / a.shape[0]
    if var_a < 1e-12:
        return src.astype(np.float32)
    cov = (bc.T @ ac) / a.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    d = np.eye(cov.shape[0])
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        d[-1, -1] = -1
    R = U @ d @ Vt
    s = (S * np.diag(d)).sum() / var_a
    t = mu_b - s * R @ mu_a
    out = (s * (src @ R.T)) + t
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Color palette (LUNA Fig 3 uses glasbey; fall back to tab20)
# ---------------------------------------------------------------------------


def _palette_for(cats: List[str], scheme: str = "luna") -> List:
    import matplotlib.pyplot as plt
    n = max(len(cats), 1)
    if scheme == "luna":
        try:
            import colorcet as cc  # type: ignore
            return list(cc.glasbey[:n])
        except ImportError:
            logger.info(
                "  colorcet not installed; falling back to tab20. "
                "Install with: pip install colorcet"
            )
    cmap = plt.get_cmap("tab20", n)
    return [cmap(i) for i in range(n)]


# ---------------------------------------------------------------------------
# Side-by-side comparison plot — raw matplotlib (no scanpy required)
# ---------------------------------------------------------------------------


def _plot_comparison(
    adata,
    color_col: str,
    out_svg: Path,
    title_prefix: str = "",
    spot_size: Optional[float] = None,
    align_for_plot: bool = True,
    palette: str = "luna",
) -> None:
    """GT vs LUNA-prediction scatter, scanpy-based for pixel parity with
    ``scgg/scripts/infer_scgg_on_cns.py:_plot_comparison``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import scanpy as sc

    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    has_gt = "spatial" in adata.obsm and adata.obsm["spatial"].shape[1] >= 2
    has_pred = "spatial_pred" in adata.obsm

    # Procrustes-align predicted coords for visualization. LUNA's
    # predictions live in the GT frame already in principle, but Umeyama
    # alignment is a no-op when already aligned and a safety net otherwise.
    # We write the aligned coords to a separate obsm key so the original
    # `spatial_pred` is preserved untouched on disk.
    pred_plot_key = "spatial_pred"
    if align_for_plot and has_gt and has_pred:
        aligned = _umeyama_align(
            adata.obsm["spatial_pred"][:, :2],
            adata.obsm["spatial"][:, :2],
            allow_reflection=True,
        )
        adata.obsm["spatial_pred_aligned"] = aligned
        pred_plot_key = "spatial_pred_aligned"

    if spot_size is None:
        spot_size = max(20.0, min(120.0, 5000.0 / np.sqrt(max(adata.n_obs, 1))))

    # Apply the chosen palette BEFORE plotting so scanpy uses it. Scanpy
    # honors `adata.uns[f"{color_col}_colors"]` over its tab20 default.
    palette_scheme = palette
    is_categorical = (
        adata.obs[color_col].dtype.name == "category"
        or adata.obs[color_col].dtype == object
    )
    palette_colors: Optional[List] = None
    if is_categorical:
        adata.obs[color_col] = adata.obs[color_col].astype("category")
        cats_pre = adata.obs[color_col].cat.categories.tolist()
        palette_colors = _palette_for(cats_pre, scheme=palette_scheme)
        adata.uns[f"{color_col}_colors"] = list(palette_colors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    common_kw = dict(
        color=color_col, show=False, size=spot_size,
        legend_loc=None, frameon=True,
    )
    if has_gt:
        sc.pl.embedding(
            adata, basis="spatial", ax=axes[0],
            title=f"{title_prefix}Ground truth", **common_kw,
        )
    else:
        axes[0].set_title(f"{title_prefix}Ground truth (none)")
        axes[0].set_axis_off()

    if has_pred:
        pred_title = (
            f"{title_prefix}LUNA prediction (aligned)"
            if pred_plot_key == "spatial_pred_aligned"
            else f"{title_prefix}LUNA prediction"
        )
        sc.pl.embedding(
            adata, basis=pred_plot_key, ax=axes[1],
            title=pred_title, **common_kw,
        )
    else:
        axes[1].set_title(f"{title_prefix}LUNA prediction (none)")
        axes[1].set_axis_off()

    # Shared legend at the bottom — matches the scgg inference plot.
    if is_categorical:
        cats = adata.obs[color_col].cat.categories.tolist()
        if palette_colors is None or len(palette_colors) < len(cats):
            palette_colors = _palette_for(cats, scheme=palette_scheme)
        patches = [
            Patch(facecolor=c, label=str(cat))
            for c, cat in zip(palette_colors, cats)
        ]
        n_cats = len(cats)
        ncol = min(max(1, (n_cats + 3) // 4), 6)
        fig.legend(
            handles=patches, labels=[str(c) for c in cats],
            loc="lower center", bbox_to_anchor=(0.5, 0.0),
            ncol=ncol, frameon=False, fontsize="small",
        )
        n_rows = (n_cats + ncol - 1) // ncol
        bottom = min(0.30, 0.05 + 0.04 * n_rows)
        fig.tight_layout(rect=(0, bottom, 1, 1))
    else:
        fig.tight_layout()

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved plot: {out_svg}")


# ---------------------------------------------------------------------------
# Per-class Spearman diagnostic
# ---------------------------------------------------------------------------


def _per_cell_spearman(
    coords_true: np.ndarray, coords_pred: np.ndarray,
) -> Dict[str, float]:
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr
    dt = cdist(coords_true, coords_true)
    dp = cdist(coords_pred, coords_pred)
    rhos: List[float] = []
    for i in range(coords_true.shape[0]):
        r, _ = spearmanr(dt[i], dp[i])
        if r is not None and not np.isnan(r):
            rhos.append(float(r))
    if not rhos:
        return {"median": float("nan"), "mean": float("nan"), "n": 0}
    return {
        "median": float(np.median(rhos)),
        "mean": float(np.mean(rhos)),
        "n": int(len(rhos)),
    }


def _per_class_spearman(
    coords_true: np.ndarray,
    coords_pred: np.ndarray,
    cell_class: Optional[np.ndarray],
    min_cells_per_class: int = 50,
) -> Dict[str, object]:
    """Median per-cell Spearman, globally + per cell class."""
    global_res = _per_cell_spearman(coords_true, coords_pred)
    out: Dict[str, object] = {
        "global_median": global_res["median"],
        "global_mean": global_res["mean"],
        "n_cells": int(coords_true.shape[0]),
        "min_cells_per_class": int(min_cells_per_class),
        "per_class": {},
    }
    if cell_class is None:
        out["note"] = "no cell_class provided; only global Spearman computed"
        return out
    cell_class = np.asarray(cell_class)
    per_class: Dict[str, Dict[str, float]] = {}
    medians: List[float] = []
    for cls in sorted({str(c) for c in cell_class}):
        mask = cell_class == cls
        n = int(mask.sum())
        if n < min_cells_per_class:
            continue
        res = _per_cell_spearman(coords_true[mask], coords_pred[mask])
        per_class[cls] = {
            "median": res["median"],
            "mean": res["mean"],
            "n_cells": n,
        }
        if not np.isnan(res["median"]):
            medians.append(res["median"])
    out["per_class"] = per_class
    out["mean_of_per_class_medians"] = (
        float(np.mean(medians)) if medians else float("nan")
    )
    return out


def _log_per_class_spearman(
    diag: Dict[str, object], section_id: str, top_n: int = 10,
) -> None:
    g = diag["global_median"]
    pcm = diag.get("mean_of_per_class_medians", float("nan"))
    logger.info(
        f"  [{section_id}] Spearman diagnostic: "
        f"global_median={g:.4f}, mean(per_class_medians)={pcm:.4f}"
    )
    if (
        not np.isnan(g) and not np.isnan(pcm) and (g - pcm) > 0.10
    ):
        logger.warning(
            f"  [{section_id}] GLOBAL Spearman is {g - pcm:+.3f} higher "
            f"than the mean of per-class medians "
            f"({g:.3f} vs {pcm:.3f}). Cell-type-collapsed embedding signature."
        )
    pc = diag.get("per_class", {}) or {}
    if pc:
        sorted_items = sorted(
            pc.items(), key=lambda kv: -kv[1]["n_cells"]
        )[:top_n]
        logger.info(
            f"  [{section_id}] per-class medians (top {len(sorted_items)} "
            "by cell count):"
        )
        for cls, stats in sorted_items:
            logger.info(
                f"    {cls:>20s}  n={stats['n_cells']:>6d}  "
                f"median={stats['median']:+.4f}  mean={stats['mean']:+.4f}"
            )


# ---------------------------------------------------------------------------
# Write LUNA predictions back into a per-section h5ad
# ---------------------------------------------------------------------------


def _write_pred_into_h5ad(
    silver_h5ad: Path,
    pred_df: pd.DataFrame,
    out_h5ad: Path,
):
    """Load original h5ad, attach obsm['spatial_pred'] from LUNA, save.

    We pass per-section integer indices (0..N-1) to LUNA, so its per-
    section ``metadata_pred.csv`` comes back with those same integer
    indices. The integer index ``i`` corresponds to position ``i`` in
    the h5ad's ``obs_names`` (we built the CSV by iterating h5ad rows in
    order). We sort by index defensively in case LUNA shuffles internally.
    """
    import anndata as ad

    adata = ad.read_h5ad(silver_h5ad)

    if len(pred_df) != adata.n_obs:
        raise ValueError(
            f"LUNA prediction count {len(pred_df)} != n_obs {adata.n_obs} "
            f"for {silver_h5ad.name}. Cell count mismatch — was the same "
            "silver h5ad used to build the test CSV?"
        )

    # Integer-index → row-position match. Sort to defend against any
    # internal reordering on LUNA's side; if the index is non-numeric
    # (legacy barcode-indexed CSV), fall through to the input order.
    if pred_df.index.dtype.kind in ("i", "u", "f"):
        pred_sorted = pred_df.sort_index()
    else:
        pred_sorted = pred_df

    coords = pred_sorted[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
    adata.obsm["spatial_pred"] = coords
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write(out_h5ad)
    return adata


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_inference(
    checkpoint: str,
    sections: List[str],
    silver_dir: Optional[str] = None,
    test_csv: Optional[str] = None,
    train_csv: Optional[str] = None,
    n_genes: Optional[int] = None,
    output_dir: Optional[str] = None,
    color_col: str = "cell_class",
    luna_repo: str = str(_DEFAULT_LUNA_REPO),
    run_name: str = "MERFISH_mouse_cortex_infer",
    log2_normalize: bool = False,
    spot_size: Optional[float] = None,
    no_align_plot: bool = False,
    per_class_spearman: bool = False,
    diagnostic_class_col: Optional[str] = None,
    palette: str = "luna",
    include_train_section: bool = False,
    extra_overrides: Optional[List[str]] = None,
) -> None:
    """Run LUNA inference; mirror of ``infer_scgg_on_cns:run_inference``.

    Two input modes:

      * **CSV mode** (recommended when the model was trained on LUNA's
        published preprocessed CSVs): pass ``test_csv`` + ``train_csv``,
        or auto-detect from the training output dir. Sections are
        filtered by the ``cell_section`` column. Plots are produced
        directly from LUNA's per-section ``metadata_pred.csv`` and
        ``metadata_true.csv`` — no h5ad needed.

      * **h5ad mode** (original): pass ``silver_dir``. Sections are
        resolved to h5ad files, gene expression is log2-normalized and
        written to a fresh test CSV, and the original h5ad is used as
        the plot backbone.

    For models trained on LUNA's CSVs, CSV mode is the **only correct
    choice** — h5ad-derived test data has a different normalization than
    the training data, so the model produces garbage on it.
    """
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # ----- Determine input mode + auto-detect paired CSVs --------------
    use_csv = (test_csv is not None and train_csv is not None)

    if not use_csv and silver_dir is None:
        # Auto-detect paired CSVs next to the checkpoint: training scripts
        # leave them at <output_dir>/work/{train,test}.csv.
        real_ckpt = ckpt_path.resolve()
        for parent in real_ckpt.parents:
            cand_train = parent / "work" / "train.csv"
            cand_test = parent / "work" / "test.csv"
            if cand_train.exists() and cand_test.exists():
                train_csv = str(cand_train)
                test_csv = str(cand_test)
                use_csv = True
                logger.info(
                    "Auto-detected paired CSVs from the training run:"
                )
                logger.info(f"  train_csv: {cand_train}")
                logger.info(f"  test_csv:  {cand_test}")
                break
        if not use_csv:
            raise ValueError(
                "Need either --silver_dir (h5ad mode) OR --test_csv + "
                "--train_csv (CSV mode). Auto-detection of paired CSVs at "
                f".../work/{{train,test}}.csv next to {ckpt_path} also "
                "failed."
            )

    # ----- Output dir ---------------------------------------------------
    if output_dir is None:
        model_ts = _timestamp_from_path(ckpt_path)
        if model_ts is None:
            model_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.warning(
                f"  checkpoint path has no luna_model/<timestamp>/ ancestor; "
                f"using fresh inference timestamp {model_ts}"
            )
        if use_csv:
            # Derive a dataset-like name from the training run's parent dir.
            ckpt_parent = ckpt_path.resolve().parents
            # ckpt_parent[0]=run_dir, [1]=luna_model, [2]=<dataset>
            base = (
                ckpt_parent[2].name if len(ckpt_parent) >= 3 else "luna_csv"
            )
        else:
            base = Path(silver_dir).name
        out_dir = _ARTIFACTS_ROOT / base / "luna_inference" / model_ts
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "infer.log", mode="w"),
        ],
        force=True,
    )
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Output dir: {out_dir}")

    luna_repo_p = Path(luna_repo)
    if not luna_repo_p.exists():
        raise FileNotFoundError(f"LUNA repo not found: {luna_repo_p}")

    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    inference_test_csv = work / "test.csv"
    inference_train_csv = work / "train.csv"

    if use_csv:
        # ---- CSV mode: filter LUNA's pre-built CSVs by section -------
        # `test_csv` / `train_csv` (args) are LUNA's full train+test
        # CSVs (typically the ones the training run used). We extract
        # only the requested sections into a single filtered CSV that
        # we'll point LUNA's test_data_path at.
        src_train_csv = Path(train_csv).resolve()
        src_test_csv = Path(test_csv).resolve()
        logger.info("CSV mode: filtering pre-built LUNA CSVs by --sections")
        logger.info(f"  source train CSV: {src_train_csv}")
        logger.info(f"  source test CSV:  {src_test_csv}")

        if include_train_section:
            extra_train_slice = "mouse1_slice99"
            if extra_train_slice not in sections:
                sections = [extra_train_slice] + list(sections)
                logger.info(
                    f"  --include_train_section: prepended "
                    f"{extra_train_slice!r}."
                )

        filt_stats = _filter_luna_csv_by_section(
            src_train_csv, src_test_csv, list(sections),
            out_csv=inference_test_csv,
        )
        logger.info(
            f"  filtered test CSV -> {inference_test_csv}  "
            f"({filt_stats['n_rows']:,} rows across "
            f"{filt_stats['n_sections']} sections: "
            f"{filt_stats['matched']})"
        )

        # LUNA needs a train_data_path too even in test_only mode
        # (for dataset setup). Symlink to the source train CSV so we
        # don't duplicate gigabytes on disk.
        if inference_train_csv.exists() or inference_train_csv.is_symlink():
            inference_train_csv.unlink()
        try:
            inference_train_csv.symlink_to(src_train_csv)
        except OSError:
            # FS without symlink support — fall through to direct path.
            inference_train_csv = src_train_csv

        # Infer n_genes from the source train CSV header if not passed
        # (uses the same dual name+dtype detection as run_luna_on_mmc).
        if n_genes is None:
            n_genes = _infer_n_genes_from_csv(src_train_csv)
            logger.info(f"  n_genes inferred from CSV = {n_genes}")
        else:
            logger.info(f"  n_genes (explicit) = {n_genes}")
    else:
        # ---- h5ad mode (legacy): build test CSV from silver h5ads ----
        silver_path = Path(silver_dir)
        section_paths = _resolve_section_files(silver_path, sections)

        if include_train_section:
            train_pick = _pick_representative_cortex_slice(silver_path, mouse_id=1)
            if train_pick is None:
                logger.warning(
                    f"  --include_train_section requested but no Mouse-1 silver "
                    f"h5ad found under {silver_path}; skipping."
                )
            elif train_pick in section_paths:
                logger.info(
                    f"  --include_train_section: {train_pick.name} already in "
                    "the requested set; not duplicating."
                )
            else:
                section_paths = [train_pick] + section_paths
                logger.info(
                    f"  --include_train_section: prepended {train_pick.name}."
                )

        logger.info(
            f"Inference on {len(section_paths)} sections: "
            f"{[p.name for p in section_paths]}"
        )

        test_files = _section_files_with_mouse(section_paths)
        logger.info(f"Writing test CSV  -> {inference_test_csv}")
        test_stats = _build_luna_csv(
            test_files, inference_test_csv, log2_normalize=log2_normalize,
        )
        n_genes = int(test_stats["n_genes"])

        # Train CSV: reuse the training-run's train.csv if available;
        # otherwise rebuild from Mouse-1 silver h5ads.
        candidate_train_csvs = [
            ckpt_path.resolve().parent / "work" / "train.csv",
            ckpt_path.resolve().parent.parent / "work" / "train.csv",
        ]
        reused = False
        for c in candidate_train_csvs:
            if c.exists():
                if inference_train_csv.exists() or inference_train_csv.is_symlink():
                    inference_train_csv.unlink()
                try:
                    inference_train_csv.symlink_to(c.resolve())
                    reused = True
                    logger.info(f"Reusing existing train CSV: {c}")
                    break
                except OSError:
                    pass
        if not reused:
            all_files = []
            for p in sorted(silver_path.iterdir()):
                mm = _SLICE_RE.match(p.name)
                if mm:
                    all_files.append((int(mm["mouse"]), int(mm["slice"]), p))
            mouse1_files = [f for f in all_files if f[0] == 1]
            if not mouse1_files:
                raise FileNotFoundError(
                    f"No existing train.csv near {ckpt_path}, and no Mouse-1 "
                    f"silver h5ads under {silver_path} to rebuild it."
                )
            logger.info(
                f"Re-building train CSV from {len(mouse1_files)} Mouse-1 "
                f"silver h5ads -> {inference_train_csv}"
            )
            _build_luna_csv(
                mouse1_files, inference_train_csv, log2_normalize=log2_normalize,
            )

    # All downstream code uses these names; rebind to whatever path we
    # actually settled on.
    train_csv = inference_train_csv  # noqa: F811
    test_csv = inference_test_csv

    # ---- 3. Invoke LUNA test mode --------------------------------------
    luna_run_dir = out_dir / "luna_run"
    luna_run_dir.mkdir(parents=True, exist_ok=True)
    test_save_dir = luna_run_dir / "test_results"
    test_save_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the checkpoint (in case it's a `best_model.ckpt` symlink)
    # and locate the original training run dir. LUNA's `test_only` flow:
    #
    #   def load_model_config(cfg, checkpoint_path):
    #       config_file = "/".join(checkpoint_path.split("/")[:-2])
    #       loading_model_cfg = safe_load(open(f"{config_file}/.hydra/config.yaml"))
    #       cfg["model"] = loading_model_cfg["model"]
    #
    # reads the *training-time* model config from the grandparent of the
    # checkpoint path (i.e., `<training_run_dir>/.hydra/config.yaml`).
    # We therefore must point LUNA at the original training checkpoints
    # dir, NOT a freshly-created isolated dir — otherwise LUNA picks up
    # the inference-time Hydra config (which has default model settings)
    # and would silently mismatch the architecture if training used any
    # model overrides.
    #
    # To limit LUNA to a single checkpoint inside that dir we set
    # `test.checkpoints_name_list=['epoch=N.ckpt']` instead of the
    # default `"all"` (which would otherwise iterate every checkpoint
    # LUNA saved during training).
    real_ckpt = ckpt_path.resolve()
    if not real_ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint symlink targets a missing file: "
            f"{ckpt_path} -> {real_ckpt}"
        )
    if not (real_ckpt.name.startswith("epoch=") and real_ckpt.name.endswith(".ckpt")):
        raise ValueError(
            f"Checkpoint name must match LUNA's `epoch=N.ckpt` pattern; "
            f"got {real_ckpt.name!r}. (LUNA's test_single_checkpoint "
            f"parses N via `name.split('=')[-1].split('.')[0]` and "
            f"returns silently on ValueError, so a mis-named checkpoint "
            f"would produce no predictions.)"
        )
    training_ckpts_dir = real_ckpt.parent              # <training>/luna_run/checkpoints
    training_run_dir = training_ckpts_dir.parent       # <training>/luna_run
    hydra_cfg_yaml = training_run_dir / ".hydra" / "config.yaml"
    if not hydra_cfg_yaml.exists():
        logger.warning(
            f"  expected training-time Hydra config at {hydra_cfg_yaml} "
            "but it doesn't exist. LUNA's test_only mode requires it to "
            "recover the training-time model architecture; the run may "
            "fail or use a mismatched model config."
        )

    # Single-quote paths so Hydra's override parser tolerates `=` (in
    # `epoch=N.ckpt`) and other special chars in the directory tree.
    def _h(v: object) -> str:
        return f"'{v}'"

    overrides = [
        f"general.name={run_name}",
        "general.mode=test_only",
        f"dataset.train_data_path={_h(train_csv.resolve())}",
        f"dataset.test_data_path={_h(test_csv.resolve())}",
        "dataset.gene_columns_start=0",
        f"dataset.gene_columns_end={n_genes}",
        f"test.checkpoints_parent_dir={_h(training_ckpts_dir.resolve())}",
        # Hydra list syntax with a quoted element so the `=` inside the
        # filename doesn't get parsed as another key/value separator.
        f"test.checkpoints_name_list=['{real_ckpt.name}']",
        f"test.save_dir={_h(test_save_dir.resolve())}",
        f"hydra.run.dir={_h(luna_run_dir.resolve())}",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)

    log_path = out_dir / "luna_stdout.log"
    rc = _invoke_luna(luna_repo_p, overrides, log_path)
    if rc != 0:
        raise RuntimeError(f"LUNA inference failed (exit {rc}). See {log_path}")

    # ---- 4. Read predictions + plot per section ------------------------
    # _read_luna_predictions returns {label: pred_df}. We also need the
    # paired metadata_true.csv for plotting / metrics — re-read it here.
    pred_sections = _read_luna_predictions(test_save_dir)
    if use_csv:
        label_to_silver: Dict[str, Optional[Path]] = {}
    else:
        label_to_silver = {
            f"mouse{m}_slice{s}": p for (m, s, p) in test_files
        }

    # Resolve per-section paths to metadata_true.csv (LUNA writes it
    # alongside metadata_pred.csv in each section subdir).
    def _true_csv_for(label: str) -> Optional[Path]:
        for pred in test_save_dir.rglob("metadata_pred.csv"):
            if pred.parent.name == label:
                t = pred.parent / "metadata_true.csv"
                return t if t.exists() else None
        return None

    summary: List[Dict[str, object]] = []
    for section_label, pred_df in pred_sections.items():
        # Resolve a plot title and an h5ad backbone (h5ad mode) OR build
        # an AnnData from LUNA's metadata CSVs (CSV mode).
        if use_csv:
            true_csv_path = _true_csv_for(section_label)
            if true_csv_path is None:
                logger.warning(
                    f"  no metadata_true.csv for LUNA section "
                    f"{section_label!r}; skipping plot."
                )
                continue
            true_df = pd.read_csv(true_csv_path, index_col=0)
            section_id = section_label
            logger.info(f"[{section_id}] LUNA -> CSV mode")
            adata = _adata_from_luna_outputs(pred_df, true_df)
            out_h5ad = out_dir / f"{section_id}_predicted.h5ad"
            adata.write(out_h5ad)
            input_source = str(true_csv_path)
        else:
            silver_for_label = label_to_silver.get(section_label)
            if silver_for_label is None:
                matches = [
                    p for k, p in label_to_silver.items() if k in section_label
                ]
                if not matches:
                    logger.warning(
                        f"  no silver h5ad matches LUNA section "
                        f"{section_label!r}; skipping plot."
                    )
                    continue
                silver_for_label = matches[0]
            section_id = _strip_known_prefix(silver_for_label.stem)
            logger.info(f"[{section_id}] LUNA -> {silver_for_label}")
            out_h5ad = out_dir / f"{section_id}_predicted.h5ad"
            adata = _write_pred_into_h5ad(silver_for_label, pred_df, out_h5ad)
            input_source = str(silver_for_label)

        coords_true_2d = (
            np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]
            if "spatial" in adata.obsm
            and adata.obsm["spatial"].shape[1] >= 2
            else None
        )

        if color_col not in adata.obs.columns:
            cat_cols = [
                c for c in adata.obs.columns
                if adata.obs[c].dtype == "object"
                or adata.obs[c].dtype.name == "category"
            ]
            color_col_eff = cat_cols[0] if cat_cols else None
            if color_col_eff:
                logger.warning(
                    f"  color column {color_col!r} missing; using "
                    f"{color_col_eff!r}"
                )
        else:
            color_col_eff = color_col

        diag_class_col = diagnostic_class_col or color_col_eff
        cell_class_arr = (
            adata.obs[diag_class_col].astype(str).to_numpy()
            if diag_class_col and diag_class_col in adata.obs.columns
            else None
        )
        spearman_diag = None
        if per_class_spearman and coords_true_2d is not None:
            spearman_diag = _per_class_spearman(
                coords_true_2d, adata.obsm["spatial_pred"], cell_class_arr,
            )
            _log_per_class_spearman(spearman_diag, section_id)

        if color_col_eff is not None:
            out_svg = out_dir / f"{section_id}_comparison.svg"
            _plot_comparison(
                adata, color_col_eff, out_svg,
                title_prefix=f"[{section_id}] ",
                spot_size=spot_size,
                align_for_plot=not no_align_plot,
                palette=palette,
            )

        rec: Dict[str, object] = {
            "section_id": section_id,
            "input_path": input_source,
            "output_h5ad": str(out_h5ad),
            "n_cells": int(adata.n_obs),
            "color_col": color_col_eff,
        }
        if spearman_diag is not None:
            rec["spearman_diagnostic"] = spearman_diag
        summary.append(rec)

    with open(out_dir / "inference_metadata.json", "w") as f:
        json.dump(
            {
                "method": "LUNA",
                "checkpoint": str(ckpt_path),
                "input_mode": "csv" if use_csv else "h5ad",
                "silver_dir": (None if use_csv else str(silver_dir)),
                "source_train_csv": (str(train_csv) if use_csv else None),
                "source_test_csv": (str(test_csv) if use_csv else None),
                "sections": summary,
            },
            f, indent=2, default=str,
        )
    logger.info(f"Wrote LUNA inference artifacts to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to the LUNA .ckpt file (produced by run_luna_on_mmc.py).",
    )
    p.add_argument(
        "--silver_dir", default=None,
        help="Per-slice silver h5ad directory (h5ad mode). Either this OR "
             "(--train_csv + --test_csv) must be provided — or, when the "
             "checkpoint sits under .../luna_model/<TS>/, paired CSVs are "
             "auto-detected from <TS>/work/{train,test}.csv.",
    )
    p.add_argument(
        "--train_csv", default=None,
        help="Pre-built LUNA train CSV (CSV mode). Pair with --test_csv. "
             "REQUIRED when the model was trained on LUNA's preprocessed "
             "CSVs (h5ad-derived test data has a different normalization "
             "and will produce garbage predictions).",
    )
    p.add_argument(
        "--test_csv", default=None,
        help="Pre-built LUNA test CSV. Pair with --train_csv.",
    )
    p.add_argument(
        "--n_genes", type=int, default=None,
        help="Number of gene columns in the pre-built CSVs (CSV mode). "
             "Auto-inferred from the train CSV header when omitted.",
    )
    p.add_argument(
        "--sections", nargs="+", default=None,
        help=(
            "Section ids / filenames / globs / presets. Accepts: "
            "bare ids ('mouse2_slice1'), filenames "
            "('mmc_mouse2_slice1.h5ad'), globs ('mouse2_*'), or the "
            "special values: 'all' (every silver h5ad), 'all_test' "
            "(mouse2_*), 'paper' (mouse2_slice99 — LUNA Fig 3c). "
            "Default: 'all_test'."
        ),
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Where to write predicted h5ads and comparison SVGs. Default: "
             "/nfs/team361/sb75/scgg-reproducibility/artifacts/"
             "<silver_dir_name>/luna_inference/<model_timestamp>/.",
    )
    p.add_argument("--color", default="cell_class",
                   help="adata.obs column for plot coloring.")
    p.add_argument(
        "--luna_repo", default=str(_DEFAULT_LUNA_REPO),
        help=f"Path to the LUNA repository. Default: {_DEFAULT_LUNA_REPO}",
    )
    p.add_argument(
        "--run_name", default="MERFISH_mouse_cortex_infer",
        help="general.name in LUNA's Hydra config.",
    )
    p.add_argument(
        "--log2_normalize", action="store_true",
        help="Apply log2(x+1) when building the test CSV. OFF by default "
             "to match LUNA's published CSVs, which carry non-integer "
             "per-cell-normalized counts in the [0, ~250] range — NOT "
             "log-transformed values. Only use this for ablation or when "
             "you know your checkpoint was trained on log2-transformed "
             "input.",
    )
    p.add_argument(
        "--spot_size", type=float, default=None,
        help="Marker size (default auto-scales with cell count).",
    )
    p.add_argument(
        "--no_align_plot", action="store_true",
        help="Don't Procrustes-align predicted coords to GT before plotting.",
    )
    p.add_argument(
        "--per_class_spearman", action="store_true",
        help="Compute median per-cell Spearman per cell class (plus global).",
    )
    p.add_argument(
        "--diagnostic_class_col", default=None,
        help="Override which obs column drives per-class Spearman.",
    )
    p.add_argument(
        "--palette", default="luna", choices=("luna", "tab20"),
        help="Color palette for the categorical cell-class plots.",
    )
    p.add_argument(
        "--include_train_section", action="store_true",
        help="Also run inference on one representative Mouse-1 slice as a "
             "sanity check (train-side reconstruction quality).",
    )
    p.add_argument(
        "--luna_override", action="append", default=[],
        help="Extra Hydra overrides. Repeatable.",
    )
    args = p.parse_args()

    # Default --sections: 'all_test' in h5ad mode (resolves to every
    # mouse2_*.h5ad). In CSV mode 'all_test' isn't supported (we filter
    # by exact cell_section names), so require the user to be explicit.
    if args.sections is not None:
        sections = args.sections
    elif args.train_csv is None and args.test_csv is None:
        sections = ["all_test"]
    else:
        raise SystemExit(
            "--sections must be specified when using --train_csv/--test_csv "
            "(CSV mode). Examples:\n"
            "  --sections mouse2_slice99 mouse2_slice119 mouse1_slice1\n"
            "  --sections mouse2_slice99\n"
            "(The 'all_test' / 'paper' presets only work in h5ad mode.)"
        )

    try:
        run_inference(
            checkpoint=args.checkpoint,
            sections=sections,
            silver_dir=args.silver_dir,
            test_csv=args.test_csv,
            train_csv=args.train_csv,
            n_genes=args.n_genes,
            output_dir=args.output_dir,
            color_col=args.color,
            luna_repo=args.luna_repo,
            run_name=args.run_name,
            log2_normalize=args.log2_normalize,
            spot_size=args.spot_size,
            no_align_plot=args.no_align_plot,
            per_class_spearman=args.per_class_spearman,
            diagnostic_class_col=args.diagnostic_class_col,
            palette=args.palette,
            include_train_section=args.include_train_section,
            extra_overrides=args.luna_override,
        )
    except Exception:
        logger.exception("LUNA inference failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

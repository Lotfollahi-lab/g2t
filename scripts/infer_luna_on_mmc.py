#!/usr/bin/env python
"""
Run LUNA inference on held-out silver sections using a trained LUNA checkpoint.

Mirror of ``scgg/scripts/infer_scgg_on_cns.py`` (the scGG inference
script). This script is **self-contained**: no dependency on the scgg
package. Meant to be run *in the LUNA Python environment* — same env as
``run_luna_on_mmc.py``.

Input is the **same silver h5ad directory** that the scGG pipeline reads
from (``--silver_dir``). The two methods apply their own
method-specific normalization on top:

  * scGG: ``normalize_total(1e4) + log1p + scale`` (via scanpy in
    ``scgg.data.luna_cortex``).
  * LUNA: ``log2(x + 1)`` (this script, when ``--no_log2_normalize`` is
    off — the default).

So the silver layer is exactly identical for both; only the normalization
applied downstream differs, which is intentional and matches each
method's published expectations.

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
# Build LUNA-format CSVs from per-slice h5ads
# ---------------------------------------------------------------------------


def _build_luna_csv(
    files: List[Tuple[int, int, Path]],
    out_csv: Path,
    log2_normalize: bool = True,
) -> Dict[str, object]:
    """Concatenate per-slice h5ads into a LUNA-format CSV."""
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
    silver_dir: str,
    sections: List[str],
    output_dir: Optional[str] = None,
    color_col: str = "cell_class",
    luna_repo: str = str(_DEFAULT_LUNA_REPO),
    run_name: str = "MERFISH_mouse_cortex_infer",
    log2_normalize: bool = True,
    spot_size: Optional[float] = None,
    no_align_plot: bool = False,
    per_class_spearman: bool = False,
    diagnostic_class_col: Optional[str] = None,
    palette: str = "luna",
    include_train_section: bool = False,
    extra_overrides: Optional[List[str]] = None,
) -> None:
    """Run LUNA inference; mirror of ``infer_scgg_on_cns:run_inference``."""
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    silver_path = Path(silver_dir)
    if output_dir is None:
        model_ts = _timestamp_from_path(ckpt_path)
        if model_ts is None:
            model_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.warning(
                f"  checkpoint path has no luna_model/<timestamp>/ ancestor; "
                f"using fresh inference timestamp {model_ts}"
            )
        out_dir = _ARTIFACTS_ROOT / silver_path.name / "luna_inference" / model_ts
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

    # ---- 1. Resolve sections --------------------------------------------
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

    # ---- 2. Build test CSV + ensure train CSV is available -------------
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    test_csv = work / "test.csv"
    train_csv = work / "train.csv"

    test_files = _section_files_with_mouse(section_paths)
    logger.info(f"Writing test CSV  -> {test_csv}")
    test_stats = _build_luna_csv(
        test_files, test_csv, log2_normalize=log2_normalize,
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
            if train_csv.exists() or train_csv.is_symlink():
                train_csv.unlink()
            try:
                train_csv.symlink_to(c.resolve())
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
            f"silver h5ads -> {train_csv}"
        )
        _build_luna_csv(mouse1_files, train_csv, log2_normalize=log2_normalize)

    # ---- 3. Invoke LUNA test mode --------------------------------------
    luna_run_dir = out_dir / "luna_run"
    luna_run_dir.mkdir(parents=True, exist_ok=True)
    test_save_dir = luna_run_dir / "test_results"
    test_save_dir.mkdir(parents=True, exist_ok=True)

    # LUNA's `test_model()` (when general.mode='test_only') iterates over
    # checkpoints found in `test.checkpoints_parent_dir`, applies
    # `test.checkpoints_name_list` (default 'all') to pick which to run,
    # and parses each filename via `name.split("=")[-1].split(".")[0]`
    # to extract an epoch number — so the file MUST be named
    # `epoch=N.ckpt`. Our wrapper accepts a single checkpoint path; we
    # isolate it by symlinking into a dedicated dir under our output
    # location, preserving the original filename (resolving any
    # `best_model.ckpt` symlink to its actual `epoch=N.ckpt` target).
    real_ckpt = ckpt_path.resolve()
    if not real_ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint symlink targets a missing file: "
            f"{ckpt_path} -> {real_ckpt}"
        )
    if not real_ckpt.name.startswith("epoch=") or not real_ckpt.name.endswith(".ckpt"):
        logger.warning(
            f"  checkpoint target {real_ckpt.name!r} does not match "
            "LUNA's expected 'epoch=N.ckpt' pattern; LUNA's "
            "test_single_checkpoint will return silently. Pass a "
            "checkpoint with that naming, or rename your target."
        )
    ckpt_isolation_dir = luna_run_dir / "single_ckpt"
    ckpt_isolation_dir.mkdir(parents=True, exist_ok=True)
    linked = ckpt_isolation_dir / real_ckpt.name
    if linked.exists() or linked.is_symlink():
        linked.unlink()
    try:
        linked.symlink_to(real_ckpt)
    except OSError as e:
        raise RuntimeError(
            f"Could not symlink {real_ckpt} -> {linked}: {e}. "
            "LUNA's test phase requires a writable dir holding the "
            "checkpoint under its `epoch=N.ckpt` name."
        )

    # Single-quote paths so Hydra's override parser tolerates `=` and
    # other special chars (LUNA's `epoch=N.ckpt` filenames have a literal
    # `=` that would otherwise be split as a key-value separator).
    def _h(v: object) -> str:
        return f"'{v}'"

    overrides = [
        f"general.name={run_name}",
        "general.mode=test_only",
        f"dataset.train_data_path={_h(train_csv.resolve())}",
        f"dataset.test_data_path={_h(test_csv.resolve())}",
        "dataset.gene_columns_start=0",
        f"dataset.gene_columns_end={n_genes}",
        f"test.checkpoints_parent_dir={_h(ckpt_isolation_dir.resolve())}",
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
    pred_sections = _read_luna_predictions(test_save_dir)
    label_to_silver = {f"mouse{m}_slice{s}": p for (m, s, p) in test_files}

    summary: List[Dict[str, object]] = []
    for section_label, pred_df in pred_sections.items():
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
            "input_path": str(silver_for_label),
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
                "silver_dir": str(silver_path),
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
        "--silver_dir", required=True,
        help="Per-slice silver h5ad directory.",
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
        "--no_log2_normalize", action="store_true",
        help="Skip log2(x+1) on the test CSV.",
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

    sections = args.sections or ["all_test"]

    try:
        run_inference(
            checkpoint=args.checkpoint,
            silver_dir=args.silver_dir,
            sections=sections,
            output_dir=args.output_dir,
            color_col=args.color,
            luna_repo=args.luna_repo,
            run_name=args.run_name,
            log2_normalize=not args.no_log2_normalize,
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

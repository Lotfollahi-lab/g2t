#!/usr/bin/env python
"""
Train scgg on a silver h5ad directory (LUNA paper Figure 3 split when
``--data_dir`` points at the MERFISH mouse cortex silver tree, but any
``*_train.h5ad`` / ``*_test.h5ad`` silver layout works).

Sister script of ``scgg/scripts/run_luna_train.py`` — byte-identical
except for the engine-specific configuration block at the top of the
file. scgg currently runs the **vendored copy of LUNA** under
``scgg/src/`` (which we will start modifying forward), while
``run_luna_train.py`` invokes the **external (pristine) LUNA checkout**
as the immutable baseline. Treat them as two separate methods to be
benchmarked against each other.

Input is a silver h5ad directory (``--data_dir``); the h5ad's ``.X``
is written to LUNA's CSV format as-is.

Pipeline
--------
  1. Discover ``*_train.h5ad`` (training) and ``*_test.h5ad`` (held
     out for inference) under ``--data_dir``. Layout produced by
     ``build_h5ad_from_luna_csv.py``; works for any dataset (cortex,
     ABC, CNS, ...).
  2. Convert per-section h5ads to LUNA's expected CSV layout
     (gene columns first, then ``coord_X`` / ``coord_Y`` /
     ``cell_section`` / ``cell_class``).
  3. Invoke the vendored scgg/src/ engine in ``general.mode=train_only``;
     ``run_scgg_inference.py`` evaluates checkpoints in ``test_only``
     mode on the held-out test data.
  4. After training, pin the latest checkpoint to ``best_model.ckpt``.

Output layout
-------------
``--output_dir`` defaults to
``/nfs/team361/sb75/scgg-reproducibility/artifacts/<data_dir.name>/scgg_model/<YYYYMMDD_HHMMSS>/``.

Usage
-----
    source /nfs/team361/sb75/.venvs/scgg/bin/activate
    python scripts/run_scgg_train.py \
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \
        --epochs 1000 --batch_size 6
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("scgg_train")


# ---------------------------------------------------------------------------
# Runtime tracking
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Resource sampling helpers — used by ``_RuntimeTracker`` to track peak
# GPU memory and process-tree RSS per phase. See the matching block
# in run_luna_train.py for the full rationale; the two scripts keep
# this code byte-equivalent so a future improvement to one needs the
# same diff applied to the other.
# ---------------------------------------------------------------------------


def _sample_gpu_mib_via_nvidia_smi() -> Optional[float]:
    """Sum of ``memory.used`` across CUDA-visible GPUs, in MiB. None
    when nvidia-smi is unavailable, fails, or produces unparseable
    output. Respects ``CUDA_VISIBLE_DEVICES``.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    per_gpu: Dict[int, float] = {}
    for line in result.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            per_gpu[int(parts[0])] = float(parts[1])
        except (ValueError, TypeError):
            continue
    if not per_gpu:
        return None
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis:
        try:
            visible_idx = {int(x) for x in vis.split(",") if x.strip()}
            filtered = {k: v for k, v in per_gpu.items() if k in visible_idx}
            if filtered:
                return sum(filtered.values())
        except ValueError:
            pass
    return sum(per_gpu.values())


def _get_gpu_name_and_count() -> Tuple[Optional[str], int]:
    """Return ``(gpu_name, n_visible_gpus)`` from ``nvidia-smi``. See
    the matching helper in run_luna_train.py for the full docstring;
    kept byte-equivalent across the two scripts.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None, 0
    if result.returncode != 0:
        return None, 0
    per_gpu: Dict[int, str] = {}
    for line in result.stdout.splitlines():
        parts = [x.strip() for x in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            per_gpu[int(parts[0])] = parts[1]
        except (ValueError, TypeError):
            continue
    if not per_gpu:
        return None, 0
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis:
        try:
            visible_idx = sorted(int(x) for x in vis.split(",") if x.strip())
            visible_names = [per_gpu[i] for i in visible_idx if i in per_gpu]
            if visible_names:
                return visible_names[0], len(visible_names)
        except ValueError:
            pass
    sorted_keys = sorted(per_gpu.keys())
    return per_gpu[sorted_keys[0]], len(per_gpu)


def _sample_rss_mib_total() -> Optional[float]:
    """Process-tree RSS in MiB (this process + descendants). Captures
    the LUNA training subprocess's RSS too. Falls back to
    ``resource.getrusage`` on the parent process only when psutil
    isn't installed.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        try:
            import resource
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return r / (1024.0 * 1024.0)
            return r / 1024.0
        except Exception:  # noqa: BLE001
            return None
    try:
        proc = psutil.Process()
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return None


class _ResourceSampler:
    """Daemon thread that polls GPU + RSS at a fixed interval and
    tracks the running peak per phase. See run_luna_train.py for the
    full docstring; kept byte-equivalent across the two scripts.
    """

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = float(interval_s)
        self.peak_gpu_mib: Optional[float] = None
        self.peak_rss_mib: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        gpu = _sample_gpu_mib_via_nvidia_smi()
        rss = _sample_rss_mib_total()
        if gpu is not None:
            self.peak_gpu_mib = (
                gpu if self.peak_gpu_mib is None else max(self.peak_gpu_mib, gpu)
            )
        if rss is not None:
            self.peak_rss_mib = (
                rss if self.peak_rss_mib is None else max(self.peak_rss_mib, rss)
            )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:  # noqa: BLE001
                pass
            if self._stop.wait(self.interval_s):
                break

    def start(self) -> None:
        try:
            self._sample()
        except Exception:  # noqa: BLE001
            pass
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 1.0)
        try:
            self._sample()
        except Exception:  # noqa: BLE001
            pass
        self._thread = None


class _RuntimeTracker:
    """Records phase wall-clock + peak GPU + peak RSS, writes to CSV.

    Use as::

        tracker = _RuntimeTracker()
        out_csv = out_dir / "runtime.csv"

        tracker.start("csv_build")
        ...do csv build...
        tracker.end("csv_build", out_csv)

        tracker.start("training")
        ...invoke LUNA...
        tracker.end("training", out_csv)

    During each open phase, a background ``_ResourceSampler`` thread
    polls GPU memory (via ``nvidia-smi``) and process-tree RSS (via
    ``psutil``) every ``RUNTIME_SAMPLE_S`` seconds (default 2.0) and
    tracks the running peak. Both columns are fail-soft: missing
    nvidia-smi → no GPU column; missing psutil → parent-only RSS via
    ``resource.getrusage``.

    ``end()`` flushes the running CSV after each phase. The CSV always
    includes a final ``total`` row with duration-sum and peak-max
    across phases.
    """

    def __init__(self):
        self.phases: List[Dict[str, object]] = []
        self.overall_t0 = time.time()
        self._open: Dict[str, Tuple[float, "_ResourceSampler"]] = {}
        try:
            self._sample_interval_s = float(
                os.environ.get("RUNTIME_SAMPLE_S", "2.0")
            )
        except (ValueError, TypeError):
            self._sample_interval_s = 2.0

    def start(self, name: str) -> None:
        sampler = _ResourceSampler(self._sample_interval_s)
        sampler.start()
        self._open[name] = (time.time(), sampler)
        logger.info(f"[runtime] start phase: {name}")

    def end(self, name: str, flush_to: Optional[Path] = None) -> None:
        entry = self._open.pop(name, None)
        if entry is None:
            logger.warning(f"[runtime] end({name!r}) called without a matching start; skipping")
            return
        t0, sampler = entry
        sampler.stop()
        t1 = time.time()
        peak_gpu = sampler.peak_gpu_mib
        peak_rss = sampler.peak_rss_mib
        self.phases.append({
            "phase": name,
            "start": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
            "end": datetime.fromtimestamp(t1).isoformat(timespec="seconds"),
            "duration_s": round(t1 - t0, 3),
            "peak_gpu_mib": round(peak_gpu, 1) if peak_gpu is not None else "",
            "peak_rss_mib": round(peak_rss, 1) if peak_rss is not None else "",
        })
        gpu_str = f"  peak_gpu={peak_gpu:.0f}MiB" if peak_gpu is not None else ""
        rss_str = f"  peak_rss={peak_rss:.0f}MiB" if peak_rss is not None else ""
        logger.info(
            f"[runtime] end   phase: {name}  ({t1 - t0:.1f}s){gpu_str}{rss_str}"
        )
        if flush_to is not None:
            self.write_csv(flush_to)

    @contextmanager
    def phase(self, name: str, flush_to: Optional[Path] = None):
        """Alternative context-manager API for new code; equivalent
        to ``start`` + ``end`` with automatic re-raise on exceptions.
        """
        self.start(name)
        try:
            yield
        finally:
            self.end(name, flush_to=flush_to)

    def peak_summary(self) -> Dict[str, Optional[float]]:
        """Max GPU / RSS across all completed phases. Useful for
        stamping the aggregate_metrics.json without re-parsing the
        CSV.
        """
        gpu_vals = [r["peak_gpu_mib"] for r in self.phases
                    if isinstance(r.get("peak_gpu_mib"), (int, float))]
        rss_vals = [r["peak_rss_mib"] for r in self.phases
                    if isinstance(r.get("peak_rss_mib"), (int, float))]
        return {
            "peak_gpu_mib": round(max(gpu_vals), 1) if gpu_vals else None,
            "peak_rss_mib": round(max(rss_vals), 1) if rss_vals else None,
        }

    def write_compute_requirements_csv(
        self,
        out_path: Path,
        *,
        method: str,
        run_timestamp: str,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        """Single-row CSV summarising the run's compute cost. See the
        matching method in run_luna_train.py for the full docstring;
        kept byte-equivalent across the two scripts.
        """
        import socket
        peak = self.peak_summary()
        gpu_name, n_gpus = _get_gpu_name_and_count()

        row: Dict[str, object] = {
            "method": method,
            "run_timestamp": run_timestamp,
            "host": socket.gethostname(),
            "gpu_name": gpu_name or "",
            "n_gpus_visible": n_gpus,
            "total_duration_s": round(time.time() - self.overall_t0, 3),
            "peak_gpu_mib": (
                peak["peak_gpu_mib"]
                if peak["peak_gpu_mib"] is not None else ""
            ),
            "peak_rss_mib": (
                peak["peak_rss_mib"]
                if peak["peak_rss_mib"] is not None else ""
            ),
        }
        if extra:
            for k, v in extra.items():
                if k in row:
                    continue
                row[k] = v

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)

    def write_csv(self, out_path: Path) -> None:
        rows = list(self.phases)
        gpu_vals = [r["peak_gpu_mib"] for r in rows
                    if isinstance(r.get("peak_gpu_mib"), (int, float))]
        rss_vals = [r["peak_rss_mib"] for r in rows
                    if isinstance(r.get("peak_rss_mib"), (int, float))]
        rows.append({
            "phase": "total",
            "start": datetime.fromtimestamp(self.overall_t0).isoformat(timespec="seconds"),
            "end": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.time() - self.overall_t0, 3),
            "peak_gpu_mib": round(max(gpu_vals), 1) if gpu_vals else "",
            "peak_rss_mib": round(max(rss_vals), 1) if rss_vals else "",
        })
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "phase", "start", "end", "duration_s",
                "peak_gpu_mib", "peak_rss_mib",
            ])
            w.writeheader()
            for r in rows:
                w.writerow(r)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Engine-specific configuration. This is the ONLY block that differs
# between run_luna_train.py and its sister run_scgg_train.py — the rest
# of the file is byte-identical between the two. Treat them as two
# separate methods (LUNA baseline vs scgg, where scgg currently is a
# copy of LUNA we will modify forward). Keep this top-of-file
# tractable so future engine swaps are a small targeted diff.
# ---------------------------------------------------------------------------

ENGINE_NAME = "scgg"               # used in output dir subpath
ENGINE_DISPLAY = "scgg"            # human-readable, used in log lines + plot titles
ENGINE_OUTPUT_SUBDIR = "scgg_model"  # <artifacts_root>/<dataset>/<this>/<TS>/
# The default repo for the LUNA model code. The luna variant points at
# the external (pristine) LUNA checkout; the scgg variant points at
# the vendored LUNA copy under scgg/src/.
_ENGINE_REPO_DEFAULT = Path(__file__).resolve().parent.parent / "src"

# Artifacts root: honour SCGG_ARTIFACTS_ROOT (the same env var the LSF
# submitter exports). Without this lookup, setting the env var only
# redirects the LSF log dir while the python script kept writing to
# the hardcoded NFS path — splitting a single run across two
# filesystems silently.
_ARTIFACTS_ROOT = Path(os.environ.get(
    "SCGG_ARTIFACTS_ROOT",
    "/nfs/team361/sb75/scgg-reproducibility/artifacts",
))

_EPOCH_RE = re.compile(r"epoch=(\d+)")


# ---------------------------------------------------------------------------
# Silver-h5ad discovery
# ---------------------------------------------------------------------------
#
# The silver layout produced by ``build_h5ad_from_luna_csv.py`` is:
#   <silver_dir>/<section_label>_train.h5ad   <- training cells
#   <silver_dir>/<section_label>_test.h5ad    <- held-out test cells
#
# Discovery is suffix-based, so it works for any dataset (cortex, ABC,
# CNS, ...) without dataset-specific filename regexes. The section
# label inside each h5ad is whatever obs['cell_section'] says (which
# build_h5ad_from_luna_csv preserves verbatim from the source CSV);
# we fall back to the filename stem with the suffix stripped if
# obs['cell_section'] is missing or non-uniform.


def _discover_split_files(silver_dir: Path, split: str) -> List[Path]:
    """Return sorted ``*_{split}.h5ad`` paths under ``silver_dir``."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
    return sorted(silver_dir.glob(f"*_{split}.h5ad"))


def _section_label_from_filename(path: Path) -> str:
    """Filename-stem fallback for the section label. Strip the
    ``_train`` / ``_test`` suffix and return the rest.
    """
    stem = path.stem
    for suf in ("_train", "_test"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


# ---------------------------------------------------------------------------
# Pred-vs-truth plotting (self-contained — no scgg.evaluation dep so
# this script runs in the LUNA env where the scgg package isn't
# installed). Inlined deliberately so the two run_*_train.py scripts
# stay byte-identical in their helpers as they diverge in the engine
# they invoke.
# ---------------------------------------------------------------------------


def _umeyama_align(
    src: np.ndarray, dst: np.ndarray, allow_reflection: bool = True,
) -> np.ndarray:
    """Best similarity transform (rotate + scale + translate, plus
    optional reflection) mapping ``src`` onto ``dst``. Visual A/B
    aid only — the loss is rotation-invariant so the prediction
    frame may differ from GT even when structure is correct.
    """
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
    return ((s * (src @ R.T)) + t).astype(np.float32)


def _palette_for(cats: List[str], scheme: str = "glasbey") -> list:
    """RGB(A) colors for ``cats``. Prefers ``colorcet.glasbey`` (high
    distinctness — LUNA's paper figures use it); falls back to
    matplotlib's tab20 if colorcet is missing."""
    import matplotlib.pyplot as plt
    n = max(len(cats), 1)
    if scheme == "glasbey":
        try:
            import colorcet as cc  # type: ignore
            return list(cc.glasbey[:n])
        except ImportError:
            logger.info(
                "colorcet not installed; falling back to tab20. "
                "Install with: pip install colorcet"
            )
    cmap = plt.get_cmap("tab20", n)
    return [cmap(i) for i in range(n)]


def _plot_pred_vs_truth(
    coords_true: np.ndarray,
    coords_pred: np.ndarray,
    cell_class: Optional[np.ndarray],
    out_path: Path,
    title_prefix: str = "",
    method_label: str = "prediction",
    align_for_plot: bool = True,
    spot_size: Optional[float] = None,
    palette: str = "glasbey",
) -> None:
    """Side-by-side scatter of ground truth vs prediction.

    Inlined from scgg.evaluation.visualization.plot_pred_vs_truth so
    this script has no scgg-package dependency. If you change the plot
    here, mirror the change in the sister run_*_train.py script (and
    in scgg.evaluation.visualization if you want the package helper
    to stay in sync).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Editable text in vector outputs. ``svg.fonttype="none"``
    # writes text as <text> elements (Inkscape / Illustrator can
    # select and re-style them); the default rasterises text into
    # <path> outlines, which is what makes "SVG but not editable".
    # ``pdf.fonttype=42`` embeds TrueType fonts in the PDF for the
    # same reason.
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    out_path.parent.mkdir(parents=True, exist_ok=True)

    coords_true = np.asarray(coords_true, dtype=np.float64)
    coords_pred = np.asarray(coords_pred, dtype=np.float64)
    if coords_true.shape != coords_pred.shape:
        raise ValueError(
            f"coords_true and coords_pred shape mismatch: "
            f"{coords_true.shape} vs {coords_pred.shape}"
        )
    n = coords_true.shape[0]
    if spot_size is None:
        spot_size = max(1.0, min(20.0, 1500.0 / np.sqrt(max(n, 1))))

    coords_pred_plot = (
        _umeyama_align(coords_pred, coords_true, allow_reflection=True)
        if align_for_plot else coords_pred
    )
    aligned_suffix = " (aligned)" if align_for_plot else ""

    if cell_class is not None:
        cell_class = np.asarray(cell_class).astype(str)
        cats = sorted(set(cell_class))
        colors = _palette_for(cats, scheme=palette)
        cat_to_color = dict(zip(cats, colors))
        point_colors = [cat_to_color[c] for c in cell_class]
    else:
        cats = []
        cat_to_color = {}
        point_colors = "#666666"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, xy, title in (
        (axes[0], coords_true, f"{title_prefix}Ground truth"),
        (axes[1], coords_pred_plot, f"{title_prefix}{method_label}{aligned_suffix}"),
    ):
        ax.scatter(xy[:, 0], xy[:, 1], c=point_colors, s=spot_size, linewidths=0)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#999")

    if cats:
        n_cats = len(cats)
        ncol = min(max(1, (n_cats + 3) // 4), 6)
        patches = [
            Patch(facecolor=cat_to_color[c], label=str(c)) for c in cats
        ]
        fig.legend(
            handles=patches, loc="lower center",
            bbox_to_anchor=(0.5, 0.0), ncol=ncol,
            frameon=False, fontsize="small",
        )
        n_rows = (n_cats + ncol - 1) // ncol
        bottom = min(0.30, 0.05 + 0.04 * n_rows)
        fig.tight_layout(rect=(0, bottom, 1, 1))
    else:
        fig.tight_layout()

    # SVG only — editable text via the rcParams set above is
    # enough for figure work; PDFs were redundant.
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Build LUNA-format CSVs from per-slice h5ads
# ---------------------------------------------------------------------------


def _build_luna_csv(
    files: List[Path],
    out_csv: Path,
    log2_normalize: bool = False,
    embedding_field: Optional[str] = None,
) -> Dict[str, object]:
    """Concatenate per-section h5ads into one CSV in LUNA's input format.

    Each file in ``files`` is one section. The section label comes
    from the h5ad's ``obs['cell_section']`` (must be uniform per
    file); if missing or non-uniform we fall back to the filename
    stem with the ``_train`` / ``_test`` suffix stripped.

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

    Pretrained gene encoder path
    ----------------------------
    When ``embedding_field`` is set (e.g. ``"pca_64"``,
    ``"ae_128"``), the "gene" columns of the resulting CSV are
    filled from ``adata.obsm[embedding_field]`` instead of
    ``adata.X``. The downstream training pipeline doesn't need to
    change — it sees a CSV with ``embedding_dim`` "gene" columns and
    runs LUNA's standard gene encoder on top of them. The encoder
    is now upstream of training: see ``precompute_embeddings.py``.
    The synthetic "gene names" written into the CSV header (and
    used by gene-panel-consistency checks) are
    ``"<embedding_field>_<idx>"``; this stays unique across runs and
    makes it obvious in downstream CSV inspection that you're not
    looking at real gene names.
    """
    import anndata as ad
    import scipy.sparse as sp

    gene_names: Optional[List[str]] = None

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    # Collect per-section DataFrames into memory so we can sort all of
    # them together by `_bronze_row_pos` before writing. Reconstructing
    # bronze's exact pre-sort row order is the only way to make LUNA's
    # unstable `sort_values("cell_section")` produce identical post-sort
    # output for both bronze-direct and h5ad-derived inputs.
    per_section_dfs: List[pd.DataFrame] = []
    has_bronze_pos = True  # gets set to False if any h5ad is missing it

    for path in files:
        adata = ad.read_h5ad(path)

        # Choose the matrix that fills the "gene" columns of LUNA's
        # CSV. Default is adata.X (raw gene counts). When the user
        # asks for a pretrained encoder via ``embedding_field``, we
        # instead read from adata.obsm[field] — same shape contract
        # downstream, just (n_cells, embedding_dim) instead of
        # (n_cells, n_genes).
        if embedding_field is not None:
            if embedding_field not in adata.obsm:
                raise KeyError(
                    f"--embedding_field={embedding_field!r} requested but "
                    f"{path.name} has no adata.obsm[{embedding_field!r}]. "
                    f"Run scripts/precompute_embeddings.py first to populate."
                )
            X = np.asarray(adata.obsm[embedding_field], dtype=np.float64)
            local_gene_names = [
                f"{embedding_field}_{i}" for i in range(X.shape[1])
            ]
        else:
            X = adata.X
            if sp.issparse(X):
                X = X.toarray()
            # Preserve the h5ad's native precision (float64 after the
            # build_h5ad_from_luna_csv float64 fix). Casting to float32
            # here would introduce LSB rounding on top of LUNA's `.float()`.
            X = np.asarray(X, dtype=np.float64)
            if log2_normalize:
                X = np.log2(X + 1.0)
            local_gene_names = list(adata.var_names)

        if gene_names is None:
            gene_names = local_gene_names
        elif local_gene_names != gene_names:
            raise ValueError(
                f"{'Embedding-dim' if embedding_field else 'Gene-panel'} "
                f"mismatch in {path.name}: expected "
                f"{len(gene_names)} columns, got {len(local_gene_names)}"
            )

        # Section label: prefer obs['cell_section'] (preserved verbatim
        # by build_h5ad_from_luna_csv from the source CSV); fall back
        # to the filename stem if obs is missing the column or has
        # heterogeneous values.
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

        # ----- Within-section cell ordering -----------------------------
        # LUNA's data_module calls `sort_values("cell_section")` with the
        # default unstable `kind="quicksort"`. For equal-section rows,
        # the post-sort order depends on the pre-sort INPUT order. The
        # bronze CSV has sections INTERLEAVED across its rows; if we
        # write our fresh CSV with sections CONTIGUOUS (which is what
        # _build_luna_csv naturally produces), pandas' unstable sort
        # gives a different post-sort within-section ordering than it
        # gives for bronze. Result: 99.98% of rows mismatch.
        #
        # Fix: stamp every cell with its original bronze CSV row
        # position (`_bronze_row_pos`, written by
        # `build_h5ad_from_luna_csv.py`), then below — AFTER we've
        # collected every section — we sort ALL rows by
        # `_bronze_row_pos`. This reconstructs bronze's exact pre-sort
        # row order; LUNA's `sort_values` then produces bit-identical
        # output on both inputs.
        if "cell_id" in adata.obs.columns:
            cell_ids = adata.obs["cell_id"].to_numpy()
        else:
            cell_ids = np.arange(len(X))

        if "_bronze_row_pos" in adata.obs.columns:
            bronze_row_pos = adata.obs["_bronze_row_pos"].to_numpy()
        else:
            # Legacy silver that didn't store _bronze_row_pos. We can't
            # reconstruct bronze's order; fall back to cell_id sort
            # (a poor approximation; LUNA-on-h5ad will NOT match
            # LUNA-on-bronze in this case, only approximately).
            has_bronze_pos = False
            order = np.argsort(cell_ids, kind="stable")
            X = X[order]
            xy = xy[order]
            cell_class = cell_class[order]
            cell_ids = cell_ids[order]
            bronze_row_pos = cell_ids  # any monotone proxy; unused later

        df = pd.DataFrame(X, columns=gene_names)
        df["coord_X"] = xy[:, 0]
        df["coord_Y"] = xy[:, 1]
        df["cell_section"] = section_label
        df["cell_class"] = cell_class
        # Use the original bronze cell_id as the CSV index (not 0..N-1)
        # so LUNA's `cell_ID = torch.tensor(input_data.index)` matches
        # bronze when the cell_ids are integer-typed (MERFISH cortex).
        # For datasets where cell_ids are strings (CNS scRNA cell
        # barcodes), torch.tensor on a string index raises
        # `ValueError: too many dimensions 'str'`, taking the whole
        # data_module init down. Detect that case up front and fall
        # back to integer row positions for the index, keeping the
        # original strings as a separate metadata column so callers
        # can still round-trip them downstream if they need to.
        try:
            df.index = pd.to_numeric(cell_ids)
        except (ValueError, TypeError):
            logger.info(
                f"    {section_label}: cell_ids are non-numeric "
                f"({type(cell_ids[0]).__name__}); using integer row "
                f"position as CSV index, original ids preserved in "
                f"'cell_id_orig' column."
            )
            df["cell_id_orig"] = cell_ids.astype(str)
            df.index = np.arange(len(df), dtype=np.int64)
        df.index.name = "cell_id"
        # Carry _bronze_row_pos through; we'll drop it before writing.
        df["_bronze_row_pos"] = bronze_row_pos

        per_section_dfs.append(df)
        logger.info(f"    collected {len(df):>6,} cells from {section_label}")

    # ----- Concatenate, reorder to bronze row order, write -------------
    if not per_section_dfs:
        raise RuntimeError("no sections collected — no h5ads matched the file list")

    big_df = pd.concat(per_section_dfs, axis=0)

    if has_bronze_pos:
        # Sort by _bronze_row_pos ASC (stable) to replay bronze CSV row
        # order EXACTLY. After this, the fresh CSV is identical to
        # bronze at the row level (modulo metadata columns we don't
        # carry forward).
        big_df = big_df.sort_values("_bronze_row_pos", kind="stable")
    else:
        logger.warning(
            "  no _bronze_row_pos found in any h5ad — rebuilt this silver "
            "with the latest build_h5ad_from_luna_csv.py to enable "
            "bit-identical LUNA-on-h5ad ≡ LUNA-on-bronze. Writing fresh "
            "CSV with sections contiguous (cell_id-ascending within); "
            "LUNA's unstable sort_values will produce a DIFFERENT "
            "within-section order than for the bronze CSV."
        )

    # Drop the helper column before writing.
    big_df = big_df.drop(columns=["_bronze_row_pos"])

    # Single write with header — replaces the per-section chunked append
    # logic. `float_format="%.17g"` gives lossless float64 round-trip.
    big_df.to_csv(out_csv, header=True, float_format="%.17g")
    rows_total = len(big_df)
    logger.info(
        f"  wrote {rows_total:,} cells total to {out_csv} "
        f"({'bronze-row-pos sorted' if has_bronze_pos else 'cell_id sorted, sections contiguous (LEGACY)'})"
    )

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
    mode: str = "train_and_test",
    wandb_project: Optional[str] = None,
    run_timestamp: Optional[str] = None,
) -> int:
    """Run LUNA via our monkey-patching launcher (``_luna_runner.py``),
    in the same Python env.

    The launcher imports LUNA's modules from ``luna_repo`` and patches
    ``DataModule.__init__`` to load only the splits the requested
    ``mode`` actually needs. That's how ``mode=train_only`` skips the
    test CSV entirely (LUNA's stock ``main.py`` always loads both).

    External LUNA files stay pristine; every change lives in the
    launcher process.
    """
    luna_repo = luna_repo.resolve()
    main_py = luna_repo / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"LUNA main.py not found: {main_py}")

    runner = Path(__file__).resolve().parent / "_luna_runner.py"
    if not runner.exists():
        raise FileNotFoundError(f"_luna_runner.py launcher missing: {runner}")

    cmd = [
        sys.executable, str(runner),
        "--luna_repo", str(luna_repo),
        "--mode", mode,
    ]
    if wandb_project:
        cmd += ["--wandb_project", wandb_project]
    if run_timestamp:
        # Forward the train-script's wall-clock timestamp so wandb
        # tags / config / summary include it — that's the canonical
        # link from a wandb run to its on-disk artifacts dir
        # (which is named by the same timestamp).
        cmd += ["--run_timestamp", run_timestamp]
    for o in overrides:
        cmd += ["--override", o]
    logger.info(f"Invoking LUNA via launcher (mode={mode}):")
    for arg in cmd:
        logger.info(f"    {arg}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "wb") as f:
        proc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, check=False,
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
    plots_dir: Optional[Path] = None,
) -> List[Dict[str, float]]:
    """Compute per-section Spearman; return one row per section.

    When ``plots_dir`` is given, also writes a GT-vs-prediction side-
    by-side scatter to ``plots_dir/<label>.svg`` for each evaluated
    section. Plotting failures are logged but never crash the eval
    loop — metrics always get computed.
    """
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

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
        if plots_dir is not None:
            cell_class = (
                true["cell_class"].astype(str).to_numpy()
                if "cell_class" in true.columns else None
            )
            try:
                _plot_pred_vs_truth(
                    coords_true=coords_true,
                    coords_pred=coords_pred,
                    cell_class=cell_class,
                    out_path=plots_dir / f"{label}.svg",
                    title_prefix=f"{label}  |  ",
                    method_label=f"{ENGINE_DISPLAY} prediction",
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  plot failed for {label}: {e}")
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
    luna_repo: str = str(_ENGINE_REPO_DEFAULT),
    run_name: str = "MERFISH_mouse_cortex",
    log2_normalize: bool = False,
    wandb_mode: str = "online",
    wandb_project: str = "",  # if empty, the script's injected default applies
    extra_overrides: Optional[List[str]] = None,
    train_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    n_genes: Optional[int] = None,
    skip_training: bool = False,
    load_checkpoint: Optional[str] = None,
    make_plots: bool = False,
    output_subdir: Optional[str] = None,
    embedding_field: Optional[str] = None,
    n_inference_samples: int = 1,
    run_timestamp: Optional[str] = None,
) -> Dict[str, float]:
    """Train LUNA on Mouse 1, evaluate on Mouse 2.

    Args mirror scgg/scripts/run_scgg.py:run_benchmark
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

    # Three sources for the run TS, in order of precedence:
    #   1. ``run_timestamp`` arg — explicit override (set by
    #      run_scgg_pipeline.py when it received --run_timestamp).
    #   2. Regex-extracted from --load_checkpoint path (test-only mode
    #      against a prior model; pairs inference dir with model dir).
    #   3. Current wall clock — fresh train run with no external pin.
    # Without source (1) the wandb-side timestamp (which goes into
    # config / summary / tags via _luna_runner) silently disagreed
    # with the on-disk artifacts dir TS when the pipeline pinned a
    # different one. Mirror the LUNA-train fix from task #72.
    if run_timestamp is not None:
        # Accept YYYYMMDD_HHMMSS optionally followed by a
        # _xxx uniquifier suffix (added by submit_pipeline.sh to
        # prevent sub-second collisions when a for-loop fires many
        # jobs in the same wall-clock second). The suffix is part of
        # the directory name and must be preserved end-to-end.
        if not re.fullmatch(r"\d{8}_\d{6}(?:_[A-Za-z0-9]+)?", run_timestamp):
            raise ValueError(
                f"run_timestamp must match YYYYMMDD_HHMMSS or "
                f"YYYYMMDD_HHMMSS_<suffix>; got {run_timestamp!r}."
            )
        run_ts = run_timestamp
    elif skip_training and load_checkpoint is not None:
        # Same shape as above for the checkpoint-path extraction.
        m = re.search(r"(\d{8}_\d{6}(?:_[A-Za-z0-9]+)?)", str(load_checkpoint))
        run_ts = m.group(1) if m else datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        if use_prebuilt:
            # Derive a sensible default from the train CSV's parent dir name.
            base = Path(train_csv).resolve().parent.name or "luna_paper_csvs"
        else:
            base = Path(data_dir).name
        # Inference wrappers pass output_subdir="..._inference"
        # so re-using the same training pipeline writes to a
        # different subtree. Defaults to ENGINE_OUTPUT_SUBDIR
        # for training runs.
        subdir = output_subdir or ENGINE_OUTPUT_SUBDIR
        out = _ARTIFACTS_ROOT / base / subdir / run_ts
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

    # Phase timing — flushed to <out>/runtime.csv at the end of each
    # phase so a crash mid-run still leaves a useful CSV behind.
    tracker = _RuntimeTracker()
    runtime_csv = out / "runtime.csv"

    tracker.start("csv_build")
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
        # Suffix-based split: any *_train.h5ad in the silver dir is a
        # training file, *_test.h5ad is held out for inference. Works
        # for any dataset that was produced by build_h5ad_from_luna_csv.
        data_path = Path(data_dir)
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

        train_csv_path = work / "train.csv"
        test_csv_path = work / "test.csv"
        if train_csv_path.exists() and test_csv_path.exists():
            logger.info("LUNA CSVs already exist under work/; reusing")
            head = pd.read_csv(train_csv_path, nrows=1, index_col=0)
            n_genes = len(head.columns) - 4
        else:
            if embedding_field is not None:
                logger.info(
                    f"Using pretrained embeddings from "
                    f"adata.obsm[{embedding_field!r}] in place of raw genes"
                )
            logger.info(f"Writing train CSV -> {train_csv_path}")
            train_stats = _build_luna_csv(
                train_files, train_csv_path, log2_normalize=log2_normalize,
                embedding_field=embedding_field,
            )
            logger.info(
                f"  train: {train_stats['n_rows']:,} rows, "
                f"{train_stats['n_genes']} "
                f"{'embedding-dims' if embedding_field else 'genes'}, "
                f"{train_stats['n_sections']} sections"
            )
            logger.info(f"Writing test CSV  -> {test_csv_path}")
            test_stats = _build_luna_csv(
                test_files, test_csv_path, log2_normalize=log2_normalize,
                embedding_field=embedding_field,
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
    tracker.end("csv_build", flush_to=runtime_csv)

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

    # Decoupled by default: train-only mode in train script, test-only
    # in the inference wrapper. The launcher
    # (``scripts/_luna_runner.py``) patches LUNA's DataModule so the
    # unused split's CSV is never loaded — critical for datasets where
    # one of the CSVs would crash the data_module (e.g. CNS scRNA
    # cells with string IDs).
    mode = "test_only" if skip_training else "train_only"
    overrides = [
        f"general.name={run_name}",
        f"general.seed={seed}",
        f"general.wandb={wandb_mode}",
        # Hydra '+' prefix: append-or-update. ``wandb_project`` is a
        # runner-side patch (see scripts/_luna_runner.py docstring),
        # not declared in the upstream LUNA ``general`` config schema.
        # The scgg-vendored config happens to declare it today, but
        # using ``+`` is strictly safer (works regardless) and
        # matches the LUNA-side runner's fix.
        f"+general.wandb_project={wandb_project or 'scgg'}",
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
    if n_inference_samples != 1:
        # Pipe the CLI value into Hydra so cfg.test.n_inference_samples
        # is set correctly inside _luna_runner. The actual ensembling
        # happens in utils/diffusion_model/test/test.py.
        overrides.append(f"test.n_inference_samples={int(n_inference_samples)}")
    if skip_training:
        if load_checkpoint is None:
            raise ValueError(
                "skip_training=True requires load_checkpoint to point at a "
                "LUNA-format .ckpt path."
            )
        ckpt_path = Path(load_checkpoint).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--load_checkpoint not found: {ckpt_path}")
        overrides.append(f"test.checkpoints_parent_dir={_h(ckpt_path.parent)}")
        overrides.append(f"test.checkpoints_name_list=[{_h(ckpt_path.name)}]")
    if extra_overrides:
        overrides.extend(extra_overrides)

    log_path = out / "luna_stdout.log"
    tracker.start("training")
    try:
        rc = _invoke_luna(
            luna_repo_p, overrides, log_path,
            mode=mode,
            wandb_project=(wandb_project or None),
            run_timestamp=run_ts,
        )
    finally:
        tracker.end("training", flush_to=runtime_csv)
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
    # Train mode (mode == "train_only") writes a checkpoint but no
    # predictions, so there's nothing to evaluate. Inference mode
    # (mode == "test_only") and the legacy combined mode both produce
    # `metadata_pred.csv` files we can score against ground truth.
    per_slice: List[Dict[str, float]] = []
    if mode != "train_only":
        tracker.start("evaluation")
        sections = _read_luna_predictions(test_save_dir)
        plots_dir = (out / "plots") if make_plots else None
        per_slice = _evaluate_predictions(sections, plots_dir=plots_dir)
        tracker.end("evaluation", flush_to=runtime_csv)
    else:
        logger.info(
            "mode=train_only: skipping evaluation phase (no predictions "
            "were written). Run `run_luna_inference.py --checkpoint ...` "
            "against the saved checkpoint to score on the test split."
        )

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
    tracker.start("write_artifacts")
    if per_slice:
        fieldnames = sorted({k for r in per_slice for k in r.keys()})
        # per_slice_metrics.csv : one row per test slice (raw values).
        with open(out / "per_slice_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in per_slice:
                w.writerow(r)

    # Aggregate metrics across slices: mean / median / std / min /
    # max for every numeric per-slice column, plus the headline
    # ``spearman_mean_of_medians`` and ``n_test_slices``. Lives
    # both as a single-row metrics.csv (easy to glob and pandas
    # together with training-side metrics.csv from earlier in the
    # pipeline) and as aggregate_metrics.json (already used by
    # downstream notebooks).
    agg = {
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

    # Resource cost: peak GPU memory + peak process-tree RSS across
    # all phases. Surfaced here so the comparison-friendly numbers
    # in aggregate_metrics.json / metrics.csv include resource usage
    # alongside accuracy. See the matching block in run_luna_train.py.
    peak = tracker.peak_summary()
    if peak["peak_gpu_mib"] is not None:
        agg["peak_gpu_mib"] = peak["peak_gpu_mib"]
    if peak["peak_rss_mib"] is not None:
        agg["peak_rss_mib"] = peak["peak_rss_mib"]

    # metrics.csv : single-row aggregate. Same data as
    # aggregate_metrics.json, just in tabular form so you can
    # `pd.concat([...])` across many runs without parsing JSON.
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg.keys()))
        w.writeheader()
        w.writerow(agg)
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

    # compute_requirements.csv — single-row, paper-shape resource
    # summary. See the matching block in run_luna_train.py for the
    # rationale.
    tracker.write_compute_requirements_csv(
        out / "compute_requirements.csv",
        method="scgg",
        run_timestamp=run_ts,
        extra={
            "epochs": epochs,
            "batch_size": batch_size,
            "n_genes": n_genes,
        },
    )
    tracker.end("write_artifacts", flush_to=runtime_csv)

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
        "--wandb_project", default=None,
        help="wandb project name. Default is set per-engine "
             "(scgg = \"scgg\"; luna = \"luna\").",
    )
    p.add_argument(
        "--wandb_mode", default="online",
        choices=("disabled", "online", "offline", "dryrun"),
        help="general.wandb override. Default 'disabled' to avoid LUNA "
             "crashing when the host isn't logged into WandB. Pass "
             "'online' to match LUNA's upstream default.",
    )
    p.add_argument(
        "--luna_repo", default=str(_ENGINE_REPO_DEFAULT),
        help=f"Path to the LUNA repository. Default: {_ENGINE_REPO_DEFAULT}",
    )
    # Two argparse names for the same Hydra `general.name` value:
    # `--wandb_run_name` to match run_scgg.py's CLI surface (so users
    # can flip between the two scripts without rewriting their command
    # lines), and `--run_name` kept as an alias because earlier
    # invocations and docs reference it.
    p.add_argument(
        "--wandb_run_name", "--run_name",
        dest="wandb_run_name",
        default="MERFISH_mouse_cortex",
        help="Sets general.name in LUNA's Hydra config (drives the "
             "wandb run name and LUNA's output dir basename). "
             "Aliases: --run_name.",
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
        "--override", "--luna_override",
        dest="override",
        action="extend", nargs="+", default=[],
        help="Extra Hydra overrides. Accepts ONE OR MORE key=value "
             "tokens per --override (space-separated), and the flag "
             "itself is repeatable. So all three of these work:\n"
             "  --override train.lr=1e-4\n"
             "  --override train.lr=1e-4 model.framework=flow_matching\n"
             "  --override train.lr=1e-4 --override model.framework=flow_matching\n"
             "'--luna_override' is kept as a backward-compat alias.",
    )
    p.add_argument(
        "--skip_training", action="store_true",
        help="Run LUNA in test-only mode against a previously trained "
             "checkpoint (requires --load_checkpoint). Used internally "
             "by inference_luna.py.",
    )
    p.add_argument(
        "--load_checkpoint", default=None,
        help="Path to a LUNA .ckpt — only used with --skip_training.",
    )
    p.add_argument(
        "--plots", action="store_true",
        help="Write per-section ground-truth-vs-prediction comparison "
             "plots (svg) into <out_dir>/plots/. OFF by default during "
             "training; ON by default in inference_luna.py.",
    )
    p.add_argument(
        "--embedding_field", default=None,
        help="adata.obsm key containing PRECOMPUTED per-cell embeddings "
             "to use in place of raw gene counts (e.g. 'pca_64', "
             "'ae_128', 'scgpt'). Run scripts/precompute_embeddings.py "
             "first to populate this obsm field on every silver h5ad. "
             "When set, the LUNA CSV's 'gene' columns are filled from "
             "obsm[<field>] and the model trains on embeddings rather "
             "than raw genes — the gene encoder becomes an adapter "
             "over a pretrained representation. Default: None "
             "(use raw genes, byte-equivalent to prior runs).",
    )
    p.add_argument(
        "--n_inference_samples", type=int, default=1,
        help="At inference, draw this many samples per slice and "
             "report the per-cell mean as the final prediction. "
             "Per-cell std is also saved to metadata_pred_std.csv "
             "for uncertainty quantification. Default 1 (no "
             "ensembling). Typical values: 5-10. Linearly scales "
             "inference wall-clock.",
    )
    p.add_argument(
        "--run_timestamp", default=None,
        help="Optional YYYYMMDD_HHMMSS run timestamp. When set, the "
             "script does NOT generate a fresh wall-clock TS; it uses "
             "this one for the artifacts subdir AND threads it through "
             "to _luna_runner so wandb config/summary/tags carry the "
             "same TS. Forwarded by run_scgg_pipeline.py so the LSF "
             "submitter can pin one timestamp across LSF logs + "
             "training dir + inference dir + wandb run.",
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
            run_name=args.wandb_run_name,
            log2_normalize=args.log2_normalize,
            wandb_mode=args.wandb_mode,
            # 'scgg' is the fixed default; --wandb_project overrides ad-hoc.
            wandb_project=args.wandb_project or "scgg",
            extra_overrides=args.override,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            n_genes=args.n_genes,
            skip_training=args.skip_training,
            load_checkpoint=args.load_checkpoint,
            make_plots=args.plots,
            embedding_field=args.embedding_field,
            n_inference_samples=args.n_inference_samples,
            run_timestamp=args.run_timestamp,
        )
    except Exception:
        logger.exception("scgg training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

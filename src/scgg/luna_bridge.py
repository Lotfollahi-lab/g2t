"""
Bridge utilities for invoking LUNA from our scgg-side code.

LUNA (Liu et al., bioRxiv 2025) runs in its own Python 3.9 / torch-2.0.1
venv that is incompatible with the scgg env. We talk to it via subprocess
and exchange data through CSVs in the format LUNA expects: gene columns
first, then ``coord_X``, ``coord_Y``, ``cell_section``, ``cell_class``.

This module factors the conversion + invocation + checkpoint-discovery
helpers shared by:
  * scgg/scripts/run_luna_on_mmc.py            (train LUNA on the cortex)
  * scgg/scripts/infer_luna_on_mmc.py          (infer with a trained LUNA)
  * scgg-reproducibility/.../run_luna_cortex_benchmark.py  (legacy combined script)

The cortex slice-filename pattern accepts both the current ``mmc_`` prefix
and the legacy ``merfish_mouse_cortex_`` prefix so a directory mid-rename
keeps working.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default artifact root (matches scgg's run_luna_cortex_benchmark.py).
_ARTIFACTS_ROOT = Path("/nfs/team361/sb75/scgg-reproducibility/artifacts")


# ---------------------------------------------------------------------------
# Silver-h5ad discovery (cortex layout)
# ---------------------------------------------------------------------------


_SLICE_RE = re.compile(
    r"^(?:mmc|merfish_mouse_cortex)_mouse(?P<mouse>\d+)_slice(?P<slice>\d+)\.h5ad$"
)


def enumerate_slice_files(silver_dir: Path) -> List[Tuple[int, int, Path]]:
    """Discover (mouse, slice, path) tuples under a cortex silver dir.

    Accepts both ``mmc_mouseM_sliceS.h5ad`` and the legacy
    ``merfish_mouse_cortex_mouseM_sliceS.h5ad`` naming.
    """
    out: List[Tuple[int, int, Path]] = []
    for p in sorted(silver_dir.iterdir()):
        m = _SLICE_RE.match(p.name)
        if not m:
            continue
        out.append((int(m["mouse"]), int(m["slice"]), p))
    return out


def split_by_mouse(
    files: List[Tuple[int, int, Path]], mouse_id: int,
) -> List[Tuple[int, int, Path]]:
    """Return only the files matching the given mouse id."""
    return [f for f in files if f[0] == mouse_id]


# ---------------------------------------------------------------------------
# CSV building (LUNA input format)
# ---------------------------------------------------------------------------


def build_luna_csv(
    files: List[Tuple[int, int, Path]],
    out_csv: Path,
    log2_normalize: bool = True,
) -> Dict[str, object]:
    """Concatenate per-slice h5ads into a single LUNA-format CSV.

    LUNA expects:
      * gene columns first (positions ``0..n_genes-1``)
      * then ``coord_X``, ``coord_Y``, ``cell_section``, ``cell_class``
      * index = original cell barcode

    Args:
        files: list of (mouse_id, slice_id, h5ad_path) tuples.
        out_csv: destination CSV path.
        log2_normalize: apply ``log2(x + 1)`` to expression. LUNA expects
            log-space input; only flip this off if your silver h5ads are
            already log-normalized.

    Returns a dict with `n_rows`, `n_genes`, `n_sections`, `gene_names`.
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
        else:
            if list(adata.var_names) != gene_names:
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
        "gene_names": gene_names or [],
    }


# ---------------------------------------------------------------------------
# LUNA invocation (subprocess into LUNA venv)
# ---------------------------------------------------------------------------


def invoke_luna(
    luna_venv: Path,
    luna_repo: Path,
    overrides: List[str],
    cwd: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> int:
    """Run LUNA's main.py via the LUNA venv's Python with Hydra overrides.

    The caller is responsible for setting ``hydra.run.dir`` and any
    ``general.mode`` / ``dataset.*`` / ``test.*`` overrides via the
    ``overrides`` list. Returns the subprocess exit code.

    Raises FileNotFoundError if the venv python or main.py is missing.
    """
    luna_python = luna_venv / "bin" / "python"
    main_py = luna_repo / "main.py"
    if not luna_python.exists():
        raise FileNotFoundError(f"LUNA venv python not found: {luna_python}")
    if not main_py.exists():
        raise FileNotFoundError(f"LUNA main.py not found: {main_py}")

    cmd = [str(luna_python), str(main_py), *overrides]
    logger.info("Invoking LUNA:")
    for arg in cmd:
        logger.info(f"    {arg}")

    t0 = time.time()
    cwd = cwd or luna_repo
    if log_path is None:
        proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "wb") as f:
            proc = subprocess.run(
                cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, check=False,
            )
    elapsed = (time.time() - t0) / 60.0
    logger.info(
        f"LUNA exited with code {proc.returncode} after {elapsed:.1f} min"
        + (f" (log: {log_path})" if log_path else "")
    )
    if proc.returncode != 0 and log_path is not None and log_path.exists():
        with open(log_path) as f:
            tail = f.read().splitlines()[-50:]
        for line in tail:
            logger.error(f"  | {line}")
    return proc.returncode


# ---------------------------------------------------------------------------
# Checkpoint discovery (find the best .ckpt in a LUNA run dir)
# ---------------------------------------------------------------------------


_EPOCH_RE = re.compile(r"epoch=(\d+)")


def find_latest_checkpoint(luna_run_dir: Path) -> Optional[Path]:
    """Return the latest-epoch checkpoint under a LUNA Hydra run dir.

    LUNA saves checkpoints as ``{run_dir}/checkpoints/epoch=<N>.ckpt``.
    If multiple are present (LUNA defaults to `save_top_k_models=40`),
    we return the one with the largest epoch number — typically the
    final one and the one the LUNA notebooks use for inference.
    """
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


def find_run_dir_from_checkpoint(ckpt_path: Path) -> Path:
    """Given .../<run_dir>/checkpoints/epoch=N.ckpt, return <run_dir>."""
    return ckpt_path.resolve().parent.parent


# ---------------------------------------------------------------------------
# Read LUNA's per-section prediction outputs
# ---------------------------------------------------------------------------


def read_luna_predictions(
    test_save_dir: Path,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Walk LUNA's test outputs and return {section_label: {pred, true}}.

    LUNA writes one subdirectory per section, each containing
    ``metadata_pred.csv`` and ``metadata_true.csv``. We search
    recursively because LUNA sometimes nests an extra level by
    checkpoint name.
    """
    pred_files = list(test_save_dir.rglob("metadata_pred.csv"))
    if not pred_files:
        raise FileNotFoundError(
            f"No metadata_pred.csv files under {test_save_dir}. "
            "Did LUNA's test phase finish successfully?"
        )

    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for pred_path in sorted(pred_files):
        true_path = pred_path.parent / "metadata_true.csv"
        if not true_path.exists():
            logger.warning(
                f"  {pred_path.parent.name}: missing metadata_true.csv; skipping"
            )
            continue
        section_label = pred_path.parent.name
        pred = pd.read_csv(pred_path, index_col=0)
        true = pd.read_csv(true_path, index_col=0)
        if not pred.index.equals(true.index):
            common = pred.index.intersection(true.index)
            pred = pred.loc[common]
            true = true.loc[common]
        out[section_label] = {"pred": pred, "true": true}
    return out


# ---------------------------------------------------------------------------
# Timestamps (mirror the layout scgg uses)
# ---------------------------------------------------------------------------


def fresh_run_timestamp() -> str:
    """Return a ``YYYYMMDD_HHMMSS`` string for naming a new run dir."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


_RUN_TS_RE = re.compile(r"^\d{8}_\d{6}$")


def timestamp_from_path(p: Path) -> Optional[str]:
    """Walk ``p``'s parents and return the first YYYYMMDD_HHMMSS segment."""
    for parent in p.resolve().parents:
        if _RUN_TS_RE.match(parent.name):
            return parent.name
    return None

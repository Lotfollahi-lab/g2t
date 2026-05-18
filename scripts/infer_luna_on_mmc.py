#!/usr/bin/env python
"""
Run LUNA inference on held-out silver sections using a LUNA checkpoint.

Mirror of ``scgg/scripts/infer_scgg_on_cns.py`` (which runs scGG
inference), but the model is LUNA itself — invoked as a subprocess into
the LUNA Python 3.9 / torch-2.0.1 venv. Outputs use the same comparison-
plot / per-class-Spearman infrastructure as the scgg inference script so
LUNA and scGG predictions are directly visually comparable.

Pipeline
--------
  1. Resolve ``--sections`` to a list of silver h5ad paths (same logic as
     the scgg inference script: ``paper`` => mouse2_slice99, ``all_test``
     => every mouse2_*, bare IDs, globs, etc.).
  2. Build a combined test CSV in LUNA's expected format for ONLY those
     sections. Re-build the original training CSV (Mouse 1) as well so
     LUNA's dataset bootstrap has both paths.
  3. Invoke LUNA's ``main.py`` with ``general.mode=test`` and
     ``test.checkpoint_path=<--checkpoint>``.
  4. Read LUNA's per-section ``metadata_pred.csv``, write the predictions
     back into a copy of the original h5ad as ``obsm['spatial_pred']``.
  5. Produce the comparison plot + optional per-class Spearman diagnostic
     using the SAME helpers as ``infer_scgg_on_cns.py`` (palette, spot
     sizing, Procrustes alignment for the plot).

Output layout
-------------
Default ``--output_dir`` resolves to
``/nfs/team361/sb75/scgg-reproducibility/artifacts/<silver_dir.name>/luna_inference/<model_timestamp>/``
where ``<model_timestamp>`` is extracted from the checkpoint path
(the ``YYYYMMDD_HHMMSS`` segment under ``luna_model/`` written by
``run_luna_on_mmc.py``). Falls back to a fresh timestamp if the
checkpoint isn't under that canonical layout.

Usage
-----
    python scripts/infer_luna_on_mmc.py \\
        --checkpoint /nfs/.../luna_model/<TS>/best_model.ckpt \\
        --silver_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --sections paper \\
        --include_train_section \\
        --per_class_spearman
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from scgg.luna_bridge import (
    _ARTIFACTS_ROOT,
    build_luna_csv,
    enumerate_slice_files,
    find_run_dir_from_checkpoint,
    fresh_run_timestamp,
    invoke_luna,
    read_luna_predictions,
    split_by_mouse,
    timestamp_from_path,
)

logger = logging.getLogger("luna_infer")


# ---------------------------------------------------------------------------
# Reuse the scgg inference script's plotting + diagnostic helpers so the
# LUNA outputs are visually directly comparable. We load the script as a
# module rather than sys.path-injecting to avoid polluting the namespace.
# ---------------------------------------------------------------------------


def _load_scgg_infer_module():
    here = Path(__file__).resolve().parent
    target = here / "infer_scgg_on_cns.py"
    if not target.exists():
        raise FileNotFoundError(
            f"Cannot locate sibling script {target} — required for plotting."
        )
    spec = importlib.util.spec_from_file_location("_scgg_infer", target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_scgg_infer = _load_scgg_infer_module()
_plot_comparison = _scgg_infer._plot_comparison
_per_class_spearman = _scgg_infer._per_class_spearman
_log_per_class_spearman = _scgg_infer._log_per_class_spearman
_strip_known_prefix = _scgg_infer._strip_known_prefix
_resolve_section_files = _scgg_infer._resolve_section_files
_pick_representative_cortex_slice = _scgg_infer._pick_representative_cortex_slice


# ---------------------------------------------------------------------------
# Build CSVs for LUNA test mode
# ---------------------------------------------------------------------------


def _section_files_from_paths(
    paths: List[Path],
) -> List[tuple[int, int, Path]]:
    """Parse (mouse, slice, path) tuples from a list of silver h5ads.

    Uses the same regex as luna_bridge.enumerate_slice_files so both
    naming prefixes are accepted.
    """
    from scgg.luna_bridge import _SLICE_RE
    out: List[tuple[int, int, Path]] = []
    for p in paths:
        m = _SLICE_RE.match(p.name)
        if not m:
            raise ValueError(
                f"Section file does not match LUNA cortex naming pattern: "
                f"{p.name}. Expected mmc_mouse{{M}}_slice{{S}}.h5ad or "
                f"merfish_mouse_cortex_mouse{{M}}_slice{{S}}.h5ad."
            )
        out.append((int(m["mouse"]), int(m["slice"]), p))
    return out


# ---------------------------------------------------------------------------
# Write LUNA predictions back into a per-section h5ad
# ---------------------------------------------------------------------------


def _write_pred_into_h5ad(
    silver_h5ad: Path,
    pred_df: pd.DataFrame,
    out_h5ad: Path,
) -> "anndata.AnnData":  # noqa: F821
    """Load the original silver h5ad, attach ``obsm['spatial_pred']``,
    save to ``out_h5ad``, and return the in-memory AnnData."""
    import anndata as ad

    adata = ad.read_h5ad(silver_h5ad)

    # Align prediction rows to the h5ad's cell barcodes if possible.
    if not pred_df.index.empty and adata.obs_names.isin(pred_df.index).any():
        common = adata.obs_names.intersection(pred_df.index)
        if len(common) < adata.n_obs:
            logger.warning(
                f"    only {len(common)} / {adata.n_obs} cells matched by "
                "barcode; the rest will get NaN predictions."
            )
        pred_aligned = pred_df.reindex(adata.obs_names)
        coords = pred_aligned[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
    else:
        # Fall back to row-order match (length must agree).
        if len(pred_df) != adata.n_obs:
            raise ValueError(
                f"Prediction row count {len(pred_df)} != n_obs "
                f"{adata.n_obs} for {silver_h5ad.name}, and no barcode "
                "overlap to fall back on."
            )
        coords = pred_df[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)

    adata.obsm["spatial_pred"] = coords
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write(out_h5ad)
    return adata


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_luna_inference(
    checkpoint: str,
    silver_dir: str,
    sections: List[str],
    output_dir: Optional[str] = None,
    color_col: str = "cell_class",
    device: Optional[str] = None,
    luna_venv: str = "/nfs/team361/sb75/.venvs/luna",
    luna_repo: str = "/nfs/team361/sb75/code/LUNA",
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
    """Run LUNA inference on selected sections.

    Signature mirrors ``scgg.scripts.infer_scgg_on_cns.run_inference``
    where applicable. LUNA-specific extras (``luna_venv``, ``luna_repo``,
    ``run_name``, ``log2_normalize``, ``extra_overrides``) replace the
    gene-namespace / mygene knobs that don't apply to the same-panel
    cortex case.
    """
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    silver_path = Path(silver_dir)
    # Default output_dir: pin to the model's training timestamp.
    if output_dir is None:
        model_ts = timestamp_from_path(ckpt_path)
        if model_ts is None:
            model_ts = fresh_run_timestamp()
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

    luna_venv_p = Path(luna_venv)
    luna_repo_p = Path(luna_repo)

    # ---- 1. Resolve sections --------------------------------------------
    section_paths = _resolve_section_files(silver_path, sections)

    if include_train_section:
        train_pick = _pick_representative_cortex_slice(silver_path, mouse_id=1)
        if train_pick is None:
            logger.warning(
                "  --include_train_section requested but no Mouse-1 silver "
                f"h5ad found under {silver_path}; skipping."
            )
        elif train_pick in section_paths:
            logger.info(
                f"  --include_train_section: {train_pick.name} already "
                "in the requested set; not duplicating."
            )
        else:
            section_paths = [train_pick] + section_paths
            logger.info(
                f"  --include_train_section: prepended {train_pick.name} "
                "as a train-side sanity check."
            )

    logger.info(
        f"Inference on {len(section_paths)} sections: "
        f"{[p.name for p in section_paths]}"
    )

    # ---- 2. Build LUNA test CSV (and reproduce the train CSV) ----------
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    test_csv = work / "test.csv"
    train_csv = work / "train.csv"

    # Test CSV from the requested sections.
    test_files = _section_files_from_paths(section_paths)
    logger.info(f"Writing test CSV  -> {test_csv}")
    test_stats = build_luna_csv(test_files, test_csv, log2_normalize=log2_normalize)
    logger.info(
        f"  test : {test_stats['n_rows']:,} rows, "
        f"{test_stats['n_genes']} genes, "
        f"{test_stats['n_sections']} sections"
    )
    n_genes = int(test_stats["n_genes"])

    # Train CSV: LUNA's dataset loader expects both paths even in test
    # mode. Reuse the training-run's train.csv if it's still around;
    # otherwise rebuild from the silver dir's Mouse 1 slices.
    candidate_train_csvs = [
        ckpt_path.resolve().parent / "work" / "train.csv",
        ckpt_path.resolve().parent.parent / "work" / "train.csv",
        find_run_dir_from_checkpoint(ckpt_path).parent / "work" / "train.csv",
    ]
    reused = False
    for c in candidate_train_csvs:
        if c.exists():
            # Symlink so LUNA reads the canonical file rather than a copy.
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
        all_files = enumerate_slice_files(silver_path)
        train_files_silver = split_by_mouse(all_files, mouse_id=1)
        if not train_files_silver:
            raise FileNotFoundError(
                f"No existing train CSV found near {ckpt_path}, and no "
                f"Mouse-1 silver h5ads under {silver_path} to rebuild it."
            )
        logger.info(
            f"Re-building train CSV from {len(train_files_silver)} "
            f"Mouse-1 silver h5ads -> {train_csv}"
        )
        build_luna_csv(train_files_silver, train_csv, log2_normalize=log2_normalize)

    # ---- 3. Invoke LUNA in test mode ------------------------------------
    luna_run_dir = out_dir / "luna_run"
    luna_run_dir.mkdir(parents=True, exist_ok=True)
    test_save_dir = luna_run_dir / "test_results"
    test_save_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"general.name={run_name}",
        "general.mode=test",
        f"dataset.train_data_path={train_csv.resolve()}",
        f"dataset.test_data_path={test_csv.resolve()}",
        "dataset.gene_columns_start=0",
        f"dataset.gene_columns_end={n_genes}",
        f"test.checkpoint_path={ckpt_path.resolve()}",
        f"test.save_dir={test_save_dir.resolve()}",
        f"hydra.run.dir={luna_run_dir.resolve()}",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)

    log_path = out_dir / "luna_stdout.log"
    rc = invoke_luna(
        luna_venv=luna_venv_p,
        luna_repo=luna_repo_p,
        overrides=overrides,
        cwd=luna_repo_p,
        log_path=log_path,
    )
    if rc != 0:
        raise RuntimeError(f"LUNA inference failed (exit {rc}). See {log_path}")

    # ---- 4. Read predictions + plot per section -------------------------
    pred_sections = read_luna_predictions(test_save_dir)

    # LUNA writes sections under their `cell_section` label
    # (e.g., "mouse2_slice99"). Build a label -> path map for the silver
    # files so we can pair them up.
    label_to_silver: Dict[str, Path] = {
        f"mouse{m}_slice{s}": p for (m, s, p) in test_files
    }

    summary: List[Dict[str, object]] = []
    for section_label, dfs in pred_sections.items():
        silver_path_for_label = label_to_silver.get(section_label)
        if silver_path_for_label is None:
            # LUNA may strip / re-prefix labels; try a fuzzy match.
            matches = [
                p for k, p in label_to_silver.items() if k in section_label
            ]
            if not matches:
                logger.warning(
                    f"  no silver h5ad matches LUNA section label "
                    f"{section_label!r}; skipping plot."
                )
                continue
            silver_path_for_label = matches[0]
        section_id = _strip_known_prefix(silver_path_for_label.stem)

        logger.info(f"[{section_id}] LUNA -> {silver_path_for_label}")
        out_h5ad = out_dir / f"{section_id}_predicted.h5ad"
        adata = _write_pred_into_h5ad(
            silver_path_for_label, dfs["pred"], out_h5ad,
        )

        coords_true_2d = (
            np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]
            if "spatial" in adata.obsm
            and adata.obsm["spatial"].shape[1] >= 2
            else None
        )

        # Color column resolution mirrors infer_scgg_on_cns.py
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

        # Per-class Spearman diagnostic (uses LUNA's 2-D coords directly).
        diag_class_col = diagnostic_class_col or color_col_eff
        cell_class_arr = (
            adata.obs[diag_class_col].astype(str).to_numpy()
            if diag_class_col and diag_class_col in adata.obs.columns
            else None
        )
        spearman_diag = None
        if per_class_spearman and coords_true_2d is not None:
            spearman_diag = _per_class_spearman(
                coords_true_2d,
                adata.obsm["spatial_pred"],
                cell_class_arr,
            )
            _log_per_class_spearman(spearman_diag, section_id)

        # Comparison plot — identical visual style to scgg inference.
        if color_col_eff is not None:
            out_svg = out_dir / f"{section_id}_comparison.svg"
            _plot_comparison(
                adata, color_col_eff, out_svg,
                title_prefix=f"[{section_id}] LUNA: ",
                spot_size=spot_size,
                align_for_plot=not no_align_plot,
                palette=palette,
            )

        rec: Dict[str, object] = {
            "section_id": section_id,
            "input_path": str(silver_path_for_label),
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
        help="Per-slice silver h5ad directory (LUNA cortex split).",
    )
    p.add_argument(
        "--sections", nargs="+", default=None,
        help=(
            "Section ids / filenames / glob patterns to run on. Accepts: "
            "bare ids ('mouse2_slice1'), filenames "
            "('mmc_mouse2_slice1.h5ad'), globs ('mouse2_*'), or the "
            "special values: 'all' (every silver h5ad), 'all_test' "
            "(mouse2_*), 'paper' (mouse2_slice99 — LUNA Fig 3c). Default "
            "is 'all_test' for the cortex layout."
        ),
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Where to write predicted h5ads and comparison SVGs. Default "
             "derives from --silver_dir's basename + the model's run "
             "timestamp: /nfs/team361/sb75/scgg-reproducibility/artifacts/"
             "<silver_dir_name>/luna_inference/<model_timestamp>/.",
    )
    p.add_argument("--color", default="cell_class",
                   help="adata.obs column to color plots by (default: cell_class).")
    p.add_argument("--device", default=None,
                   help="cuda|cpu (auto if omitted). Passed to LUNA via Hydra.")
    p.add_argument(
        "--luna_venv", default="/nfs/team361/sb75/.venvs/luna",
        help="Path to the uv venv created by setup_luna_env.sh.",
    )
    p.add_argument(
        "--luna_repo", default="/nfs/team361/sb75/code/LUNA",
        help="Path to the cloned LUNA repository.",
    )
    p.add_argument(
        "--run_name", default="MERFISH_mouse_cortex_infer",
        help="Sets general.name in LUNA's Hydra config.",
    )
    p.add_argument(
        "--no_log2_normalize", action="store_true",
        help="Skip log2(x+1) normalization when writing the test CSV.",
    )
    p.add_argument(
        "--spot_size", type=float, default=None,
        help="Marker size in the comparison plot (default auto-scales).",
    )
    p.add_argument(
        "--no_align_plot", action="store_true",
        help="Don't Procrustes-align predicted coords to GT before plotting. "
             "LUNA outputs coords in the GT frame already, so alignment is "
             "usually a no-op, but it's kept on by default for symmetry "
             "with the scgg inference script.",
    )
    p.add_argument(
        "--per_class_spearman", action="store_true",
        help="Compute median per-cell Spearman restricted to within-class "
             "pairs (plus global). Same diagnostic as the scgg inference "
             "script — useful for directly comparing LUNA vs scGG.",
    )
    p.add_argument(
        "--diagnostic_class_col", default=None,
        help="Override which obs column drives per-class Spearman (default: "
             "--color value).",
    )
    p.add_argument(
        "--palette", default="luna", choices=("luna", "tab20"),
        help="Color palette for the categorical cell-class plots.",
    )
    p.add_argument(
        "--include_train_section", action="store_true",
        help="Also run inference on one representative Mouse-1 slice "
             "alongside the requested sections. Same sanity check as the "
             "scgg inference script.",
    )
    p.add_argument(
        "--luna_override", action="append", default=[],
        help="Extra Hydra overrides to pass to LUNA. Repeatable.",
    )
    args = p.parse_args()

    # Default sections: 'all_test' for the cortex layout.
    sections = args.sections or ["all_test"]

    try:
        run_luna_inference(
            checkpoint=args.checkpoint,
            silver_dir=args.silver_dir,
            sections=sections,
            output_dir=args.output_dir,
            color_col=args.color,
            device=args.device,
            luna_venv=args.luna_venv,
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

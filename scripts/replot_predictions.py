"""Regenerate per-slice ``<slice>.svg`` ground-truth-vs-prediction plots
for one or more inference output directories — WITHOUT re-running
inference.

Designed for the case where ``metadata_pred.csv`` /
``metadata_true.csv`` already exist on disk (every run_*_inference.py
writes them eagerly per slice) but the plots themselves came out with
the wrong palette / colors / spot size / etc. Typical trigger: the
celery env shipped without ``colorcet``, so its plots fell back to
``tab20`` instead of LUNA's ``glasbey``. After
``pip install colorcet`` into the celery venv, this script
re-renders the SVGs in-place.

Works uniformly for luna / scgg / celery / novosparc, because all four
pipelines lay out their per-slice CSVs the same way::

    <RUN_OUTPUT_DIR>/
        plots/
            <slice_label>.svg            <-- (re-)written here
        luna_run/test_results/<wandb_run_name>/<method>/
            <slice_label>/
                metadata_pred.csv         <-- read
                metadata_true.csv         <-- read

This script just globs ``**/metadata_pred.csv`` underneath each given
output dir, pairs each with its sibling ``metadata_true.csv``, and
calls ``run_luna_train._plot_pred_vs_truth`` — the SAME function the
pipelines themselves use. So the resulting SVGs are byte-equivalent to
what re-running inference would produce, minus the actual GPU work.

Example::

    source /nfs/team361/sb75/.venvs/celery/bin/activate
    python scripts/replot_predictions.py \\
        /nfs/.../mmc_luna/celery_inference/20260602_074322 \\
        /nfs/.../mmc_luna/celery_inference/20260602_074327

You can pass any number of output dirs — the script walks each
independently. Failures on individual slices are logged and skipped,
they don't abort the run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Reuse LUNA's plotter — it's the canonical one that scgg, novosparc
# and celery_inference all already import. Importing it here pulls in
# matplotlib (and tries colorcet for glasbey); the user is expected
# to run this script INSIDE the env they want the palette from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_luna_train  # noqa: E402

_plot_pred_vs_truth = run_luna_train._plot_pred_vs_truth

logger = logging.getLogger("replot")


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _detect_method_label(out_dir: Path) -> str:
    """Best-effort method label for the plot title.

    Reads ``out_dir/manifest.json`` or ``out_dir/config.yaml`` if
    present; otherwise infers from the parent dir name (which contains
    the method, e.g. ``celery_inference``, ``luna_inference``,
    ``scgg_inference``). Falls back to ``"prediction"``.
    """
    # Cheapest signal: parent dir name pattern ``<method>_inference``.
    for ancestor in (out_dir, *out_dir.parents):
        name = ancestor.name
        for method, label in (
            ("celery", "CeLEry"),
            ("luna",   "LUNA"),
            ("scgg",   "scGG"),
            ("novosparc", "novoSpaRc"),
        ):
            if name.startswith(method + "_"):
                return label
    return "prediction"


def replot_one_output_dir(out_dir: Path, *, method_label: str | None = None) -> int:
    """Re-render every slice plot under ``out_dir``. Returns # of
    plots successfully written."""
    out_dir = out_dir.resolve()
    if not out_dir.exists():
        logger.error(f"  not found: {out_dir}")
        return 0

    method_label = method_label or _detect_method_label(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Find every metadata_pred.csv under this output dir. We glob
    # broadly so the script doesn't have to know the
    # luna_run/test_results/<run_name>/<method>/<slice>/ tree.
    pred_csvs = sorted(out_dir.rglob("metadata_pred.csv"))
    if not pred_csvs:
        logger.warning(f"  no metadata_pred.csv under {out_dir}; skipping")
        return 0

    logger.info(
        f"  {out_dir.name}: found {len(pred_csvs)} slice CSV pair(s) "
        f"  (method_label={method_label!r})"
    )

    written = 0
    for pred_csv in pred_csvs:
        slice_dir = pred_csv.parent
        slice_label = slice_dir.name
        true_csv = slice_dir / "metadata_true.csv"
        if not true_csv.exists():
            logger.warning(f"    {slice_label}: missing metadata_true.csv, skipping")
            continue

        try:
            pred = pd.read_csv(pred_csv, index_col=0)
            true = pd.read_csv(true_csv, index_col=0)
            _plot_pred_vs_truth(
                coords_true=true[["coord_X", "coord_Y"]].to_numpy(),
                coords_pred=pred[["coord_X", "coord_Y"]].to_numpy(),
                cell_class=(
                    true["cell_class"].astype(str).to_numpy()
                    if "cell_class" in true.columns else None
                ),
                out_path=plots_dir / f"{slice_label}.svg",
                title_prefix=slice_label,
                method_label=method_label,
            )
            written += 1
            logger.debug(f"    wrote plots/{slice_label}.svg")
        except Exception as exc:
            logger.warning(f"    {slice_label}: plot failed: {exc}")
            continue

    logger.info(f"  {out_dir.name}: wrote {written}/{len(pred_csvs)} plots → {plots_dir}")
    return written


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "output_dirs", nargs="+", type=Path,
        help="One or more inference output dirs (each contains "
             "plots/ and luna_run/test_results/...). Plots will be "
             "(re-)written into each <out_dir>/plots/.",
    )
    p.add_argument(
        "--method_label", default=None,
        help="Override the method label in the plot title. Default: "
             "inferred from each output dir's parent name (e.g. "
             "'celery_inference' → 'CeLEry').",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="DEBUG-level logging (lists every slice as it's written).",
    )
    args = p.parse_args()
    _setup_logging(args.verbose)

    # Tell the user up front which palette they'll actually get.
    try:
        import importlib
        importlib.import_module("colorcet")
        logger.info("colorcet found — using LUNA's glasbey palette ✓")
    except ImportError:
        logger.warning(
            "colorcet NOT installed in this env — _plot_pred_vs_truth "
            "will silently fall back to matplotlib tab20. "
            "If you wanted the LUNA/scgg palette, install colorcet "
            "FIRST:  pip install colorcet  (then re-run this script)."
        )

    total = 0
    for d in args.output_dirs:
        total += replot_one_output_dir(d, method_label=args.method_label)

    logger.info(f"done. wrote {total} plot(s) across {len(args.output_dirs)} output dir(s).")
    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

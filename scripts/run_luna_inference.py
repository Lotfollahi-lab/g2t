"""Run inference with a previously-trained LUNA checkpoint.

Thin wrapper around ``run_luna_train.run_benchmark`` with ``skip_training=True``,
so the underlying pipeline (silver-h5ad discovery, CSV materialisation,
Hydra invocation, per-slice evaluation, runtime tracking) is shared
verbatim with the training script — nothing about how predictions are
computed or scored drifts between train and inference.

Reads the silver dir's ``*_test.h5ad`` files and runs LUNA's
``general.mode=test_only`` against the supplied ``.ckpt``. Writes the
same artifacts as ``run_luna_train.py`` (per_slice_metrics.csv,
aggregate_metrics.json, runtime.csv, config.yaml, ...).

Example::

    python scripts/inference_luna.py \\
        --data_dir   /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --checkpoint /nfs/.../luna_run/checkpoints/epoch=999.ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse the full pipeline from run_luna_train.py — same silver discovery,
# CSV builder, LUNA subprocess invocation, evaluation, runtime
# tracking. No duplication.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_luna_train  # noqa: E402  (local-script import, intentional)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data_dir", default=None,
        help="Silver dir containing *_test.h5ad files for evaluation. "
             "Use this OR (--train_csv + --test_csv).",
    )
    p.add_argument(
        "--train_csv", default=None,
        help="Pre-built LUNA-format train CSV (only used to satisfy "
             "LUNA's data_module wiring; rows are NOT trained on).",
    )
    p.add_argument(
        "--test_csv", default=None,
        help="Pre-built LUNA-format test CSV. Pair with --train_csv.",
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to a LUNA .ckpt produced by a previous run_luna_train.py run.",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Where to write per-slice metrics, runtime.csv, etc. "
             "Default: <ARTIFACTS_ROOT>/<data_dir_name>/<ENGINE_OUTPUT_SUBDIR>/<TS>/  (defaults: luna_model/...).",
    )
    p.add_argument("--seed", type=int, default=0,
                   help="LUNA general.seed (default 0 — matches paper).")
    p.add_argument("--n_genes", type=int, default=None,
                   help="Explicit gene-column count. Default: inferred "
                        "from CSV header.")
    p.add_argument(
        "--wandb_run_name", "--run_name",
        dest="wandb_run_name",
        default="MERFISH_LUNA_inference",
        help="Sets general.name in LUNA's Hydra config. Aliases: --run_name.",
    )
    p.add_argument("--wandb_mode", default="disabled",
                   choices=("disabled", "online", "offline", "dryrun"),
                   help="LUNA general.wandb (default 'disabled').")
    p.add_argument(
        "--luna_repo", default=str(run_luna_train._ENGINE_REPO_DEFAULT),
        help=f"Path to the external LUNA repo. Default: "
             f"{run_luna_train._ENGINE_REPO_DEFAULT}",
    )
    p.add_argument(
        "--override", "--luna_override",
        dest="override",
        action="extend", nargs="+", default=[],
        help="Extra Hydra overrides. Accepts ONE OR MORE key=value "
             "tokens per --override (space-separated), and the flag "
             "itself is repeatable. '--luna_override' is a "
             "backward-compat alias.",
    )
    p.add_argument(
        "--no_plots", action="store_true",
        help="Skip per-section ground-truth-vs-prediction plots. Plots "
             "ON by default in inference mode (in <out_dir>/plots/).",
    )
    p.add_argument(
        "--exclude_test_files", default=None,
        help="Comma-separated *_test.h5ad basenames to drop from the "
             "test set. Useful for skipping too-large slices that "
             "GPU-OOM during inference (e.g. CNS sagittal1/2/3 + "
             "spinalcord). Forwarded to run_luna_train.run_benchmark "
             "which performs the filtering before assembling test.csv.",
    )
    args = p.parse_args()

    # Inference uses run_benchmark in skip-training mode. epochs/batch_size
    # are ignored by LUNA when mode=test_only but the API still wants them.
    # output_subdir routes inference outputs into a different artifacts
    # subtree (luna_inference/...) so they don't clobber training runs
    # (luna_model/...).
    try:
        run_luna_train.run_benchmark(
            data_dir=args.data_dir,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            n_genes=args.n_genes,
            output_dir=args.output_dir,
            seed=args.seed,
            luna_repo=args.luna_repo,
            run_name=args.wandb_run_name,
            wandb_mode=args.wandb_mode,
            extra_overrides=args.override,
            skip_training=True,
            load_checkpoint=args.checkpoint,
            make_plots=not args.no_plots,
            output_subdir="luna_inference",
            exclude_test_files=(
                [s.strip() for s in args.exclude_test_files.split(",") if s.strip()]
                if args.exclude_test_files else None
            ),
        )
    except Exception:
        run_luna_train.logger.exception("LUNA inference failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

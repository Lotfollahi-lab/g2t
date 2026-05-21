"""Run inference with a previously-trained scgg (vendored LUNA) checkpoint.

Thin wrapper around ``run_scgg_train.run_benchmark`` with
``skip_training=True``. Uses the vendored LUNA under
``scgg/src/`` — the modifiable copy we'll edit forward from the
baseline. Pair this with ``inference_luna.py`` (which uses the
external LUNA) for A/B comparisons.

Reads the silver dir's ``*_test.h5ad`` files and runs LUNA's
``general.mode=test_only`` against the supplied ``.ckpt``. Writes the
same artifacts as ``run_scgg_train.py`` (per_slice_metrics.csv,
aggregate_metrics.json, runtime.csv, config.yaml, ...).

Example::

    python scripts/inference_scgg.py \\
        --data_dir   /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --checkpoint /nfs/.../luna_run/checkpoints/epoch=999.ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse the full pipeline from run_scgg_train.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_scgg_train  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data_dir", required=True,
        help="Silver dir containing *_test.h5ad files for evaluation. "
             "(*_train.h5ad files in the same dir are needed to size "
             "LUNA's data_module — rows are NOT trained on.)",
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to a LUNA .ckpt produced by a previous run_scgg_train.py run.",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Where to write per-slice metrics, runtime.csv, etc. "
             "Default: <ARTIFACTS_ROOT>/<data_dir_name>/model/<TS>/.",
    )
    p.add_argument("--seed", type=int, default=0,
                   help="LUNA general.seed (default 0).")
    p.add_argument("--wandb_run_name", default="scgg_inference",
                   help="LUNA general.name (drives wandb run name + run-dir basename).")
    p.add_argument("--no_wandb", action="store_true",
                   help="Set LUNA's general.wandb=disabled (default).")
    p.add_argument("--wandb_online", action="store_true",
                   help="Set LUNA's general.wandb=online — requires "
                        "`wandb login` on the host.")
    p.add_argument("--contact_percentile", type=float, default=0.01,
                   help="Percentile for LUNA's contact F1 metric.")
    p.add_argument("--skip_rssd", action="store_true",
                   help="Skip Kabsch RSSD (faster).")
    p.add_argument(
        "--extra_override", action="append", default=None, metavar="KEY=VALUE",
        help="Extra Hydra override(s) passed through to LUNA. Repeatable.",
    )
    p.add_argument(
        "--no_plots", action="store_true",
        help="Skip per-section ground-truth-vs-prediction plots. Plots "
             "ON by default in inference mode (in <out_dir>/plots/).",
    )
    args = p.parse_args()

    if args.wandb_online and args.no_wandb:
        p.error("--wandb_online and --no_wandb are mutually exclusive.")
    wandb_mode = "online" if args.wandb_online else "disabled"

    try:
        run_scgg_train.run_benchmark(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            wandb_mode=wandb_mode,
            wandb_run_name=args.wandb_run_name,
            contact_percentile=args.contact_percentile,
            compute_rssd=not args.skip_rssd,
            skip_training=True,
            load_checkpoint=args.checkpoint,
            extra_overrides=args.extra_override,
            make_plots=not args.no_plots,
        )
    except Exception:
        run_scgg_train.logger.exception("scgg inference failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

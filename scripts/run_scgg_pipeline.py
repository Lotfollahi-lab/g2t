#!/usr/bin/env python
"""Train + immediately evaluate: single-command scGG pipeline.

Wraps ``run_scgg_train.py`` and ``run_scgg_inference.py`` into one
invocation. A single wall-clock timestamp is generated upfront and
pinned through both steps, so the trained model and its inference
artifacts pair up by eye::

    artifacts/<dataset>/scgg_model/<TS>/      ← training output
    artifacts/<dataset>/scgg_inference/<TS>/  ← inference output

Typical use::

    python scgg/scripts/run_scgg_pipeline.py \\
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --wandb_run_name scgg_mmc_fm_sinkhorn \\
        --override model.framework=flow_matching \\
                   model.loss.sinkhorn.enabled=true

For inference-only knobs (e.g. switching to the Heun sampler at eval
time without changing training) use ``--inference_override``::

    python scgg/scripts/run_scgg_pipeline.py \\
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --wandb_run_name scgg_mmc_fm \\
        --override model.framework=flow_matching \\
        --inference_override model.flow_matching.sampler=heun \\
                             model.flow_matching.n_sampling_steps=25

Use ``--skip_inference`` to train only.

The wrapper exits non-zero if either step fails — i.e. an inference
failure won't be swallowed by an earlier training success.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


logger = logging.getLogger("scgg_pipeline")

_THIS_DIR = Path(__file__).resolve().parent
_TRAIN_SCRIPT = _THIS_DIR / "run_scgg_train.py"
_INFER_SCRIPT = _THIS_DIR / "run_scgg_inference.py"
# Mirrors run_scgg_train.py's _ARTIFACTS_ROOT default. Override via
# the ``SCGG_ARTIFACTS_ROOT`` env var.
_DEFAULT_ARTIFACTS_ROOT = "/nfs/team361/sb75/scgg-reproducibility/artifacts"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---------------- Dataset source (one of these required) ----------------
    p.add_argument(
        "--data_dir", default=None,
        help="Silver h5ad directory (suffix-based train/test discovery). "
             "Use this OR (--train_csv + --test_csv).",
    )
    p.add_argument(
        "--train_csv", default=None,
        help="Pre-built LUNA-format train CSV. Pair with --test_csv.",
    )
    p.add_argument(
        "--test_csv", default=None,
        help="Pre-built LUNA-format test CSV.",
    )
    p.add_argument(
        "--n_genes", type=int, default=None,
        help="Number of gene columns in pre-built CSVs. Auto-inferred when omitted.",
    )

    # ---------------- Train-side knobs ----------------
    p.add_argument("--epochs", type=int, default=1000,
                   help="train.n_epochs override (LUNA paper default: 1000).")
    p.add_argument("--batch_size", type=int, default=6,
                   help="train.batch_size override.")
    p.add_argument("--lr", type=float, default=None,
                   help="Optional train.lr override.")
    p.add_argument("--seed", type=int, default=0,
                   help="general.seed override. Passed to both steps.")
    p.add_argument(
        "--log2_normalize", action="store_true",
        help="Apply log2 normalisation when building CSVs from h5ad. "
             "Default OFF — LUNA training expects raw counts.",
    )
    p.add_argument(
        "--override", "--luna_override",
        dest="override",
        action="extend", nargs="+", default=[],
        help="Hydra overrides for the TRAINING run. Accepts space-"
             "separated key=value tokens; flag is repeatable. The "
             "model.* part of these is also captured in the saved "
             ".hydra/config.yaml so inference picks it up automatically "
             "— no need to repeat model-affecting overrides for "
             "inference.",
    )

    # ---------------- WandB ----------------
    p.add_argument("--wandb_project", default=None,
                   help="wandb project name. Used by both steps.")
    p.add_argument("--wandb_mode", default="online",
                   choices=("disabled", "online", "offline", "dryrun"),
                   help="Train-side wandb mode. Inference defaults to the "
                        "same value unless --inference_wandb_mode is set.")
    p.add_argument(
        "--wandb_run_name", "--run_name", dest="wandb_run_name",
        default=None,
        help="Base name. Training uses this verbatim; inference appends "
             "--inference_run_name_suffix.",
    )
    p.add_argument(
        "--luna_repo", default=None,
        help="Path to LUNA / scgg src tree. Default: scgg's vendored copy "
             "(set by run_scgg_train.py).",
    )

    # ---------------- Inference-side knobs ----------------
    p.add_argument(
        "--inference_override",
        action="extend", nargs="+", default=[],
        help="Hydra overrides applied ONLY to the inference step (in "
             "addition to the train-time overrides that are already "
             "captured in the checkpoint's .hydra/config.yaml). Use "
             "this for inference-only knobs like "
             "`model.flow_matching.sampler=heun` or "
             "`model.flow_matching.n_sampling_steps=25`.",
    )
    p.add_argument(
        "--no_inference_plots", action="store_true",
        help="Skip per-section ground-truth-vs-prediction plots in the "
             "inference output. Plots are ON by default at inference time.",
    )
    p.add_argument(
        "--inference_wandb_mode", default=None,
        choices=("disabled", "online", "offline", "dryrun"),
        help="Override wandb_mode for the inference step. Default: same "
             "as training's --wandb_mode.",
    )
    p.add_argument(
        "--inference_run_name_suffix", default="_inference",
        help="Appended to --wandb_run_name when calling inference. "
             "Default '_inference' makes inference runs visually grouped "
             "with their training counterpart in the wandb UI.",
    )

    # ---------------- Pretrained gene encoder ----------------
    p.add_argument(
        "--embedding_field", default=None,
        help="adata.obsm key containing PRECOMPUTED per-cell embeddings "
             "to use in place of raw gene counts (e.g. 'pca_64', "
             "'ae_128', 'scvi_10'). Run scripts/precompute_embeddings.py "
             "first to populate this obsm field on every silver h5ad. "
             "Forwarded to BOTH the train and inference subprocess "
             "calls so the inference path sees the same input "
             "representation as training.",
    )

    # ---------------- Multi-sample inference ----------------
    p.add_argument(
        "--n_inference_samples", type=int, default=1,
        help="Multi-sample inference ensembling. When > 1, both the "
             "training's built-in test step AND the separate inference "
             "subprocess draw N samples per slice and report per-cell "
             "mean. Per-cell std is saved alongside for UQ.",
    )

    # ---------------- Wrapper control ----------------
    p.add_argument(
        "--skip_inference", action="store_true",
        help="Stop after training. Use when you want to inspect "
             "intermediates before evaluating.",
    )

    return p


def _dataset_name_from_args(args: argparse.Namespace) -> str:
    """Mirror run_scgg_train.py's logic for naming the artifacts subtree."""
    if args.data_dir:
        return Path(args.data_dir).name
    if args.train_csv:
        return Path(args.train_csv).resolve().parent.name or "luna_paper_csvs"
    return "unknown"


def _build_train_cmd(
    args: argparse.Namespace,
    train_output_dir: Path,
) -> list[str]:
    """Construct the run_scgg_train.py argv. ``--output_dir`` is pinned
    here so the pipeline knows exactly where the checkpoint will land.
    """
    cmd = [sys.executable, str(_TRAIN_SCRIPT)]
    if args.data_dir:
        cmd += ["--data_dir", args.data_dir]
    if args.train_csv:
        cmd += ["--train_csv", args.train_csv]
    if args.test_csv:
        cmd += ["--test_csv", args.test_csv]
    if args.n_genes is not None:
        cmd += ["--n_genes", str(args.n_genes)]
    cmd += [
        "--output_dir", str(train_output_dir),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--wandb_mode", args.wandb_mode,
    ]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.log2_normalize:
        cmd += ["--log2_normalize"]
    if args.wandb_project:
        cmd += ["--wandb_project", args.wandb_project]
    if args.wandb_run_name:
        cmd += ["--wandb_run_name", args.wandb_run_name]
    if args.luna_repo:
        cmd += ["--luna_repo", args.luna_repo]
    if args.embedding_field:
        cmd += ["--embedding_field", args.embedding_field]
    if args.n_inference_samples and args.n_inference_samples != 1:
        cmd += ["--n_inference_samples", str(args.n_inference_samples)]
    if args.override:
        cmd += ["--override", *args.override]
    return cmd


def _build_inference_cmd(
    args: argparse.Namespace,
    checkpoint: Path,
) -> list[str]:
    """Construct the run_scgg_inference.py argv. Inference INHERITS
    the timestamp from the checkpoint path (regex in run_scgg_train.py),
    so we don't pass --output_dir explicitly — the inference artifacts
    land at artifacts/<dataset>/scgg_inference/<TS>/ automatically.
    """
    cmd = [sys.executable, str(_INFER_SCRIPT)]
    if args.data_dir:
        cmd += ["--data_dir", args.data_dir]
    if args.train_csv:
        cmd += ["--train_csv", args.train_csv]
    if args.test_csv:
        cmd += ["--test_csv", args.test_csv]
    if args.n_genes is not None:
        cmd += ["--n_genes", str(args.n_genes)]
    cmd += [
        "--checkpoint", str(checkpoint),
        "--seed", str(args.seed),
        "--wandb_mode", args.inference_wandb_mode or args.wandb_mode,
    ]
    if args.wandb_project:
        cmd += ["--wandb_project", args.wandb_project]
    if args.wandb_run_name:
        infer_name = f"{args.wandb_run_name}{args.inference_run_name_suffix}"
        cmd += ["--wandb_run_name", infer_name]
    if args.luna_repo:
        cmd += ["--luna_repo", args.luna_repo]
    if args.embedding_field:
        cmd += ["--embedding_field", args.embedding_field]
    if args.n_inference_samples and args.n_inference_samples != 1:
        cmd += ["--n_inference_samples", str(args.n_inference_samples)]
    if args.no_inference_plots:
        cmd += ["--no_plots"]
    if args.inference_override:
        cmd += ["--override", *args.inference_override]
    return cmd


def _find_checkpoint(train_output_dir: Path) -> Path | None:
    """Resolve the checkpoint to load for inference. ``best_model.ckpt``
    is the stable symlink that ``run_scgg_train.py`` creates at the
    end of training; if for any reason it's missing (e.g. training
    crashed before the symlink was written), scan
    ``luna_run/checkpoints/`` for the latest .ckpt.
    """
    stable = train_output_dir / "best_model.ckpt"
    if stable.exists():
        return stable
    run_dir = train_output_dir / "luna_run"
    ckpts = sorted(run_dir.glob("checkpoints/*.ckpt"))
    if not ckpts:
        return None
    return ckpts[-1]


def main() -> int:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ---- Validate dataset source ----
    if not args.data_dir and not (args.train_csv and args.test_csv):
        sys.exit(
            "Must pass either --data_dir, or both --train_csv and --test_csv."
        )
    if (args.train_csv is None) != (args.test_csv is None):
        sys.exit("--train_csv and --test_csv must be passed together.")

    # ---- Pin a single timestamp and on-disk location ----
    # Generated here (not in run_scgg_train.py) so we know exactly
    # where the checkpoint will land and can build the inference
    # command deterministically.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_root = Path(os.environ.get(
        "SCGG_ARTIFACTS_ROOT", _DEFAULT_ARTIFACTS_ROOT,
    ))
    dataset_name = _dataset_name_from_args(args)
    train_output_dir = artifacts_root / dataset_name / "scgg_model" / timestamp
    infer_output_dir = artifacts_root / dataset_name / "scgg_inference" / timestamp

    logger.info("=" * 60)
    logger.info(f"scgg pipeline timestamp: {timestamp}")
    logger.info(f"  train artifacts → {train_output_dir}")
    logger.info(f"  infer artifacts → {infer_output_dir}")
    logger.info("=" * 60)

    # ---- 1. Training ----
    train_cmd = _build_train_cmd(args, train_output_dir)
    logger.info(f"[1/2] Training (mode=train_and_test). Command:")
    for token in train_cmd:
        logger.info(f"    {token}")
    rc = subprocess.run(train_cmd).returncode
    if rc != 0:
        logger.error(f"Training failed (exit code {rc}). Aborting pipeline.")
        return rc

    # ---- Find checkpoint ----
    checkpoint = _find_checkpoint(train_output_dir)
    if checkpoint is None:
        logger.error(
            f"No checkpoint found under {train_output_dir} after training "
            f"completed successfully. Expected {train_output_dir}/best_model.ckpt "
            f"or a *.ckpt under {train_output_dir}/luna_run/checkpoints/."
        )
        return 1
    logger.info(f"Resolved checkpoint: {checkpoint}")

    # ---- Skip inference if requested ----
    if args.skip_inference:
        logger.info("--skip_inference set; pipeline complete (training only).")
        return 0

    # ---- 2. Inference ----
    infer_cmd = _build_inference_cmd(args, checkpoint)
    logger.info(f"[2/2] Inference. Command:")
    for token in infer_cmd:
        logger.info(f"    {token}")
    rc = subprocess.run(infer_cmd).returncode
    if rc != 0:
        logger.error(f"Inference failed (exit code {rc}).")
        return rc

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info(f"  Training artifacts:  {train_output_dir}")
    logger.info(f"  Inference artifacts: {infer_output_dir}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

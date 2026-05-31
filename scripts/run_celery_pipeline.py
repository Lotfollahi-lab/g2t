"""run_celery_pipeline.py — train CeLEry then score on the held-out mouse.

End-to-end orchestrator that mirrors run_luna_pipeline.py / run_scgg_pipeline.py:

    [1/2] subprocess: run_celery_train.py   (writes model.obj — single global model)
    [2/2] subprocess: run_celery_inference.py (writes metrics.csv + per_slice_*.csv)

Both training and inference artifacts land under the same per-run
timestamp directory layout the other three methods use:

    <ARTIFACTS_ROOT>/<dataset>/celery_model/<TS>/
    <ARTIFACTS_ROOT>/<dataset>/celery_inference/<TS>/

The single ``<TS>`` is generated once at startup (or inherited from
``--run_timestamp``) and pinned across both subprocesses so the train
+ infer pair is visible at a glance under the artifacts tree.

CLI surface mirrors run_luna_pipeline.py — every flag the LSF
submitter forwards is accepted. CeLEry-specific knobs
(``--lr``, ``--hidden_dims``, ``--num_workers``) are exposed for
hyperparameter tuning; defaults match the CeLEry paper's MMC setup.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


_THIS_DIR = Path(__file__).resolve().parent
_TRAIN_SCRIPT = _THIS_DIR / "run_celery_train.py"
_INFER_SCRIPT = _THIS_DIR / "run_celery_inference.py"

_DEFAULT_ARTIFACTS_ROOT = Path("/nfs/team361/sb75/scgg-reproducibility/artifacts")


logger = logging.getLogger("celery_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Argument parser — keeps cross-pipeline parity
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Data ----
    p.add_argument(
        "--data_dir", required=True,
        help="Silver h5ad directory with *_train.h5ad + *_test.h5ad files.",
    )

    # ---- Training hyperparameters ----
    p.add_argument(
        "--epochs", type=int, default=None,
        help="num_epochs_max for CeLEry's Fit_cord. Default 500 in "
             "run_celery_train.py; override here to bump/cut.",
    )
    p.add_argument(
        "--batch_size", type=int, default=None,
        help="CeLEry Fit_cord batch_size. Default 4 in train script.",
    )
    p.add_argument(
        "--lr", type=float, default=None,
        help="CeLEry initial_learning_rate. Default 1e-3.",
    )
    p.add_argument(
        "--hidden_dims", type=int, nargs="+", default=None,
        help="CeLEry MLP hidden-layer widths. Default [30, 25, 15].",
    )
    p.add_argument(
        "--num_workers", type=int, default=None,
        help="DataLoader num_workers. Default 0 (no fork-RSS amplification).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Seed for both reference-slice selection AND CeLEry's "
             "internal seednum. Same-seed runs reproduce the exact "
             "per-test-slice reference assignment.",
    )

    # ---- Wandb ----
    p.add_argument("--wandb_project", default=None)
    p.add_argument(
        "--wandb_mode", default="online",
        choices=("disabled", "online", "offline", "dryrun"),
    )
    p.add_argument("--wandb_run_name", default=None)

    # ---- Inference ----
    p.add_argument(
        "--n_inference_samples", type=int, default=1,
        help="Multi-sample ensembling. CeLEry is deterministic, so "
             "gains beyond 1 are typically negligible; flag exists "
             "for cross-pipeline parity with scgg/luna.",
    )
    p.add_argument(
        "--no_inference_plots", action="store_true",
        help="Skip per-slice GT-vs-pred scatter plots at inference time.",
    )
    p.add_argument(
        "--inference_run_name_suffix", default="_inference",
        help="Appended to --wandb_run_name for the inference wandb run.",
    )
    p.add_argument(
        "--inference_wandb_mode", default=None,
        choices=(None, "disabled", "online", "offline", "dryrun"),
        help="Override wandb_mode just for the inference step.",
    )

    # ---- Wrapper control ----
    p.add_argument(
        "--skip_inference", action="store_true",
        help="Stop after training.",
    )
    p.add_argument(
        "--skip_training", action="store_true",
        help="Skip the train block; only run inference. Requires "
             "--checkpoint=<best_model.ckpt symlink or model.obj path>.",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="REQUIRED when --skip_training is set. Path to the "
             "training run's best_model.ckpt symlink, model.obj file, "
             "or the directory containing model.obj.",
    )
    p.add_argument(
        "--exclude_test_files", default=None,
        help="Comma-separated *_test.h5ad basenames to drop from "
             "both train and inference (CeLEry trains one model per "
             "test slice, so excluding a test slice also skips its "
             "model). Forwarded to both subprocesses.",
    )
    p.add_argument(
        "--run_timestamp", default=None,
        help="Optional YYYYMMDD_HHMMSS timestamp. Pinned across train "
             "+ infer + LSF logs. Set by submit_pipeline.sh.",
    )

    return p


def _dataset_name_from_args(args: argparse.Namespace) -> str:
    return Path(args.data_dir).name


def _build_train_cmd(
    args: argparse.Namespace,
    train_output_dir: Path,
    timestamp: str,
) -> List[str]:
    cmd = [sys.executable, str(_TRAIN_SCRIPT)]
    cmd += [
        "--data_dir", args.data_dir,
        "--output_dir", str(train_output_dir),
        "--run_timestamp", timestamp,
        "--seed", str(args.seed),
        "--wandb_mode", args.wandb_mode,
    ]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        cmd += ["--batch_size", str(args.batch_size)]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.hidden_dims is not None:
        cmd += ["--hidden_dims", *[str(d) for d in args.hidden_dims]]
    if args.num_workers is not None:
        cmd += ["--num_workers", str(args.num_workers)]
    if args.wandb_project:
        cmd += ["--wandb_project", args.wandb_project]
    if args.wandb_run_name:
        cmd += ["--wandb_run_name", args.wandb_run_name]
    if args.exclude_test_files:
        cmd += ["--exclude_test_files", args.exclude_test_files]
    return cmd


def _build_inference_cmd(
    args: argparse.Namespace,
    checkpoint: Path,
    infer_output_dir: Path,
) -> List[str]:
    cmd = [sys.executable, str(_INFER_SCRIPT)]
    cmd += [
        "--data_dir", args.data_dir,
        "--checkpoint", str(checkpoint),
        "--output_dir", str(infer_output_dir),
        "--seed", str(args.seed),
        "--wandb_mode", args.inference_wandb_mode or args.wandb_mode,
    ]
    if args.wandb_project:
        cmd += ["--wandb_project", args.wandb_project]
    if args.wandb_run_name:
        infer_name = f"{args.wandb_run_name}{args.inference_run_name_suffix}"
        cmd += ["--wandb_run_name", infer_name]
    if args.n_inference_samples and args.n_inference_samples != 1:
        cmd += ["--n_inference_samples", str(args.n_inference_samples)]
    if args.no_inference_plots:
        cmd += ["--no_plots"]
    if args.exclude_test_files:
        cmd += ["--exclude_test_files", args.exclude_test_files]
    return cmd


def _find_checkpoint(train_output_dir: Path) -> Optional[Path]:
    """Resolve the model.obj produced by the multi-slice training run.

    The training script writes a ``best_model.ckpt`` symlink that
    points at ``model.obj`` (a single CeLEry pickle, since the new
    protocol trains ONE global model on all training slices
    concatenated). Prefer the symlink; fall back to ``model.obj``
    directly if the symlink is missing.
    """
    stable = train_output_dir / "best_model.ckpt"
    if stable.exists():
        return stable
    model_obj = train_output_dir / "model.obj"
    if model_obj.is_file():
        return model_obj
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _build_arg_parser().parse_args()

    if args.skip_training and not args.checkpoint:
        sys.exit("--skip_training requires --checkpoint=<celery_models or best_model.ckpt>.")
    if args.skip_training and args.skip_inference:
        sys.exit("--skip_training AND --skip_inference would do nothing.")

    # ---- Resolve timestamp ----
    # Same three-source TS logic as scgg / luna:
    #   1. --run_timestamp (LSF submitter sets this)
    #   2. Inherit from --checkpoint path when --skip_training
    #   3. Fresh wall-clock
    if args.run_timestamp:
        if not re.fullmatch(r"\d{8}_\d{6}(?:_[A-Za-z0-9]+)?", args.run_timestamp):
            sys.exit(
                f"--run_timestamp must match YYYYMMDD_HHMMSS or "
                f"YYYYMMDD_HHMMSS_<suffix>; got {args.run_timestamp!r}"
            )
        timestamp = args.run_timestamp
    elif args.skip_training and args.checkpoint:
        m = re.search(r"\d{8}_\d{6}(?:_[A-Za-z0-9]+)?", str(args.checkpoint))
        if m:
            timestamp = m.group(0)
            logger.info(
                f"[--skip_training] inherited TS={timestamp!r} from "
                f"--checkpoint so inference artifacts pair with the "
                f"original training run."
            )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.warning(
                f"[--skip_training] could not extract a TS from "
                f"--checkpoint={args.checkpoint!r}; using fresh "
                f"TS={timestamp!r}."
            )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    artifacts_root = Path(os.environ.get(
        "CELERY_ARTIFACTS_ROOT",
        os.environ.get(
            "SCGG_ARTIFACTS_ROOT",
            str(_DEFAULT_ARTIFACTS_ROOT),
        ),
    ))
    dataset_name = _dataset_name_from_args(args)
    train_output_dir = artifacts_root / dataset_name / "celery_model" / timestamp
    infer_output_dir = artifacts_root / dataset_name / "celery_inference" / timestamp

    logger.info("=" * 60)
    logger.info(f"CeLEry pipeline timestamp: {timestamp}")
    logger.info(f"  train artifacts → {train_output_dir}")
    logger.info(f"  infer artifacts → {infer_output_dir}")
    logger.info("=" * 60)

    # ---- 1. Training ----
    if args.skip_training:
        checkpoint = Path(args.checkpoint).resolve()
        if not checkpoint.exists():
            logger.error(f"--checkpoint={checkpoint!s} does not exist.")
            return 1
        logger.info(
            f"[1/2] --skip_training set; using provided checkpoint: {checkpoint}"
        )
    else:
        train_cmd = _build_train_cmd(args, train_output_dir, timestamp)
        logger.info(f"[1/2] Training. Command:")
        for token in train_cmd:
            logger.info(f"    {token}")
        rc = subprocess.run(train_cmd).returncode
        if rc != 0:
            logger.error(f"Training failed (exit code {rc}). Aborting pipeline.")
            return rc

        checkpoint = _find_checkpoint(train_output_dir)
        if checkpoint is None:
            logger.error(
                f"No CeLEry checkpoints found under {train_output_dir} "
                f"after training completed successfully. Expected "
                f"{train_output_dir}/best_model.ckpt → celery_models/ "
                f"symlink, or the celery_models/ dir directly."
            )
            return 1
        logger.info(f"Resolved checkpoint: {checkpoint}")

    # ---- Skip inference if requested ----
    if args.skip_inference:
        logger.info("--skip_inference set; pipeline complete (training only).")
        return 0

    # ---- 2. Inference ----
    infer_cmd = _build_inference_cmd(args, checkpoint, infer_output_dir)
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

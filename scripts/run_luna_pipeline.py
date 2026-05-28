#!/usr/bin/env python
"""Train + immediately evaluate: single-command LUNA pipeline.

Wraps ``run_luna_train.py`` and ``run_luna_inference.py`` into one
invocation. A single wall-clock timestamp is generated upfront and
pinned through both steps, so the trained model and its inference
artifacts pair up by eye::

    artifacts/<dataset>/luna_model/<TS>/      ← training output
    artifacts/<dataset>/luna_inference/<TS>/  ← inference output

Mirrors ``run_scgg_pipeline.py`` for the LUNA baseline so you can
flip between the two with the same command-line surface — useful for
producing baseline numbers next to scgg ablation results without
keeping two sets of invocations in your shell history.

Typical use::

    python scgg/scripts/run_luna_pipeline.py \\
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --wandb_run_name luna_mmc_baseline

Bit-exact LUNA paper reproduction (pre-built CSVs)::

    python scgg/scripts/run_luna_pipeline.py \\
        --train_csv /nfs/.../MERFISH_mouse_cortex_train.csv \\
        --test_csv  /nfs/.../MERFISH_mouse_cortex_test.csv \\
        --wandb_run_name luna_paper_reproduction

For inference-only knobs use ``--inference_override``::

    python scgg/scripts/run_luna_pipeline.py \\
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --wandb_run_name luna_mmc_baseline \\
        --inference_override test.something=value

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


logger = logging.getLogger("luna_pipeline")

_THIS_DIR = Path(__file__).resolve().parent
_TRAIN_SCRIPT = _THIS_DIR / "run_luna_train.py"
_INFER_SCRIPT = _THIS_DIR / "run_luna_inference.py"
# Mirrors run_luna_train.py's _ARTIFACTS_ROOT default. Override via
# the ``LUNA_ARTIFACTS_ROOT`` env var. (Keeping a separate env var
# from scgg's so users can route LUNA baselines and scgg experiments
# to different filesystems if they want, though by default they
# share the same root.)
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
        help="Pre-built LUNA-format train CSV (e.g. LUNA's published "
             "MERFISH_mouse_cortex_train.csv). Pair with --test_csv. "
             "Required for bit-exact LUNA-paper reproduction.",
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
    p.add_argument("--epochs", type=int, default=None,
                   help="train.n_epochs override. When omitted, the "
                        "underlying run_luna_train.py default (1000, "
                        "the LUNA paper value) applies — so a Hydra "
                        "override 'train.n_epochs=X' in --override "
                        "wins. Pass this flag only when you want to "
                        "set epochs explicitly via the wrapper.")
    p.add_argument("--batch_size", type=int, default=None,
                   help="train.batch_size override. When omitted, the "
                        "underlying run_luna_train.py default (6, the "
                        "LUNA paper value) applies — so a Hydra "
                        "override 'train.batch_size=X' in --override "
                        "wins. Pass this flag only when you want to "
                        "set batch_size explicitly via the wrapper.")
    p.add_argument("--lr", type=float, default=None,
                   help="Optional train.lr override. LUNA's published "
                        "default for the cortex experiment is 5e-4.")
    p.add_argument("--seed", type=int, default=0,
                   help="general.seed override. Passed to both steps. "
                        "Default 0 matches LUNA's published config.")
    p.add_argument(
        "--log2_normalize", action="store_true",
        help="Apply log2(x+1) when building CSVs from h5ad. "
             "Default OFF — LUNA training expects raw counts; "
             "log-transformed inputs collapse to ~0 Spearman.",
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
    p.add_argument(
        "--train_plots", action="store_true",
        help="Pass LUNA train's --plots flag through, writing per-section "
             "ground-truth-vs-prediction plots into the training output "
             "dir. OFF by default (matches run_luna_train.py default).",
    )

    # ---------------- WandB ----------------
    p.add_argument("--wandb_project", default=None,
                   help="wandb project name. Used by both steps. "
                        "LUNA-side default 'luna' when omitted.")
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
        help="Path to the LUNA repository checkout. Default: the path "
             "hard-coded in run_luna_train.py "
             "(/nfs/team361/sb75/scgg-reproducibility/analysis/"
             "benchmarking/luna).",
    )

    # ---------------- Inference-side knobs ----------------
    p.add_argument(
        "--inference_override",
        action="extend", nargs="+", default=[],
        help="Hydra overrides applied ONLY to the inference step (in "
             "addition to the train-time overrides that are already "
             "captured in the checkpoint's .hydra/config.yaml).",
    )
    p.add_argument(
        "--no_inference_plots", action="store_true",
        help="Skip per-section ground-truth-vs-prediction plots in the "
             "inference output. Plots are ON by default at inference "
             "time (run_luna_inference.py default).",
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

    # ---------------- Wrapper control ----------------
    p.add_argument(
        "--skip_inference", action="store_true",
        help="Stop after training. Use when you want to inspect "
             "intermediates before evaluating.",
    )
    p.add_argument(
        "--skip_training", action="store_true",
        help="Skip the train block and only run inference. Requires "
             "--checkpoint=<path to an existing .ckpt>. Common use: "
             "re-running inference on an existing model with different "
             "hyperparameters (e.g. dataset.num_workers=0 to avoid OOM "
             "on large datasets). The dataset/data_dir/seed/wandb args "
             "behave exactly as in a normal pipeline call.",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to a LUNA .ckpt. REQUIRED when --skip_training is "
             "set; ignored otherwise (the pipeline auto-discovers the "
             "training output's checkpoint in the normal train→infer "
             "flow).",
    )
    p.add_argument(
        "--run_timestamp", default=None,
        help="Optional ``YYYYMMDD_HHMMSS`` timestamp to use as the run "
             "label. When set, the pipeline does NOT generate a fresh "
             "timestamp at startup; it uses this one to construct both "
             "the training and inference output dirs AND threads it "
             "through to ``_luna_runner.py`` so the wandb run carries "
             "the same TS in its config/summary/tags. Used by the LSF "
             "submitter (submit_pipeline.sh) to align its own log dir "
             "with the pipeline's artifacts dir. Format-checked: must "
             "match ``YYYYMMDD_HHMMSS`` (8 digits + underscore + 6 "
             "digits).",
    )

    return p


def _dataset_name_from_args(args: argparse.Namespace) -> str:
    """Mirror run_luna_train.py's logic for naming the artifacts subtree."""
    if args.data_dir:
        return Path(args.data_dir).name
    if args.train_csv:
        return Path(args.train_csv).resolve().parent.name or "luna_paper_csvs"
    return "unknown"


def _build_train_cmd(
    args: argparse.Namespace,
    train_output_dir: Path,
    timestamp: str,
) -> list[str]:
    """Construct the run_luna_train.py argv. ``--output_dir`` is pinned
    here so the pipeline knows exactly where the checkpoint will land.
    The ``timestamp`` is forwarded explicitly so the train script logs
    it to wandb (config + summary + tags) — that way the wandb run
    pairs up with its on-disk artifacts dir by TS.
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
        "--run_timestamp", timestamp,
        "--seed", str(args.seed),
        "--wandb_mode", args.wandb_mode,
    ]
    # Only forward --epochs / --batch_size when the user explicitly
    # passed them — so a Hydra override of train.n_epochs /
    # train.batch_size in --override actually wins, instead of being
    # silently clobbered by the wrapper's default value being placed
    # BEFORE the override list on the cmdline.
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        cmd += ["--batch_size", str(args.batch_size)]
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
    if args.train_plots:
        cmd += ["--plots"]
    if args.override:
        cmd += ["--override", *args.override]
    return cmd


def _build_inference_cmd(
    args: argparse.Namespace,
    checkpoint: Path,
    infer_output_dir: Path,
) -> list[str]:
    """Construct the run_luna_inference.py argv. ``--output_dir`` is
    pinned here explicitly (mirrors the scgg pipeline) so the
    inference subprocess writes to the SAME filesystem the pipeline
    has chosen — previously inference relied on regex-extraction of
    the timestamp from the checkpoint path + the hardcoded
    ``_ARTIFACTS_ROOT`` inside run_luna_train, which silently split
    a run across two filesystems whenever ``LUNA_ARTIFACTS_ROOT``
    was set to redirect outputs (the env-var lookup added in
    run_luna_train.py:525 is honoured, but only by THAT process).
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
        "--output_dir", str(infer_output_dir),
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
    if args.no_inference_plots:
        cmd += ["--no_plots"]
    if args.inference_override:
        cmd += ["--override", *args.inference_override]
    return cmd


def _find_checkpoint(train_output_dir: Path) -> Path | None:
    """Resolve the checkpoint to load for inference. ``best_model.ckpt``
    is the stable symlink that ``run_luna_train.py`` creates at the
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
    # Generated here (not in run_luna_train.py) so we know exactly
    # where the checkpoint will land and can build the inference
    # command deterministically. run_luna_inference.py regex-parses
    # this same timestamp out of the checkpoint path, so the inference
    # artifacts land at artifacts/<dataset>/luna_inference/<TS>/.
    # When --run_timestamp is passed (e.g. by the LSF submitter), use
    # it verbatim so the LSF log dir + on-disk artifacts dir + wandb
    # run config all share the same TS.
    if args.run_timestamp:
        import re as _re
        if not _re.fullmatch(r"\d{8}_\d{6}(?:_[A-Za-z0-9]+)?", args.run_timestamp):
            sys.exit(
                f"--run_timestamp must match YYYYMMDD_HHMMSS or "
                f"YYYYMMDD_HHMMSS_<suffix>; got {args.run_timestamp!r}."
            )
        timestamp = args.run_timestamp
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_root = Path(os.environ.get(
        "LUNA_ARTIFACTS_ROOT", _DEFAULT_ARTIFACTS_ROOT,
    ))
    dataset_name = _dataset_name_from_args(args)
    train_output_dir = artifacts_root / dataset_name / "luna_model" / timestamp
    infer_output_dir = artifacts_root / dataset_name / "luna_inference" / timestamp

    logger.info("=" * 60)
    logger.info(f"luna pipeline timestamp: {timestamp}")
    logger.info(f"  train artifacts → {train_output_dir}")
    logger.info(f"  infer artifacts → {infer_output_dir}")
    logger.info("=" * 60)

    # ---- Validate skip_training / checkpoint pairing ----
    if args.skip_training and not args.checkpoint:
        sys.exit("--skip_training requires --checkpoint=<path>.")
    if args.skip_training and args.skip_inference:
        sys.exit(
            "--skip_training AND --skip_inference would do nothing. "
            "Pick at most one."
        )

    # ---- 1. Training (or skip + use provided checkpoint) ----
    if args.skip_training:
        checkpoint = Path(args.checkpoint).resolve()
        if not checkpoint.is_file():
            logger.error(
                f"--checkpoint={checkpoint!s} does not exist."
            )
            return 1
        logger.info(
            f"[1/2] --skip_training set; using provided checkpoint: "
            f"{checkpoint}"
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

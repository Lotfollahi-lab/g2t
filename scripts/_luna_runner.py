"""Launcher for the external (pristine) LUNA repo that adds
``train_only`` / ``test_only`` modes via runtime monkey-patching of
LUNA's ``DataModule`` and ``main``. The external LUNA files stay
untouched on disk — every change lives inside this process.

Why this exists
---------------
LUNA's stock ``main.py`` only knows ``train_and_test`` and
``test_only``. Worse, its ``DataModule.__init__`` always loads BOTH
the train and test CSVs at construction time, so a CSV that's
unusable for one phase (e.g. CNS scRNA test cells with string-typed
cell-IDs that crash ``torch.tensor(...)`` ) takes down the entire
run, even if you only wanted to train.

This launcher patches:
  1. ``datasets.data_module.DataModule.__init__`` — load only the
     split that the current ``general.mode`` actually needs.
  2. ``main.main``'s mode dispatch — accept ``general.mode=train_only``
     and route it to ``train_model`` alone (no ``test_model`` call).

Invoked by ``run_luna_train.py`` / ``run_luna_inference.py`` via
subprocess::

    python _luna_runner.py \\
        --luna_repo /path/to/external/luna \\
        --mode train_only \\
        --override key=value [--override key=value ...]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("luna_runner")


def _patch_datamodule(target_mode: str) -> None:
    """Monkey-patch ``datasets.data_module.DataModule.__init__`` so that
    only the split needed by ``target_mode`` is materialised.

    Must run AFTER ``sys.path`` has been adjusted to include the
    external LUNA repo (otherwise the import below fails).
    """
    from datasets import data_module as dm

    OrigDataModule = dm.DataModule
    OrigInit = OrigDataModule.__init__

    load_train = target_mode in ("train_only", "train_and_test")
    load_test = target_mode in ("test_only", "train_and_test")

    def patched_init(self, cfg):
        # Inline reimpl of upstream LUNA's DataModule.__init__ that
        # loads only the splits we asked for. This matches the
        # equivalent patch we applied directly to scgg/src/datasets/
        # data_module.py — the two stay in lockstep on purpose.
        if load_train:
            train_data = self.data_loading(cfg, "train")
            self.train_dataset = self._initialize_dataset("train", train_data, cfg)
        else:
            self.train_dataset = None

        if load_test:
            test_data = self.data_loading(cfg, "test")
            self.test_dataset = self._initialize_dataset("test", test_data, cfg)
        else:
            self.test_dataset = None

        if cfg.dataset.validation_data_path:
            val_data = self.data_loading(cfg, "validation")
            self.validation_dataset = self._initialize_dataset(
                "validation", val_data, cfg
            )
        else:
            self.validation_dataset = None

        train_stats = self.train_dataset.statistics if self.train_dataset else None
        test_stats = self.test_dataset.statistics if self.test_dataset else train_stats
        if train_stats is None:
            train_stats = test_stats
        self.statistics = {
            "train": train_stats,
            "validation": (
                self.validation_dataset.statistics
                if self.validation_dataset else None
            ),
            "test": test_stats,
        }
        # Call AbstractDataModule's __init__ via super() — same as the
        # upstream class does.
        super(OrigDataModule, self).__init__(
            cfg,
            train_dataset=self.train_dataset,
            val_dataset=self.validation_dataset if self.validation_dataset else None,
            test_dataset=self.test_dataset,
        )

    OrigDataModule.__init__ = patched_init
    logger.info(
        f"[luna_runner] patched DataModule.__init__ for mode={target_mode!r} "
        f"(load_train={load_train}, load_test={load_test})"
    )


def _extract_output_dir(overrides) -> Optional[str]:
    """Pull a ``hydra.run.dir=...`` override out of the list, returning
    the path. ``compose`` (which we use instead of ``@hydra.main``)
    does NOT honour ``hydra.run.dir``, so we read it ourselves and
    chdir there before training — that keeps LUNA's
    ``checkpoints_parent_dir = os.path.join(os.getcwd(), "checkpoints")``
    pointing where the caller asked for.
    """
    for o in overrides:
        if o.startswith("hydra.run.dir="):
            v = o.split("=", 1)[1]
            return v.strip().strip("'\"")
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--luna_repo", required=True,
                   help="Path to the LUNA checkout (must contain "
                        "main.py and configs/). Either the external "
                        "LUNA repo or scgg's vendored copy.")
    p.add_argument(
        "--mode", required=True,
        choices=("train_only", "test_only", "train_and_test"),
        help="Which phases to run. train_only skips DataModule test "
             "loading entirely.",
    )
    p.add_argument(
        "--override", action="append", default=[],
        help="Hydra override (key=value). Repeatable.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    luna_repo = Path(args.luna_repo).resolve()
    if not (luna_repo / "main.py").exists():
        raise FileNotFoundError(f"LUNA main.py missing under {luna_repo}")
    config_dir = luna_repo / "configs"
    if not config_dir.exists():
        raise FileNotFoundError(f"LUNA configs/ missing under {luna_repo}")

    # sys.path so LUNA's absolute imports (`from models.X import ...`)
    # resolve. cwd is also set to luna_repo briefly so any module-level
    # path resolution in LUNA picks up its own tree.
    sys.path.insert(0, str(luna_repo))
    os.chdir(str(luna_repo))

    # Patch DataModule BEFORE LUNA's main is imported / runs.
    _patch_datamodule(args.mode)

    from hydra import compose, initialize_config_dir
    from main import set_seed, train_model, test_model  # type: ignore
    from utils.diffusion_model.setup.setup import setup_dataset  # type: ignore

    # Compose the config from an ABSOLUTE path — using @hydra.main would
    # resolve config_path relative to THIS file (scgg/scripts/), which
    # is the wrong tree. initialize_config_dir takes an absolute path,
    # so it works whether luna_repo is the external LUNA checkout or
    # the vendored scgg/src/ copy.
    with initialize_config_dir(
        version_base="1.3", config_dir=str(config_dir), job_name="luna_runner",
    ):
        cfg = compose(config_name="config", overrides=list(args.override))

    # Force the mode override regardless of what Hydra resolved (the
    # user may have left it on the experiment default).
    cfg.general.mode = args.mode

    # compose() does NOT auto-create a Hydra runtime output dir or
    # chdir to it. Read the hydra.run.dir override ourselves, mkdir
    # it, and chdir so LUNA's relative-path bookkeeping
    # (`os.path.join(os.getcwd(), "checkpoints")`) lands in the right
    # place.
    output_dir = _extract_output_dir(args.override) or os.getcwd()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    cfg.general.local_saved_path = output_dir

    set_seed(cfg.general.seed)
    datamodule, dataset_infos = setup_dataset(cfg)
    if args.mode == "train_and_test":
        train_model(cfg, datamodule, dataset_infos)
        test_model(cfg, datamodule, dataset_infos)
    elif args.mode == "train_only":
        train_model(cfg, datamodule, dataset_infos)
    elif args.mode == "test_only":
        test_model(cfg, datamodule, dataset_infos)
    else:  # pragma: no cover — argparse already constrains this
        raise ValueError(f"unknown mode {args.mode!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

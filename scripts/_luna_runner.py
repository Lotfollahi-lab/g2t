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


def _run_train_only(cfg, train_model_fn, setup_dataset_fn) -> None:
    """train_only: setup_dataset → train_model. No test phase."""
    import hydra
    cfg.general.local_saved_path = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    datamodule, dataset_infos = setup_dataset_fn(cfg)
    train_model_fn(cfg, datamodule, dataset_infos)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--luna_repo", required=True,
                   help="Path to the external LUNA checkout (must contain "
                        "main.py and configs/).")
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

    # cwd / sys.path so LUNA's absolute imports (`from models.X import ...`)
    # work, and Hydra finds configs/.
    sys.path.insert(0, str(luna_repo))
    os.chdir(str(luna_repo))

    # Patch DataModule BEFORE LUNA's main is imported / runs.
    _patch_datamodule(args.mode)

    # Now we replicate LUNA's main(), but with mode dispatch we control.
    import hydra
    from omegaconf import DictConfig
    from main import set_seed, train_model, test_model  # type: ignore
    from utils.diffusion_model.setup.setup import setup_dataset  # type: ignore

    # Hand the Hydra overrides through sys.argv (Hydra's @hydra.main
    # picks them up from there).
    sys.argv = ["main.py"] + list(args.override)

    @hydra.main(
        version_base="1.3", config_path="./configs", config_name="config",
    )
    def _entry(cfg: DictConfig):
        # Force the mode override regardless of what Hydra resolved
        # (the user may have left it on the experiment default).
        cfg.general.mode = args.mode
        set_seed(cfg.general.seed)
        cfg.general.local_saved_path = (
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
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

    _entry()
    return 0


if __name__ == "__main__":
    sys.exit(main())

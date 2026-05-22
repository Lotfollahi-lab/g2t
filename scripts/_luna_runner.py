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


def _patch_test_single_checkpoint() -> None:
    """LUNA's stock ``test_single_checkpoint`` silently bails out when
    the checkpoint filename doesn't match ``epoch=N.ckpt``::

        try:
            cfg.test.epoch_index = int(checkpoint_path.split("=")[-1].split(".")[0])
        except ValueError:
            return   # silently!

    That fires for anything sensible-but-non-standard you'd pass:
    ``best_model.ckpt`` (our symlink target name), ``last.ckpt``,
    ``checkpoint.ckpt``, etc. The whole test phase becomes a no-op,
    LUNA exits 0, and the caller sees "no metadata_pred.csv" with no
    explanation. We replace the silent return with a fallback that
    sets ``epoch_index = 0`` and continues, so the test always runs.
    """
    import os
    import main as luna_main_mod  # noqa: WPS433

    def patched_test_single_checkpoint(
        cfg, dataset_infos, *positional, **kwargs,
    ):
        # Signature note: upstream LUNA's signature is
        # (cfg, datamodule, dataset_infos, checkpoint_path, dataloader_test).
        # We accept varargs/kwargs so any version of LUNA we wrap stays
        # callable, then re-extract the parts we need by name or position.
        datamodule = positional[0] if positional else kwargs["datamodule"]
        checkpoint_path = (
            positional[1] if len(positional) > 1
            else kwargs.get("checkpoint_path") or kwargs["checkpoint"]
        )
        dataloader_test = (
            positional[2] if len(positional) > 2
            else kwargs.get("dataloader_test") or kwargs.get("dataloaders_test")
        )

        cfg.test.checkpoint_path = checkpoint_path
        cfg.test.test_save_parent_path = os.path.join(
            cfg.test.save_dir, cfg.general.name
        )
        print(f"Testing checkpoint: {checkpoint_path}")
        try:
            cfg.test.epoch_index = int(
                checkpoint_path.split("=")[-1].split(".")[0]
            )
        except ValueError:
            cfg.test.epoch_index = 0
            print(
                f"  (filename {os.path.basename(checkpoint_path)!r} doesn't "
                f"match epoch=N.ckpt — defaulting epoch_index to 0 and "
                f"continuing instead of LUNA's silent skip.)"
            )
        print(f"Epoch index: {cfg.test.epoch_index}")
        if cfg.general.mode == "test_only":
            luna_main_mod.load_model_config(cfg, checkpoint_path)
        model = luna_main_mod.setup_model(
            cfg, dataset_infos, checkpoint_path=checkpoint_path,
        )
        callbacks = luna_main_mod.setup_callbacks(cfg, datamodule)
        trainer = luna_main_mod.setup_trainer(cfg, callbacks)
        trainer.test(
            model, ckpt_path=checkpoint_path, dataloaders=dataloader_test,
        )

    # Wrap so we get the upstream calling convention back. test_model
    # invokes `test_single_checkpoint(cfg, datamodule, dataset_infos,
    # checkpoint_path, dataloaders_test)` positionally — bind the
    # positional args through to our patched body.
    def adapter(cfg, datamodule, dataset_infos, checkpoint_path, dataloader_test):
        return patched_test_single_checkpoint(
            cfg, dataset_infos, datamodule, checkpoint_path, dataloader_test,
        )

    luna_main_mod.test_single_checkpoint = adapter
    logger.info(
        "[luna_runner] patched test_single_checkpoint "
        "(non-epoch=N filenames no longer silently skipped)"
    )


def _patch_load_model_config() -> None:
    """LUNA's stock ``load_model_config`` reads
    ``<checkpoint_dir>/../.hydra/config.yaml`` and crashes if the file
    is missing. Training runs from before we started writing this
    snapshot (or runs launched via ``compose()`` rather than
    ``@hydra.main``) don't have it. Make the load lenient so we
    keep the currently-composed ``cfg.model`` in that case — same
    fix we made in the vendored scgg/src/main.py.
    """
    import os
    import main as luna_main_mod  # noqa: WPS433
    from yaml import safe_load

    def patched_load_model_config(cfg, checkpoint_path):
        config_file = "/".join(checkpoint_path.split("/")[:-2])
        config_yaml = f"{config_file}/.hydra/config.yaml"
        if not os.path.exists(config_yaml):
            print(
                f"[load_model_config] no snapshot at {config_yaml} — "
                f"keeping the currently composed cfg.model."
            )
            return
        loading_model_cfg = safe_load(open(config_yaml))
        cfg["model"] = loading_model_cfg["model"]

    luna_main_mod.load_model_config = patched_load_model_config
    logger.info(
        "[luna_runner] patched load_model_config "
        "(missing .hydra/config.yaml no longer crashes)"
    )


def _patch_setup_model() -> None:
    """Monkey-patch upstream LUNA's ``setup_model`` so it only loads a
    checkpoint when ``general.mode == "test_only"``.

    Upstream LUNA's ``setup_model`` reads::

        if cfg.general.mode == "train_and_test":
            pass
        else:
            cfg, _ = get_resume(cfg, dataset_infos, checkpoint_path)

    Anything that isn't ``train_and_test`` — including our new
    ``train_only`` — falls into ``else``, where ``checkpoint_path`` is
    ``None`` for a from-scratch run and ``torch.load(None)`` blows up.
    The vendored scgg copy already has the corrected
    ``if mode == "test_only"`` check; this patch mirrors the same fix
    onto the external LUNA at runtime so the external files stay
    pristine on disk.
    """
    from utils.diffusion_model.setup import setup as setup_mod

    OrigSetupModel = setup_mod.setup_model

    def patched_setup_model(cfg, dataset_infos, checkpoint_path=None):
        # Only resume from checkpoint when testing — every other mode
        # starts from a fresh model.
        if cfg.general.mode == "test_only":
            cfg, _ = setup_mod.get_resume(cfg, dataset_infos, checkpoint_path)
        return setup_mod.FullDenoisingDiffusion(cfg=cfg, dataset_infos=dataset_infos)

    setup_mod.setup_model = patched_setup_model
    # Re-import the binding `main.train_model` uses so its closed-over
    # `setup_model` reference also picks up the patch.
    import main as luna_main_mod  # noqa: WPS433
    if hasattr(luna_main_mod, "setup_model"):
        luna_main_mod.setup_model = patched_setup_model
    logger.info("[luna_runner] patched setup_model (resume only on test_only)")


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

    # Patch setup_model AFTER main is imported so we can also rebind
    # main.setup_model (train_model uses the name imported into main's
    # module namespace at import time). Same for the test-side patches.
    _patch_setup_model()
    _patch_test_single_checkpoint()
    _patch_load_model_config()

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

    # Write the .hydra/config.yaml snapshot that @hydra.main would have
    # produced for free. LUNA's `test_single_checkpoint` -> `load_model_config`
    # reads <checkpoint_dir>/../.hydra/config.yaml at inference time to
    # restore the exact model config used during training; without
    # this file, test_only crashes with FileNotFoundError. Done only
    # for the train modes — test_only relies on the snapshot left
    # behind by the original training run, not the inference run dir.
    if args.mode in ("train_only", "train_and_test"):
        from omegaconf import OmegaConf
        hydra_dir = Path(output_dir) / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, hydra_dir / "config.yaml")
        logger.info(
            f"[luna_runner] wrote config snapshot for later inference: "
            f"{hydra_dir / 'config.yaml'}"
        )

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

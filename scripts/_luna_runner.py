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


def _patch_setup_wandb(
    force_project: Optional[str],
    run_timestamp: Optional[str] = None,
) -> None:
    """Monkey-patch ``utils.data.misc.setup_wandb`` to use a fixed
    project name (and optionally tag the run with our training
    timestamp).

    LUNA's stock ``setup_wandb`` defaults to
    ``f'MolDiffusion_{dataset_name}'`` and we tried adding a
    ``cfg.general.wandb_project`` Hydra override on top of that —
    which doesn't reliably reach the call site (depends on which
    LUNA tree is on ``sys.path`` and whether ``general.wandb_project``
    is in the schema). This patch sidesteps that whole chain: each
    train/inference wrapper passes ``--wandb_project <name>`` to the
    launcher, which replaces ``setup_wandb`` with a version that
    always uses ``<name>``. Predictable; no fallback to
    ``MolDiffusion_*``.

    ``run_timestamp`` (optional, ``YYYYMMDD_HHMMSS``): if passed, the
    same wall-clock timestamp that names the on-disk artifacts dir
    is also injected into the wandb run as
      * ``wandb.config['run_timestamp']`` (searchable/filterable),
      * a wandb tag (visually obvious in the run list),
      * ``wandb.summary['run_timestamp']`` (sortable column in the
        run table).
    This is the canonical link from a wandb run back to its on-disk
    artifacts subtree at ``.../<engine>_model/<run_timestamp>/``.

    If ``force_project`` is None (e.g. someone invokes
    ``_luna_runner.py`` directly without the flag) we leave the
    upstream ``setup_wandb`` untouched.
    """
    if not force_project:
        return
    import wandb as wandb_mod
    import omegaconf as oc
    from utils.data import misc as misc_mod  # type: ignore

    def patched_setup_wandb(cfg):
        config_dict = oc.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True,
        )
        # Inject the training timestamp into the run config so the
        # wandb run pairs unambiguously with its on-disk artifacts
        # dir (also named by this timestamp). Done by mutating the
        # dict we hand to wandb.init rather than by writing to
        # `cfg` — keeps the Hydra-composed config snapshot on disk
        # unchanged.
        if run_timestamp:
            if isinstance(config_dict, dict):
                config_dict.setdefault("run_timestamp", run_timestamp)
        entity = getattr(cfg.general, "wandb_entity", None) or None
        kwargs = {
            "name": cfg.general.name,
            "project": force_project,
            "entity": entity,
            "config": config_dict,
            "reinit": True,
            "mode": cfg.general.wandb,
        }
        if run_timestamp:
            # Tag form is human-friendly in the run-list UI.
            kwargs["tags"] = [f"ts:{run_timestamp}"]
        print(
            f"[setup_wandb] init (forced): project={force_project!r}, "
            f"entity={entity!r}, mode={cfg.general.wandb!r}, "
            f"name={cfg.general.name!r}, run_timestamp={run_timestamp!r}"
        )
        wandb_mod.init(**kwargs)
        # Also publish as a summary metric so it shows up as a
        # sortable column in the wandb run table. `summary` only
        # exists after init succeeded and only when wandb is in
        # online/offline mode (disabled mode returns a stub).
        if run_timestamp and getattr(wandb_mod, "run", None) is not None:
            try:
                wandb_mod.run.summary["run_timestamp"] = run_timestamp
            except Exception:  # noqa: BLE001 — best-effort decoration
                pass
        wandb_mod.save("*.txt")
        return cfg

    misc_mod.setup_wandb = patched_setup_wandb
    # FullDenoisingDiffusion imports setup_wandb into its module
    # namespace at import time — rebind that too.
    import diffusion_model as luna_diffusion_mod  # type: ignore
    if hasattr(luna_diffusion_mod, "setup_wandb"):
        luna_diffusion_mod.setup_wandb = patched_setup_wandb
    # Same for utils.diffusion_model.setup.setup, which also imports it.
    from utils.diffusion_model.setup import setup as setup_mod  # type: ignore
    if hasattr(setup_mod, "setup_wandb"):
        setup_mod.setup_wandb = patched_setup_wandb
    logger.info(
        f"[luna_runner] patched setup_wandb (force project={force_project!r})"
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


def _patch_setup_callbacks() -> None:
    """Inject a ``MetricsCsvCallback`` into LUNA's callback list so
    per-epoch loss values land on disk at ``<out_dir>/metrics.csv``.

    LUNA's PyTorch Lightning trainer ships with no on-disk metrics
    logger by default — its ``setup_trainer`` doesn't pass a
    ``logger=`` argument, and even PL's default ``CSVLogger`` puts
    things under ``lightning_logs/<name>/version_N/metrics.csv``,
    three directories deep and not very discoverable. We want a
    single ``metrics.csv`` next to the rest of the artifacts
    (``runtime.csv``, ``aggregate_metrics.json``, …).

    The callback:
      * writes one row per training/validation epoch
      * row contains ``step``, ``epoch``, ``phase``, and every numeric
        entry in ``trainer.callback_metrics`` (i.e. all the
        ``train_loss/*``, ``train_epoch/*``, ``val_loss/*`` keys our
        ``LossFunction`` already emits)
      * appends across the run; columns grow as new logged keys
        appear (column set is rewritten on each row to keep things
        simple — fine for the row counts we produce, ~few × n_epochs).
    """
    import csv as _csv
    import pytorch_lightning as pl_local
    from utils.diffusion_model.setup import setup as setup_mod  # type: ignore

    class MetricsCsvCallback(pl_local.Callback):
        def __init__(self, out_path: Path):
            super().__init__()
            self.out_path = out_path
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self._rows: list[dict] = []

        def _snapshot(self, trainer, phase: str) -> None:
            row = {
                "phase": phase,
                "epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
            }
            for k, v in trainer.callback_metrics.items():
                try:
                    if hasattr(v, "detach"):
                        row[k] = float(v.detach().cpu().item())
                    else:
                        row[k] = float(v)
                except Exception:
                    # non-numeric metric — stringify so the column is preserved
                    row[k] = str(v)
            self._rows.append(row)
            # Stable column set across writes (sorted for determinism + a
            # consistent ordering across re-runs in the same out_dir).
            fieldnames = sorted({k for r in self._rows for k in r.keys()})
            with open(self.out_path, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                for r in self._rows:
                    w.writerow(r)

        def on_train_epoch_end(self, trainer, pl_module):
            self._snapshot(trainer, phase="train_epoch")

        def on_validation_epoch_end(self, trainer, pl_module):
            # Only meaningful when validation is actually configured;
            # PL still calls this hook with empty metrics if not.
            if trainer.sanity_checking:
                return
            self._snapshot(trainer, phase="val_epoch")

    OrigSetupCallbacks = setup_mod.setup_callbacks

    def patched_setup_callbacks(cfg, datamodule):
        callbacks = list(OrigSetupCallbacks(cfg, datamodule))
        # cfg.general.local_saved_path is the Hydra run dir, which our
        # run_*_train.py wrappers set to <out_dir>/luna_run/. metrics.csv
        # should live one level up next to runtime.csv etc.
        run_dir = Path(cfg.general.local_saved_path)
        out_dir = run_dir.parent

        # Force-pin ModelCheckpoint dirpaths to an absolute path under
        # the Hydra run dir.
        #
        # LUNA's create_model_checkpoint_callbacks uses dirpath=
        # "checkpoints" (relative). Lightning resolves relative
        # dirpaths against Trainer.default_root_dir, which itself
        # falls back to os.getcwd(). We've seen runs (notably the
        # regression-framework smoke run) silently fail to write
        # any checkpoint — the cwd at Trainer construction time
        # doesn't always agree with where run_scgg_train.py expects
        # to find them. Rewriting any relative dirpath to
        # <luna_run>/checkpoints/ here eliminates the ambiguity:
        # the file lands EXACTLY where _find_latest_checkpoint
        # looks for it.
        ckpt_target_dir = run_dir / "checkpoints"
        ckpt_target_dir.mkdir(parents=True, exist_ok=True)
        rewritten = 0
        for cb in callbacks:
            # Lazy import the type check so we don't grab a stale
            # PL reference if pl_local was patched mid-import.
            if isinstance(cb, pl_local.callbacks.ModelCheckpoint):
                existing = getattr(cb, "dirpath", None)
                if existing is None or not Path(str(existing)).is_absolute():
                    cb.dirpath = str(ckpt_target_dir)
                    rewritten += 1
        if rewritten > 0:
            logger.info(
                f"[luna_runner] pinned {rewritten} ModelCheckpoint(s) to "
                f"absolute dirpath {ckpt_target_dir}"
            )

        metrics_csv = out_dir / "metrics.csv"
        callbacks.append(MetricsCsvCallback(metrics_csv))
        logger.info(
            f"[luna_runner] MetricsCsvCallback will write per-epoch losses "
            f"to {metrics_csv}"
        )

        # Optional EMA callback. Default decay=0 means OFF (and we
        # never instantiate the callback, so train.py is byte-
        # equivalent to before). Anything in (0, 1) turns it on with
        # that decay; canonical values are 0.999 / 0.9999.
        ema_decay = float(getattr(cfg.train, "ema_decay", 0.0))
        if ema_decay > 0.0:
            try:
                from utils.diffusion_model.ema import EMACallback  # type: ignore
            except ImportError as e:
                # Fail loudly — if the user asked for EMA, silently
                # dropping it would be the wrong default.
                raise RuntimeError(
                    f"train.ema_decay={ema_decay} but EMACallback could "
                    f"not be imported: {e}. Check that the scgg src tree "
                    f"is on sys.path."
                )
            ema_every = int(getattr(cfg.train, "ema_every_n_steps", 1))
            callbacks.append(EMACallback(decay=ema_decay, every_n_steps=ema_every))
            logger.info(
                f"[luna_runner] EMACallback enabled (decay={ema_decay}, "
                f"every_n_steps={ema_every}). val/test/checkpoint-save "
                f"will use EMA weights."
            )
        return callbacks

    setup_mod.setup_callbacks = patched_setup_callbacks
    # train_model already imported setup_callbacks into main's namespace;
    # rebind that too so the patched version is the one called.
    import main as luna_main_mod  # type: ignore
    if hasattr(luna_main_mod, "setup_callbacks"):
        luna_main_mod.setup_callbacks = patched_setup_callbacks
    logger.info("[luna_runner] patched setup_callbacks (adds MetricsCsvCallback)")


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
    p.add_argument(
        "--wandb_project", default=None,
        help="If set, force wandb runs into this project regardless of "
             "what cfg.general.wandb_project resolves to. Set per-engine "
             "by the run_*_train.py / run_*_inference.py wrappers "
             "(scgg -> 'scgg', luna -> 'luna').",
    )
    p.add_argument(
        "--run_timestamp", default=None,
        help="YYYYMMDD_HHMMSS string. Injected into wandb.config, "
             "wandb tags, and wandb.summary so the wandb run pairs "
             "unambiguously with its on-disk artifacts dir (which is "
             "named by the same timestamp). Set automatically by the "
             "run_*_train.py wrappers.",
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
    # module namespace at import time). Same for load_model_config —
    # the lenient version is harmless when the .hydra/config.yaml IS
    # present and rescues runs that pre-date that snapshot being
    # written. We deliberately do NOT patch test_single_checkpoint's
    # silent-return-on-ValueError: the user wants a malformed
    # checkpoint path to fail loudly, not default to epoch 0.
    _patch_setup_model()
    _patch_load_model_config()
    _patch_setup_callbacks()
    _patch_setup_wandb(args.wandb_project, run_timestamp=args.run_timestamp)

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

#!/usr/bin/env python
"""Launch (or print) a scgg benchmark run from a committed experiment recipe.

Each run is fully specified by one YAML in ``experiments/`` — the single source
of truth, versioned in the repo, so a run can be reproduced by reading or
re-launching that file WITHOUT hunting for a config on disk. The launcher
flattens the recipe into the exact ``submit_pipeline.sh`` command (the same
``--override "k=v k=v ..."`` the pipeline already expects).

Usage:
  python scripts/launch_experiment.py --list
  python scripts/launch_experiment.py <name> --dry-run     # print the command
  python scripts/launch_experiment.py <name>               # submit it

See experiments/README.md for the recipe schema.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

# Built-in default; override per-recipe (``submit_script:``) or via $SCGG_SUBMIT.
_DEFAULT_SUBMIT = (
    "/nfs/team361/sb75/scgg-reproducibility/analysis/benchmarking/lsf/"
    "submit_pipeline.sh"
)
_REPO = Path(__file__).resolve().parent.parent
_EXP_DIR = _REPO / "experiments"
_REQUIRED = ("method", "data_dir", "epochs", "seed", "gpu_model")


def _fmt(v) -> str:
    """Serialise an override value for the Hydra CLI. Booleans -> true/false
    (YAML parses `off`/`no` as False, so string flags must be quoted in the
    recipe; those arrive here as real strings and pass through unchanged)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _flatten_override(ov) -> str:
    if not ov:
        return ""
    if not isinstance(ov, dict):
        sys.exit("[launch] 'override' must be a mapping of key: value")
    return " ".join(f"{k}={_fmt(v)}" for k, v in ov.items())


def _list_recipes():
    return sorted(p.stem for p in _EXP_DIR.glob("*.yaml"))


def _load(name: str) -> dict:
    path = _EXP_DIR / f"{name}.yaml"
    if not path.exists():
        sys.exit(
            f"[launch] no recipe '{name}' at {path}\n"
            f"Available: {', '.join(_list_recipes()) or '(none)'}"
        )
    with open(path) as fh:
        rec = yaml.safe_load(fh) or {}
    if not isinstance(rec, dict):
        sys.exit(f"[launch] recipe '{name}' is not a YAML mapping")
    return rec


def _build_cmd(name: str, rec: dict) -> list:
    missing = [k for k in _REQUIRED if k not in rec]
    if missing:
        sys.exit(f"[launch] recipe '{name}' missing required field(s): "
                 f"{', '.join(missing)}")
    submit = (rec.get("submit_script")
              or os.environ.get("SCGG_SUBMIT", _DEFAULT_SUBMIT))
    cmd = [
        "bash", str(submit),
        "--method", str(rec["method"]),
        "--data_dir", str(rec["data_dir"]),
        "--seed", str(rec["seed"]),
        "--epochs", str(rec["epochs"]),
        "--gpu_model", str(rec["gpu_model"]),
        "--wandb_run_name", str(rec.get("wandb_run_name", name)),
    ]
    if rec.get("queue"):
        cmd += ["--queue", str(rec["queue"])]
    if rec.get("group"):
        cmd += ["--group", str(rec["group"])]
    override = _flatten_override(rec.get("override"))
    if override:
        cmd += ["--override", override]
    cmd += [str(x) for x in (rec.get("extra_flags") or [])]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("name", nargs="?",
                    help="recipe name (file stem in experiments/)")
    ap.add_argument("--list", action="store_true",
                    help="list available recipes and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the submit command instead of running it")
    a = ap.parse_args()

    if a.list or not a.name:
        print("Available experiment recipes (experiments/*.yaml):")
        for n in _list_recipes():
            print("  " + n)
        return 0

    rec = _load(a.name)
    cmd = _build_cmd(a.name, rec)
    printable = " ".join(shlex.quote(c) for c in cmd)
    if a.dry_run:
        print(printable)
        return 0
    print("[launch] " + printable, file=sys.stderr)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

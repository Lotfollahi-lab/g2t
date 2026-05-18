#!/usr/bin/env python
"""
Ablation runner for the LUNA cortex benchmark.

Sweeps a matrix of (variant, seed) combinations, writes per-run results to
their own subdirectories, and produces a comparison CSV + Markdown summary
table at the end. Crash-resilient: if one variant fails, the rest continue,
and `--skip_existing` (on by default) lets you re-run the script and pick
up where the last invocation left off.

Quick smoke check with shorter training:

    python scripts/run_ablation.py \\
        --data_dir /nfs/team361/sb75/DATASETS/silver/merfish_mouse_cortex_luna \\
        --output_root ./results/ablation_quick \\
        --quick

Full sweep (3 seeds × all variants ≈ 25 h on one GPU):

    python scripts/run_ablation.py \\
        --data_dir /nfs/team361/sb75/DATASETS/silver/merfish_mouse_cortex_luna \\
        --output_root ./results/ablation_v1 \\
        --seeds 42,43,44

Single variant for fast iteration:

    python scripts/run_ablation.py \\
        --data_dir ... --output_root ... \\
        --variants v3_aux_med

Listing the configured variants:

    python scripts/run_ablation.py --list_variants
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
import time
import traceback
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger("scgg.ablation")


# ---------------------------------------------------------------------------
# Variant matrix — each entry is a deep-merge overlay on configs/default.yaml.
# Add / remove entries here to extend the sweep.
# ---------------------------------------------------------------------------

VARIANTS: Dict[str, dict] = {
    # ---- baselines / controls ----------------------------------------------
    # Current best as of v4 — cross-attention head + SupCon + distance regression.
    "v0_baseline": {},
    # Control: no cross-cell attention, just per-cell MLP.
    "v0_mlp_head": {
        "model": {"metric_head": {"type": "mlp"}},
    },

    # ---- loss-weight sweep -------------------------------------------------
    "v1_dist_5x": {
        "training": {"loss": {"distance_regression": {"weight": 5.0}}},
    },
    "v1_dist_10x": {
        "training": {"loss": {"distance_regression": {"weight": 10.0}}},
    },

    # ---- loss ablations ----------------------------------------------------
    "v2_pure_distance": {
        "training": {"loss": {"contrastive": {"weight": 0.0}}},
    },
    "v2_pure_contrastive": {
        "training": {"loss": {"distance_regression": {"enabled": False}}},
    },

    # ---- cell-class auxiliary ----------------------------------------------
    "v3_aux_low": {
        "training": {"loss": {"cell_class_aux": {"enabled": True, "weight": 0.3}}},
    },
    "v3_aux_med": {
        "training": {"loss": {"cell_class_aux": {"enabled": True, "weight": 0.5}}},
    },
    "v3_aux_high": {
        "training": {"loss": {"cell_class_aux": {"enabled": True, "weight": 1.0}}},
    },

    # ---- capacity ----------------------------------------------------------
    "v4_bigger_head": {
        "model": {"metric_head": {"n_layers": 4, "n_heads": 8, "embed_dim": 128}},
    },

    # ---- temperature -------------------------------------------------------
    "v5_temp_005": {
        "training": {"loss": {"contrastive": {"temperature": 0.05}}},
    },
    "v5_temp_010": {
        "training": {"loss": {"contrastive": {"temperature": 0.10}}},
    },

    # ---- speculative combinations -----------------------------------------
    # Best-guess combo: aux + heavy distance + bigger head
    "v6_combo_aux_dist": {
        "training": {"loss": {
            "distance_regression": {"weight": 5.0},
            "cell_class_aux": {"enabled": True, "weight": 0.5},
        }},
    },
    "v6_combo_bigger_aux_dist": {
        "model": {"metric_head": {"n_layers": 4, "n_heads": 8, "embed_dim": 128}},
        "training": {"loss": {
            "distance_regression": {"weight": 5.0},
            "cell_class_aux": {"enabled": True, "weight": 0.5},
        }},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge — overlay overrides base."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_default_config() -> dict:
    """Read the package-installed default.yaml."""
    text = resources.files("scgg.configs").joinpath("default.yaml").read_text()
    return yaml.safe_load(text)


def setup_logging(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "ablation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a"),
        ],
        force=True,
    )


def _flatten_metrics(d: dict) -> Dict[str, float]:
    """Drop nested values (we only want scalar metrics in the comparison row)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v)
        elif isinstance(v, str):
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------------


def run_single(
    variant_name: str,
    seed: int,
    args: argparse.Namespace,
    output_root: Path,
    default_cfg: dict,
) -> Optional[Dict[str, float]]:
    """Run one (variant, seed) combination. Returns the aggregate metrics dict
    or None on failure / skip."""
    overlay = VARIANTS[variant_name]
    run_dir = output_root / f"{variant_name}__seed{seed}"
    agg_path = run_dir / "aggregate_metrics.json"

    if args.skip_existing and agg_path.exists():
        logger.info(f"  -> already complete; loading from {agg_path}")
        try:
            with open(agg_path) as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  failed to load existing aggregate: {e}; re-running")

    # Build the merged config: default + overlay + CLI top-overrides.
    cfg = deep_merge(default_cfg, overlay)
    cfg["training"]["epochs"] = int(args.epochs)
    cfg["training"]["batch_size"] = int(args.batch_size)
    cfg["training"]["wandb"] = bool(args.wandb)
    if args.wandb:
        cfg["training"]["wandb_project"] = args.wandb_project
        cfg["training"]["wandb_run_name"] = f"{variant_name}__seed{seed}"

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Import here so the heavy deps load lazily (and a botched scgg install
    # surfaces with a clear error instead of crashing all 33 runs).
    # The benchmark module sits next to this script — add the script dir to
    # sys.path so it imports under any invocation style.
    _scripts_dir = str(Path(__file__).resolve().parent)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from run_luna_cortex_benchmark import run_benchmark  # noqa: E402

    t_start = time.time()
    try:
        agg = run_benchmark(
            data_dir=args.data_dir,
            output_dir=str(run_dir),
            config_path=str(cfg_path),
            seed=seed,
            device=args.device,
            wandb=bool(args.wandb),
            wandb_run_name=f"{variant_name}__seed{seed}",
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"  RUN FAILED: {variant_name} seed={seed}: {e}")
        traceback.print_exc()
        # Record the failure so we don't silently skip on retry.
        (run_dir / "FAILED").write_text(f"{type(e).__name__}: {e}\n")
        return None
    t_elapsed = time.time() - t_start
    agg["training_time_min"] = t_elapsed / 60.0
    # `run_benchmark` already wrote aggregate_metrics.json — re-write so our
    # `training_time_min` augmentation is persisted.
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)

    logger.info(
        f"  done in {t_elapsed/60:.1f} min  "
        f"spearman_mean_of_medians={agg.get('spearman_mean_of_medians', float('nan')):.4f}"
    )
    return agg


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


_PREFERRED_COLS = [
    "variant",
    "seed",
    "spearman_mean_of_medians",
    "spearman_median_of_medians",
    "spearman_std_of_medians",
    "spearman_mean_of_means",
    "precision_mean",
    "f1_mean",
    "absolute_rssd_mean",
    "mean_rssd_mean",
    "training_time_min",
    "n_slices",
    "total_cells",
]


def write_comparison_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    columns = [c for c in _PREFERRED_COLS if c in all_keys] + sorted(
        all_keys - set(_PREFERRED_COLS)
    )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def write_summary(
    rows: List[dict],
    output_root: Path,
    luna_target: float = 0.448,
) -> None:
    """Aggregate across seeds: write `summary.csv` (mean ± std per variant)
    and `summary.md` (sorted Markdown table)."""
    if not rows:
        logger.warning("No results to summarize.")
        return

    # Group by variant
    by_variant: Dict[str, List[dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)

    metric_keys = [
        "spearman_mean_of_medians",
        "spearman_median_of_medians",
        "spearman_mean_of_means",
        "precision_mean",
        "f1_mean",
        "absolute_rssd_mean",
        "mean_rssd_mean",
        "training_time_min",
    ]

    summary_rows = []
    for v, runs in by_variant.items():
        row = {"variant": v, "n_seeds": len(runs)}
        for k in metric_keys:
            vals = [r[k] for r in runs if k in r and r[k] is not None]
            vals = [float(x) for x in vals if isinstance(x, (int, float, np.integer, np.floating))]
            if vals:
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
            else:
                row[f"{k}_mean"] = float("nan")
                row[f"{k}_std"] = 0.0
        summary_rows.append(row)

    # CSV
    summary_csv = output_root / "summary.csv"
    cols = ["variant", "n_seeds"] + [
        f"{k}_{stat}" for k in metric_keys for stat in ("mean", "std")
    ]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary_rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # Markdown summary sorted by Spearman descending
    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda r: r.get("spearman_mean_of_medians_mean", -1),
        reverse=True,
    )
    md_path = output_root / "summary.md"
    with open(md_path, "w") as f:
        f.write("# ScGG ablation summary — LUNA cortex benchmark\n\n")
        f.write(f"LUNA paper headline: **{luna_target:.4f}**.\n")
        f.write("Headline metric: mean across 31 test slices of per-slice "
                "median per-cell Spearman.\n\n")
        f.write("| Variant | n_seeds | Spearman (mean ± std) | Δ vs LUNA | Precision | F1 | RSSD | Time (min) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in summary_rows_sorted:
            spr = r["spearman_mean_of_medians_mean"]
            spr_s = r["spearman_mean_of_medians_std"]
            delta = spr - luna_target
            prec = r.get("precision_mean_mean", float("nan"))
            f1 = r.get("f1_mean_mean", float("nan"))
            rssd = r.get("absolute_rssd_mean_mean", float("nan"))
            tmin = r.get("training_time_min_mean", float("nan"))
            f.write(
                f"| `{r['variant']}` | {r['n_seeds']} | "
                f"**{spr:.4f}** ± {spr_s:.4f} | "
                f"{delta:+.4f} | {prec:.4f} | {f1:.4f} | {rssd:.2f} | {tmin:.1f} |\n"
            )
        f.write("\n")
        f.write("Variants in this matrix:\n\n")
        for v in VARIANTS:
            f.write(f"- `{v}` — overlay: `{json.dumps(VARIANTS[v])}`\n")
    logger.info(f"Wrote summary CSV   -> {summary_csv}")
    logger.info(f"Wrote summary table -> {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a matrix of ScGG ablations on the LUNA cortex benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", help="Per-slice h5ad directory (LUNA cortex split).")
    p.add_argument("--output_root", help="Directory to write per-run results + summary.")
    p.add_argument("--seeds", default="42",
                   help="Comma-separated list of random seeds (default: 42).")
    p.add_argument("--variants", default="all",
                   help="Comma-separated variant names, or 'all'. See --list_variants.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--device", default=None, help="cuda|cpu (auto if omitted)")
    p.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip (variant, seed) combos with existing aggregate_metrics.json (default).",
    )
    p.add_argument(
        "--no_skip_existing", dest="skip_existing", action="store_false",
        help="Re-run all combinations even if results already exist.",
    )
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 50 epochs instead of --epochs (fast smoke / iteration).")
    p.add_argument("--wandb", action="store_true",
                   help="Log each run to wandb (under project --wandb_project).")
    p.add_argument("--wandb_project", default="scgg-ablation",
                   help="Wandb project name (only if --wandb).")
    p.add_argument("--list_variants", action="store_true",
                   help="Print available variant names and exit.")
    args = p.parse_args()

    if args.list_variants:
        for name, overlay in VARIANTS.items():
            print(f"  {name:30s}  overlay={json.dumps(overlay)}")
        sys.exit(0)

    if not args.data_dir or not args.output_root:
        p.error("--data_dir and --output_root are required (unless --list_variants).")

    return args


def filter_variants(args: argparse.Namespace) -> List[str]:
    if args.variants == "all":
        return list(VARIANTS.keys())
    selected = [v.strip() for v in args.variants.split(",")]
    unknown = [v for v in selected if v not in VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown variants: {unknown}. "
            f"Available: {list(VARIANTS.keys())}"
        )
    return selected


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = 50

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(output_root)

    variants_to_run = filter_variants(args)
    seeds = [int(s) for s in args.seeds.split(",")]

    default_cfg = load_default_config()

    total = len(variants_to_run) * len(seeds)
    logger.info(f"Ablation plan: {len(variants_to_run)} variants × {len(seeds)} seeds "
                f"= {total} runs")
    logger.info(f"Variants: {variants_to_run}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output: {output_root.resolve()}")

    all_rows: List[dict] = []
    run_idx = 0
    t_total = time.time()

    for variant_name in variants_to_run:
        for seed in seeds:
            run_idx += 1
            logger.info("")
            logger.info("=" * 72)
            logger.info(f"[{run_idx}/{total}] variant={variant_name}  seed={seed}")
            logger.info("=" * 72)

            agg = run_single(variant_name, seed, args, output_root, default_cfg)
            if agg is None:
                continue

            row = {"variant": variant_name, "seed": seed}
            row.update(_flatten_metrics(agg))
            all_rows.append(row)

            # Incrementally write the comparison CSV after every run.
            write_comparison_csv(all_rows, output_root / "comparison.csv")

    logger.info(f"\nTotal wall time: {(time.time()-t_total)/60:.1f} min")
    write_summary(all_rows, output_root)
    logger.info("Ablation complete.")


if __name__ == "__main__":
    main()

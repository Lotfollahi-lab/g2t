#!/usr/bin/env python
"""
End-to-end LUNA Figure 3 reproduction with ScGG.

Trains ScGG (contrastive mode by default) on the 33 Mouse 1 slices and
evaluates per-slice on the 31 Mouse 2 slices, reporting the exact LUNA
metrics (median Spearman, contact precision/F1, Kabsch RSSD).

Default report row matches LUNA's headline: mean across 31 test slices of
the per-slice MEDIAN of per-cell Spearman (LUNA reports 44.8% for this).

Usage:

    python -m scgg.scripts.run_luna_cortex_benchmark \\
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --epochs 200 \\
        --batch_size 8192 \\
        --output_dir ./results/luna_cortex_run1

Or call from a notebook by importing `run_benchmark()`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from importlib import resources

logger = logging.getLogger("scgg.luna_cortex_benchmark")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: Optional[Path]) -> dict:
    if path is not None and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f)
    ref = resources.files("scgg.configs").joinpath("default.yaml")
    return yaml.safe_load(ref.read_text())


def _split_train_val(
    section_ids: np.ndarray,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a fraction of unique sections for validation."""
    unique = np.unique(section_ids)
    if val_fraction <= 0 or len(unique) < 2:
        return np.ones(section_ids.shape, dtype=bool), np.zeros(section_ids.shape, dtype=bool)
    n_val = max(1, int(round(val_fraction * len(unique))))
    val = rng.choice(unique, size=n_val, replace=False)
    val_mask = np.isin(section_ids, val)
    train_mask = ~val_mask
    return train_mask, val_mask


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _embed_section(model, gene_expr: np.ndarray, device: torch.device) -> np.ndarray:
    """Compute the metric embedding (or 2-D coord if flow_matching) for one section."""
    model.eval()
    ge = torch.from_numpy(gene_expr).float().to(device)
    if model.objective == "contrastive":
        emb = model.embed_batched(ge)
    else:
        # flow_matching ablation: integrate the ODE to get 2-D coords
        emb = model.generate_embeddings(ge)
    return emb.detach().cpu().numpy()


def _evaluate_split(
    model,
    *,
    gene_expr: np.ndarray,
    coords: np.ndarray,
    section_ids: np.ndarray,
    cell_class: Optional[np.ndarray],
    section_map: Dict[int, str],
    device: torch.device,
    contact_percentile: float,
    compute_rssd: bool,
) -> List[Dict[str, float]]:
    """Run per-slice evaluation and return a list of per-slice metric dicts."""
    from scgg.evaluation.luna_metrics import evaluate_slice

    rows: List[Dict[str, float]] = []
    for sid in np.unique(section_ids):
        mask = section_ids == sid
        if mask.sum() < 10:
            logger.info(f"  skipping section {section_map.get(int(sid), sid)} (n<10)")
            continue
        true_xy = coords[mask]
        ge = gene_expr[mask]
        pred = _embed_section(model, ge, device)
        cls = cell_class[mask] if cell_class is not None else None

        row = evaluate_slice(
            true_xy, pred, cls,
            contact_percentile=contact_percentile,
            compute_rssd=compute_rssd,
            rssd_projection="pca",
        )
        row["section_id"] = int(sid)
        row["section_label"] = section_map.get(int(sid), str(sid))
        rows.append(row)
        logger.info(
            f"  {row['section_label']:30s}  "
            f"spr_median={row['spearman_per_cell_median']:.4f}  "
            f"spr_mean={row['spearman_per_cell_mean']:.4f}  "
            f"prec={row['precision']:.4f}  "
            f"rssd={row.get('absolute_rssd', float('nan')):.2f}"
        )
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


_ARTIFACTS_ROOT = Path("/nfs/team361/sb75/scgg-reproducibility/artifacts")


def run_benchmark(
    data_dir: str,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    seed: int = 42,
    device: Optional[str] = None,
    wandb: Optional[bool] = None,
    wandb_run_name: Optional[str] = None,
    val_fraction: float = 0.1,
    contact_percentile: float = 0.01,
    compute_rssd: bool = True,
    n_top_hvg: Optional[int] = None,
    normalize: bool = True,
    scale: bool = True,
    skip_training: bool = False,
    load_checkpoint: Optional[str] = None,
    class_stratified_distance: Optional[bool] = None,
    k_default: Optional[int] = None,
) -> Dict[str, float]:
    """Run the full LUNA Figure 3 benchmark and return the aggregated metrics.

    Args:
        data_dir: Path to the per-slice h5ad directory (LUNA cortex split).
            Files like mmc_mouse{1,2}_slice{N}.h5ad (or the legacy
            merfish_mouse_cortex_mouse{1,2}_slice{N}.h5ad). Mouse 1
            is treated as TRAIN, Mouse 2 as TEST.
        output_dir: Where to write the trained checkpoint + per-slice and
            aggregated metrics. None (default) derives the path from
            data_dir's basename plus a per-run timestamp:
            {ARTIFACTS_ROOT}/{data_dir.name}/model/{YYYYMMDD_HHMMSS}/
            (e.g. /nfs/team361/sb75/scgg-reproducibility/artifacts/mmc_luna/
            model/20260518_213045/). The timestamp lets multiple training
            runs coexist; the matching inference outputs live under
            inference/{YYYYMMDD_HHMMSS}/.
        config_path: Optional override path to a scgg config YAML.
        epochs / batch_size / lr / wandb / wandb_run_name: CLI overrides.
        val_fraction: Fraction of TRAIN slices held out for validation.
        contact_percentile: Percentile for LUNA's contact F1 metric.
        compute_rssd: Skip Kabsch RSSD if False (faster; for metric embeddings
            it requires a 2-D PCA projection).
        n_top_hvg: Optional HVG selection (seurat_v3 on train counts).
        skip_training: If True, only run evaluation (must provide load_checkpoint).
        load_checkpoint: Path to a saved checkpoint to evaluate (skips training).
        class_stratified_distance: If True, switch the DistanceRegression loss
            from all-pairs Pearson to per-cell-class Pearson averaged across
            classes — forces the model to encode within-class spatial
            geometry instead of just clustering by cell type. None (default)
            leaves whatever the config specifies untouched.
        k_default: k for the GT spatial kNN graph that defines SupCon
            positives. None leaves the config default (10) alone. In laminar
            tissues like cortex, small k means a cell's positives are almost
            all in the same layer with similar tangential positions, so the
            model can satisfy SupCon without learning tangential structure.
            k=30–50 makes positives extend across tangential extents and
            forces the model to encode them.

    Returns:
        Dict of aggregated metrics; identical structure to
        scgg.evaluation.luna_metrics.aggregate_slices().
    """
    from scgg.data.luna_cortex import load_luna_cortex
    from scgg.data.dataset import SpatialTranscriptomicsDataset
    from scgg.model.scgg import ScGG
    from scgg.training.trainer import Trainer
    from scgg.evaluation.luna_metrics import aggregate_slices

    data_path = Path(data_dir)
    # Stamp the run so multiple training attempts don't overwrite each
    # other and so inference can pin to a specific model version.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        out_dir = _ARTIFACTS_ROOT / data_path.name / "model" / run_timestamp
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "benchmark.log", mode="w"),
        ],
        force=True,
    )
    logger.info(f"Run timestamp: {run_timestamp}")
    logger.info(f"Output dir:    {out_dir}")

    cfg = _load_config(Path(config_path) if config_path else None)

    # CLI overrides
    if epochs is not None:
        cfg["training"]["epochs"] = int(epochs)
    if batch_size is not None:
        cfg["training"]["batch_size"] = int(batch_size)
    if lr is not None:
        cfg["training"]["lr"] = float(lr)
    if wandb is not None:
        cfg["training"]["wandb"] = bool(wandb)
    if wandb_run_name is not None:
        cfg["training"]["wandb_run_name"] = wandb_run_name
    cfg["training"]["checkpoint_dir"] = str(out_dir / "checkpoints")

    if class_stratified_distance is not None:
        cfg.setdefault("training", {}).setdefault("loss", {}).setdefault(
            "distance_regression", {}
        )["class_stratified"] = bool(class_stratified_distance)
        logger.info(
            f"distance_regression.class_stratified overridden via CLI: "
            f"{bool(class_stratified_distance)}"
        )

    if k_default is not None:
        cfg.setdefault("graph", {})["k_default"] = int(k_default)
        logger.info(f"graph.k_default overridden via CLI: {int(k_default)}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {dev}")

    # ---- Load data ------------------------------------------------------
    logger.info(f"Loading LUNA cortex from {data_dir}")
    data = load_luna_cortex(
        data_dir,
        normalize=normalize,
        scale=scale,
        n_top_hvg=n_top_hvg,
    )
    n_genes = data["gene_expr_train"].shape[1]
    logger.info(
        f"Train: {data['gene_expr_train'].shape[0]} cells, "
        f"{len(data['section_map_train'])} sections | "
        f"Test:  {data['gene_expr_test'].shape[0]} cells, "
        f"{len(data['section_map_test'])} sections | "
        f"Genes: {n_genes}"
    )

    rng = np.random.default_rng(seed)
    train_mask, val_mask = _split_train_val(
        data["section_ids_train"], val_fraction, rng
    )

    k_default = int(cfg["graph"]["k_default"])
    cc_train = data.get("cell_class_id_train")
    train_ds = SpatialTranscriptomicsDataset(
        gene_expr=data["gene_expr_train"][train_mask],
        coords=data["coords_train"][train_mask],
        section_ids=data["section_ids_train"][train_mask],
        config=cfg["data"],
        k_values=[k_default],
        is_train=True,
        cell_class=(cc_train[train_mask] if cc_train is not None else None),
    )
    val_ds = None
    if val_mask.any():
        val_ds = SpatialTranscriptomicsDataset(
            gene_expr=data["gene_expr_train"][val_mask],
            coords=data["coords_train"][val_mask],
            section_ids=data["section_ids_train"][val_mask],
            config=cfg["data"],
            k_values=[k_default],
            is_train=False,
            cell_class=(cc_train[val_mask] if cc_train is not None else None),
        )
    if data.get("class_names"):
        logger.info(
            f"Cell-class labels available: {len(data['class_names'])} classes "
            f"({data['class_names'][:5]}{' ...' if len(data['class_names']) > 5 else ''})"
        )

    # ---- Build model + trainer -----------------------------------------
    model = ScGG(n_genes=n_genes, config=cfg)
    logger.info(
        f"Built ScGG (objective={model.objective}) with "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} params"
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=cfg,
        device=dev,
    )

    if load_checkpoint is not None:
        trainer.load_checkpoint(load_checkpoint)
    elif not skip_training:
        t0 = time.time()
        trainer.train()
        logger.info(f"Training finished in {(time.time() - t0) / 60:.1f} min")
    else:
        logger.warning("skip_training=True with no checkpoint — evaluating untrained model")

    # ---- Evaluate on Mouse 2 -------------------------------------------
    logger.info(f"Evaluating on {len(data['section_map_test'])} test slices "
                 f"(Mouse 2, {data['gene_expr_test'].shape[0]} cells)")
    per_slice = _evaluate_split(
        model,
        gene_expr=data["gene_expr_test"],
        coords=data["coords_test"],
        section_ids=data["section_ids_test"],
        cell_class=data["cell_class_test"],
        section_map=data["section_map_test"],
        device=dev,
        contact_percentile=contact_percentile,
        compute_rssd=compute_rssd,
    )

    # ---- Aggregate + write results -------------------------------------
    agg = aggregate_slices(per_slice)

    headline = agg.get("spearman_mean_of_medians", float("nan"))
    luna_reported = 0.448
    logger.info("=" * 72)
    logger.info("LUNA Figure 3 reproduction — aggregated metrics across test slices")
    logger.info("=" * 72)
    for k, v in agg.items():
        logger.info(f"  {k:34s} = {v}")
    logger.info("-" * 72)
    logger.info(
        f"Headline (LUNA-equivalent mean-of-per-slice-median Spearman): "
        f"{headline:.4f}   |   LUNA paper: {luna_reported:.4f}"
    )
    if not np.isnan(headline):
        delta = (headline - luna_reported) * 100
        logger.info(f"Delta vs LUNA: {delta:+.2f} percentage points")

    # CSV
    fieldnames = sorted({k for r in per_slice for k in r.keys()})
    with open(out_dir / "per_slice_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_slice:
            w.writerow(r)
    # JSON aggregate
    with open(out_dir / "aggregate_metrics.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)
    # Config snapshot
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    logger.info(f"Wrote results to {out_dir}")
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Run LUNA Figure 3 reproduction with ScGG")
    p.add_argument(
        "--data_dir", required=True,
        help="Path to per-slice h5ad directory (LUNA cortex split).",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Where to write the trained checkpoint + per-slice / aggregate "
             "metrics. Default derives from --data_dir's basename and adds "
             "a timestamp: /nfs/team361/sb75/scgg-reproducibility/artifacts/"
             "<data_dir_name>/model/<YYYYMMDD_HHMMSS>/.",
    )
    p.add_argument("--config", default=None, help="Optional config YAML override.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda|cpu (auto if omitted)")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--contact_percentile", type=float, default=0.01)
    p.add_argument("--skip_rssd", action="store_true",
                   help="Skip Kabsch RSSD (faster — no PCA projection for embeddings)")
    p.add_argument("--n_top_hvg", type=int, default=None)
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--no_scale", action="store_true")
    p.add_argument("--skip_training", action="store_true")
    p.add_argument("--load_checkpoint", default=None)
    p.add_argument(
        "--class_stratified_distance", action="store_true",
        help="Switch the DistanceRegression loss from all-pairs Pearson to "
             "per-cell-class Pearson averaged across classes. Use this when "
             "the inference UMAP diagnostic shows the embedding has "
             "collapsed to a cell-type classifier (i.e. global Spearman is "
             "decent but the mean of per-class median Spearmans is near zero).",
    )
    p.add_argument(
        "--k_default", type=int, default=None,
        help="k for the GT spatial kNN graph (SupCon positives). Defaults "
             "to the config value (10). Try 30 or 50 in laminar tissues to "
             "force the model to encode tangential position instead of just "
             "depth-band membership.",
    )
    args = p.parse_args()

    run_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        wandb=False if args.no_wandb else None,
        wandb_run_name=args.wandb_run_name,
        val_fraction=args.val_fraction,
        contact_percentile=args.contact_percentile,
        compute_rssd=not args.skip_rssd,
        n_top_hvg=args.n_top_hvg,
        normalize=not args.no_normalize,
        scale=not args.no_scale,
        skip_training=args.skip_training,
        load_checkpoint=args.load_checkpoint,
        class_stratified_distance=(
            True if args.class_stratified_distance else None
        ),
        k_default=args.k_default,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Train scGG on the ABC MERFISH atlas — Zhuang-ABCA-1 (Animal 1).

This is the training step required for LUNA's Section 3.3 reproduction
(de novo reconstruction of scRNA-seq data). LUNA trains on Animal 1 of the
ABC atlas (2.85 M cells across 147 slices, 1,122 genes) and applies the
resulting model to the dissociated scRNA-seq CNS atlas at inference time.

This script:
  1. Loads per-section silver h5ads produced by prepare_abc_silver.py
  2. Builds train (and optional val) datasets, optionally holding out a
     random subset of sections for validation
  3. Instantiates scGG with sensible ABC-scale defaults
  4. Trains for the requested number of epochs and saves checkpoints

Differences from the cortex training:
  * ABC slices are larger (avg ~19 k, max ~30 k cells) than cortex slices
    (max ~7.4 k). With our current cross-attention metric head and
    batch_size=8192, attention is mini-batch-local rather than slice-wide.
    Two options to mitigate:
      a) Increase batch_size to 16384+ if GPU memory permits.
      b) Switch metric_head.type to "mlp" (slightly less expressive but
         O(N) memory).
  * The label vocabulary is larger (27-31 broad classes; 338 subclasses).
    The cell-class auxiliary loss can use any of cell_class / subclass /
    supertype via --label_column.

Usage:
    python scripts/train_scgg_on_abc.py \\
        --silver_dir /nfs/team361/sb75/DATASETS/silver/abc_luna \\
        --output_dir ./results/abc_animal1_v1 \\
        --epochs 200 --batch_size 8192 \\
        --wandb_run_name abc_animal1_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger("scgg.train_abc")


def _load_config(path: Optional[Path]) -> dict:
    if path is not None and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f)
    ref = resources.files("scgg.configs").joinpath("default.yaml")
    return yaml.safe_load(ref.read_text())


def _split_train_val_by_section(
    section_ids: np.ndarray,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(section_ids)
    if val_fraction <= 0 or len(unique) < 2:
        return (
            np.ones(section_ids.shape, dtype=bool),
            np.zeros(section_ids.shape, dtype=bool),
        )
    n_val = max(1, int(round(val_fraction * len(unique))))
    val = rng.choice(unique, size=n_val, replace=False)
    val_mask = np.isin(section_ids, val)
    return ~val_mask, val_mask


def run_training(
    silver_dir: str,
    output_dir: str,
    config_path: Optional[str],
    epochs: int,
    batch_size: int,
    lr: Optional[float],
    seed: int,
    device: Optional[str],
    wandb: Optional[bool],
    wandb_run_name: Optional[str],
    val_fraction: float,
    n_top_hvg: Optional[int],
    normalize: bool,
    scale: bool,
    label_column: str,
    cell_class_aux: bool,
    cell_class_aux_weight: float,
    distance_loss_weight: float,
    metric_head_type: str,
    embed_dim: int,
    n_layers: int,
    n_heads: int,
    sections: Optional[list[str]],
) -> None:
    from scgg.data.abc import load_abc_animal1
    from scgg.data.dataset import SpatialTranscriptomicsDataset
    from scgg.model.scgg import ScGG
    from scgg.training.trainer import Trainer

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "train.log", mode="w"),
        ],
        force=True,
    )

    cfg = _load_config(Path(config_path) if config_path else None)

    # CLI overrides
    cfg["training"]["epochs"] = int(epochs)
    cfg["training"]["batch_size"] = int(batch_size)
    if lr is not None:
        cfg["training"]["lr"] = float(lr)
    if wandb is not None:
        cfg["training"]["wandb"] = bool(wandb)
    if wandb_run_name is not None:
        cfg["training"]["wandb_run_name"] = wandb_run_name
    cfg["training"]["checkpoint_dir"] = str(out_dir / "checkpoints")

    # Model overrides
    mh = cfg["model"].setdefault("metric_head", {})
    mh["type"] = metric_head_type
    mh["embed_dim"] = int(embed_dim)
    if metric_head_type == "cross_attention":
        mh["n_layers"] = int(n_layers)
        mh["n_heads"] = int(n_heads)

    # Loss overrides
    loss_cfg = cfg["training"].setdefault("loss", {})
    loss_cfg.setdefault("distance_regression", {})["weight"] = float(distance_loss_weight)
    loss_cfg["distance_regression"]["enabled"] = True
    if cell_class_aux:
        cc = loss_cfg.setdefault("cell_class_aux", {})
        cc["enabled"] = True
        cc["weight"] = float(cell_class_aux_weight)

    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {dev}")

    # ---- Load ABC silver data ------------------------------------------
    logger.info(f"Loading ABC Animal 1 from {silver_dir}")
    t0 = time.time()
    data = load_abc_animal1(
        silver_dir,
        n_top_hvg=n_top_hvg,
        normalize=normalize,
        scale=scale,
        section_filter=sections,
        label_column=label_column,
    )
    n_genes = data["gene_expr"].shape[1]
    n_sections = len(data["section_map"])
    logger.info(
        f"Loaded in {time.time()-t0:.1f}s: "
        f"{data['gene_expr'].shape[0]:,} cells, "
        f"{n_genes:,} genes, {n_sections} sections"
    )

    rng = np.random.default_rng(seed)
    train_mask, val_mask = _split_train_val_by_section(
        data["section_ids"], val_fraction, rng
    )
    logger.info(
        f"Train/val split (by section): "
        f"{train_mask.sum():,} train cells, {val_mask.sum():,} val cells"
    )

    k_default = int(cfg["graph"]["k_default"])
    cc_id = data.get("cell_class_id")
    train_ds = SpatialTranscriptomicsDataset(
        gene_expr=data["gene_expr"][train_mask],
        coords=data["coords"][train_mask],
        section_ids=data["section_ids"][train_mask],
        config=cfg["data"],
        k_values=[k_default],
        is_train=True,
        cell_class=(cc_id[train_mask] if cc_id is not None else None),
    )
    val_ds = None
    if val_mask.any():
        val_ds = SpatialTranscriptomicsDataset(
            gene_expr=data["gene_expr"][val_mask],
            coords=data["coords"][val_mask],
            section_ids=data["section_ids"][val_mask],
            config=cfg["data"],
            k_values=[k_default],
            is_train=False,
            cell_class=(cc_id[val_mask] if cc_id is not None else None),
        )

    # ---- Build + train -------------------------------------------------
    model = ScGG(n_genes=n_genes, config=cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Built scGG ({model.objective}): {n_params:,} parameters")

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=cfg,
        device=dev,
    )
    logger.info(
        f"use_distance_loss={trainer.use_distance_loss} "
        f"contrastive_w={trainer.contrastive_weight} "
        f"distance_w={trainer.distance_loss_weight} "
        f"use_cellclass_aux={trainer.use_cellclass_aux}"
    )

    # Persist the frozen config and a summary of the data
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(out_dir / "data_summary.json", "w") as f:
        json.dump({
            "silver_dir": silver_dir,
            "n_cells": int(data["gene_expr"].shape[0]),
            "n_genes": int(n_genes),
            "n_sections": int(n_sections),
            "class_names": data.get("class_names"),
            "n_classes": int(data.get("n_classes") or 0),
            "train_cells": int(train_mask.sum()),
            "val_cells": int(val_mask.sum()),
            "label_column": label_column,
            "normalize": normalize,
            "scale": scale,
            # The exact gene order the model was trained on. Critical for
            # downstream inference on data with a different gene panel
            # (e.g. scRNA-seq): the inference script must reorder/pad the
            # query expression matrix to match this order. ABC uses Ensembl
            # gene IDs in var_names; gene_symbols are the human-readable
            # aliases, persisted so inference can map symbol -> Ensembl
            # when a scRNA-seq h5ad uses the symbol namespace.
            "gene_names": list(data.get("gene_names", [])),
            "gene_symbols": list(data.get("gene_symbols") or []),
        }, f, indent=2, default=str)

    t_start = time.time()
    trainer.train()
    elapsed = (time.time() - t_start) / 60.0
    logger.info(f"Training finished in {elapsed:.1f} min")

    # Save final state explicitly (best is already saved as best_model.pt)
    final_path = Path(cfg["training"]["checkpoint_dir"]) / "final_model.pt"
    trainer._save_checkpoint(epoch=int(cfg["training"]["epochs"]) - 1, is_best=False)
    logger.info(f"Final checkpoint: {final_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--silver_dir",
        default="/nfs/team361/sb75/DATASETS/silver/abc_luna",
    )
    p.add_argument(
        "--output_dir",
        default="./results/abc_animal1",
    )
    p.add_argument("--config", default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--val_fraction", type=float, default=0.05,
                   help="Random fraction of TRAINING sections held out for val.")
    p.add_argument("--n_top_hvg", type=int, default=None,
                   help="Optional HVG selection on counts.")
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--no_scale", action="store_true")
    p.add_argument("--label_column", default="cell_class",
                   choices=["cell_class", "subclass", "supertype"],
                   help="Which ABC taxonomy level to use as class label.")
    p.add_argument("--no_cell_class_aux", action="store_true",
                   help="Disable the cell-class auxiliary classifier.")
    p.add_argument("--cell_class_aux_weight", type=float, default=0.5)
    p.add_argument("--distance_loss_weight", type=float, default=1.0)
    p.add_argument("--metric_head_type", default="cross_attention",
                   choices=["mlp", "cross_attention"])
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2,
                   help="Transformer layers in the cross-attention head.")
    p.add_argument("--n_heads", type=int, default=4,
                   help="Attention heads in the cross-attention head.")
    p.add_argument("--sections", default=None,
                   help="Comma-separated subset of section labels (default: all).")
    args = p.parse_args()

    run_training(
        silver_dir=args.silver_dir,
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
        n_top_hvg=args.n_top_hvg,
        normalize=not args.no_normalize,
        scale=not args.no_scale,
        label_column=args.label_column,
        cell_class_aux=not args.no_cell_class_aux,
        cell_class_aux_weight=args.cell_class_aux_weight,
        distance_loss_weight=args.distance_loss_weight,
        metric_head_type=args.metric_head_type,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        sections=([s.strip() for s in args.sections.split(",")] if args.sections else None),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

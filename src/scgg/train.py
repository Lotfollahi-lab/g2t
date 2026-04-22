#!/usr/bin/env python
"""
ScGG Training Script.

Usage:
    python -m scgg.train --config scgg/configs/default.yaml --data_path data/merfish/brain.h5ad

    # Quick test with synthetic data
    python -m scgg.train --synthetic --epochs 10
"""

import argparse
import logging
import yaml
import numpy as np
import torch
from pathlib import Path
from importlib import resources

from scgg.model.scgg import ScGG
from scgg.data.dataset import SpatialTranscriptomicsDataset
from scgg.data.merfish import load_merfish_data
from scgg.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scgg.train")


def generate_synthetic_data(
    n_sections: int = 5,
    n_cells_per_section: int = 2000,
    n_genes: int = 200,
    n_cell_types: int = 5,
    seed: int = 42,
) -> dict:
    """Generate synthetic spatial transcriptomics data for testing.

    Creates a dataset with known spatial patterns:
    - Cell types are arranged in spatial clusters.
    - Gene expression is informative of cell type (and thus spatial location).
    - Some genes are spatially variable (gradient across tissue).
    """
    rng = np.random.default_rng(seed)

    all_expr = []
    all_coords = []
    all_sections = []
    all_cell_types = []

    for sec in range(n_sections):
        n_cells = n_cells_per_section + rng.integers(-200, 200)

        # Generate cell type assignments with spatial clustering
        # Place cell type centers at random locations
        type_centers = rng.uniform(-5, 5, (n_cell_types, 2)).astype(np.float32)

        # Assign cells to types based on spatial proximity
        coords = rng.uniform(-5, 5, (n_cells, 2)).astype(np.float32)
        # Add some structure: cells are drawn from Gaussian clusters
        cell_types = rng.integers(0, n_cell_types, n_cells)
        for i in range(n_cells):
            ct = cell_types[i]
            coords[i] = type_centers[ct] + rng.normal(0, 1.0, 2).astype(np.float32)

        # Generate gene expression based on cell type + spatial position
        # Base expression per cell type
        type_profiles = rng.normal(0, 1, (n_cell_types, n_genes)).astype(np.float32)
        expr = type_profiles[cell_types]

        # Add spatially variable genes (first 20 genes have spatial gradients)
        for g in range(min(20, n_genes)):
            direction = rng.normal(0, 1, 2)
            direction = direction / np.linalg.norm(direction)
            spatial_signal = coords @ direction.astype(np.float32) * 0.5
            expr[:, g] += spatial_signal

        # Add noise
        expr += rng.normal(0, 0.3, expr.shape).astype(np.float32)

        all_expr.append(expr)
        all_coords.append(coords)
        all_sections.append(np.full(n_cells, sec, dtype=np.int64))
        all_cell_types.append(cell_types)

    return {
        "gene_expr": np.concatenate(all_expr, axis=0),
        "coords": np.concatenate(all_coords, axis=0),
        "section_ids": np.concatenate(all_sections, axis=0),
        "gene_names": [f"gene_{i}" for i in range(n_genes)],
        "section_map": {i: f"section_{i}" for i in range(n_sections)},
        "cell_types": np.concatenate(all_cell_types, axis=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Train ScGG")
    parser.add_argument("--config", type=str, default="scgg/configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to .h5ad data file")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--section_col", type=str, default="brain_section_label",
                        help="Column in .obs with section labels")
    parser.add_argument("--cell_type_col", type=str, default="cell_type",
                        help="Column in .obs with cell type annotations")
    parser.add_argument("--seed", type=int, default=42)
    # Wandb overrides
    parser.add_argument("--wandb", action="store_true", default=None,
                        help="Enable wandb logging (overrides config)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging (overrides config)")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Override wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Override wandb run name")
    parser.add_argument("--wandb_tags", nargs="*", default=None,
                        help="Wandb tags (space-separated)")
    args = parser.parse_args()

    # Load config — resolve bundled default.yaml via importlib if user path not found
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
    else:
        # Try to load bundled default config from the installed package
        try:
            config_ref = resources.files("scgg.configs").joinpath("default.yaml")
            config_text = config_ref.read_text()
            config = yaml.safe_load(config_text)
            logger.info("Loaded bundled default config from scgg.configs")
        except Exception as e:
            logger.warning(f"Config {config_path} not found and bundled config failed ({e}), using hardcoded defaults")
            config = {
                "model": {
                    "encoder": {"hidden_dims": [512, 256], "embed_dim": 128, "dropout": 0.1, "norm": "layernorm"},
                    "section_encoder": {"hidden_dim": 128, "embed_dim": 64, "n_attention_heads": 4, "subsample_size": 4096},
                    "velocity_net": {"hidden_dims": [512, 512, 256], "time_embed_dim": 64, "k_embed_dim": 16, "dropout": 0.1},
                    "spatial_dim": 2,
                    "flow": {"sigma_min": 1e-4, "solver": "euler", "n_steps": 100},
                },
                "graph": {"k_default": 10, "k_train_range": [5, 30], "backend": "faiss", "metric": "euclidean"},
                "data": {"coord_normalize": "per_section", "coord_augment_rotation": True, "coord_center": True},
                "training": {
                    "seed": 42, "epochs": 200, "batch_size": 8192, "lr": 1e-3,
                    "weight_decay": 1e-4, "optimizer": "adamw", "scheduler": "cosine",
                    "warmup_epochs": 10, "grad_clip": 1.0,
                    "loss": {"flow_matching": 1.0, "contrastive_spatial": 0.1,
                             "contrastive_temperature": 0.1, "contrastive_n_negatives": 64},
                    "checkpoint_dir": "./checkpoints", "save_every": 10,
                    "log_every": 100, "eval_every": 5, "wandb": True,
                },
                "evaluation": {"k_values": [5, 10, 15, 20], "n_samples": 5},
            }

    # Apply CLI overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr

    # Wandb overrides
    if args.no_wandb:
        config["training"]["wandb"] = False
    elif args.wandb:
        config["training"]["wandb"] = True
    if args.wandb_project is not None:
        config["training"]["wandb_project"] = args.wandb_project
    if args.wandb_run_name is not None:
        config["training"]["wandb_run_name"] = args.wandb_run_name
    if args.wandb_tags is not None:
        config["training"]["wandb_tags"] = args.wandb_tags

    logger.info(f"Wandb logging: {'enabled' if config['training'].get('wandb', False) else 'disabled'}")

    # Seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    if args.synthetic:
        logger.info("Generating synthetic data for testing")
        data = generate_synthetic_data()
    elif args.data_path:
        data = load_merfish_data(
            args.data_path,
            section_col=args.section_col,
            cell_type_col=args.cell_type_col,
            normalize=config["data"].get("normalize", True),
            scale=config["data"].get("scale", True),
            n_top_hvg=config["data"].get("n_top_hvg", None),
        )
    else:
        logger.error("Provide --data_path or --synthetic")
        return

    n_genes = data["gene_expr"].shape[1]
    logger.info(f"Data: {data['gene_expr'].shape[0]} cells, {n_genes} genes")

    # Train/val split by section
    unique_sections = np.unique(data["section_ids"])
    n_sections = len(unique_sections)

    if n_sections >= 3:
        n_val = max(1, n_sections // 5)
        perm = np.random.permutation(unique_sections)
        val_sections = set(perm[:n_val])
        train_sections = set(perm[n_val:])
    elif n_sections == 1:
        # Single section: split cells 80/20 into two pseudo-sections
        logger.info("Single section detected, splitting cells 80/20 for train/val")
        n_cells = len(data["section_ids"])
        perm = np.random.permutation(n_cells)
        split = int(0.8 * n_cells)
        # Re-label: training cells as section 0, val cells as section 1
        new_section_ids = np.zeros(n_cells, dtype=np.int64)
        new_section_ids[perm[split:]] = 1
        data["section_ids"] = new_section_ids
        unique_sections = np.array([0, 1])
        train_sections = {0}
        val_sections = {1}
    else:
        train_sections = set(unique_sections)
        val_sections = set()

    train_mask = np.isin(data["section_ids"], list(train_sections))
    val_mask = np.isin(data["section_ids"], list(val_sections))

    # k values for training
    k_range = config["graph"]["k_train_range"]
    k_values = list(range(k_range[0], k_range[1] + 1, 5))
    if config["graph"]["k_default"] not in k_values:
        k_values.append(config["graph"]["k_default"])

    # Create datasets
    train_dataset = SpatialTranscriptomicsDataset(
        gene_expr=data["gene_expr"][train_mask],
        coords=data["coords"][train_mask],
        section_ids=data["section_ids"][train_mask],
        config=config["data"],
        k_values=k_values,
        is_train=True,
    )

    val_dataset = None
    if val_mask.any():
        val_dataset = SpatialTranscriptomicsDataset(
            gene_expr=data["gene_expr"][val_mask],
            coords=data["coords"][val_mask],
            section_ids=data["section_ids"][val_mask],
            config=config["data"],
            k_values=[config["graph"]["k_default"]],
            is_train=False,
        )

    # Create model
    model = ScGG(n_genes=n_genes, config=config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
    )

    trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
ScGG Evaluation Script.

Loads a trained model and evaluates it on spatial transcriptomics data,
computing graph-level and biological metrics.

Usage:
    python -m scgg.evaluate \
        --checkpoint checkpoints/best_model.pt \
        --data_path data/merfish/brain.h5ad \
        --k 10 \
        --output_dir results/

    # Evaluate with uncertainty quantification (multiple samples)
    python -m scgg.evaluate \
        --checkpoint checkpoints/best_model.pt \
        --data_path data/merfish/brain.h5ad \
        --n_samples 5
"""

import argparse
import logging
import json
import yaml
import numpy as np
import torch
from pathlib import Path

from scgg.model.scgg import ScGG
from scgg.data.merfish import load_merfish_data
from scgg.data.spatial_graph import build_ground_truth_graph
from scgg.evaluation.graph_metrics import evaluate_graph
from scgg.evaluation.bio_metrics import (
    compare_spatial_autocorrelation,
    ligand_receptor_enrichment,
)
from scgg.evaluation.visualization import (
    plot_spatial_embedding,
    plot_graph_comparison,
    plot_edge_confidence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scgg.evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ScGG")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to evaluation .h5ad file")
    parser.add_argument("--k", type=int, default=10,
                        help="kNN parameter for graph construction")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of flow samples (>1 for uncertainty)")
    parser.add_argument("--n_steps", type=int, default=100,
                        help="ODE integration steps")
    parser.add_argument("--solver", type=str, default="euler",
                        choices=["euler", "midpoint", "rk4"])
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--section_col", type=str, default="brain_section_label")
    parser.add_argument("--cell_type_col", type=str, default="cell_type")
    parser.add_argument("--max_sections", type=int, default=None,
                        help="Max sections to evaluate")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Load data
    logger.info(f"Loading evaluation data from {args.data_path}")
    data = load_merfish_data(
        args.data_path,
        section_col=args.section_col,
        normalize=config["data"].get("normalize", True),
        scale=config["data"].get("scale", True),
        n_top_hvg=config["data"].get("n_top_hvg", None),
    )

    n_genes = data["gene_expr"].shape[1]

    # Create model and load weights
    model = ScGG(n_genes=n_genes, config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Get cell types if available
    cell_types = None
    if args.cell_type_col in data["adata"].obs.columns:
        cell_types = data["adata"].obs[args.cell_type_col].astype("category").cat.codes.values

    # Evaluate per section
    unique_sections = np.unique(data["section_ids"])
    if args.max_sections:
        unique_sections = unique_sections[:args.max_sections]

    all_results = []

    for sec_id in unique_sections:
        sec_name = data["section_map"].get(sec_id, str(sec_id))
        logger.info(f"Evaluating section {sec_name} (id={sec_id})")

        mask = data["section_ids"] == sec_id
        sec_expr = data["gene_expr"][mask]
        sec_coords = data["coords"][mask]
        sec_types = cell_types[mask] if cell_types is not None else None

        n_cells_sec = sec_expr.shape[0]
        logger.info(f"  {n_cells_sec} cells")

        # Build ground truth graph
        gt_adj = build_ground_truth_graph(sec_coords, k=args.k, symmetric=True)

        # Generate predicted graph
        gene_expr_tensor = torch.tensor(sec_expr, dtype=torch.float32, device=device)

        if args.n_samples == 1:
            pred_adj = model.generate_graph(
                gene_expr_tensor, k=args.k,
                n_steps=args.n_steps, solver=args.solver,
            )
        else:
            pred_adj, confidence_adj = model.generate_graph(
                gene_expr_tensor, k=args.k,
                n_steps=args.n_steps, solver=args.solver,
                n_samples=args.n_samples,
            )

        # Graph metrics
        graph_results = evaluate_graph(pred_adj, gt_adj, sec_types)

        # Biological metrics
        bio_results = compare_spatial_autocorrelation(
            pred_adj, gt_adj, sec_expr, n_genes=100
        )
        graph_results.update(bio_results)

        # Ligand-receptor enrichment
        lr_results = ligand_receptor_enrichment(
            pred_adj, sec_expr, data["gene_names"]
        )
        graph_results.update(lr_results)

        graph_results["section"] = sec_name
        graph_results["n_cells"] = n_cells_sec
        all_results.append(graph_results)

        logger.info(
            f"  Edge F1: {graph_results['f1']:.4f} | "
            f"Moran's I corr: {graph_results['morans_i_correlation']:.4f}"
        )

        # Generate visualizations for first few sections
        if len(all_results) <= 3:
            # Get predicted embeddings for visualization
            pred_embeddings = model.generate_embeddings(
                gene_expr_tensor, k=args.k, n_steps=args.n_steps
            ).cpu().numpy()

            plot_graph_comparison(
                pred_embeddings, sec_coords, pred_adj, gt_adj,
                cell_types=sec_types,
                save_path=str(output_dir / f"graph_comparison_{sec_name}.png"),
            )

            plot_spatial_embedding(
                pred_embeddings, cell_types=sec_types,
                title=f"Predicted Embedding - {sec_name}",
                save_path=str(output_dir / f"embedding_{sec_name}.png"),
            )

            if args.n_samples > 1:
                plot_edge_confidence(
                    pred_embeddings, confidence_adj,
                    save_path=str(output_dir / f"confidence_{sec_name}.png"),
                )

    # Aggregate results
    summary = {}
    numeric_keys = [k for k in all_results[0] if isinstance(all_results[0][k], (int, float))]
    for key in numeric_keys:
        values = [r[key] for r in all_results]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))

    logger.info("\n=== Evaluation Summary ===")
    for key in ["f1", "precision", "recall", "morans_i_correlation",
                 "composition_cosine", "degree_wasserstein"]:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in summary:
            logger.info(f"  {key}: {summary[mean_key]:.4f} ± {summary[std_key]:.4f}")

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {"summary": summary, "per_section": all_results},
            f, indent=2, default=str,
        )
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

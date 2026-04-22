"""
Biological evaluation metrics for spatial graph quality.

These metrics assess whether the predicted spatial graph preserves
biologically meaningful spatial patterns, beyond pure graph structure.
"""

import numpy as np
from scipy import sparse
from typing import Dict, Optional, List


def spatial_autocorrelation(
    adj: sparse.csr_matrix,
    gene_expr: np.ndarray,
    gene_names: Optional[List[str]] = None,
    n_genes: Optional[int] = None,
) -> Dict[str, float]:
    """Compute Moran's I spatial autocorrelation on the predicted graph.

    Moran's I measures whether a gene's expression is spatially structured
    (high I = spatially autocorrelated, near 0 = random). By computing it
    on the predicted graph and comparing with the ground truth graph, we
    assess whether the predicted spatial structure preserves gene expression
    patterns.

    Args:
        adj: Adjacency matrix (predicted or ground truth).
        gene_expr: Expression matrix, shape (n_cells, n_genes_total).
        gene_names: Gene names for reporting.
        n_genes: Number of genes to evaluate (selects highest-variance genes).

    Returns:
        Dict with 'morans_i_mean', 'morans_i_std', and per-gene values.
    """
    n_cells = adj.shape[0]

    # Weight matrix (row-normalized adjacency)
    W = adj.copy().astype(np.float64)
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums = np.maximum(row_sums, 1e-12)
    W = sparse.diags(1.0 / row_sums) @ W

    # Select genes to evaluate
    if n_genes is not None and gene_expr.shape[1] > n_genes:
        variances = np.var(gene_expr, axis=0)
        top_idx = np.argsort(variances)[-n_genes:]
        gene_expr = gene_expr[:, top_idx]
        if gene_names is not None:
            gene_names = [gene_names[i] for i in top_idx]

    morans_i_values = []
    total_weight = W.sum()

    for g in range(gene_expr.shape[1]):
        x = gene_expr[:, g].astype(np.float64)
        x_mean = x.mean()
        x_centered = x - x_mean
        denominator = np.sum(x_centered ** 2)

        if denominator < 1e-12:
            morans_i_values.append(0.0)
            continue

        # Moran's I = (N / W) * (sum_ij w_ij * (x_i - mean)(x_j - mean)) / sum_i (x_i - mean)^2
        numerator = x_centered @ W @ x_centered
        I = (n_cells / total_weight) * (numerator / denominator)
        morans_i_values.append(float(I))

    morans_i_values = np.array(morans_i_values)

    results = {
        "morans_i_mean": float(np.mean(morans_i_values)),
        "morans_i_std": float(np.std(morans_i_values)),
        "morans_i_median": float(np.median(morans_i_values)),
    }

    return results


def compare_spatial_autocorrelation(
    pred_adj: sparse.csr_matrix,
    true_adj: sparse.csr_matrix,
    gene_expr: np.ndarray,
    n_genes: int = 100,
) -> Dict[str, float]:
    """Compare Moran's I between predicted and ground truth graphs.

    Args:
        pred_adj: Predicted adjacency.
        true_adj: Ground truth adjacency.
        gene_expr: Expression matrix.
        n_genes: Number of top-variance genes to evaluate.

    Returns:
        Dict with correlation and distance of Moran's I values.
    """
    # Select same genes for both
    variances = np.var(gene_expr, axis=0)
    top_idx = np.argsort(variances)[-n_genes:]
    expr_subset = gene_expr[:, top_idx]

    pred_results = spatial_autocorrelation(pred_adj, expr_subset)
    true_results = spatial_autocorrelation(true_adj, expr_subset)

    # Per-gene Moran's I comparison
    pred_morans = []
    true_morans = []

    n_cells = pred_adj.shape[0]

    # Recompute per-gene for correlation
    for adj, values_list in [(pred_adj, pred_morans), (true_adj, true_morans)]:
        W = adj.copy().astype(np.float64)
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums = np.maximum(row_sums, 1e-12)
        W = sparse.diags(1.0 / row_sums) @ W
        total_weight = W.sum()

        for g in range(expr_subset.shape[1]):
            x = expr_subset[:, g].astype(np.float64)
            x_mean = x.mean()
            x_centered = x - x_mean
            denom = np.sum(x_centered ** 2)
            if denom < 1e-12:
                values_list.append(0.0)
                continue
            numer = x_centered @ W @ x_centered
            values_list.append(float((n_cells / total_weight) * (numer / denom)))

    pred_morans = np.array(pred_morans)
    true_morans = np.array(true_morans)

    # Pearson correlation of Moran's I values across genes
    if np.std(pred_morans) > 0 and np.std(true_morans) > 0:
        correlation = np.corrcoef(pred_morans, true_morans)[0, 1]
    else:
        correlation = 0.0

    # L1 distance
    l1_distance = np.mean(np.abs(pred_morans - true_morans))

    return {
        "morans_i_correlation": float(correlation),
        "morans_i_l1_distance": float(l1_distance),
        "pred_morans_i_mean": float(pred_morans.mean()),
        "true_morans_i_mean": float(true_morans.mean()),
    }


def ligand_receptor_enrichment(
    adj: sparse.csr_matrix,
    gene_expr: np.ndarray,
    gene_names: List[str],
    lr_pairs: Optional[List[tuple]] = None,
) -> Dict[str, float]:
    """Evaluate ligand-receptor co-expression enrichment in spatial neighbors.

    For known ligand-receptor pairs, checks whether cells expressing a ligand
    are more likely to neighbor cells expressing the corresponding receptor
    in the predicted graph than expected by chance. This is a biologically
    grounded metric for spatial graph quality.

    Args:
        adj: Adjacency matrix.
        gene_expr: Expression matrix, shape (n_cells, n_genes).
        gene_names: Gene names.
        lr_pairs: List of (ligand, receptor) gene name tuples. If None,
            uses a small set of well-known pairs.

    Returns:
        Dict with enrichment scores.
    """
    if lr_pairs is None:
        # Common ligand-receptor pairs (mouse gene names)
        lr_pairs = [
            ("Wnt5a", "Fzd1"),
            ("Dkk1", "Lrp6"),
            ("Shh", "Ptch1"),
            ("Bmp4", "Bmpr1a"),
            ("Fgf10", "Fgfr2"),
            ("Dll1", "Notch1"),
            ("Pdgfa", "Pdgfra"),
            ("Vegfa", "Flt1"),
            ("Cxcl12", "Cxcr4"),
            ("Sema3a", "Nrp1"),
        ]

    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    adj_csr = adj.tocsr()

    enrichment_scores = []

    for ligand, receptor in lr_pairs:
        if ligand not in gene_name_to_idx or receptor not in gene_name_to_idx:
            continue

        lig_idx = gene_name_to_idx[ligand]
        rec_idx = gene_name_to_idx[receptor]

        lig_expr = gene_expr[:, lig_idx]
        rec_expr = gene_expr[:, rec_idx]

        # Binarize: expressed if above median
        lig_on = lig_expr > np.median(lig_expr)
        rec_on = rec_expr > np.median(rec_expr)

        # For cells expressing ligand, what fraction of neighbors express receptor?
        lig_cells = np.where(lig_on)[0]
        if len(lig_cells) == 0:
            continue

        neighbor_rec_fracs = []
        for i in lig_cells:
            neighbors = adj_csr[i].indices
            if len(neighbors) == 0:
                continue
            frac = rec_on[neighbors].mean()
            neighbor_rec_fracs.append(frac)

        if not neighbor_rec_fracs:
            continue

        observed = np.mean(neighbor_rec_fracs)
        expected = rec_on.mean()  # background rate

        if expected > 0:
            enrichment = observed / expected
        else:
            enrichment = 1.0

        enrichment_scores.append(enrichment)

    return {
        "lr_enrichment_mean": float(np.mean(enrichment_scores)) if enrichment_scores else 1.0,
        "lr_enrichment_std": float(np.std(enrichment_scores)) if enrichment_scores else 0.0,
        "lr_pairs_evaluated": len(enrichment_scores),
    }

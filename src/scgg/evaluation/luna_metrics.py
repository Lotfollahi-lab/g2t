"""
Faithful reimplementations of the three metrics LUNA reports on the MERFISH
mouse primary motor cortex benchmark (Figure 3 + Supplementary Fig. 10):

  1. compute_spearman_correlation — per-cell Spearman of pairwise-distance
     rows, aggregated as (mean, median) over cells per slice.
  2. compute_contact — percentile-thresholded contact precision / F1 on
     flattened pairwise distance matrices.
  3. compute_RSSD — per-cell-class Kabsch-aligned Root Sum Squared Deviation
     using scipy.spatial.transform.Rotation.align_vectors after Z-padding
     coordinates to 3-D.

The implementations mirror LUNA's `metrics/evaluation_statistics.py` and
`utils/data/load.py` (per the references in the public repo). They accept
numpy arrays directly so they can score either a 2-D coordinate prediction
(LUNA, CeLEry, novoSpaRc) or a d-dimensional metric embedding (ScGG
contrastive mode). For Spearman/contact this is dimension-agnostic; RSSD
requires 2-D and a `embedding_to_2d` helper is provided.

Aggregation across slices follows the LUNA paper's Figure 3 convention: the
headline number is the MEAN over test slices of the per-slice MEDIAN of the
per-cell Spearman. Both aggregations (mean of medians, mean of means, median
of medians, median of means) are reported in `aggregate_slices`.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distance matrices
# ---------------------------------------------------------------------------


def compute_distance(coords_or_embed: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix.

    Matches LUNA's ``compute_distance``: ``cdist(pos.T, pos.T)`` on
    ``pos = [coord_X, coord_Y]``. We accept an (N, d) array directly so this
    is dimension-agnostic — for a ScGG metric embedding (d=32) the row-wise
    Spearman ranking is unchanged by the dim because Spearman is rank-based.

    Args:
        coords_or_embed: (N, d) array of coordinates or embeddings.

    Returns:
        (N, N) pairwise Euclidean distance matrix (float32).
    """
    x = np.asarray(coords_or_embed, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected (N, d) array, got shape {x.shape}")
    return cdist(x, x).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Spearman
# ---------------------------------------------------------------------------


def compute_spearman_correlation(
    coords_true: np.ndarray,
    coords_pred: np.ndarray,
    progress: bool = False,
) -> Dict[str, np.ndarray | float]:
    """Per-cell Spearman correlation of pairwise-distance rows.

    LUNA's exact formulation: build the full NxN Euclidean pairwise distance
    matrix for both prediction and ground truth; for each cell i, take row i
    of the true matrix and row i of the predicted matrix (length-N distance
    vectors) and compute the Spearman rank correlation between them.

    Args:
        coords_true: (N, 2) ground-truth 2-D coordinates.
        coords_pred: (N, d) predicted coordinates OR embeddings (d arbitrary).
        progress: emit a tqdm progress bar over cells if True.

    Returns:
        Dict with:
          per_cell:  (N,) per-cell Spearman values (NaNs for degenerate cells)
          per_cell_p:(N,) p-values
          mean:      float, mean over cells
          median:    float, median over cells
    """
    coords_true = np.asarray(coords_true, dtype=np.float32)
    coords_pred = np.asarray(coords_pred, dtype=np.float32)
    if coords_true.shape[0] != coords_pred.shape[0]:
        raise ValueError(
            f"row count mismatch: true has {coords_true.shape[0]} cells, "
            f"pred has {coords_pred.shape[0]}"
        )

    n = coords_true.shape[0]
    dt = compute_distance(coords_true)
    dp = compute_distance(coords_pred)

    rng = range(n)
    if progress:
        try:
            from tqdm import tqdm
            rng = tqdm(rng, desc="per-cell Spearman")
        except ImportError:
            pass

    rho = np.empty(n, dtype=np.float64)
    pval = np.empty(n, dtype=np.float64)
    for i in rng:
        r, p = spearmanr(dt[i], dp[i])
        rho[i] = r
        pval[i] = p

    rho_ok = rho[~np.isnan(rho)]
    return {
        "per_cell": rho,
        "per_cell_p": pval,
        "mean": float(rho_ok.mean()) if rho_ok.size else float("nan"),
        "median": float(np.median(rho_ok)) if rho_ok.size else float("nan"),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Contact precision / F1
# ---------------------------------------------------------------------------


def compute_contact(
    coords_true: np.ndarray,
    coords_pred: np.ndarray,
    percentile: float = 0.01,
) -> Dict[str, float]:
    """Percentile-thresholded contact precision / F1.

    Matches LUNA's ``compute_contact``: flatten the pairwise-distance matrices
    of both true and predicted, drop the zero (self) entries, threshold each
    at its own ``percentile`` quantile, treat "below threshold" as a contact,
    and compute precision and F1.

    Because both sides are thresholded at the same percentile of their own
    distribution, predicted-positive count == true-positive count → precision
    == recall == F1. (This is why LUNA's CSVs show identical precision and F1
    values to 15 decimals.)

    Args:
        coords_true: (N, 2) ground-truth coordinates.
        coords_pred: (N, d) predicted coordinates or embeddings.
        percentile: fraction of all pairs treated as "in contact" (default 0.01).

    Returns:
        Dict with 'precision', 'f1', 'recall', 'percentile'.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    dt = compute_distance(coords_true).reshape(-1)
    dp = compute_distance(coords_pred).reshape(-1)

    # LUNA filters out entries where both are zero (drops self-pairs and
    # any double zeros). We replicate exactly.
    nonzero = np.nonzero(dt * dp)[0]
    dt = dt[nonzero]
    dp = dp[nonzero]

    if dt.size == 0:
        return {
            "precision": float("nan"), "f1": float("nan"),
            "recall": float("nan"), "percentile": percentile,
        }

    t_th = np.quantile(dt, percentile)
    p_th = np.quantile(dp, percentile)

    labels = (dt < t_th).astype(np.int8)
    preds = (dp < p_th).astype(np.int8)

    if labels.sum() == 0 or preds.sum() == 0:
        return {
            "precision": 0.0, "f1": 0.0, "recall": 0.0, "percentile": percentile
        }

    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "percentile": percentile,
    }


# ---------------------------------------------------------------------------
# RSSD (Kabsch / Root Sum Squared Deviation)
# ---------------------------------------------------------------------------


def _filter_nan_inf_pair(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop rows where EITHER input has NaN/Inf, preserving row correspondence.

    LUNA's reference code filters `a` and `b` independently, which silently
    breaks the row alignment if only one has a NaN row. We always use the
    intersection mask so `a[i]` and `b[i]` stay paired.
    """
    mask = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    return a[mask], b[mask]


def compute_kabsch_rssd(
    coords_true_2d: np.ndarray,
    coords_pred_2d: np.ndarray,
) -> float:
    """Kabsch-aligned Root Sum Squared Deviation on 2-D coordinates.

    Pads the input to 3-D with Z=0 (LUNA's convention), filters NaN/Inf rows
    (using a joint mask to preserve row correspondence), then calls
    scipy.spatial.transform.Rotation.align_vectors which performs the
    Kabsch–Umeyama optimal rotation and returns the RSSD.

    Args:
        coords_true_2d: (N, 2).
        coords_pred_2d: (N, 2). Must have the same N as coords_true_2d.

    Returns:
        RSSD = sqrt(sum_i ||R · p_pred,i - p_true,i||^2), or +inf if SVD
        fails to converge.
    """
    a = np.asarray(coords_true_2d, dtype=np.float64)
    b = np.asarray(coords_pred_2d, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.shape[1] != 2:
        raise ValueError(f"expected (N, 2), got {a.shape}")

    a = np.pad(a, ((0, 0), (0, 1)), mode="constant")  # -> (N, 3)
    b = np.pad(b, ((0, 0), (0, 1)), mode="constant")
    a, b = _filter_nan_inf_pair(a, b)
    if a.shape[0] == 0:
        return float("inf")

    try:
        _rot, rssd, _sens = R.align_vectors(a, b, return_sensitivity=True)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(rssd)


def compute_RSSD(
    coords_true_2d: np.ndarray,
    coords_pred_2d: np.ndarray,
    cell_class: Optional[Sequence] = None,
) -> Dict[str, float]:
    """LUNA's compute_RSSD on a single slice.

    LUNA reports three numbers per slice:
      * absolute_rssd: Kabsch RSSD over ALL cells together.
      * sum_rssd:      sum over per-cell-class Kabsch RSSDs.
      * mean_rssd:     mean over per-cell-class Kabsch RSSDs.

    The per-class variants align each cell class independently — useful when
    rotation/translation invariance is class-specific. If cell_class is None,
    only absolute_rssd is computed; sum_rssd / mean_rssd are returned as NaN.

    Args:
        coords_true_2d: (N, 2) ground truth.
        coords_pred_2d: (N, 2) prediction.
        cell_class: optional length-N sequence of class labels for the
            per-class flavor.

    Returns:
        Dict with 'absolute_rssd', 'sum_rssd', 'mean_rssd', 'n_classes'.
    """
    out: Dict[str, float] = {}
    out["absolute_rssd"] = compute_kabsch_rssd(coords_true_2d, coords_pred_2d)

    if cell_class is None:
        out["sum_rssd"] = float("nan")
        out["mean_rssd"] = float("nan")
        out["n_classes"] = 0
        return out

    cls = np.asarray(cell_class)
    rssds: List[float] = []
    for c in np.unique(cls):
        mask = cls == c
        if mask.sum() < 3:
            # Kabsch needs at least 3 non-collinear points; LUNA effectively
            # skips degenerate classes via filter_nan_inf shrinking the set.
            continue
        rssds.append(
            compute_kabsch_rssd(coords_true_2d[mask], coords_pred_2d[mask])
        )
    rssds = [r for r in rssds if np.isfinite(r)]
    if not rssds:
        out["sum_rssd"] = float("nan")
        out["mean_rssd"] = float("nan")
        out["n_classes"] = 0
    else:
        out["sum_rssd"] = float(np.sum(rssds))
        out["mean_rssd"] = float(np.mean(rssds))
        out["n_classes"] = int(len(rssds))
    return out


# ---------------------------------------------------------------------------
# Embedding -> 2-D helper (only needed for RSSD on metric embeddings)
# ---------------------------------------------------------------------------


def embedding_to_2d(embedding: np.ndarray, method: str = "pca") -> np.ndarray:
    """Project a d-dimensional embedding to 2-D for RSSD.

    Spearman and contact metrics are dimension-agnostic and do NOT need this
    step (they operate on rank-of-distance vectors). Only RSSD requires 2-D
    because the Kabsch alignment must operate in the same space as the
    ground truth.

    Args:
        embedding: (N, d) embedding.
        method: 'pca' (fast, linear) or 'mds' (distance-preserving, slower).

    Returns:
        (N, 2) projection.
    """
    x = np.asarray(embedding, dtype=np.float32)
    if x.shape[1] == 2:
        return x

    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(x).astype(np.float32)
    if method == "mds":
        from sklearn.manifold import MDS
        return MDS(
            n_components=2, dissimilarity="euclidean", normalized_stress="auto"
        ).fit_transform(x).astype(np.float32)
    raise ValueError(f"unknown projection method: {method!r}")


# ---------------------------------------------------------------------------
# Per-slice evaluation + across-slice aggregation
# ---------------------------------------------------------------------------


def evaluate_slice(
    coords_true: np.ndarray,
    coords_pred: np.ndarray,
    cell_class: Optional[Sequence] = None,
    contact_percentile: float = 0.01,
    compute_rssd: bool = True,
    rssd_projection: str = "pca",
) -> Dict[str, float]:
    """Compute all three LUNA metrics on one slice.

    Args:
        coords_true: (N, 2) ground-truth 2-D coordinates.
        coords_pred: (N, d) prediction (2-D coords OR d-dim metric embedding).
            If d > 2 and compute_rssd is True, the embedding is projected to
            2-D via `rssd_projection` for the RSSD computation only; Spearman
            and contact use the embedding directly.
        cell_class: optional per-cell class labels for per-class RSSD.
        contact_percentile: percentile threshold for `compute_contact`.
        compute_rssd: skip the RSSD computation if False (no projection).
        rssd_projection: 'pca' or 'mds' if the embedding is not 2-D.

    Returns:
        Flat dict of metrics for this slice.
    """
    out: Dict[str, float] = {}

    spr = compute_spearman_correlation(coords_true, coords_pred)
    out["spearman_per_cell_mean"] = spr["mean"]
    out["spearman_per_cell_median"] = spr["median"]

    contact = compute_contact(coords_true, coords_pred, contact_percentile)
    out["precision"] = contact["precision"]
    out["f1"] = contact["f1"]
    out["recall"] = contact["recall"]
    out["contact_percentile"] = contact["percentile"]

    if compute_rssd:
        if coords_pred.shape[1] == 2:
            pred2d = coords_pred
        else:
            pred2d = embedding_to_2d(coords_pred, method=rssd_projection)
        rssd = compute_RSSD(coords_true, pred2d, cell_class)
        out["absolute_rssd"] = rssd["absolute_rssd"]
        out["sum_rssd"] = rssd["sum_rssd"]
        out["mean_rssd"] = rssd["mean_rssd"]
        out["n_classes_rssd"] = rssd["n_classes"]
        out["rssd_projection"] = rssd_projection

    out["n_cells"] = int(coords_true.shape[0])
    return out


def aggregate_slices(
    per_slice: Iterable[Dict[str, float]],
) -> Dict[str, float]:
    """Aggregate per-slice metric dicts across all test slices.

    LUNA Figure 3 reports the mean across slices of the per-slice MEDIAN
    Spearman (44.8 % for LUNA). We additionally report the other three
    aggregations so the user can cross-check against any variant.

    The function silently passes through any keys it doesn't recognize.
    """
    rows = list(per_slice)
    if not rows:
        return {}

    def _stack(key: str) -> np.ndarray:
        return np.array(
            [r[key] for r in rows if key in r and not np.isnan(r[key])],
            dtype=np.float64,
        )

    out: Dict[str, float] = {}

    # Spearman aggregations — the headline LUNA number is mean_of_medians.
    med = _stack("spearman_per_cell_median")
    mean = _stack("spearman_per_cell_mean")
    if med.size:
        out["spearman_mean_of_medians"] = float(med.mean())     # LUNA Fig.3
        out["spearman_median_of_medians"] = float(np.median(med))
        out["spearman_std_of_medians"] = float(med.std())
    if mean.size:
        out["spearman_mean_of_means"] = float(mean.mean())
        out["spearman_median_of_means"] = float(np.median(mean))

    for key in ("precision", "f1", "recall",
                "absolute_rssd", "sum_rssd", "mean_rssd"):
        v = _stack(key)
        if v.size:
            out[f"{key}_mean"] = float(v.mean())
            out[f"{key}_median"] = float(np.median(v))
            out[f"{key}_std"] = float(v.std())

    n = _stack("n_cells")
    if n.size:
        out["total_cells"] = int(n.sum())
        out["n_slices"] = int(n.size)

    return out

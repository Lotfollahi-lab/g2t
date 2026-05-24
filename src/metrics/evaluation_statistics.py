from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy.linalg import svd
from utils.data.load import compute_distance, to_dataframe


# Utility Functions


def align_point_clouds(base, target):
    """Align target to base using Procrustes analysis (rotation only)."""
    # Ensure data are centered at the origin
    base_centered = base - np.mean(base, axis=0)
    target_centered = target - np.mean(target, axis=0)

    # SVD for rotation matrix
    U, _, Vt = svd(np.dot(target_centered.T, base_centered))
    R = np.dot(U, Vt)  # Calculate the rotation matrix

    # Apply rotation to the target
    aligned_target = np.dot(target_centered, R)
    return aligned_target + np.mean(base, axis=0)  # Re-add the mean of the base


def filter_nan_inf(array: np.ndarray) -> np.ndarray:
    """Replace NaN and Inf values with zeros."""
    nan_mask = np.isnan(array) | np.isinf(array)
    array[nan_mask] = 0
    return array


def compute_kabsch_rotation(
    metadata_true: np.ndarray, metadata_pred: np.ndarray
) -> Tuple[object, float, float]:
    """Apply the Kabsch algorithm to compute the optimal rotation.

    Robustness fixes vs the upstream LUNA implementation:

    1. **Degenerate-set guard.** ``compute_RSSD`` calls this function
       per cell-class subset of a slice (see
       ``compute_RSSD`` below). Rare cell types can have only 0 or
       1 cell in a given slice, but ``R.align_vectors`` needs at
       least 2 vector pairs to fit a rotation. We short-circuit
       those cases to ``NaN`` rather than crashing the whole test
       phase.

    2. **No sensitivity matrix.** The original code passed
       ``return_sensitivity=True`` to ``align_vectors`` and then
       discarded the ``sens`` output at every call site. Requesting
       it forces a stricter N>=2 + finite-weight precondition
       inside scipy and was the root cause of the
       "Cannot return sensitivity matrix with an infinite weight
       or one vector pair" ValueError. Dropping that flag is a
       free fix.

    3. **Broader exception catch.** ``np.linalg.LinAlgError`` only
       covers SVD non-convergence; the upstream code missed other
       scipy ValueErrors (degenerate vectors, NaN weights, etc.).
       Catch both, return ``+inf`` so the downstream NaN/Inf
       filter in ``compute_RSSD`` drops the slice from the
       aggregate.
    """
    # Guard 1: empty or single-vector input — Kabsch is undefined.
    n_true = int(np.asarray(metadata_true).shape[0])
    n_pred = int(np.asarray(metadata_pred).shape[0])
    if n_true < 2 or n_pred < 2:
        return np.eye(3), float("nan"), float("nan")
    try:
        # Guard 2: drop return_sensitivity=True — sens is unused at
        # every call site and demanding it adds a scipy precondition
        # we don't need.
        rot, rssd = R.align_vectors(metadata_true, metadata_pred)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Kabsch alignment failed ({type(e).__name__}: {e}); "
              f"returning +inf to be filtered downstream.")
        return np.eye(3), float("inf"), float("inf")
    return rot, float(rssd), float("nan")


def prepare_metadata_for_kabsch(
    metadata_true: pd.DataFrame, metadata_pred: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare metadata for Kabsch algorithm by padding Z axis."""
    metadata_true_filtered = np.pad(
        metadata_true[["coord_X", "coord_Y"]].to_numpy(), ((0, 0), (0, 1)), mode="constant"
    )
    metadata_pred_filtered = np.pad(
        metadata_pred[["coord_X", "coord_Y"]].to_numpy(), ((0, 0), (0, 1)), mode="constant"
    )

    metadata_true_filtered = filter_nan_inf(metadata_true_filtered)
    metadata_pred_filtered = filter_nan_inf(metadata_pred_filtered)

    return metadata_true_filtered, metadata_pred_filtered


# Core Computations


def compute_spearman_correlation(
    metadata_true: pd.DataFrame, metadata_pred: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Computes Spearman correlation between two sets of metadata."""
    n = len(metadata_true)
    true_distances = compute_distance(metadata_true)
    pred_distances = compute_distance(metadata_pred)

    spearman_corr, spearman_p = [], []

    for i in tqdm(range(n)):
        corr, pval = spearmanr(true_distances[i], pred_distances[i])
        spearman_corr.append(corr)
        spearman_p.append(pval)

    spr_v = np.array(spearman_corr)
    spr_p = np.array(spearman_p)
    spr_avg = np.mean(spr_v)
    spr_median = np.median(spr_v)

    return spr_v, spr_p, spr_avg, spr_median


def compute_contact(
    distances_true: np.ndarray, distances_pred: np.ndarray, percentile: float
) -> Tuple[float, float]:
    """Computes precision and F1 scores based on the given percentile."""
    labels = distances_true.flatten()
    predictions = distances_pred.flatten()

    nonzero_indices = np.nonzero(labels * predictions)
    labels = labels[nonzero_indices]
    predictions = predictions[nonzero_indices]

    labels_threshold = np.quantile(labels, percentile)
    predictions_threshold = np.quantile(predictions, percentile)

    labels = labels < labels_threshold
    predictions = predictions < predictions_threshold

    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)

    return precision, f1


def compute_lisi(
    X: np.ndarray,
    metadata: pd.DataFrame,
    label_colnames: Iterable[str],
    perplexity: float = 30,
) -> np.ndarray:
    """Computes the Local Inverse Simpson Index (LISI) for each column in metadata."""
    n_cells = metadata.shape[0]
    n_labels = len(label_colnames)
    knn = NearestNeighbors(n_neighbors=int(perplexity * 3), algorithm="kd_tree").fit(X)
    distances, indices = knn.kneighbors(X)

    indices = indices[:, 1:]
    distances = distances[:, 1:]

    lisi_df = np.zeros((n_cells, n_labels))

    for i, label in enumerate(label_colnames):
        labels = pd.Categorical(metadata[label])
        n_categories = len(labels.categories)
        simpson = compute_simpson(
            distances.T, indices.T, labels, n_categories, perplexity
        )
        lisi_df[:, i] = 1 / simpson

    return lisi_df


def compute_kabsch_algorithm(
    metadata_true: pd.DataFrame, metadata_pred: pd.DataFrame
) -> Tuple[np.ndarray, float, float]:
    """Compute the Kabsch algorithm for aligning two sets of vectors."""
    metadata_true_filtered, metadata_pred_filtered = prepare_metadata_for_kabsch(
        metadata_true, metadata_pred
    )
    return compute_kabsch_rotation(metadata_true_filtered, metadata_pred_filtered)


def compute_RSSD(
    metadata_true: pd.DataFrame, metadata_pred: pd.DataFrame
) -> Tuple[float, float, float, float]:
    """Compute RSSD metrics comparing the true and predicted dataframes."""
    metadata_true_dict, metadata_pred_dict = {}, {}
    num_graph = (
        1
        if isinstance(metadata_true, pd.DataFrame)
        else metadata_true.positions.shape[0]
    )

    if isinstance(metadata_true, pd.DataFrame) and isinstance(
        metadata_pred, pd.DataFrame
    ):
        num_graph = 1
        metadata_pred_dict[0] = metadata_pred
        metadata_true_dict[0] = metadata_true
    else:
        num_graph = metadata_true.positions.shape[0]
        for i in range(num_graph):
            metadata_true_dict[i] = to_dataframe(
                metadata_true.cell_class[i].squeeze().detach().cpu().numpy(),
                metadata_true.positions[i].squeeze().detach().cpu().numpy(),
                metadata_true.cell_ID[i].squeeze().detach().cpu().numpy(),
            )
            metadata_pred_dict[i] = to_dataframe(
                metadata_true.cell_class[i].squeeze().detach().cpu().numpy(),
                metadata_pred.positions[i].squeeze().detach().cpu().numpy(),
                metadata_true.cell_ID[i].squeeze().detach().cpu().numpy(),
            )

    classes_rsd, num_cells_per_class = [], []

    for i in range(num_graph):
        rot, absolute_rssd, _ = compute_kabsch_algorithm(
            metadata_true_dict[i], metadata_pred_dict[i]
        )
        classes = set(metadata_true_dict[i]["cell_class"])

        for c in classes:
            metadata_true_c = metadata_true_dict[i][metadata_true_dict[i]["cell_class"] == c]
            metadata_pred_c = metadata_pred_dict[i][metadata_pred_dict[i]["cell_class"] == c]
            _, rssd, _ = compute_kabsch_algorithm(metadata_true_c, metadata_pred_c)
            classes_rsd.append(rssd)
            num_cells_per_class.append(len(metadata_true_c))

    # Filter NaN / Inf out of the aggregation. compute_kabsch_rotation
    # returns NaN for degenerate cell-class subsets (1 or 0 cells)
    # and Inf for SVD failure; either would poison np.sum / np.mean
    # if left in. Falling back to NaN for whole-slice RSSD when ALL
    # classes are degenerate (genuinely uninformative slice).
    classes_rsd_valid = [
        r for r in classes_rsd
        if r is not None and not (np.isnan(r) or np.isinf(r))
    ]
    if classes_rsd_valid:
        sum_rssd = float(np.sum(classes_rsd_valid))
        mean_rssd = float(np.mean(classes_rsd_valid))
    else:
        sum_rssd = float("nan")
        mean_rssd = float("nan")

    return sum_rssd, mean_rssd, absolute_rssd


# Supporting Functions


def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    n_categories: int,
    perplexity: float,
    tol: float = 1e-5,
) -> np.ndarray:
    """Computes Simpson's index for LISI."""
    n = distances.shape[1]
    simpson = np.zeros(n)
    logU = np.log(perplexity)

    for i in range(n):
        beta = 1
        betamin, betamax = -np.inf, np.inf
        P = np.exp(-distances[:, i] * beta)
        P_sum = np.sum(P)

        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
            P /= P_sum
        Hdiff = H - logU

        for _ in range(50):
            if abs(Hdiff) < tol:
                break
            if Hdiff > 0:
                betamin = beta
                beta = 2 * beta if not np.isfinite(betamax) else (beta + betamax) / 2
            else:
                betamax = beta
                beta = beta / 2 if not np.isfinite(betamin) else (beta + betamin) / 2

            P = np.exp(-distances[:, i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
                P /= P_sum
            Hdiff = H - logU

        if H == 0:
            simpson[i] = -1
        for label_category in labels.categories:
            ix = indices[:, i]
            q = labels[ix] == label_category
            if np.any(q):
                simpson[i] += np.sum(P[q]) ** 2

    return simpson

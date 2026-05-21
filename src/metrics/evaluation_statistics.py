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
) -> Tuple[np.ndarray, float, float]:
    """Apply the Kabsch algorithm to compute the optimal rotation."""
    try:
        rot, rssd, sens = R.align_vectors(
            metadata_true, metadata_pred, return_sensitivity=True
        )
        
    except np.linalg.LinAlgError:
        print("SVD did not converge.")
        return +np.inf, +np.inf, +np.inf
    return rot, rssd, sens


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

    sum_rssd = np.sum(classes_rsd)
    mean_rssd = np.mean(classes_rsd)

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

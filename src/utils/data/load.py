import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.spatial.distance import cdist

from utils.data.dataholder import DataHolder


def create_anndata(
    metadata_true: pd.DataFrame, metadata_pred: pd.DataFrame
) -> sc.AnnData:
    """
    Creates an AnnData object from given metadata.

    Parameters:
    metadata_true (pd.DataFrame): DataFrame containing the true metadata.
    metadata_pred (pd.DataFrame): DataFrame containing the predicted metadata.

    Returns:
    sc.AnnData: The created AnnData object.
    """
    true_points = np.column_stack(
        (metadata_true["coord_X"].values, metadata_true["coord_Y"].values)
    )
    pred_points = np.column_stack(
        (metadata_pred["coord_X"].values, metadata_pred["coord_Y"].values)
    )

    adata = sc.AnnData(X=pred_points)
    adata.obs["cell_class"] = pd.Categorical(metadata_true["c"])
    adata.obsm["true_points"] = true_points
    adata.obsm["spatial"] = pred_points
    return adata


def clean_data(data: np.array) -> np.array:
    """
    Cleans the given dataset by replacing NaN and infinite values with zeros.

    Parameters:
    data (np.array): A NumPy array containing the dataset.

    Returns:
    np.array: The cleaned dataset with only finite values.
    """
    # Check for NaN values and replace them
    if np.isnan(data).any():
        data = np.nan_to_num(data)

    # Check for infinite values and replace them
    if np.isinf(data).any():
        data = np.where(np.isinf(data), 0, data)

    return data

def standardise_dataframe_colnames(raw_data):
    """
    Standardizes column names in the DataFrame according to the dataset name.

    This function renames columns in the provided DataFrame based on the dataset name.
    For example, for the 'merfish' dataset, it renames certain columns to standard names
    like 'x', 'y', and 'c'.

    Args:
        dataset_name (str): Name of the dataset.
        raw_data (pd.DataFrame): The raw data DataFrame with original column names.

    Returns:
        pd.DataFrame: The DataFrame with standardized column names.
    """
    raw_data = raw_data.rename(
        columns={
            "x": "coord_X", # x coordinate
            "y": "coord_Y", # y coordinate
            "subclass": "cell_class", # cell class
            "region": "cell_section", # cell section
        }
    )
    return raw_data

def character_to_int(input: pd.DataFrame, uniques: list) -> tuple[list, dict]:
    """
    Converts categorical data in a DataFrame to integer values based on a list of unique categories.

    This function creates two mappings: one for converting class labels to integers (class_to_int)
    and another for the reverse mapping (int_to_class). It then converts the input DataFrame's
    class labels into corresponding integer values.

    Parameters:
    input (pd.DataFrame): A pandas DataFrame containing categorical data.
    uniques (list): A list of unique class labels.

    Returns:
    tuple:
        - A list of integer values representing the class labels in the input DataFrame.
    - A dictionary mapping integer values back to the original class labels.
    """
    # Create a mapping from class labels to integers
    class_to_int = {label: i for i, label in enumerate(uniques)}

    # Create a reverse mapping from integers to class labels
    int_to_class = {i: label for i, label in enumerate(uniques)}

    # Convert class labels in the input DataFrame to integers
    cell_class_integer = [
        class_to_int[label] if label in class_to_int else None for label in input
    ]

    return cell_class_integer, int_to_class



def log2_norm(gene_names: list, raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies log2 normalization to specified columns in the DataFrame.

    Args:
        gene_names (list of str): The column names in the DataFrame to normalize.
        raw_data (pd.DataFrame): The DataFrame containing the gene expression data.

    Returns:
        pd.DataFrame: The DataFrame with normalized specified columns.
    """
    raw_data[gene_names] = raw_data[gene_names].apply(lambda x: np.log2(1 + x))
    return raw_data


def detect_nan_rows(pos: np.ndarray) -> np.ndarray:
    """
    Detects rows with NaN values in the position.

    Args:
        pos (torch.Tensor): A 2D tensor in which to detect NaN values.

    Returns:
        torch.Tensor: A boolean tensor where True indicates the presence of NaN in the row.
    """
    return torch.isnan(pos).any(dim=1)


def position_normalize(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the given positions to a range between -0.5 and 0.5.

    Args:
        pos (np.ndarray): An array of positions to be normalized.

    Returns:
        np.ndarray: The normalized positions.
    """

    for key in ["coord_X", "coord_Y"]:
        if "cell_section" in input_data.columns:
            groups = input_data.groupby("cell_section")[key]
            min_, max_ = groups.transform("min"), groups.transform("max")
        else:
            min_, max_ = input_data[key].min(), input_data[key].max()
        input_data[key] = (input_data[key] - min_) / (max_ - min_) - 0.5
    return input_data


def to_dataframe(cell_class: list, position: np.ndarray, index=None) -> pd.DataFrame:
    """
    Converts cell class labels and their positions into a pandas DataFrame.

    This function creates a DataFrame with columns for cell class and x, y
    coordinates of the cell positions.

    Args:
        cell_class (list): The class labels of the cells.
        position (np.ndarray): The x, y coordinates of the cells.

    Returns:
        pd.DataFrame: A DataFrame containing the cell class and positions.
    """
    if index is None:
        index = range(len(cell_class))
    metadata = pd.DataFrame(
        data={
            "cell_class": cell_class,
            "coord_X": position[:, 0],
            "coord_Y": position[:, 1],
        },
        index=index,
    )
    return metadata


def cell_class_decoding(batch: DataHolder, cell_class_decoder: dict) -> list:
    """
    Decodes the cell classes from the given cell class array.

    This function takes a numpy array of cell classes and decodes them into a list of cell class characters.

    Args:
        batch (DataHolder): A batch of data.
        cell_class_decoder (dict): A dictionary mapping cell class integers to characters.

    Returns:
        list: A list of cell class characters.
    """
    cell_class = batch.cell_class[batch.node_mask]
    cell_class_int = cell_class.squeeze().cpu().numpy()
    cell_class_character = [cell_class_decoder[item] for item in cell_class_int]

    return cell_class_character


def remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    Remove mean from x based on node_mask.

    Parameters:
        - x (torch.Tensor): Input tensor of shape (bs x n x d).
        - node_mask (torch.Tensor): Boolean mask tensor of shape (bs x n).

    Returns:
        torch.Tensor: Tensor with mean removed based on the node_mask.

    Raises:
        AssertionError: If the dtype of node_mask is not torch.bool.
        AssertionError: If the absolute sum of masked values is greater than 1e-5.
    """
    assert node_mask.dtype == torch.bool, f"Wrong type {node_mask.dtype}"

    # Expand node_mask dimensions
    node_mask = node_mask.unsqueeze(-1)

    # Calculate the absolute sum of masked values
    masked_max_abs_value = (x * (~node_mask)).abs().sum().item()

    # Check if the absolute sum is within the acceptable range
    assert masked_max_abs_value < 1e-5, f"Error {masked_max_abs_value} too high"

    # Calculate the count of unmasked nodes
    N = node_mask.sum(1, keepdims=True)

    # Calculate the mean along the second dimension
    mean = torch.sum(x, dim=1, keepdim=True) / N

    # Subtract mean from x for masked nodes
    x = x - mean * node_mask

    return x


def remove_last_set_of_duplicates(arr: np.ndarray) -> np.ndarray:
    """
    Removes the last set of duplicate items based on the first element of each sub-item in an array.

    This function iterates through the array and keeps adding unique items (based on the first element)
    to the result. When a duplicate is found, it stops and returns the accumulated items.

    Args:
        arr (np.ndarray): An array of arrays or tuples, where duplicates are determined by the first element.

    Returns:
        np.ndarray: An array with the last set of duplicates removed.
    """
    result = []
    # Iterate through the array in reverse order
    for item in arr:
        if item[0] not in [r[0] for r in result]:
            result.append(item)
        else:
            break

    return np.vstack(result)


def compute_distance(metadata: pd.DataFrame) -> np.ndarray:
    """
    Computes the pairwise distance matrix for the given metadata.

    This function takes a pandas DataFrame containing 'x' and 'y' coordinates of points
    and calculates the pairwise Euclidean distance between each pair of points.

    Args:
        metadata (pd.DataFrame): A DataFrame containing 'x' and 'y' coordinates of points.

    Returns:
        np.ndarray: A 2D array representing the pairwise distance matrix of the points.
    """
    # Convert metadata to numpy array of positions
    pos = np.array([metadata["coord_X"], metadata["coord_Y"]])

    # Compute and return the pairwise distance matrix
    return cdist(pos.T, pos.T)

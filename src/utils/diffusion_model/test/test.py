import json
import os
import pathlib
import time

import numpy as np
import pandas as pd
import wandb
import torch
from metrics.evaluation_plot import plot_scatter_visualization
from metrics.evaluation_statistics import (
    compute_contact,
    compute_RSSD,
    align_point_clouds,
    compute_spearman_correlation,
)
from utils.data.dataholder import DataHolder
from utils.data.load import (
    cell_class_decoding,
    compute_distance,
    position_normalize,
    to_dataframe,
)
from utils.data.misc import setup_wandb, to_batch
from utils.diffusion_model.sample.sample import sample_graphs


def on_test_epoch_start_func(self) -> None:
    """
    Callback function called at the start of the test.
    """
    if self.local_rank == 0:
        setup_wandb(self.cfg)


def test_step_func(self, data: DataHolder, i: int) -> None:
    """
    Executes a test step for a single data batch.
    """
    batches = to_batch(data)
    root_dir = get_root_dir(self.cfg)
    test_save_path = get_test_save_path(root_dir, self.cfg)
    batch_size = len(batches.positions)

    # Evaluate on all test batches
    for index in range(batch_size):
        process_single_batch(self, batches, index, test_save_path)


def on_test_epoch_end_func(self) -> None:
    """
    Callback function called at the end of the test.
    """
    pass


def get_root_dir(cfg):
    """
    Determines the root directory for saving the evaluation results.
    """
    if cfg.test.save_dir is None:
        return cfg.test.checkpoints_parent_dir
    else:
        root_dir = cfg.test.test_save_parent_path
        os.makedirs(root_dir, exist_ok=True)
        return root_dir


def get_test_save_path(root_dir, cfg):
    """
    Creates a directory for saving the evaluation results for the test epoch.
    """
    epoch_index = cfg.test.epoch_index
    model_name = cfg.test.checkpoint_path.split("/")[-3].split("_")[0]
    test_save_path = os.path.join(
        root_dir, f"model_{model_name}_epoch_{str(epoch_index)}"
    )
    os.makedirs(test_save_path, exist_ok=True)
    return test_save_path


def process_single_batch(self, batches, index, test_save_path):
    """
    Processes a single batch, samples predictions, and saves evaluation results.
    """
    batch = batches.get_batch(index=index, batch_size=1)
    positions_pred_whole = sample_model_predictions(self, batch)

    for sample_index, positions_pred in enumerate(positions_pred_whole):
        process_single_sample(self, batch, positions_pred, test_save_path, sample_index)


def sample_model_predictions(self, batch):
    """
    Samples predictions from the diffusion model.
    """
    start_time = time.time()
    positions_pred_whole = sample_graphs(self, batch=batch, test=True)
    sample_time = time.time() - start_time
    print(f"Sampling on one graph took {sample_time} seconds.")
    return positions_pred_whole


def process_single_sample(self, batch, positions_pred, test_save_path, sample_index):
    """
    Processes a single sample, decodes cell classes, evaluates, and saves results.
    """
    positions_pred = positions_pred.cpu().numpy()
    positions_true = batch.positions[0].cpu().numpy()
    node_mask = batch.node_mask[0].cpu().numpy()
    cell_ID = batch.cell_ID[0].cpu().numpy()

    positions_pred, positions_true, cell_ID = mask_positions(
        positions_pred, positions_true, cell_ID, node_mask
    )

    mapping_dict, cell_class_decoder = get_mappings(self)
    corresponding_region = get_corresponding_region(mapping_dict, positions_pred)

    save_dir = get_unique_dir(test_save_path, corresponding_region)

    cell_class = cell_class_decoding(batch, cell_class_decoder)

    metadata_true, metadata_pred_normalized, unique_cell_classes = prepare_data(
        positions_pred, positions_true, cell_ID, cell_class, save_dir
    )

    log_dict = perform_evaluation(
        self.cfg.test.epoch_index,
        metadata_true,
        metadata_pred_normalized,
        unique_cell_classes,
        save_dir,
    )
    log_dict.update(
        {
            "checkpoint_path": self.cfg.test.checkpoint_path,
            "save_idx": f"{corresponding_region}_{sample_index}",
        }
    )

    file_path = os.path.join(test_save_path, f"{self.cfg.general.name}_results.csv")
    append_to_dataframe(log_dict, file_path)


def mask_positions(positions_pred, positions_true, cell_ID, node_mask):
    """
    Masks and filters positions and cell IDs based on the node mask.
    """
    positions_pred = positions_pred[node_mask]
    positions_true = positions_true[node_mask]
    cell_ID = cell_ID.squeeze()[node_mask.squeeze()]
    return positions_pred, positions_true, cell_ID


def get_mappings(self):
    """
    Retrieves mappings for regions and cell class decoders.
    """
    mapping_dict = self.dataset_infos.num_cell_to_region_mapping_dict
    cell_class_decoder = self.dataset_infos.cell_class_decoder
    return mapping_dict, cell_class_decoder


def get_corresponding_region(mapping_dict, positions_pred):
    """
    Determines the corresponding region based on the size of predicted positions.
    """
    try:
        return mapping_dict[positions_pred.shape[0]]
    except KeyError:
        return "unknown"


def get_unique_dir(test_save_path, corresponding_region):
    """
    Creates a unique directory for saving evaluation results.
    """
    result = 0
    dir_path = os.path.join(test_save_path, f"{corresponding_region}_{result}")
    while os.path.exists(dir_path):
        result += 1
        dir_path = os.path.join(test_save_path, f"{corresponding_region}_{result}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def prepare_data(positions_pred, positions_true, cell_ID, cell_class, dir_path):
    """
    Prepares and saves the data for analysis, converting to DataFrames and normalizing.
    """
    positions_pred = positions_pred[: len(positions_true)]
    cell_class = cell_class[: len(positions_true)]

    metadata_true = to_dataframe(cell_class, positions_true, index=cell_ID)
    metadata_pred_unnormalized = to_dataframe(cell_class, positions_pred, index=cell_ID)
    metadata_pred = position_normalize(metadata_pred_unnormalized.copy())

    metadata_true.fillna(0, inplace=True)
    metadata_pred.fillna(0, inplace=True)

    save_dataframes(metadata_true, metadata_pred, dir_path)

    unique_cell_classes = metadata_true["cell_class"].unique()
    unique_cell_classes.sort()

    return metadata_true, metadata_pred, unique_cell_classes


def save_dataframes(metadata_true, metadata_pred_normalized, dir_path):
    """
    Saves true and predicted metadata to CSV files.
    """
    metadata_true.to_csv(os.path.join(dir_path, "metadata_true.csv"))
    metadata_pred_normalized.to_csv(os.path.join(dir_path, "metadata_pred.csv"))


def append_to_dataframe(log_dict, file_path):
    """
    Appends log data to an existing DataFrame or creates a new one.
    """
    df = pd.DataFrame([log_dict])

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        updated_df = df

    updated_df.to_csv(file_path, index=False)


def perform_evaluation(
    epoch, metadata_true, metadata_pred_normalized, unique_cell_classes, dir_path
):
    """
    Evaluates model predictions using various metrics and generates a log dictionary.
    """
    plot_scatter_visualization(
        metadata_true, metadata_pred_normalized, unique_cell_classes, dir_path
    )

    spr_v, spr_p, spr_avg, spr_median = compute_spearman_correlation(
        metadata_true, metadata_pred_normalized
    )
    sum_rssd, mean_rssd, absolute_rssd = compute_RSSD(
        metadata_true, metadata_pred_normalized
    )

    distances_true = compute_distance(metadata_true)
    distances_pred = compute_distance(metadata_pred_normalized)
    precision, f1 = compute_contact(distances_true, distances_pred, percentile=0.1)

    log_dict = {
        "test/epoch": epoch,
        "test/precision": precision,
        "test/Spearman's Rank Correlation (Median)": spr_median,
        "test/Spearman's Rank Correlation (Average)": spr_avg,
        "test/F1": f1,
        "test/rssd_absolute": absolute_rssd,
        "test/mean_rssd": mean_rssd,
        "test/sum_rssd": sum_rssd,
        "num_cells": len(metadata_true),
    }
    return log_dict

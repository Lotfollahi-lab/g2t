"""
PyTorch dataset for spatial transcriptomics data.

Handles section-based sampling, coordinate augmentation (random rotation,
centering, normalization), and precomputation of ground truth spatial graphs.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from typing import Optional, Dict, List, Tuple, Any
import logging

from .spatial_graph import build_ground_truth_graph

logger = logging.getLogger(__name__)


class SpatialTranscriptomicsDataset(Dataset):
    """Dataset that samples batches of cells from tissue sections.

    Each __getitem__ returns a batch of cells from a single section,
    along with the ground truth spatial graph for that section.

    This section-based sampling ensures:
    1. Spatial graphs are within-section (no cross-section edges).
    2. Section-level embeddings are meaningful.
    3. Coordinate augmentation is applied per-section.

    Args:
        gene_expr: Expression matrix, shape (n_cells, n_genes).
        coords: Spatial coordinates, shape (n_cells, 2).
        section_ids: Section labels, shape (n_cells,).
        config: Data configuration dict.
        k_values: List of k values for precomputing ground truth graphs.
        is_train: Whether this is a training set (enables augmentation).
    """

    def __init__(
        self,
        gene_expr: np.ndarray,
        coords: np.ndarray,
        section_ids: np.ndarray,
        config: dict,
        k_values: List[int] = [10],
        is_train: bool = True,
        cell_class: Optional[np.ndarray] = None,
    ):
        self.gene_expr = gene_expr
        self.raw_coords = coords.copy()
        self.section_ids = section_ids
        self.config = config
        self.is_train = is_train
        # Optional per-cell integer class label (e.g., cortex cell type).
        # -1 = unknown / out-of-vocabulary; downstream aux loss masks these out.
        self.cell_class = (
            np.asarray(cell_class, dtype=np.int64) if cell_class is not None else None
        )
        if self.cell_class is not None:
            # Number of classes = max id + 1 (ignoring -1 unknowns).
            valid = self.cell_class[self.cell_class >= 0]
            self.n_classes = int(valid.max()) + 1 if valid.size else 0
        else:
            self.n_classes = 0

        # Build section index
        self.sections = np.unique(section_ids)
        self.section_indices = {
            s: np.where(section_ids == s)[0] for s in self.sections
        }

        # Normalize coordinates per section
        self.coords = self._normalize_coords(coords, section_ids)

        # Precompute ground truth graphs for each section and k value
        logger.info(f"Precomputing ground truth graphs for k={k_values}")
        self.gt_graphs: Dict[Tuple[int, int], sparse.csr_matrix] = {}
        for s in self.sections:
            idx = self.section_indices[s]
            sec_coords = self.raw_coords[idx]  # use raw coords for GT graph
            for k in k_values:
                if len(idx) > k:
                    self.gt_graphs[(s, k)] = build_ground_truth_graph(
                        sec_coords, k=k, symmetric=True
                    )

        self.k_values = k_values

    def _normalize_coords(
        self, coords: np.ndarray, section_ids: np.ndarray
    ) -> np.ndarray:
        """Normalize coordinates.

        Modes:
          * ``per_section`` (legacy): zero-mean / unit-std per section,
            using a SCALAR std across both axes (preserves aspect ratio).
          * ``per_section_minmax`` (LUNA-aligned, recommended): per
            section, scale EACH AXIS independently to [-0.5, 0.5] via
            ``(x - min) / (max - min) - 0.5``. Matches LUNA's
            ``utils.data.load.position_normalize``. The model's prior
            ``N(0, I)`` covers this box comfortably; the slice's aspect
            ratio is squashed to 1, which is fine because the model is
            trained against per-axis min-maxed targets.
          * ``global``: zero-mean / unit-std across the whole dataset.
        """
        normalized = coords.copy()
        mode = self.config.get("coord_normalize", "per_section")

        if mode == "per_section":
            for s in np.unique(section_ids):
                mask = section_ids == s
                sec_coords = coords[mask]
                center = sec_coords.mean(axis=0)
                std = sec_coords.std()
                if std > 0:
                    normalized[mask] = (sec_coords - center) / std
                else:
                    normalized[mask] = sec_coords - center
        elif mode == "per_section_minmax":
            for s in np.unique(section_ids):
                mask = section_ids == s
                sec_coords = coords[mask]
                mn = sec_coords.min(axis=0)
                mx = sec_coords.max(axis=0)
                rng = mx - mn
                # Guard against degenerate axes (rng == 0).
                rng = np.where(rng > 0, rng, 1.0)
                normalized[mask] = (sec_coords - mn) / rng - 0.5
        elif mode == "global":
            center = coords.mean(axis=0)
            std = coords.std()
            normalized = (coords - center) / std if std > 0 else coords - center

        return normalized.astype(np.float32)

    def __len__(self) -> int:
        """Number of sections (each section is one 'sample')."""
        return len(self.sections)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a section's data.

        Returns a dict with all cells from one section, including gene
        expression, coordinates, and metadata. The training loop handles
        batching within sections.
        """
        section_id = self.sections[idx]
        cell_indices = self.section_indices[section_id]

        expr = self.gene_expr[cell_indices]
        coords = self.coords[cell_indices].copy()

        # Coordinate augmentation during training
        if self.is_train:
            if self.config.get("coord_center", True):
                coords = coords - coords.mean(axis=0)

            if self.config.get("coord_augment_rotation", True):
                angle = np.random.uniform(0, 2 * np.pi)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
                coords = coords @ rot.T

        # Select a random k for this section
        k = np.random.choice(self.k_values)
        gt_graph_key = (section_id, k)

        section_class = (
            torch.tensor(self.cell_class[cell_indices], dtype=torch.long)
            if self.cell_class is not None
            else None
        )

        out = {
            "gene_expr": torch.tensor(expr, dtype=torch.float32),
            "coords": torch.tensor(coords, dtype=torch.float32),
            "section_id": section_id,
            "cell_indices": torch.tensor(cell_indices, dtype=torch.long),
            "k_target": k,
            "gt_graph_key": gt_graph_key,
        }
        if section_class is not None:
            out["cell_class"] = section_class
        return out


def collate_section_batch(batch: List[Dict]) -> Dict:
    """Custom collate that keeps sections separate (no padding/stacking).

    Since sections have different numbers of cells, we can't stack them
    into a single tensor. Instead, return a list of section data.
    """
    # For single section (batch_size=1 in DataLoader)
    if len(batch) == 1:
        return batch[0]

    # For multiple sections, return list
    return batch


def create_cell_batches(
    section_data: Dict,
    batch_size: int,
    shuffle: bool = True,
) -> List[Dict]:
    """Split a section's cells into mini-batches for training.

    Args:
        section_data: Output of SpatialTranscriptomicsDataset.__getitem__.
        batch_size: Maximum cells per mini-batch.
        shuffle: Whether to shuffle cells.

    Returns:
        List of mini-batch dicts, each containing gene_expr, coords,
        cell_indices (within section), and k_target.
    """
    n_cells = section_data["gene_expr"].shape[0]

    if shuffle:
        perm = torch.randperm(n_cells)
    else:
        perm = torch.arange(n_cells)

    batches = []
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        idx = perm[start:end]
        entry = {
            "gene_expr": section_data["gene_expr"][idx],
            "coords": section_data["coords"][idx],
            "batch_indices_in_section": idx,
            "k_target": section_data["k_target"],
            "section_id": section_data["section_id"],
            "gt_graph_key": section_data["gt_graph_key"],
        }
        if "cell_class" in section_data:
            entry["cell_class"] = section_data["cell_class"][idx]
        batches.append(entry)

    return batches

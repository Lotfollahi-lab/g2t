import torch


def to_device(tensor, device):
    """
    Moves the tensor to the specified device if the tensor is not None.
    """
    return tensor.to(device) if tensor is not None else None


def apply_mask(tensor, mask, mask_dim=-1):
    """
    Applies a mask to a tensor by broadcasting the mask along the last dimension.
    """
    return tensor * mask.unsqueeze(mask_dim) if tensor is not None else None


def center_positions(positions, mask):
    """
    Centers the positions by subtracting the mean for the masked nodes.
    """
    for i in range(positions.shape[0]):
        masked_positions = positions[i][mask[i]]
        positions[i][mask[i]] = masked_positions - masked_positions.mean(dim=0)
    return positions


class DataHolder:
    def __init__(
        self,
        positions: torch.Tensor,
        node_features: torch.Tensor,
        diffusion_time: int,
        cell_ID=None,
        cell_class=None,
        t_int=None,
        t=None,
        node_mask=None,
    ) -> None:
        """
        Initializes a DataHolder object.
        """
        self.positions = positions
        self.node_features = node_features
        self.cell_class = cell_class
        self.cell_ID = cell_ID
        self.t_int = t_int
        self.t = t
        self.diffusion_time = diffusion_time if diffusion_time is not None else t
        self.node_mask = node_mask

    def device_as(self, node_features: torch.Tensor) -> "DataHolder":
        """
        Adjusts the device of all tensors to match `node_features`.
        """
        device = node_features.device
        self.positions = to_device(self.positions, device)
        self.node_features = to_device(self.node_features, device)
        self.cell_class = to_device(self.cell_class, device)
        self.cell_ID = to_device(self.cell_ID, device)
        return self

    def mask(self, node_mask=None) -> "DataHolder":
        """
        Applies a node mask to node features and positions.
        """
        if node_mask is None:
            assert self.node_mask is not None
            node_mask = self.node_mask

        node_features_mask = node_mask.unsqueeze(-1)  # bs, n, 1

        self.node_features = apply_mask(self.node_features, node_mask)
        self.positions = apply_mask(self.positions, node_mask)
        if self.positions is not None:
            self.positions = center_positions(self.positions, node_mask)

        self.cell_class = apply_mask(self.cell_class, node_mask)
        self.cell_ID = apply_mask(self.cell_ID, node_mask)

        return self

    def collapse(self) -> "DataHolder":
        """
        Collapses the node_features tensor by applying argmax and sets masked values to -1.
        """
        copy = self.copy()
        copy.node_features = torch.argmax(self.node_features, dim=-1)
        copy.node_features[self.node_mask == 0] = -1
        return copy

    def copy(self) -> "DataHolder":
        """
        Creates a copy of the DataHolder object.
        """
        return DataHolder(
            positions=self.positions.clone() if self.positions is not None else None,
            node_features=self.node_features.clone()
            if self.node_features is not None
            else None,
            cell_class=self.cell_class.clone() if self.cell_class is not None else None,
            diffusion_time=self.diffusion_time,
            cell_ID=self.cell_ID.clone() if self.cell_ID is not None else None,
            t_int=self.t_int,
            t=self.t,
            node_mask=self.node_mask.clone() if self.node_mask is not None else None,
        )

    def get_batch(batches: "DataHolder", index: int, batch_size: int) -> "DataHolder":
        """
        Extracts a single batch from the batches object.
        """
        extract = lambda x: x[index].unsqueeze(0) if batch_size == 1 else x[index]

        dense_data = DataHolder(
            node_features=extract(batches.node_features),
            positions=extract(batches.positions),
            node_mask=extract(batches.node_mask),
            cell_ID=extract(batches.cell_ID) if batches.cell_ID is not None else None,
            cell_class=extract(batches.cell_class),
            diffusion_time=None,  # Adjust as needed
        )
        return dense_data

    def __repr__(self) -> str:
        """
        Returns a string representation of the DataHolder object.
        """
        return (
            f"positions: {self.positions.shape if isinstance(self.positions, torch.Tensor) else self.positions} -- "
            + f"node_features: {self.node_features.shape if isinstance(self.node_features, torch.Tensor) else self.node_features} -- "
            + f"cell_class: {self.cell_class.shape if isinstance(self.cell_class, torch.Tensor) else self.cell_class} -- "
            + f"cell_ID: {self.cell_ID.shape if isinstance(self.cell_ID, torch.Tensor) else self.cell_ID} -- "
            + f"t_int: {self.t_int} -- "
            + f"t: {self.t} -- "
            + f"node_mask: {self.node_mask.shape if isinstance(self.node_mask, torch.Tensor) else self.node_mask}"
        )

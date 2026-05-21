import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from utils.data.load import remove_mean_with_mask


class MLP(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        # Define the first hidden layer
        self.hidden_layer1 = nn.Linear(input_feature_dim, hidden_dim1)
        self.hidden_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))

        # Apply the output layer
        x = self.output_layer(x)

        return x


class PositionsMLP(nn.Module):
    def __init__(self, hidden_dim: int, eps=1e-5) -> None:
        """
        Initializes a PositionsMLP model.

        This model is designed to transform input positions using a Multi-Layer Perceptron (MLP)
        applied to the Euclidean norm of the input positions. It is particularly useful for tasks
        involving node positions in a graph-based neural network.

        Parameters:
        - hidden_dim (int): The number of hidden units in the MLP layers.
        - eps (float, optional): A small epsilon value to avoid division by zero. Default is 1e-5.
        """
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, pos: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the PositionsMLP model.

        Parameters:
        - pos (torch.Tensor): Input tensor representing positions.
          Shape: (batch_size, num_nodes, 3), where 3 corresponds to the x, y, z coordinates.
        - node_mask (torch.Tensor): Binary mask indicating active nodes.
          Shape: (batch_size, num_nodes).

        Returns:
        - torch.Tensor: Transformed positions based on the MLP model.
          Shape: (batch_size, num_nodes, 3), representing the updated x, y, z coordinates.
        """
        norm = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        new_norm = self.mlp(norm)  # bs, n, 1
        new_pos = pos * new_norm / (norm + self.eps)

        new_pos = new_pos * node_mask.unsqueeze(-1)
        new_pos = remove_mean_with_mask(new_pos, node_mask)
        return new_pos


class PositionNorm(nn.Module):
    def __init__(self, eps: float = 1e-5, device=None, dtype=None) -> None:
        """
        Initializes a PositionNorm layer.

        This layer performs normalization on input positions. It scales positions based on the mean norm of active
        nodes, allowing for a normalized representation.

        Parameters:
        - eps (float, optional): A small epsilon value to avoid division by zero. Default is 1e-5.
        - device (str, optional): Device on which the layer's parameters and computations should be placed.
        - dtype (torch.dtype, optional): Data type of the layer's parameters.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.normalized_shape = (1,)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets the weight parameters of the layer to ones.
        """
        init.ones_(self.weight)

    def forward(self, pos: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the PositionNorm layer.

        Parameters:
        - pos (torch.Tensor): Input tensor representing positions.
        - node_mask (torch.Tensor): Binary mask indicating active nodes.
          Shape: (batch_size, num_nodes).

        Returns:
        - torch.Tensor: Normalized positions based on the mean norm of active nodes.
        """
        norm = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        mean_norm = torch.sum(norm, dim=1, keepdim=True) / torch.sum(
            node_mask, dim=1, keepdim=True
        )  # bs, 1, 1

        # Note that self.weight is a parameter of shape (1, 1, 1) and is broadcasted to (bs, 1, 1)
        new_pos = self.weight * pos / (mean_norm + self.eps)
        return new_pos

    def extra_repr(self) -> str:
        """
        Returns a string representation of the layer's extra attributes.
        """
        return "{normalized_shape}, eps={eps}".format(**self.__dict__)


def masked_softmax(x, mask, **kwargs):
    if torch.sum(mask) == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)

import torch.nn as nn

from models.layers import PositionsMLP
from models.transformer import TransformerLayer
from utils.data.dataholder import DataHolder
import torch


class Model(nn.Module):
    """
    Model class for the neural network architecture.

    Attributes:
        n_layers (int): Number of transformer layers.
        input_dimensions_node_features (int): Input dimensions for node features.
        input_dimensions_diffusion_time (int): Input dimensions for diffusion time.
        output_dimensions_node_features (int): Output dimensions for node features.
        output_dimensions_diffusion_time (int): Output dimensions for diffusion time.
        mlp_in_node_features (nn.Sequential): MLP for processing input node features.
        mlp_in_diffusion_time (nn.Sequential): MLP for processing input diffusion time.
        mlp_in_position (PositionsMLP): MLP for processing input positions.
        transformer_layers (nn.ModuleList): List of TransformerLayer instances.
        mlp_out_node_features (nn.Sequential): MLP for processing output node features.
        mlp_out_pos (PositionsMLP): MLP for processing output positions.

    Methods:
        __init__(input_dims, n_layers: int, hidden_mlp_dims: dict, hidden_dims: dict, output_dims)
        forward(data: DataHolder) -> DataHolder
    """

    def __init__(
        self,
        input_dims,
        n_layers: int,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims,
        positionMLP_eps: float = 1e-9,
    ) -> None:
        """
        Constructor to initialize the Model instance.

        Args:
            input_dims: Input dimensions.
            n_layers (int): Number of transformer layers.
            hidden_mlp_dims (dict): Dimensions for hidden MLP layers.
            hidden_dims (dict): Dimensions for hidden layers.
            output_dims: Output dimensions.

        Returns:
            None
        """
        super().__init__()
        self.n_layers = n_layers
        self.input_dimensions_node_features = input_dims["node_features_dimensions"]
        self.input_dimensions_diffusion_time = input_dims["diffusion_time_dimensions"]
        self.output_dimensions_node_features = output_dims["node_features_dimensions"]
        self.output_dimensions_diffusion_time = output_dims["diffusion_time_dimensions"]
        self.positionMLP_eps = positionMLP_eps

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()

        # MLP for processing input node features
        self.mlp_in_node_features = nn.Sequential(
            nn.Linear(self.input_dimensions_node_features, hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
        )

        # MLP for processing input diffusion time
        self.mlp_in_diffusion_time = nn.Sequential(
            nn.Linear(self.input_dimensions_diffusion_time, hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
        )

        # MLP for processing input positions
        self.mlp_in_position = PositionsMLP(hidden_mlp_dims["pos"])

        # List of TransformerLayer instances
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    node_features_dimensions=hidden_dims["dx"],
                    diffusion_time_dimensions=hidden_dims["dy"],
                    delta_dimensions=hidden_dims["dd"],
                    num_heads=hidden_dims["num_heads"],
                    dim_ff_node_features=hidden_dims["dim_ffX"],
                    dim_ff_diffusion_time=hidden_dims["dim_ffy"],
                    last_layer=False,
                )
                for _ in range(n_layers)
            ]
        )

        # MLP for processing output node features
        self.mlp_out_node_features = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["output_features_to_pos_dims"]),
        )

        self.mlp_out_pos_norm = nn.Sequential(
            nn.Linear(
                hidden_dims["output_features_to_pos_dims"] + 3, hidden_mlp_dims["X"]
            ),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], 1),
        )

        # MLP for processing output positions
        self.mlp_out_pos = PositionsMLP(hidden_mlp_dims["pos"])

    def forward(self, data: DataHolder) -> DataHolder:
        """
        Forward pass of the neural network.

        Args:
            data (DataHolder): Input data.

        Returns:
            DataHolder: Output data.
        """
        node_mask = data.node_mask
        node_features = data.node_features
        diffusion_time = data.diffusion_time
        positions = data.positions

        add_diffusion_time_to_out = diffusion_time[
            ..., : self.output_dimensions_diffusion_time
        ]

        # Process input features using MLPs
        transformed_features = DataHolder(
            node_features=self.mlp_in_node_features(node_features),
            diffusion_time=self.mlp_in_diffusion_time(diffusion_time),
            positions=self.mlp_in_position(positions, node_mask),
            node_mask=node_mask,
        ).mask()

        # Apply transformer layers
        for layer in self.transformer_layers:
            transformed_features = layer(transformed_features)

        # Process output features using MLPs
        transformed_node_features = self.mlp_out_node_features(
            transformed_features.node_features
        )

        pos = transformed_features.positions
        norm = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        new_norm = self.mlp_out_pos_norm(
            torch.cat([transformed_node_features, pos, norm], dim=-1)
        )  # bs, n, 1
        new_pos = pos * new_norm / (norm + self.positionMLP_eps)

        new_pos = new_pos * node_mask.unsqueeze(-1)
        new_pos = new_pos - torch.mean(new_pos, dim=1, keepdim=True)
        pos = new_pos

        # Add input features to output
        transformed_node_features = transformed_node_features
        diffusion_time = add_diffusion_time_to_out

        # Create output DataHolder
        out = DataHolder(
            node_features=transformed_node_features,
            diffusion_time=diffusion_time,
            positions=pos,
            node_mask=node_mask,
        ).mask()

        return out

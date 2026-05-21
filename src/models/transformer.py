import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from models.layers import PositionNorm
from models.self_attention import SelfAttention
from utils.data.dataholder import DataHolder


class TransformerLayer(nn.Module):
    """
    Transformer that updates node, edge, and global features.

    Parameters:
    - node_features_dimensions (int): Dimensionality of node features.
    - diffusion_time_dimensions (int): Dimensionality of diffusion time information.
    - num_heads (int): Number of attention heads in the multi-head attention.
    - dim_ff_node_features (int): Dimension of the feedforward network model for node features after self-attention.
    - dim_ff_diffusion_time (int): Dimension of the feedforward network model for diffusion time after self-attention.
    - dropout (float): Dropout probability. 0 to disable.
    - layer_norm_eps (float): Epsilon value in layer normalizations.
    - device: Device for the model parameters.
    - dtype: Data type for the model parameters.
    - last_layer (bool): Flag indicating whether this layer is the last layer in the model.
    """

    def __init__(
        self,
        node_features_dimensions: int,
        delta_dimensions: int,
        diffusion_time_dimensions: int,
        num_heads: int,
        dim_ff_node_features: int = 2048,
        dim_ff_diffusion_time: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
        last_layer=False,
    ) -> None:
        """
        Initialize TransformerLayer.

        Parameters:
        - node_features_dimensions (int): Dimensionality of node features.
        - diffusion_time_dimensions (int): Dimensionality of diffusion time information.
        - num_heads (int): Number of attention heads in the multi-head attention.
        - dim_ff_node_features (int): Dimension of the feedforward network model for node features after self-attention.
        - dim_ff_diffusion_time (int): Dimension of the feedforward network model for diffusion time after self-attention.
        - dropout (float): Dropout probability. 0 to disable.
        - layer_norm_eps (float): Epsilon value in layer normalizations.
        - device: Device for the model parameters.
        - dtype: Data type for the model parameters.
        - last_layer (bool): Flag indicating whether this layer is the last layer in the model.
        """
        kw = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = SelfAttention(
            node_features_dimensions=node_features_dimensions,
            delta_dimensions=delta_dimensions,
            diffusion_time_dimensions=diffusion_time_dimensions,
            num_heads=num_heads,
            last_layer=last_layer,
        )

        self.lin_node_features_1 = Linear(
            node_features_dimensions, dim_ff_node_features, **kw
        )
        self.lin_node_features_2 = Linear(
            dim_ff_node_features, node_features_dimensions, **kw
        )
        self.norm_node_features_1 = LayerNorm(
            node_features_dimensions, eps=layer_norm_eps, **kw
        )
        self.norm_node_features_2 = LayerNorm(
            node_features_dimensions, eps=layer_norm_eps, **kw
        )
        self.dropout_node_features_1 = Dropout(dropout)
        self.dropout_node_features_2 = Dropout(dropout)
        self.dropout_node_features_3 = Dropout(dropout)

        self.norm_positions_1 = PositionNorm(eps=1e-8, **kw)

        self.last_layer = last_layer

        if not last_layer:
            self.lin_diffusion_time_1 = Linear(
                diffusion_time_dimensions, dim_ff_diffusion_time, **kw
            )
            self.lin_diffusion_time_2 = Linear(
                dim_ff_diffusion_time, diffusion_time_dimensions, **kw
            )
            self.norm_diffusion_time_1 = LayerNorm(
                diffusion_time_dimensions, eps=layer_norm_eps, **kw
            )
            self.norm_diffusion_time_2 = LayerNorm(
                diffusion_time_dimensions, eps=layer_norm_eps, **kw
            )
            self.dropout_diffusion_time_1 = Dropout(dropout)
            self.dropout_diffusion_time_2 = Dropout(dropout)
            self.dropout_diffusion_time_3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, features: DataHolder) -> DataHolder:
        """
        Pass the input through the transformer layer.

        Again the capacity makes no sense. Why is it so much?

        Parameters:
        - features (Dataholder): Placeholder object containing node features, diffusion time, positions, and node mask.

        Returns:
        - Dataholder: Updated node features, diffusion time, positions, and node mask.
        """
        node_features = features.node_features
        diffusion_time = features.diffusion_time
        positions = features.positions
        node_mask = features.node_mask
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1

        (
            transformed_features,
            transformed_diffusion_time,
            transformed_position,
        ) = self.self_attn(
            node_features=node_features,
            diffusion_time=diffusion_time,
            positions=positions,
            node_mask=node_mask,
        )

        # Transform Positions
        transformed_position = self.norm_positions_1(transformed_position, x_mask)

        # Transform Node Features
        transformed_features = self.dropout_node_features_1(transformed_features)
        transformed_features = self.norm_node_features_1(
            node_features + transformed_features
        )

        ff_output_node_features = self.lin_node_features_2(
            self.dropout_node_features_2(
                self.activation(self.lin_node_features_1(transformed_features))
            )
        )
        ff_output_node_features = self.dropout_node_features_3(ff_output_node_features)
        transformed_features = self.norm_node_features_2(
            transformed_features + ff_output_node_features
        )

        # Transform Diffusion Time
        if not self.last_layer:
            transformed_diffusion_time = self.dropout_diffusion_time_1(
                transformed_diffusion_time
            )
            transformed_diffusion_time = self.norm_diffusion_time_1(
                diffusion_time + transformed_diffusion_time
            )

            ff_output_diffusion_time = self.lin_diffusion_time_2(
                self.dropout_diffusion_time_2(
                    self.activation(
                        self.lin_diffusion_time_1(transformed_diffusion_time)
                    )
                )
            )
            ff_output_diffusion_time = self.dropout_diffusion_time_3(
                ff_output_diffusion_time
            )
            transformed_diffusion_time = self.norm_diffusion_time_2(
                transformed_diffusion_time + ff_output_diffusion_time
            )

        out = DataHolder(
            node_features=transformed_features,
            diffusion_time=transformed_diffusion_time,
            positions=transformed_position,
            node_mask=node_mask,
        ).mask()

        return out

from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from linear_attention_transformer import LinearAttentionTransformer


class SelfAttention(nn.Module):
    """
    Self attention layer that also updates the representations on the edges.

    Parameters:
    - node_features_dimensions (torch.Tensor): Number of input features.
    - de (torch.Tensor): Dimensionality of edge embeddings.
    - diffusion_time_dimensions (torch.Tensor): Dimensionality of global features.
    - num_heads (int): Number of attention heads.
    - last_layer (bool): Flag indicating whether this layer is the last layer in the model.
    """

    def __init__(
        self,
        node_features_dimensions: torch.Tensor,
        diffusion_time_dimensions: torch.Tensor,
        delta_dimensions: torch.Tensor,
        num_heads: int,
        last_layer=False,
    ):
        """
        Initialize NodeEdgeBlock.

        Parameters:
        - node_features_dimensions (torch.Tensor): Number of input features.
        - diffusion_time_dimensions (torch.Tensor): Dimensionality of diffusion time.
        - num_heads (int): Number of attention heads.
        - last_layer (bool): Flag indicating whether this layer is the last layer in the model.
        """

        super().__init__()
        assert (
            node_features_dimensions % num_heads == 0
        ), f"node_features_dimensions: {node_features_dimensions} -- nhead: {num_heads}"
        self.node_features_dimensions = node_features_dimensions
        self.diffusion_time_dimensions = diffusion_time_dimensions
        self.delta_dimensions = delta_dimensions
        self.num_heads = num_heads

        """
        Note the below transformations are named funny. I.e. the name of the function is not exactly what it does.
        Its more like a transformation of the input to the function and then the operation is done. 
        For example: the function add_delta_to_edge_embeddings does not add delta to edge embeddings, but it computes
        the transofrmation of delta that is added to the edge embeddings. This naming convention is more or
        less followed throughout this file.
        
        However, when things will be simplified, a lot of these functions will be removed and the naming convention
        will be improved.
        """

        self.head_dim = self.node_features_dimensions // self.num_heads
        self.transform_positions_for_attn_mlp = self.mlp_out_pos_norm = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, delta_dimensions),
        )
        # Node Transformation (Attention)
        total_feature_dim = (
            node_features_dimensions + delta_dimensions + diffusion_time_dimensions
        )
        self.lin_node_features = torch.nn.Linear(
            node_features_dimensions, node_features_dimensions
        )
        self.concatenated_features = torch.nn.Linear(
            total_feature_dim, node_features_dimensions
        )
        self.attention = LinearAttentionTransformer(
            dim=node_features_dimensions, heads=num_heads, depth=1, max_seq_len=70000
        )
        self.head_features_to_position = torch.nn.Linear(num_heads * self.head_dim, 2)

        # Computing Diffusion Time
        self.last_layer = last_layer
        if not last_layer:
            self.y_y = Linear(diffusion_time_dimensions, diffusion_time_dimensions)

    def transform_positions_for_attention(self, positions, node_mask):
        positions = positions * node_mask
        norm_positions = torch.norm(positions, dim=-1, keepdim=True)
        normalized_position = positions / (norm_positions + 1e-7)

        transformed_positons = self.transform_positions_for_attn_mlp(
            normalized_position
        )
        return transformed_positons

    def transform_node_features(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        diffusion_time: torch.Tensor,
        node_mask: torch.Tensor,
        e_mask1: torch.Tensor,
        e_mask2: torch.Tensor,
    ) -> torch.Tensor:
        transformed_X = self.lin_node_features(node_features)
        transformed_delta = self.transform_positions_for_attention(positions, node_mask)
        transformed_time = diffusion_time.unsqueeze(1).expand(
            -1, node_features.size(1), -1
        )

        # Concatenate transformed features
        concatenated_features = torch.cat(
            [transformed_X, transformed_delta, transformed_time],
            dim=-1,
        )
        concatenated_features = self.concatenated_features(concatenated_features)
        head_outputs = self.attention(concatenated_features)
        return head_outputs

    def transform_diffusion_time(
        self,
        diffusion_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transforms diffusion time information based on node features, positional information, and edge masks.
        1. The choice of functions is super odd. Why are there so many linear layers? Why are there so many multiplications?
        2. Also it makes no sense to have these many transformations for just the diffusion time information.
        3. Would make sense to simplify it.

        Parameters:
        - diffusion_time (torch.Tensor): Tensor representing diffusion time information.
        Shape: (batch_size, num_nodes).

        Returns:
        - torch.Tensor: Transformed diffusion time information.
        Shape: (batch_size, num_diffusion_time_dim).
        """
        if self.last_layer:
            y = None
        else:
            y = self.y_y(diffusion_time)
        return y

    def transform_positions(self, head_outputs: torch.Tensor) -> torch.Tensor:
        """
        Transforms node positions based on edge embeddings.
        1. The choice of functions is super odd. Why are there so many linear layers? Why are there so many multiplications?
        2. Also note that the position transformations contain no diffusion time. (Updated 16th Jan, diffusion time is contained in the input edge embeddings)

        Parameters:
        - head_outputs (torch.Tensor): Tensor representing transformed node features.
        - positions (torch.Tensor): Tensor representing node positions.
        Shape: (batch_size, num_nodes, 2), where 2 corresponds to the x, y coordinates.
        - node_mask (torch.Tensor): Binary mask indicating active nodes.
        Shape: (batch_size, num_nodes).

        Returns:
        - torch.Tensor: Transformed node positions.
        Shape: (batch_size, num_nodes, 2), representing the updated x, y coordinates.
        """
        transformed_position = self.head_features_to_position(head_outputs)
        return transformed_position

    def forward(
        self,
        node_features: torch.Tensor,
        diffusion_time: torch.Tensor,
        positions: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model (Computing Attention)

        Parameters:
        - node_features (torch.Tensor): Tensor representing node features.
        Shape: (batch_size, num_nodes, feature_dim).
        - diffusion_time (torch.Tensor): Tensor representing diffusion time information.
        Shape: (batch_size, num_nodes).
        - position (torch.Tensor): Tensor representing node positions.
        Shape: (batch_size, num_nodes, 2), where 2 corresponds to the x, y coordinates.
        - node_mask (torch.Tensor): Binary mask indicating active nodes.
        Shape: (batch_size, num_nodes).

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Transformed node features, diffusion time,
        and node positions.
        - Transformed Node Features: Tensor representing updated node features.
            Shape: (batch_size, num_nodes, feature_dim).
        - Transformed Diffusion Time: Tensor representing updated diffusion time information.
            Shape: (batch_size, num_diffusion_time_dim).
        - Transformed Positions: Tensor representing updated node positions.
            Shape: (batch_size, num_nodes, 2), representing the updated x, y coordinates.
        """

        bs, n, _ = node_features.shape
        node_mask = node_mask.unsqueeze(-1)  # [bs, n, 1]
        edge_mask_1 = node_mask.unsqueeze(2)  # [bs, n, 1, 1]
        edge_mask_2 = node_mask.unsqueeze(1)  # [bs, 1, n, 1]

        # Create the transformed diffusion time based on the transformed node features, diffusion time, and edge embeddings
        transformed_diffusion_time = self.transform_diffusion_time(
            diffusion_time=diffusion_time,
        )

        transformed_node_features = self.transform_node_features(
            node_features=node_features,
            positions=positions,
            diffusion_time=transformed_diffusion_time,
            node_mask=node_mask,
            e_mask1=edge_mask_1,
            e_mask2=edge_mask_2,
        )

        # Create the transformed positions based on the edge embeddings and the node positions
        transformed_positions = self.transform_positions(
            head_outputs=transformed_node_features,
        )

        return (
            transformed_node_features,
            transformed_diffusion_time,
            transformed_positions,
        )

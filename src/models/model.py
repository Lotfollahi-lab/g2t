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
        input_activation: str = "relu",
        input_layernorm: bool = False,
        input_dropout: float = 0.0,
    ) -> None:
        """
        Constructor to initialize the Model instance.

        Args:
            input_dims: Input dimensions.
            n_layers (int): Number of transformer layers.
            hidden_mlp_dims (dict): Dimensions for hidden MLP layers.
            hidden_dims (dict): Dimensions for hidden layers.
            output_dims: Output dimensions.
            input_activation: Activation function used inside the
                node-features input MLP. ``"relu"`` (default) preserves
                the historic behaviour. ``"gelu"`` / ``"silu"`` pass
                real-valued (both-sign) signals through without zeroing
                the negative half — important when the node features
                are dense pretrained embeddings (e.g. Nicheformer,
                scGPT) rather than non-negative gene values.
            input_layernorm: If True, prepend a LayerNorm before the
                node-features input MLP. Useful for pretrained
                embeddings whose per-cell scale isn't controlled by
                the data loader (calibrates the input statistics so
                the first Linear sees a well-conditioned distribution).
                Default False (preserves historic behaviour for
                raw-gene inputs, which the dataloader already scales).
            input_dropout: If > 0, insert a Dropout(p=input_dropout)
                right after the optional LayerNorm and before the
                first Linear. Acts as input-level regularisation —
                especially helpful for deterministic pretrained
                embeddings where the model can otherwise memorise
                per-cell embedding patterns. Default 0.0 (off).

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
        self.input_activation_name = str(input_activation).lower()
        self.input_layernorm = bool(input_layernorm)
        self.input_dropout = float(input_dropout)

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()

        # Node-features input MLP. The historic 2-layer Linear→ReLU
        # design is preserved when all knobs are at their defaults
        # (activation=relu, layernorm=False, dropout=0.0) — same param
        # count, same arithmetic, byte-identical to before.
        #
        # When ``input_activation`` is gelu/silu, the two ReLU steps
        # are replaced with the chosen activation. This matters for
        # dense, real-valued node features (pretrained embeddings):
        # ReLU zeros out negative entries; GELU/SiLU pass them
        # through with smooth saturation. For non-negative inputs
        # (raw gene values), ReLU vs GELU is near-identity so the
        # baseline-on-gene-expression behaviour barely shifts.
        #
        # LayerNorm and Dropout, when on, are inserted BEFORE the
        # first Linear — they pre-condition the input rather than
        # acting on the hidden representation. This is the standard
        # DiT/Llama-style "norm at the boundary" pattern.
        layers: list = []
        if self.input_layernorm:
            layers.append(nn.LayerNorm(self.input_dimensions_node_features))
        if self.input_dropout > 0.0:
            layers.append(nn.Dropout(self.input_dropout))
        layers.append(
            nn.Linear(self.input_dimensions_node_features, hidden_mlp_dims["X"])
        )
        layers.append(self._make_activation())
        layers.append(nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]))
        layers.append(self._make_activation())
        self.mlp_in_node_features = nn.Sequential(*layers)

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

    def _make_activation(self) -> nn.Module:
        """Return a fresh instance of the configured input activation.

        Resolution: ``self.input_activation_name`` set from the
        ``input_activation`` constructor kwarg. Allowed values:
        ``"relu"`` (historic default), ``"gelu"``, ``"silu"``.
        Raises ValueError on unknown name — fail-loud rather than
        silently falling back to ReLU.
        """
        name = self.input_activation_name
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(
            f"Unknown input_activation={name!r}. Expected one of "
            f"'relu', 'gelu', 'silu'."
        )

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

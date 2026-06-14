import logging
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, scatter

logger = logging.getLogger(__name__)

class CustomGCNConv(MessagePassing):
    """Custom implementation of standard Graph Convolutional Network (GCN) layer.

    Subclasses PyG's MessagePassing base class and implements explicit
    message(), aggregate(), and update() functions.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        # Initialize with 'add' aggregation as standard GCN accumulates neighbor signals
        super().__init__(aggr="add")
        
        # Weight parameter and bias
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes weight and bias parameters."""
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass of the custom GCN layer.

        Step 1: Add self-loops to edge index (A_tilde = A + I)
        Step 2: Linear transform node features (X' = X W)
        Step 3: Compute symmetric degree normalization coefficients (D_tilde^-1/2)
        Step 4: Propagate messages using custom message, aggregate, and update
        """
        # 1. Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 2. Linear transformation of input features
        x = self.lin(x)

        # 3. Compute symmetric normalization factor: 1 / sqrt(deg_i * deg_j)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        
        # Replace infinity with 0 (for any isolated nodes, though Cora doesn't have them)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        
        # Compute normalized weight for each edge (j -> i)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 4. Propagate messages
        # x: [num_nodes, out_channels]
        # norm: [num_edges]
        out = self.propagate(edge_index, x=x, norm=norm)

        # 5. Add bias
        out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        """Constructs message from neighbor j to node i.

        x_j: Node features of the source node (neighbor), shape [num_edges, out_channels]
        norm: Edge normalization factor, shape [num_edges]
        """
        return norm.view(-1, 1) * x_j

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None
    ) -> Tensor:
        """Aggregates messages from neighbors.

        inputs: Messages from neighbors (from message()), shape [num_edges, out_channels]
        index: Target node indices (col), shape [num_edges]
        """
        # Explicit aggregate using PyG's scatter utility (sums neighbor messages)
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")

    def update(self, inputs: Tensor) -> Tensor:
        """Updates node representation after aggregation.

        inputs: Aggregated representation, shape [num_nodes, out_channels]
        """
        # For standard GCN, the update is simply the aggregated message sum
        return inputs

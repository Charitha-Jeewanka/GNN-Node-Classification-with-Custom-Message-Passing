import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, scatter, softmax

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

class CustomSAGEConv(MessagePassing):
    """Custom implementation of GraphSAGE convolution layer.

    Supports mean, max, and LSTM aggregation schemes. Projects neighbor features before
    propagation to avoid CUDA Out-of-Memory during LSTM pooling over large graphs.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregator_type: str = "mean"
    ) -> None:
        super().__init__(aggr=None)  # Disable default PyG aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregator_type = aggregator_type.lower()

        # Linear transformations for self-loop and neighbor aggregates
        self.lin_self = Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = Linear(in_channels, out_channels, bias=False)

        # For LSTM aggregator, initialize LSTM parameters mapping out_channels to out_channels
        if self.aggregator_type == "lstm":
            self.lstm_agg = nn.LSTM(out_channels, out_channels, batch_first=True)

        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes model weights."""
        self.lin_self.reset_parameters()
        self.lin_neigh.reset_parameters()
        if self.aggregator_type == "lstm":
            for name, param in self.lstm_agg.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0.0)
        self.bias.data.zero_()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass of the custom GraphSAGE layer."""
        # Project neighbor features BEFORE propagation to save GPU memory.
        # This reduces sequence dimension for LSTM from in_channels (1433) to out_channels (64).
        x_projected = self.lin_neigh(x)

        # 1. Propagate neighbor representations
        # GraphSAGE does NOT add self-loops here; the self-connection is explicitly summed
        neigh_out = self.propagate(edge_index, x=x_projected)

        # 2. Combine self representation and aggregated neighbor representations
        out = self.lin_self(x) + neigh_out

        # 3. Add bias
        out = out + self.bias
        return out

    def message(self, x_j: Tensor) -> Tensor:
        """Passes neighbor representations as messages."""
        return x_j

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None
    ) -> Tensor:
        """Aggregates neighbor messages using the configured aggregator (mean, max, lstm)."""
        device = inputs.device
        num_nodes = dim_size if dim_size is not None else int(index.max().item() + 1)

        if self.aggregator_type == "mean":
            return scatter(inputs, index, dim=self.node_dim, dim_size=num_nodes, reduce="mean")

        elif self.aggregator_type == "max":
            # Element-wise maximum
            out = scatter(inputs, index, dim=self.node_dim, dim_size=num_nodes, reduce="max")
            out[out == float("-inf")] = 0.0  # Replace -inf with 0 for nodes with 0 neighbors
            return out

        elif self.aggregator_type == "lstm":
            if inputs.size(0) == 0:
                return torch.zeros(num_nodes, self.out_channels, device=device)

            # Sort indices Contiguously for grouping
            sorted_index, perm = index.sort()
            sorted_inputs = inputs[perm]

            # Vectorized offset computation for target neighbors
            mask = torch.cat([
                torch.tensor([True], device=device),
                sorted_index[1:] != sorted_index[:-1]
            ])
            first_occ_indices = torch.arange(len(sorted_index), device=device)[mask]
            cumsum_mask = torch.cumsum(mask.long(), dim=0) - 1
            first_occ_idx_per_edge = first_occ_indices[cumsum_mask]
            offsets = torch.arange(len(sorted_index), device=device) - first_occ_idx_per_edge

            # Compute degree statistics
            deg = degree(index, num_nodes, dtype=torch.long)
            max_deg = int(deg.max().item())

            # Fill padded tensor
            pad_inputs = torch.zeros(num_nodes, max_deg, self.out_channels, device=device)
            flat_dest_idx = sorted_index * max_deg + offsets
            pad_inputs.view(-1, self.out_channels).index_copy_(0, flat_dest_idx, sorted_inputs)

            # Pass padded neighbor sequences through LSTM
            lstm_out, _ = self.lstm_agg(pad_inputs)

            # Gather final step representation for each active node
            gather_idx = (deg - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, self.out_channels)
            out = lstm_out.gather(1, gather_idx).squeeze(1)

            # Set inactive nodes (degree == 0) to zero
            out = out * (deg > 0).view(-1, 1).to(out.dtype)
            return out

        else:
            raise ValueError(f"Unsupported GraphSAGE aggregator: {self.aggregator_type}")

    def update(self, inputs: Tensor) -> Tensor:
        """Returns aggregated representation."""
        return inputs


class CustomAttentionConv(MessagePassing):
    """Custom Single-Head Graph Attention (GAT) layer.

    Formulates attention weights as dynamic, normalized edge gates,
    explicitly mirroring edge-gated convolution concepts.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        bias: bool = True
    ) -> None:
        super().__init__(node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        # Node feature projection
        self.lin = Linear(in_channels, out_channels, bias=False)

        # Source and destination attention parameters (att_src and att_dst)
        # In GAT, these are parameter vectors that multiply neighbor and self projections
        self.att_src = torch.nn.Parameter(torch.empty(1, out_channels))
        self.att_dst = torch.nn.Parameter(torch.empty(1, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes weights, attention vectors, and bias parameters."""
        self.lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass of the custom attention layer."""
        # 1. Add self-loops to edge index (nodes attend to themselves)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 2. Linear projection of input features: X' = X W
        x_proj = self.lin(x)

        # 3. Propagate messages (size is set dynamically)
        out = self.propagate(edge_index, x=x_proj, size=None)

        # 4. Add bias
        if self.bias is not None:
            out = out + self.bias
            
        return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int]
    ) -> Tensor:
        """Computes normalized attention coefficients (edge gates) and gates neighbor features."""
        # 1. Compute unnormalized attention (edge gate)
        # e_ji = LeakyReLU(a_src^T W h_j + a_dst^T W h_i)
        e = F.leaky_relu(
            (x_j * self.att_src).sum(dim=-1) + (x_i * self.att_dst).sum(dim=-1),
            negative_slope=0.2
        )

        # 2. Softmax normalization over target node incoming edges (index corresponds to target node i)
        alpha = softmax(e, index, num_nodes=size_i)

        # 3. Apply attention coefficient dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 4. Gated neighbor message: alpha_ji * (W h_j)
        return alpha.view(-1, 1) * x_j

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None
    ) -> Tensor:
        """Aggregates attention-weighted neighbor messages."""
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")

    def update(self, inputs: Tensor) -> Tensor:
        """Returns aggregated node representations."""
        return inputs

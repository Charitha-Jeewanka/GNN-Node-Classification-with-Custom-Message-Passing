import torch
import torch.nn.functional as F
from torch import Tensor

from src.layers import CustomGCNConv

class GCN(torch.nn.Module):
    """A standard 2-layer Graph Convolutional Network (GCN) model.

    Utilizes the CustomGCNConv message passing layer.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.conv1 = CustomGCNConv(in_channels, hidden_dim)
        self.conv2 = CustomGCNConv(hidden_dim, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass of the 2-layer GCN."""
        # First GCN Layer + Activation + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN Layer
        x = self.conv2(x, edge_index)
        return x

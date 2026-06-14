import torch
import torch.nn.functional as F
from torch import Tensor

from src.layers import CustomGCNConv, CustomSAGEConv, CustomAttentionConv

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

    def get_embeddings(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Extracts the learned node embeddings from the first GCN layer."""
        self.eval()
        with torch.no_grad():
            emb = self.conv1(x, edge_index)
        return emb


class GraphSAGE(torch.nn.Module):
    """A 2-layer GraphSAGE model for node classification.

    Utilizes the CustomSAGEConv message passing layer with configurable aggregation.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        aggregator_type: str = "mean",
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.conv1 = CustomSAGEConv(in_channels, hidden_dim, aggregator_type)
        self.conv2 = CustomSAGEConv(hidden_dim, out_channels, aggregator_type)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass of the 2-layer GraphSAGE."""
        # First GraphSAGE Layer + Activation + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GraphSAGE Layer
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Extracts the learned node embeddings from the first GraphSAGE layer."""
        self.eval()
        with torch.no_grad():
            # In GraphSAGE, conv1 projects neighbor features in forward.
            # Here we extract the output of conv1.
            emb = self.conv1(x, edge_index)
        return emb


class CustomAttentionGNN(torch.nn.Module):
    """A 2-layer Graph Neural Network using custom attention-weighted aggregation (Simplified GAT).

    Attaches attention scores to edges, acting as normalized edge gates.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        dropout: float = 0.5,
        attn_dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.conv1 = CustomAttentionConv(in_channels, hidden_dim, dropout=attn_dropout)
        self.conv2 = CustomAttentionConv(hidden_dim, out_channels, dropout=attn_dropout)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass of the 2-layer Custom Attention GNN."""
        # First Layer + Relu + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second Layer
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Extracts the learned node embeddings from the first Attention layer."""
        self.eval()
        with torch.no_grad():
            emb = self.conv1(x, edge_index)
        return emb

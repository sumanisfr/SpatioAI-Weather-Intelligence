"""
GNN Model Architecture for SpatioAI Spatio-Temporal Event Association (Phase 4).
Supports GraphSAGE / GAT message passing with edge-feature fused binary link prediction.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, SAGEConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class FallbackSageConv(nn.Module):
    """
    Standard PyTorch fallback GraphSAGE layer if PyG is not installed or available.
    Supports node neighborhood mean-aggregation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        out_channels = self.lin_neigh.out_features

        if edge_index.numel() == 0 or edge_index.size(1) == 0:
            # No edges: node self-transformation only
            return self.lin_self(x)

        src, dst = edge_index[0], edge_index[1]
        # Aggregate messages from src to dst
        neigh_sum = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        degree = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)

        neigh_sum.index_add_(0, dst, x[src])
        degree.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device, dtype=x.dtype))
        degree = torch.clamp(degree, min=1.0)
        neigh_mean = neigh_sum / degree

        out = self.lin_self(x) + self.lin_neigh(neigh_mean)
        return out


class SpatioTemporalGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for Extreme Weather Event Association.

    Architecture:
    1. Node Input Linear Projection
    2. Multi-layer Graph Message Passing (GraphSAGE / GAT) with LayerNorm and Residuals
    3. Edge Fusion MLP: [node_emb_src || node_emb_dst || edge_attr] -> Binary Logits
    """

    def __init__(
        self,
        node_in_dim: int = 15,
        edge_in_dim: int = 9,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = "sage",  # "sage" or "gat"
        use_edge_attr_in_conv: bool = False,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type.lower()

        # Node feature projection
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Message passing layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for _ in range(num_layers):
            if PYG_AVAILABLE and self.conv_type == "sage":
                self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif PYG_AVAILABLE and self.conv_type == "gat":
                self.conv_layers.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False))
            else:
                # Built-in pure PyTorch GraphSAGE implementation
                self.conv_layers.append(FallbackSageConv(hidden_dim, hidden_dim))

            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        # Edge classifier MLP: takes [src_node, dst_node, edge_attr]
        edge_mlp_in_dim = (2 * hidden_dim) + edge_in_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_mlp_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the GNN.

        Args:
            x: Node feature tensor of shape (N, node_in_dim)
            edge_index: Graph connectivity tensor of shape (2, E)
            edge_attr: Edge feature tensor of shape (E, edge_in_dim)

        Returns:
            torch.Tensor: Unnormalized edge association logits of shape (E,)
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D node tensor (N, D), got shape {x.shape}")

        if x.size(0) == 0 or edge_index.size(1) == 0:
            return torch.empty(0, device=x.device, dtype=x.dtype)

        # 1. Encode initial node representations
        h = self.node_encoder(x)

        # 2. Graph message passing with residual connections
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            h_in = h
            h_conv = conv(h, edge_index)
            h = norm(h_conv + h_in)  # Residual connection
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # 3. Formulate edge representations
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        h_src = h[src_idx]
        h_dst = h[dst_idx]

        # Concatenate [node_i, node_j, edge_attr_ij]
        edge_features_fused = torch.cat([h_src, h_dst, edge_attr], dim=-1)

        # 4. Predict edge association logits
        logits = self.edge_classifier(edge_features_fused).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_probabilities(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes calibrated edge association probabilities P(i -> j) in [0, 1].
        """
        self.eval()
        logits = self.forward(x, edge_index, edge_attr)
        return torch.sigmoid(logits)

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import TUDataset, ZINC, GNNBenchmarkDataset, LRGBDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
import warnings 
import itertools, copy
import random 
from utils.utility import RRWPPositionalEncoding, CosineWarmupScheduler


class FeatureModulator(nn.Module):
    """
    Neural network that modulates node features based on edge features.
    Corresponds to the ψ function in the paper.
    """
    def __init__(self, edge_dim, node_dim, hidden_dim=64, dropout=0.0):
        super(FeatureModulator, self).__init__()
        self.mlp = nn.Sequential(
            Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, node_dim)
        )

    def forward(self, edge_features):
        return self.mlp(edge_features)


class CKGConv(MessagePassing):
    """
    Convolution layer that utilizes concatenated node/edge features with
    positional encodings and applies adaptive degree scaling.
    """
    def __init__(self, node_in_dim, edge_in_dim, pe_dim, out_channels,
                 modulator_hidden_dim=64, dropout=0.0, add_self_loops=True,
                 aggr='mean'):
        """
        Args:
            node_in_dim (int): Dimension of raw node features
            edge_in_dim (int): Dimension of raw edge features
            pe_dim (int): Dimension of positional encodings
            out_channels (int): Output feature dimension
            modulator_hidden_dim (int): Hidden dimension for feature modulator
            dropout (float): Dropout probability
            add_self_loops (bool): Whether to add self-loops during message passing
            aggr (str): Aggregation scheme ('mean', 'add', etc.)
        """
        super(CKGConv, self).__init__(aggr=aggr)
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.pe_dim = pe_dim
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops

        # Combined dimensions
        self.node_feature_dim = node_in_dim + pe_dim
        self.edge_feature_dim = edge_in_dim + pe_dim

        # Feature modulator (ψ function)
        self.modulator = FeatureModulator(
            edge_dim=self.edge_feature_dim,
            node_dim=self.node_feature_dim,
            hidden_dim=modulator_hidden_dim,
            dropout=dropout
        )

        # Linear transformation
        self.linear = Linear(self.node_dim, out_channels)

        # Learnable degree scaling parameters
        self.theta1 = nn.Parameter(torch.ones(out_channels))
        self.theta2 = nn.Parameter(torch.zeros(out_channels))

        # Initialization
        self.reset_parameters()

    def reset_parameters(self):
        """Reset learnable parameters."""
        self.modulator.mlp[0].reset_parameters()
        self.modulator.mlp[3].reset_parameters()
        self.linear.reset_parameters()
        nn.init.ones_(self.theta1)
        nn.init.zeros_(self.theta2)

    def forward(self, x, x_pe, edge_index, edge_attr, edge_pe, batch=None):
        """
        Forward pass of the layer.

        Args:
            x (Tensor): Raw node features [num_nodes, node_in_dim]
            x_pe (Tensor): Node positional encodings [num_nodes, pe_dim]
            edge_index (Tensor): Graph connectivity [2, num_edges]
            edge_attr (Tensor): Raw edge features [num_edges, edge_in_dim]
            edge_pe (Tensor): Edge positional encodings [num_edges, pe_dim]
            batch (Tensor, optional): Batch vector for disconnected graphs

        Returns:
            Tensor: Updated node features [num_nodes, out_channels]
        """
        num_nodes = x.size(0)

        # Concat raw features with positional encodings
        x = torch.cat([x, x_pe], dim=-1)

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # Concat edge features with positional encodings
        edge_attr = torch.cat([edge_attr, edge_pe], dim=-1)

        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Compute node degrees (optionally includes self-loops)
        deg = degree(edge_index[0], num_nodes=num_nodes).to(x.dtype)
        deg = deg.clamp(min=1)

        # Apply adaptive degree scaling
        deg_sqrt = deg.sqrt().view(-1, 1)
        out = out * self.theta1 + deg_sqrt * (out * self.theta2)

        return out

    def message(self, x_j, edge_attr):
        """
        Message computation: modulate source node features with edge features

        Args:
            x_j (Tensor): Source node features [num_edges, node_dim]
            edge_attr (Tensor): Edge features [num_edges, edge_dim]

        Returns:
            Tensor: Computed messages
        """
        # Apply modulator to get weights for node features
        edge_weights = self.modulator(edge_attr)

        # Element-wise multiplication of source features with modulated weights
        return x_j * edge_weights

    def update(self, aggr_out):
        """
        Update function: apply linear transformation to aggregated messages

        Args:
            aggr_out (Tensor): Aggregated messages [num_nodes, node_dim]

        Returns:
            Tensor: Updated node features [num_nodes, out_channels]
        """
        return self.linear(aggr_out)


class CKGConvBlock(nn.Module):
    """
    Block consisting of a CKGConv layer followed by normalization and a feed-forward network
    with residual connections.
    """
    def __init__(self, node_in_dim, edge_in_dim, pe_dim, out_channels,
                 ffn_hidden_dim=None, modulator_hidden_dim=64,
                 dropout=0.0, norm_type='batch', add_self_loops=True, aggr='mean'):
        """
        Args:
            node_in_dim (int): Dimension of raw node features
            edge_in_dim (int): Dimension of raw edge features
            pe_dim (int): Dimension of positional encodings
            out_channels (int): Output feature dimension
            ffn_hidden_dim (int): Hidden dimension of FFN (defaults to 4*out_channels)
            modulator_hidden_dim (int): Hidden dimension for feature modulator
            dropout (float): Dropout probability
            norm_type (str): Normalization type ('batch', 'layer', or 'none')
            add_self_loops (bool): Whether to add self-loops during message passing
            aggr (str): Aggregation scheme ('mean', 'add', etc.)
        """
        super(CKGConvBlock, self).__init__()

        # Set FFN hidden dimension if not provided
        if ffn_hidden_dim is None:
            ffn_hidden_dim = 4 * out_channels

        # Convolution layer
        self.conv = CKGConv(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            pe_dim=pe_dim,
            out_channels=out_channels,
            modulator_hidden_dim=modulator_hidden_dim,
            dropout=dropout,
            add_self_loops=add_self_loops,
            aggr=aggr
        )

        # First normalization layer
        if norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
        elif norm_type == 'layer':
            self.norm1 = nn.LayerNorm(out_channels)
            self.norm2 = nn.LayerNorm(out_channels)
        else:  # 'none'
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Feed-forward network
        self.ffn = nn.Sequential(
            Linear(out_channels, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(ffn_hidden_dim, out_channels)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_pe, edge_index, edge_attr, edge_pe, batch=None):
        """
        Forward pass of the block.

        Args:
            x (Tensor): Raw node features [num_nodes, node_in_dim]
            x_pe (Tensor): Node positional encodings [num_nodes, pe_dim]
            edge_index (Tensor): Graph connectivity [2, num_edges]
            edge_attr (Tensor): Raw edge features [num_edges, edge_in_dim]
            edge_pe (Tensor): Edge positional encodings [num_edges, pe_dim]
            batch (Tensor, optional): Batch vector for disconnected graphs

        Returns:
            Tensor: Updated node features [num_nodes, out_channels]
        """
        # Apply convolution
        identity = x if x.size(1) == self.conv.out_channels else None

        x = self.conv(x, x_pe, edge_index, edge_attr, edge_pe, batch)
        x = self.norm1(x)

        # Residual connection (if dimensions match)
        if identity is not None:
            x = x + identity

        # Apply FFN with residual connection
        identity = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + identity
        x = self.norm2(x)

        return x


class CKGNet(nn.Module):
    """
    Complete CKGNet model for graph-level tasks, using RRWP positional encodings
    and CKGConv blocks.
    """
    def __init__(self, node_in_dim, edge_in_dim, pe_dim=16, hidden_channels=128,
                 num_layers=4, dropout=0.0, ffn_ratio=4, pooling='mean',
                 num_classes=1, task='classification', norm_type='batch',
                 add_self_loops=True, aggr='mean', task_type='graph', pe_type="random_walk"):
        """
        Args:
            node_in_dim (int): Dimension of raw node features
            edge_in_dim (int): Dimension of raw edge features
            pe_dim (int): Dimension of positional encodings
            hidden_channels (int): Hidden dimension in network
            num_layers (int): Number of CKGConv blocks
            dropout (float): Dropout probability
            ffn_ratio (int): Ratio to determine FFN hidden dimension (hidden_channels * ffn_ratio)
            pooling (str): Graph pooling method ('mean', 'sum', 'max', etc.)
            num_classes (int): Number of output classes/targets
            task (str): Task type ('classification' or 'regression')
            norm_type (str): Normalization type ('batch', 'layer', or 'none')
            add_self_loops (bool): Whether to add self-loops during message passing
            aggr (str): Aggregation scheme ('mean', 'add', etc.)
        """
        super(CKGNet, self).__init__()

        self.pe_dim = pe_dim
        self.task = task
        self.task_type = task_type

        # Positional encoding module
        self.pe_encoder = RRWPPositionalEncoding(K=pe_dim, pe_type=pe_type)

        # Input projection (if needed)
        self.node_proj = Linear(node_in_dim, hidden_channels)

        # Stack of CKGConv blocks
        self.blocks = nn.ModuleList()

        # First block
        self.blocks.append(CKGConvBlock(
            node_in_dim= hidden_channels,
            edge_in_dim=edge_in_dim,
            pe_dim=pe_dim,
            out_channels=hidden_channels,
            ffn_hidden_dim=hidden_channels * ffn_ratio,
            dropout=dropout,
            norm_type=norm_type,
            add_self_loops=add_self_loops,
            aggr=aggr
        ))

        # Remaining blocks
        for _ in range(num_layers - 1):
            self.blocks.append(CKGConvBlock(
                node_in_dim=hidden_channels,
                edge_in_dim=edge_in_dim,
                pe_dim=pe_dim,
                out_channels=hidden_channels,
                ffn_hidden_dim=hidden_channels * ffn_ratio,
                dropout=dropout,
                norm_type=norm_type,
                add_self_loops=add_self_loops,
                aggr=aggr,
                modulator_hidden_dim=hidden_channels
            ))

        # Set pooling function
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'add' or pooling == 'sum':
            self.pool = global_add_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}")

        # Prediction head
        self.pred_head = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(hidden_channels, num_classes)
        )

    def forward(self, data):
        """
        Forward pass of the full model.

        Args:
            data: PyG Data object containing:
                - x: Node features [num_nodes, node_in_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr (optional): Edge features [num_edges, edge_in_dim]
                - batch: Batch assignment for nodes

        Returns:
            Tensor: Predictions [batch_size, num_classes]
        """
            # Convert node features to float if they're not already
        if hasattr(data, 'x') and data.x is not None:
            if data.x.dtype != torch.float:
                data.x = data.x.float()
        
        # Convert edge attributes to float if they're not already
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.dtype != torch.float:
                data.edge_attr = data.edge_attr.float()
                
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Handle edge attributes
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr
        else:
            edge_attr = torch.ones(edge_index.size(1), 1, device=edge_index.device)

        # Compute positional encodings
        x_pe, edge_pe = self.pe_encoder(edge_index, x.size(0), batch=batch)

        # Apply input projection if needed
        x = self.node_proj(x)

        # Apply CKGConv blocks
        for block in self.blocks:
            x = block(x, x_pe, edge_index, edge_attr, edge_pe, batch)

        # Graph-level pooling 
        if self.task_type == 'graph':
            x = self.pool(x, batch)

        # Prediction head
        x = self.pred_head(x)

        # Apply activation based on task
        if self.task == 'classification' and x.size(-1) > 1:
            return F.log_softmax(x, dim=-1)

        return x

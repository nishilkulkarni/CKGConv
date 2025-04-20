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
import yaml
import argparse
from types import SimpleNamespace


def load_yaml_as_namespace(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Convert dict to Namespace
    config_namespace = argparse.Namespace(**config_dict)
    return config_namespace


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


class RRWPPositionalEncoding(nn.Module):
    """
    Computes positional encodings for nodes and edges based on different types,
    including random-walk, Laplacian, shortest-path, node degree, kernel distance,
    and structure-aware encodings as described in the respective papers.
    """
    def __init__(self, K=16, use_sparse=False, normalization='row', self_loops=True, pe_type="random_walk"):
        """
        Args:
            K (int): Number of steps (or target dimension for some projections).
            use_sparse (bool): Whether to use sparse matrix operations.
            normalization (str): 'row', 'sym', or None.
            self_loops (bool): Whether to add self-loops.
            pe_type (str): Type of positional encoding. Options include:
                           "laplacian", "shortest_path", "node_degree",
                           "kernel_distance", "random_walk", "structure_aware"
        """
        super(RRWPPositionalEncoding, self).__init__()
        self.K = K
        self.use_sparse = use_sparse
        self.normalization = normalization
        self.self_loops = self_loops
        self.pe_type = pe_type

        # For some PE types, we define the extra layers as parameters.
        if pe_type == "structure_aware":
            # Project concatenated random walk and degree features ([K+1]) to K dimensions.
            self.struct_node_linear = nn.Linear(K + 1, K)
            # Combine the projected node features from both endpoints.
            self.struct_edge_linear = nn.Linear(2 * K, K)
        elif pe_type == "laplacian":
            # For edge encoding, project concatenated eigenvector features (2*K) to K dimensions.
            self.laplacian_edge_linear = nn.Linear(2 * K, K)
        elif pe_type == "shortest_path":
            self.max_dist = 10
            self.distance_embedding = nn.Embedding(self.max_dist + 1, K)
        elif pe_type == "node_degree":
            self.node_degree_edge_linear = nn.Linear(2, K)
        # For "kernel_distance" and "random_walk" no extra parameters are required.

    def forward(self, edge_index, num_nodes, edge_weight=None, batch=None):
        device = edge_index.device
        node_pe, edge_pe = self._compute_pe_dense(edge_index, num_nodes, edge_weight, device)
        return node_pe, edge_pe

    def _compute_pe_dense(self, edge_index, num_nodes, edge_weight=None, device=None):
        # Create a dense adjacency matrix; use edge_weight if provided.
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device)
        A = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
        A = A.to(device)

        # Add self-loops if specified.
        if self.self_loops:
            A = A + torch.eye(num_nodes, device=device, dtype=A.dtype)

        # For methods using random-walk features, normalize A.
        if self.pe_type in ["structure_aware", "random_walk"]:
            deg = A.sum(dim=1)
            if self.normalization == 'row':
                deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
                M = A * deg_inv.unsqueeze(1)
            elif self.normalization == 'sym':
                deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
                M = (deg_inv_sqrt.unsqueeze(1) * A) * deg_inv_sqrt.unsqueeze(0)
            else:
                M = A.clone()

        if self.pe_type == "structure_aware":
            # Structure-Aware Encoding (Chen et al., 2022)
            I = torch.eye(num_nodes, device=device, dtype=A.dtype)
            mats = [I]
            T_power = M.clone()
            for t in range(1, self.K):
                mats.append(T_power)
                T_power = T_power @ M
            # Random-walk features from the diagonal of each power.
            node_rw = torch.stack([mat.diag() for mat in mats], dim=1)  # [num_nodes, K]
            # Degree centrality as an extra feature.
            node_deg = A.sum(dim=1).unsqueeze(1)  # [num_nodes, 1]
            node_struct = torch.cat([node_rw, node_deg], dim=1)  # [num_nodes, K+1]
            node_pe = self.struct_node_linear(node_struct)
            # For edge PE, combine the structure-aware features from both endpoints.
            i_idx, j_idx = edge_index
            edge_struct = torch.cat([node_pe[i_idx], node_pe[j_idx]], dim=1)
            edge_pe = self.struct_edge_linear(edge_struct)

        elif self.pe_type == "laplacian":
            # Laplacian PE (Dwivedi & Bresson, 2020)
            deg = A.sum(dim=1)
            deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            I = torch.eye(num_nodes, device=device, dtype=A.dtype)
            L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt
            eigvals, eigvecs = torch.linalg.eigh(L_norm)
            if num_nodes > 1 and self.K < num_nodes:
                node_pe = eigvecs[:, 1:self.K+1]
            else:
                node_pe = eigvecs[:, :self.K]
            i_idx, j_idx = edge_index
            edge_feat = torch.cat([node_pe[i_idx], node_pe[j_idx]], dim=1)
            edge_pe = self.laplacian_edge_linear(edge_feat)

        elif self.pe_type == "shortest_path":
            # Shortest-path encoding (Li et al., 2020)
            INF = 1e6
            ones = torch.ones_like(A)
            dist = torch.where(A > 0, ones, torch.full_like(A, INF))
            for i in range(num_nodes):
                dist[i, i] = 0
            for k in range(num_nodes):
                dist = torch.minimum(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))
            dist_clipped = dist.clone().long().clamp(max=self.max_dist)
            node_distance_emb = self.distance_embedding(dist_clipped)
            node_pe = node_distance_emb.mean(dim=1)
            i_idx, j_idx = edge_index
            edge_dist = dist_clipped[i_idx, j_idx]
            edge_pe = self.distance_embedding(edge_dist)

        elif self.pe_type == "node_degree":
            # Node degree centrality encoding (Ying et al., 2021)
            deg = A.sum(dim=1)
            max_deg = deg.max() if deg.max() > 0 else 1.0
            norm_deg = deg / max_deg
            node_pe = norm_deg.unsqueeze(1).repeat(1, self.K)
            i_idx, j_idx = edge_index
            edge_feat = torch.stack([norm_deg[i_idx], norm_deg[j_idx]], dim=1)
            edge_pe = self.node_degree_edge_linear(edge_feat)

        elif self.pe_type == "kernel_distance":
            # Kernel distance based encoding (Mialon et al., 2021)
            deg = A.sum(dim=1)
            deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            I = torch.eye(num_nodes, device=device, dtype=A.dtype)
            L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt
            t = 1.0
            diffusion_kernel = I.clone()
            K_mat = I.clone()
            for order in range(1, self.K):
                K_mat = K_mat @ ((-t * L_norm) / order)
                diffusion_kernel = diffusion_kernel + K_mat
            node_pe = diffusion_kernel.diag().unsqueeze(1).repeat(1, self.K)
            i_idx, j_idx = edge_index
            edge_pe = diffusion_kernel[i_idx, j_idx].unsqueeze(1).repeat(1, self.K)

        elif self.pe_type == "random_walk":
            # Random-walk encoding (Dwivedi et al., 2022)
            I = torch.eye(num_nodes, device=device, dtype=A.dtype)
            mats = [I]
            T_power = M.clone()
            for t in range(1, self.K):
                mats.append(T_power)
                T_power = T_power @ M
            i_idx, j_idx = edge_index
            edge_pe = torch.cat([mat[i_idx, j_idx].unsqueeze(1) for mat in mats], dim=1)
            node_pe = torch.stack([mat.diag() for mat in mats], dim=1)

        else:
            raise ValueError(f"Unknown pe_type: {self.pe_type}")

        # Return node and edge positional encodings (node_pe first).
        return node_pe, edge_pe


def load_datasets(dataset):
    if dataset == "ZINC":
        # ZINC provides built-in splits.
        train_dataset = ZINC(root=os.path.join('data', dataset), name=dataset, split="train")
        val_dataset = ZINC(root=os.path.join('data', dataset), name=dataset, split="val")
        test_dataset = ZINC(root=os.path.join('data', dataset), name=dataset, split="test")
    if dataset in ["MNIST", "CIFAR10", "PATTERN", "CLUSTER"]:
        train_dataset = GNNBenchmarkDataset(root=os.path.join('data', dataset), name=dataset, split="train")
        val_dataset = GNNBenchmarkDataset(root=os.path.join('data', dataset), name=dataset, split="val")
        test_dataset = GNNBenchmarkDataset(root=os.path.join('data', dataset), name=dataset, split="test")
    if dataset in ['Peptides-func', 'Peptides-struct']:
        train_dataset = LRGBDataset(root=os.path.join('data', dataset), name=dataset, split="train")
        val_dataset = LRGBDataset(root=os.path.join('data', dataset), name=dataset, split="val")
        test_dataset = LRGBDataset(root=os.path.join('data', dataset), name=dataset, split="test")
    return train_dataset, val_dataset, test_dataset


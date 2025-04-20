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


def train(model, loader, optimizer, device, task='classification', dataset_type=None, noise_val=0):
    """
    Train the model for one epoch.

    Supports graph-level and node-level tasks (pattern/cluster datasets are node-level).
    For 'Peptides-struct' dataset, the task is multiclass regression.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total_items = 0
    total_mse = 0.0
    total_mae = 0.0
    
    # Track total number of dimensions for regression
    total_dimensions = 0

    all_preds = []
    all_targets = []
    all_probs = []  # Store prediction probabilities for AP calculation

    # Weighted accuracy tracking
    if task == 'classification' and dataset_type in ['pattern', 'cluster']:
        class_correct = {}
        class_total = {}

    node_level = dataset_type in ['pattern', 'cluster']
    multiclass_regression = dataset_type == 'Peptides-struct'

    for batch_idx, data in enumerate(loader):
        batch_start = time.time()
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)

        # Determine target and count based on task and level
        if node_level:
            y_true = data.y.view(-1)
            count = data.num_nodes
        else:
            y_true = data.y
            count = data.num_graphs

        # Loss & predictions
        if task == 'classification':
            if random.random() < noise_val:
                # Store original labels for printing
                temp = y_true.clone()
                print("Possibly flipping something")
                # For multilabel, randomly flip some bits in the binary label vector
                # Creating a random noise mask (1 = flip, 0 = keep)
                noise_mask = torch.bernoulli(torch.ones_like(y_true) * 0.2)  # 20% chance to flip each label
                
                # Flip the selected labels (XOR operation)
                # y_true = y_true ^ noise_mask.to(torch.int64)
                y_true = (y_true.long() ^ noise_mask.long()).float()

                # Print which labels were flipped
                changed_indices = (noise_mask == 1).nonzero(as_tuple=True)
                # print(f"Flipping labels: original={temp[changed_indices]}, new={y_true[changed_indices]}")
            
            # Use binary cross entropy with logits for multilabel classification
            loss = F.binary_cross_entropy_with_logits(out, y_true.float())
            
            # Prediction is done with a threshold (typically 0.5)
            pred = (torch.sigmoid(out) > 0.5).float()
            
            # Store probabilities for metrics calculation
            probs = torch.sigmoid(out)
            all_probs.append(probs.detach().cpu())
            
            # Calculate correct predictions (exact matches across all labels)
            correct += int((pred == y_true).all(dim=1).sum())
            
            # Alternatively, calculate Hamming accuracy (ratio of correctly predicted labels)
            # hamming_score = (pred == y_true).float().mean()
            
            # Store predictions and targets for detailed metrics calculation later
            all_preds.append(pred.detach().cpu())
            all_targets.append(y_true.detach().cpu())

            if node_level:
                for c in torch.unique(y_true):
                    mask = (y_true == c)
                    class_correct.setdefault(c.item(), 0)
                    class_total.setdefault(c.item(), 0)
                    class_correct[c.item()] += int((pred[mask] == c).sum())
                    class_total[c.item()] += int(mask.sum())

        else:  # regression
            # Handle regression (single or multi-dimension)
            # Ensure dimensions match
            if out.size() != y_true.size():
                out = out.view(y_true.size())
            
            # Track total number of dimensions
            n_items = out.size(0)
            n_dimensions = out.size(1) if len(out.size()) > 1 else 1
            total_dimensions += n_items * n_dimensions
            
            # MSE loss
            loss = F.mse_loss(out, y_true.float())
            
            # Calculate MSE and MAE with sum reduction for tracking
            total_mse += F.mse_loss(out, y_true.float(), reduction='sum').item()
            total_mae += F.l1_loss(out, y_true.float(), reduction='sum').item()
            
            # Store predictions and targets for further analysis if needed
            all_preds.append(out.detach().cpu())
            all_targets.append(y_true.detach().cpu())

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * count
        total_items += count

    # Epoch metrics
    epoch_loss = total_loss / total_items

    if task == 'classification':
       # Calculate accuracy (exact matches)
        epoch_acc = 100.0 * correct / total_items

        # Concatenate predictions and targets from batches
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        probs = torch.cat(all_probs)

        # Calculate F1 score for multilabel classification
        # Use 'micro', 'macro', or 'samples' average depending on your needs
        epoch_f1 = f1_score(targets.numpy(), preds.numpy(), average='weighted', zero_division=0)

        # For multilabel, targets are already in the correct format (no need for one-hot encoding)
        # as they should already be binary vectors

        # Calculate average precision for multilabel classification
        epoch_ap = average_precision_score(targets.numpy(), probs.numpy(), average='weighted')

        # ROC AUC score
        roc_auc_val = roc_auc_score(targets.numpy(), probs.numpy(), average='macro')

        if node_level and class_total:
            class_accs = [class_correct[c] / class_total[c] for c in class_total]
            epoch_w_acc = 100.0 * sum(class_accs) / len(class_accs)
            print(f"TRAIN Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, Train F1 Score: {epoch_f1:.4f}, Train AP: {epoch_ap:.4f}, Train Weighted Accuracy: {epoch_w_acc:.2f}%")
            return epoch_loss, epoch_acc, epoch_f1, epoch_ap, epoch_w_acc
        else:
            print(f"TRAIN Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, Train F1 Score: {epoch_f1:.4f}, Train AP: {epoch_ap:.4f}")
            return epoch_loss, epoch_acc, epoch_f1, epoch_ap, roc_auc_val

    else:  # regression
        # Divide by total_dimensions for proper mean error
        epoch_mse = total_mse / total_dimensions
        epoch_rmse = math.sqrt(epoch_mse)
        epoch_mae = total_mae / total_dimensions
        
        if multiclass_regression:
            print(f"TRAIN Loss: {epoch_loss:.4f}, Train MSE: {epoch_mse:.4f}, Train RMSE: {epoch_rmse:.4f}, Train MAE: {epoch_mae:.4f} (Multiclass Regression)")
        else:
            print(f"TRAIN Loss: {epoch_loss:.4f}, Train MSE: {epoch_mse:.4f}, Train RMSE: {epoch_rmse:.4f}, Train MAE: {epoch_mae:.4f}")
        
        return epoch_loss, epoch_mse, epoch_mae


def evaluate(model, loader, device, task='classification', dataset_type=None, phase="Validation"):
    """
    Evaluate the model on validation or test data.

    Supports graph-level and node-level tasks (pattern/cluster datasets are node-level).
    For 'Peptides-struct' dataset, the task is multiclass regression.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_items = 0
    total_mse = 0.0
    total_mae = 0.0
    
    # Track total number of dimensions for regression
    total_dimensions = 0

    all_preds = []
    all_targets = []
    all_probs = []  # Store prediction probabilities for AP calculation

    if task == 'classification' and dataset_type in ['pattern', 'cluster']:
        class_correct = {}
        class_total = {}

    node_level = dataset_type in ['pattern', 'cluster']
    multiclass_regression = dataset_type == 'Peptides-struct'

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)

            if node_level:
                y_true = data.y.view(-1)
                count = data.num_nodes
            else:
                y_true = data.y
                count = data.num_graphs

            if task == 'classification':
                # Use binary cross entropy with logits for multilabel classification
                loss = F.binary_cross_entropy_with_logits(out, y_true.float())
                
                # Prediction is done with a threshold (typically 0.5)
                pred = (torch.sigmoid(out) > 0.5).float()
                
                # Store probabilities for metrics calculation
                probs = torch.sigmoid(out)
                all_probs.append(probs.detach().cpu())
                
                # Calculate correct predictions (exact matches across all labels)
                correct += int((pred == y_true).all(dim=1).sum())
                
                # Alternatively, calculate Hamming accuracy (ratio of correctly predicted labels)
                # hamming_score = (pred == y_true).float().mean()
                
                # Store predictions and targets for detailed metrics calculation later
                all_preds.append(pred.detach().cpu())
                all_targets.append(y_true.detach().cpu())

                if node_level:
                    for c in torch.unique(y_true):
                        mask = (y_true == c)
                        class_correct.setdefault(c.item(), 0)
                        class_total.setdefault(c.item(), 0)
                        class_correct[c.item()] += int((pred[mask] == c).sum())
                        class_total[c.item()] += int(mask.sum())

            else:  # regression
                # Handle regression (single or multi-dimension)
                # Ensure dimensions match
                if out.size() != y_true.size():
                    out = out.view(y_true.size())
                
                # Track total number of dimensions
                n_items = out.size(0)
                n_dimensions = out.size(1) if len(out.size()) > 1 else 1
                total_dimensions += n_items * n_dimensions
                
                # MSE loss with sum reduction
                loss = F.mse_loss(out, y_true.float(), reduction='sum')
                
                # Calculate MSE and MAE
                total_mse += loss.item()
                total_mae += F.l1_loss(out, y_true.float(), reduction='sum').item()
                
                # Store predictions and targets for further analysis if needed
                all_preds.append(out.detach().cpu())
                all_targets.append(y_true.detach().cpu())

            total_loss += float(loss)
            total_items += count

    eval_loss = total_loss / total_items

    if task == 'classification':
        eval_acc = 100.0 * correct / total_items
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        probs = torch.cat(all_probs)
        
        # Calculate F1 score
        eval_f1 = f1_score(targets.numpy(), preds.numpy(), average='weighted', zero_division=0)
        
        # Calculate average precision
        # Convert to one-hot encoding for multi-class
        #num_classes = probs.shape[1]
        #y_true_one_hot = torch.zeros(targets.size(0), num_classes)
        #y_true_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Calculate average precision for each class and then average
        eval_ap = average_precision_score(targets.numpy(), probs.numpy(), average='weighted')

        eval_roc_auc_val = roc_auc_score(targets.numpy(), probs.numpy(), average='macro')

        if node_level and class_total:
            class_accs = [class_correct[c] / class_total[c] for c in class_total]
            eval_w_acc = 100.0 * sum(class_accs) / len(class_accs)
            print(f"{phase} Loss: {eval_loss:.4f}, {phase} Accuracy: {eval_acc:.2f}%, {phase} F1 Score: {eval_f1:.4f}, {phase} AP: {eval_ap:.4f}, {phase} Weighted Accuracy: {eval_w_acc:.2f}%")
            return eval_loss, eval_acc, eval_f1, eval_ap, eval_w_acc
        else:
            print(f"{phase} Loss: {eval_loss:.4f}, {phase} Accuracy: {eval_acc:.2f}%, {phase} F1 Score: {eval_f1:.4f}, {phase} AP: {eval_ap:.4f}")
            return eval_loss, eval_acc, eval_f1, eval_ap

    else:  # regression
        # Divide by total_dimensions for proper mean error
        eval_mse = total_mse / total_dimensions
        eval_rmse = math.sqrt(eval_mse)
        eval_mae = total_mae / total_dimensions
        
        if multiclass_regression:
            print(f"{phase} Loss: {eval_loss:.4f}, {phase} MSE: {eval_mse:.4f}, {phase} RMSE: {eval_rmse:.4f}, {phase} MAE: {eval_mae:.4f} (Multiclass Regression)")
        else:
            print(f"{phase} Loss: {eval_loss:.4f}, {phase} MSE: {eval_mse:.4f}, {phase} RMSE: {eval_rmse:.4f}, {phase} MAE: {eval_mae:.4f}")
        
        return eval_loss, eval_mse, eval_rmse, eval_mae


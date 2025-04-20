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
import argparse
from CKGConv.utils.utility import load_datasets, load_yaml_as_namespace
from CKGConv.model.ckgnet import CKGNet
from CKGConv.utils.trainer import train, evaluate


def main(config_path):
    """
    Main function to run the CKGNet pipeline with separate train/val/test datasets.
    """
    args = load_yaml_as_namespace(config_path)
    warnings.filterwarnings("ignore")

    # for args in sweep_configs:
        # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    # Create TensorBoard writer
    from torch.utils.tensorboard import SummaryWriter
    log_dir = f"{args.log_dir}_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Print arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        # Also log hyperparameters to TensorBoard
        if arg not in ['device', 'model_path', 'log_dir']:
            writer.add_text("hyperparameters", f"{arg}: {getattr(args, arg)}", 0)

    # Setup device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    # Use TUDataset with train/val/test split
    print(f"\nLoading GNNDataset: {args.dataset}")

    train_dataset, val_dataset, test_dataset = load_datasets(args.dataset)
    
    
    # Extract dataset info
    num_node_features = train_dataset.num_node_features
    num_edge_features = train_dataset.num_edge_features
    
    # Determine num_classes or num_targets based on task
    if args.task == 'classification':
        num_classes = train_dataset.num_classes
        output_dim = num_classes
    else:  # regression
        # For regression, target dim is typically 1, but can be more
        # Assuming a single example gives you the shape of y
        sample = train_dataset[0]
        if hasattr(sample, 'y'):
            output_dim = 1 if sample.y.dim() == 0 else sample.y.size(-1)
        else:
            output_dim = 1
            print("Warning: Could not determine output dimension, using default of 1.")

    # Print dataset information
    print(f"Number of training graphs: {len(train_dataset)}")
    print(f"Number of validation graphs: {len(val_dataset)}")
    print(f"Number of test graphs: {len(test_dataset)}")
    if args.task == 'classification':
        print(f"Number of classes: {num_classes}")
    else:
        print(f"Output dimension: {output_dim}")
    print(f"Number of node features: {num_node_features}")
    print(f"Number of edge features: {num_edge_features}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model
    model = CKGNet(
        node_in_dim=num_node_features or 1,  # Use 1 if no node features
        edge_in_dim=num_edge_features or 1,  # Use 1 if no edge features
        pe_dim=args.pe_dim,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        ffn_ratio=args.ffn_ratio,
        pooling=args.pooling,
        num_classes=output_dim,
        task=args.task,
        norm_type=args.norm_type,
        add_self_loops=True,
        aggr=args.aggr,
        task_type=args.task_type,
        pe_type= args.pe_type
    ).to(device)

    # Log model graph to TensorBoard
    try:
        # Create a dummy input batch for the model graph
        dummy_batch = next(iter(train_loader)).to(device)
        # Run a forward pass to initialize parameters
        with torch.no_grad():
            _ = model(dummy_batch)
        writer.add_graph(model, dummy_batch)
    except Exception as e:
        print(f"Could not log model graph to TensorBoard: {e}")

    # Print model architecture and parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: CKGNet with {args.num_layers} layers")
    print(f"Total parameters: {total_params:,}")
    writer.add_text("model_info", f"Total parameters: {total_params:,}", 0)

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler = CosineWarmupScheduler(
    #         optimizer,
    #         warmup_epochs=args.warmup_epochs,
    #         max_epochs=args.epochs,
    #         min_lr=args.min_lr
    #     )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-4)

    # Training loop
    if args.task == 'classification':
        best_val_metric = 0  # For classification, higher accuracy is better
        monitor_metric = 'accuracy'
    else:
        best_val_metric = float('inf')  # For regression, lower MSE is better
        monitor_metric = 'mse'
        
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train for one epoch
        epoch_start = time.time()
        
        if args.task == 'classification':
            if args.dataset_type in ['pattern', 'cluster']:
                train_loss, train_acc, train_f1, train_ap, train_w_acc = train(
                    model, train_loader, optimizer, device, args.task, args.dataset_type)
            else:
                train_loss, train_acc, train_f1, train_ap, train_roc = train(
                    model, train_loader, optimizer, device, args.task, noise_val=args.noise_val)
        else:  # regression
            train_loss, train_mse, train_mae = train(
                model, train_loader, optimizer, device, args.task)
            
        epoch_time = time.time() - epoch_start

        # Evaluate on validation set
        if args.task == 'classification':
            if args.dataset_type in ['pattern', 'cluster']:
                val_loss, val_acc, val_f1, val_ap, val_w_acc = evaluate(
                    model, val_loader, device, args.task, args.dataset_type, "Validation")
            else:
                val_loss, val_acc, val_f1, val_ap = evaluate(
                    model, val_loader, device, args.task, None, "Validation")
        else:  # regression
            val_loss, val_mse, val_rmse, val_mae = evaluate(
                model, val_loader, device, args.task, None, "Validation")

        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics to TensorBoard
        writer.add_scalar('learning_rate', current_lr, epoch)
        writer.add_scalar('time/epoch', epoch_time, epoch)
        
        # Log training metrics
        writer.add_scalar('train/loss', train_loss, epoch)
        if args.task == 'classification':
            writer.add_scalar('train/accuracy', train_acc, epoch)
            writer.add_scalar('train/f1_score', train_f1, epoch)
            writer.add_scalar('train/average_precision', train_ap, epoch)
            # writer.add_scalar('train/roc', train_roc, epoch)  # Added AP to TensorBoard
            if args.dataset_type in ['pattern', 'cluster']:
                writer.add_scalar('train/weighted_accuracy', train_w_acc, epoch)
        else:  # regression
            writer.add_scalar('train/mse', train_mse, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)
            
        # Log validation metrics
        writer.add_scalar('val/loss', val_loss, epoch)
        if args.task == 'classification':
            writer.add_scalar('val/accuracy', val_acc, epoch)
            writer.add_scalar('val/f1_score', val_f1, epoch)
            writer.add_scalar('val/average_precision', val_ap, epoch)
            # writer.add_scalar('val/roc', val_roc, epoch)  # Added AP to TensorBoard
            if args.dataset_type in ['pattern', 'cluster']:
                writer.add_scalar('val/weighted_accuracy', val_w_acc, epoch)
        else:  # regression
            writer.add_scalar('val/mse', val_mse, epoch)
            writer.add_scalar('val/rmse', val_rmse, epoch)
            writer.add_scalar('val/mae', val_mae, epoch)

        # Print epoch summary
        print(f"\nEpoch Summary:")
        print(f"Time: {epoch_time:.2f}s, LR: {current_lr:.6f}")
        
        if args.task == 'classification':
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}, Train AP: {train_ap:.4f}")
            if args.dataset_type in ['pattern', 'cluster']:
                print(f"Train Weighted Acc: {train_w_acc:.2f}%")
                
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}, Val AP: {val_ap:.4f}")
            if args.dataset_type in ['pattern', 'cluster']:
                print(f"Val Weighted Acc: {val_w_acc:.2f}%")
                
            # Determine which metric to use for model selection
            if args.dataset_type in ['pattern', 'cluster']:
                current_val_metric = val_w_acc  # Use weighted accuracy
            elif args.dataset == 'Peptides-func':
                current_val_metric = val_ap
            else:
                current_val_metric = val_acc  # Use standard accuracy
        else:  # regression
            print(f"Train Loss: {train_loss:.4f}, Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}")
            
            # For regression, use MSE as the metric
            current_val_metric = val_mae

        # Check if this is the best model so far
        improved = False
        if args.task == 'classification':
            if current_val_metric > best_val_metric:
                improved = True
        else:  # regression - lower is better
            if current_val_metric < best_val_metric:
                improved = True
                
        if improved:
            best_val_metric = current_val_metric
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            if args.save_model:
                model_dir = os.path.dirname(args.model_path)
                if model_dir:
                    os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), args.model_path)
                print(f"Saved best model to {args.model_path}")
        else:
            patience_counter += 1
            if args.task == 'classification':
                print(f"No improvement for {patience_counter} epochs (best {monitor_metric}: {best_val_metric:.2f}% at epoch {best_epoch})")
            else:
                print(f"No improvement for {patience_counter} epochs (best {monitor_metric}: {best_val_metric:.4f} at epoch {best_epoch})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after epoch {epoch}")
            break

    # Log model weights histogram at end of training
    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param.clone().cpu().data.numpy(), 0)

    # Load best model for testing
    if args.save_model:
        model.load_state_dict(torch.load(args.model_path))

    # Evaluate on test set
    if args.task == 'classification':
        if args.dataset_type in ['pattern', 'cluster']:
            test_loss, test_acc, test_f1, test_ap, test_w_acc = evaluate(
                model, test_loader, device, args.task, args.dataset_type, "Test")
        else:
            test_loss, test_acc, test_f1, test_ap = evaluate(
                model, test_loader, device, args.task, None, "Test")
    else:  # regression
        test_loss, test_mse, test_rmse, test_mae = evaluate(
            model, test_loader, device, args.task, None, "Test")

    # Log test metrics to TensorBoard
    writer.add_scalar('test/loss', test_loss, 0)
    if args.task == 'classification':
        writer.add_scalar('test/accuracy', test_acc, 0)
        writer.add_scalar('test/f1_score', test_f1, 0)
        writer.add_scalar('test/average_precision', test_ap, 0)
        # writer.add_scalar('test/roc', test_roc, 0)  # Added AP to TensorBoard
        if args.dataset_type in ['pattern', 'cluster']:
            writer.add_scalar('test/weighted_accuracy', test_w_acc, 0)
    else:  # regression
        writer.add_scalar('test/mse', test_mse, 0)
        writer.add_scalar('test/rmse', test_rmse, 0)
        writer.add_scalar('test/mae', test_mae, 0)

    # Print final results
    print(f"\n{'#'*30} FINAL RESULTS {'#'*30}")
    
    if args.task == 'classification':
        metric_name = "Weighted Accuracy" if args.dataset_type in ['pattern', 'cluster'] else "Accuracy"
        print(f"Best Validation {metric_name}: {best_val_metric:.2f}% at epoch {best_epoch}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test Average Precision: {test_ap:.4f}")
        if args.dataset_type in ['pattern', 'cluster']:
            print(f"Test Weighted Accuracy: {test_w_acc:.2f}%")
    else:  # regression
        print(f"Best Validation MSE: {best_val_metric:.4f} at epoch {best_epoch}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
    
    # Add summary metrics as text
    writer.add_text("results", f"Best Validation {monitor_metric}: {best_val_metric:.4f} at epoch {best_epoch}", 0)

    # Close the TensorBoard writer
    writer.close()
    
    # Return test performance
    # if args.task == 'classification':
    #     return test_acc, test_f1, test_ap  # Return AP along with other metrics
    # else:
    #     return test_mse, test_rmse, test_mae


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

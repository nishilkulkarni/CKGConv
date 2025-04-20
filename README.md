
# CKGConv: General Graph Convolution with Continuous Kernels

This repository contains an unofficial implementation of CKGConv, a customizable continous kernel convolution Graph Neural Network (GNN) pipeline built with PyTorch Geometric. It supports both classification and regression tasks on graph-structured data. Refer to the paper for more architecture details - [arXiv](https://arxiv.org/abs/2404.13604)


## Installation

```bash
git clone https://github.com/nishilkulkarni/CKGConv.git
cd CKGConv
pip install -r requirements.txt
```

## Usage

### Load a pre-existing/Create a YAML config file

Example `config.yaml`:

```yaml
dataset: 'Peptides-func'
task: 'classification'
device: 0
epochs: 200
batch_size: 32
lr: 0.001
weight_decay: 0.0005
hidden_channels: 64
num_layers: 5
dropout: 0.3
patience: 20
save_model: true
model_path: 'models/ckgnet_best_model.pt'
log_dir: 'runs/ckgnet'
pe_dim: 16
ffn_ratio: 2
pooling: 'mean'
norm_type: 'batch'
aggr: 'add'
task_type: 'single'
pe_type: 'laplacian'
seed: 42
warmup_epochs: 10
min_lr: 0.0001
noise_val: 0.0
dataset_type: 'pattern'
```

### Run training

```bash
python main.py --config configs/mnist.yaml
```

## TensorBoard

Launch TensorBoard to visualize logs:

```bash
tensorboard --logdir results_{dataset_name}_{timestamp}
```

## License

This project is licensed under the MIT License.

## Contributions

Contributions are welcome! Feel free to fork the repo and submit a pull request.

# iDSGD-GE


## Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.10+
- PyTorch 1.13+
- CUDA 11.6+ (If you run using GPU)

## Installation

### Prepare environment

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n federated-env python=3.10 -y
    conda activate federated-env
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    ```
    or
    ```shell
    pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
    ```

## Run with Configurations

You can specify configurations to overwrite the default configurations.

```python
import src

# Define part of customized configs.
config = {
    "data": {"dataset": "Cifar10", "partition_type": "noniid_class"},
    "controller": {"rounds_iterations": 2000, "nodes_per_round": 10},
    "node": {"epoch_or_iteration": "iteration", "local_iteration": 1},
    "model": "resnet18"
}

# Define part of configs in a yaml file.
config_file = "config.yaml"
# Load and combine these two configs.
config = src.load_config(config_file, config)
# Initialize RobustFL with the new config.
src.init(config)
# Execute federated learning training.
src.run()
```

In the example above, we run training with model ResNet-18 and CIFAR-10 dataset that is partitioned into 10 nodes by label `class`.
It runs training with 10 nodes per iteration for 2000 iterations. In each iteration, each node trains 1 iteration.



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



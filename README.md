# Synthetic Noisy Datasets

## Quickstart
### PyTorch Style
```
from synthetic_noisy_datasets.datasets import NoisyMNIST
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

train_dataset = NoisyMNIST(root, 'symmetric', 0.2, transform=ToTensor())
val_dataset = MNIST(root, train=False, transform=ToTensor())
```

### PyTorch-Lightning Style
```
from synthetic_noisy_datasets.datamodules import NoisyMNIST
from torchvision.transforms import ToTensor

datamodule = NoisyMNIST(
    root,
    num_clean=100,          # small set of clean training data (something like Semi-Supervised Learning)
    batch_sizes={
        'clean': 64,        # if (num_clean == 0 or batch_sizes['clean'] == 0) then noisy-only mode
        'noisy': 448,
        'val': 512,
    },
    transforms={
        'clean': ToTensor(),
        'noisy': ToTensor(),
        'val': ToTensor(),
    },
    expand_clean=True,      # to reduce re-loading small clean data
    enumerate_noisy=True,   # (x, y) → (i, (x, y)), i is sample id (integer, 0 ~ len(dataset)-1)
)
```

# Noisy CIFAR-10/CIFAR-100

## Quickstart
```
from noisy_cifar import NoisyCIFAR10
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

root = '/datasets/CIFAR-10'
train_dataset = NoisyCIFAR10(root, 'symmetric', 0.2, transform=ToTensor())
val_dataset = CIFAR10(root, train=False, transform=ToTensor())
```

## Install
```bash
pip install noisy-cifar-owaix2quzq
```

## Arguments
- `root`: path to the original dataset
- `noise_type`: symmetric / asymmetric / human
- `noise_level`:
  - (float) [0.0 ~ 1.0] for symmetric & asymmetric
  - (str) {aggregate, random1, random2, random3, worst} for human
- `random_seed`: default `0`
- `transform`: default `None`
- `target_transform`: default `None`
- `download`: default `False`

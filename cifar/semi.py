from torchvision.datasets import CIFAR10, CIFAR100
from .base import BaseCIFAR

__all__ = ['SemiCIFAR10', 'SemiCIFAR100']


class SemiCIFAR(BaseCIFAR):
    splits = ['labeled', 'unlabeled', 'val']


class SemiCIFAR10(SemiCIFAR):
    num_classes = 10
    CIFAR = CIFAR10


class SemiCIFAR100(SemiCIFAR):
    num_classes = 100
    CIFAR = CIFAR100

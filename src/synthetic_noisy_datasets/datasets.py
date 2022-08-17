import os
from typing import Callable, Optional
import numpy as np
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from .utils import random_noisify, transition_matrix


__all__ = ['NoisyMNIST', 'NoisyCIFAR10', 'NoisyCIFAR100']


class NoisyMNIST(MNIST):
    def __init__(
        self,
        root: str,
        noise_type: str = 'symmetric',
        noise_ratio: float = 0.0,
        random_seed: int = 0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.random_seed = random_seed
        self.T = transition_matrix('MNIST', noise_type, noise_ratio)

        filename = f'noisy-{noise_type}-{noise_ratio}-{random_seed}.npy'
        path = os.path.join(root, filename)
        if os.path.isfile(path):
            self.targets = np.load(path)
        else:
            self.targets = random_noisify(self.targets, self.T, random_seed)
            np.save(path, self.targets)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', "processed")


class NoisyCIFAR10(CIFAR10):
    def __init__(
        self,
        root: str,
        noise_type: str = 'symmetric',
        noise_ratio: float = 0.0,
        random_seed: int = 0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.random_seed = random_seed
        self.T = transition_matrix('CIFAR10', noise_type, noise_ratio)

        filename = f'noisy-{noise_type}-{noise_ratio}-{random_seed}.npy'
        path = os.path.join(root, filename)
        if os.path.isfile(path):
            self.targets = np.load(path)
        else:
            self.targets = random_noisify(self.targets, self.T, random_seed)
            np.save(path, self.targets)


class NoisyCIFAR100(CIFAR100):
    def __init__(
        self,
        root: str,
        noise_type: str = 'symmetric',
        noise_ratio: float = 0.0,
        random_seed: int = 0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.random_seed = random_seed
        self.T = transition_matrix('CIFAR100', noise_type, noise_ratio)

        filename = f'noisy-{noise_type}-{noise_ratio}-{random_seed}.npy'
        path = os.path.join(root, filename)
        if os.path.isfile(path):
            self.targets = np.load(path)
        else:
            self.targets = random_noisify(self.targets, self.T, random_seed)
            np.save(path, self.targets)

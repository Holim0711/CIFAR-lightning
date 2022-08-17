import os
from typing import Callable, Optional
import numpy as np
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from .utils import random_noisify, transition_matrix


__all__ = ['NoisyMNIST', 'NoisyCIFAR10', 'NoisyCIFAR100']


def load_or_save_Ỹ(root, seed, Y, T):
    path = os.path.join(root, f'noisy-{seed}') + '.npy'
    if os.path.isfile(path):
        Ỹ = np.load(path)
    else:
        Ỹ = random_noisify(Y, T, seed)
        np.save(path, Ỹ)
    return Ỹ


class NoisyMNIST(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        noise_type: str = 'symmetric',
        noise_ratio: float = 0.0,
        random_seed: int = 0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.random_seed = random_seed
        self.T = transition_matrix('MNIST', noise_type, noise_ratio)
        self.targets = load_or_save_Ỹ(root, random_seed, self.targets, self.T)

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
        train: bool = True,
        noise_type: str = 'symmetric',
        noise_ratio: float = 0.0,
        random_seed: int = 0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.random_seed = random_seed
        self.T = transition_matrix('CIFAR10', noise_type, noise_ratio)
        self.targets = load_or_save_Ỹ(root, random_seed, self.targets, self.T)


class NoisyCIFAR100(CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool = True,
        noise_type: str = 'symmetric',
        noise_ratio: float = 0.0,
        random_seed: int = 0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.random_seed = random_seed
        self.T = transition_matrix('CIFAR100', noise_type, noise_ratio)
        self.targets = load_or_save_Ỹ(root, random_seed, self.targets, self.T)

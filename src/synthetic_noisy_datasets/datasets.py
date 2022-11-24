import os
from typing import Callable, Optional, Union
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from .utils import random_noisify, transition_matrix


__all__ = ['NoisyCIFAR10', 'NoisyCIFAR100']


class NoisyCIFAR10(CIFAR10):
    def __init__(
        self,
        root: str,
        noise_type: str,
        noise_level: Union[float, str],
        random_seed: int = 0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        show_indices: bool = False,
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
        self.noise_level = noise_level
        self.random_seed = random_seed
        self.show_indices = show_indices

        if noise_type == 'human':
            noise_level = {
                'clean': 'clean_label',
                'aggregate': 'aggre_label',
                'random1': 'random_label1',
                'random2': 'random_label2',
                'random3': 'random_label3',
                'worst': 'worse_label',
            }[noise_level]
            path = os.path.join(os.path.dirname(__file__), 'CIFARN', 'CIFAR-10_human.pt')
            self.targets = torch.load(path)[noise_level]
        else:
            T = transition_matrix('CIFAR10', noise_type, noise_level)
            self.targets = random_noisify(self.targets, T, random_seed)

    def __getitem__(self, index: int):
        xy = super().__getitem__(index)
        return (index, xy) if self.show_indices else xy

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = True,
    ):
        return DataLoader(self, batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)


class NoisyCIFAR100(CIFAR100):
    def __init__(
        self,
        root: str,
        noise_type: str,
        noise_level: Union[float, str],
        random_seed: int = 0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        show_indices: bool = False
    ):
        super().__init__(
            root,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
        self.noise_type = noise_type
        self.noise_ratio = noise_level
        self.random_seed = random_seed
        self.show_indices = show_indices

        if noise_type == 'human':
            noise_level = {
                'clean': 'clean_label',
                'noisy': 'noisy_label',
            }[noise_level]
            path = os.path.join(os.path.dirname(__file__), 'CIFARN', 'CIFAR-100_human.pt')
            self.targets = torch.load(path)['noisy_label']
        else:
            T = transition_matrix('CIFAR100', noise_type, noise_level)
            self.targets = random_noisify(self.targets, T, random_seed)

    def __getitem__(self, index: int):
        xy = super().__getitem__(index)
        return (index, xy) if self.show_indices else xy

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = True,
    ):
        return DataLoader(self, batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)

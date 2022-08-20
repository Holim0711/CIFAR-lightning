import os
import warnings
from math import ceil
from typing import Callable, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from .datasets import (
    NoisyMNIST as DatasetNoisyMNIST,
    NoisyCIFAR10 as DatasetNoisyCIFAR10,
    NoisyCIFAR100 as DatasetNoisyCIFAR100,
)
from .utils import random_select, transition_matrix


__all__ = ['NoisyMNIST', 'NoisyCIFAR10', 'NoisyCIFAR100']


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


class NoisyDataModule(LightningDataModule):

    CLEAN = None
    NOISY = None

    def __init__(
        self,
        root: str,
        num_clean: int = 0,
        noise_type: str = 'symmetric',
        noise_ratio: float = 0.0,
        random_seed: int = 0,
        transforms: dict[str, Callable] = {},
        batch_sizes: dict[str, int] = {},
        expand_clean: bool = False,
        enumerate_noisy: bool = False,
    ):
        super().__init__()
        self.root = root
        self.num_clean = num_clean
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.random_seed = random_seed

        splits = ['clean', 'noisy', 'val']
        self.transforms = {k: transforms.get(k, ToTensor()) for k in splits}
        self.batch_sizes = {k: batch_sizes.get(k, 1) for k in splits}

        for k in transforms:
            if k not in splits:
                warnings.warn(f"'{k}' in transforms is ignored")
        for k in batch_sizes:
            if k not in splits:
                warnings.warn(f"'{k}' in batch_sizes is ignored")

        dataset = self.CLEAN.__name__
        self.T = transition_matrix(dataset, noise_type, noise_ratio)

        self.expand_clean = expand_clean and self.with_clean
        self.enumerate_noisy = enumerate_noisy

    @property
    def with_clean(self):
        return self.num_clean > 0 and self.batch_sizes['clean'] > 0

    def prepare_data(self):
        self.NOISY(self.root,
                   noise_type=self.noise_type,
                   noise_ratio=self.noise_ratio,
                   random_seed=self.random_seed,
                   download=True)

    def get_dataset(self, split: str, transform: Optional[Callable] = None):
        if split == 'clean':
            d = self.CLEAN(self.root, transform=transform)
            i = random_select(d.targets, self.num_clean, self.random_seed)
            return Subset(d, i)
        elif split == 'noisy':
            return self.NOISY(self.root,
                              noise_type=self.noise_type,
                              noise_ratio=self.noise_ratio,
                              random_seed=self.random_seed,
                              transform=transform)
        elif split == 'val':
            return self.CLEAN(self.root, train=False, transform=transform)

        raise ValueError(f'Unknwon dataset split: {split}')

    def setup(self, stage=None):
        clean = self.get_dataset('clean', self.transforms['clean'])
        noisy = self.get_dataset('noisy', self.transforms['noisy'])
        val = self.get_dataset('val', self.transforms['val'])

        if self.expand_clean:
            m = ((len(noisy) * self.batch_sizes['clean']) /
                 (len(clean) * self.batch_sizes['noisy'] * 2))
            clean = ConcatDataset([clean] * max(ceil(m), 1))

        if self.enumerate_noisy:
            noisy = IndexedDataset(noisy)

        self._datasets = {'clean': clean, 'noisy': noisy, 'val': val}

    def get_dataloader(
        self,
        split: str,
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = True
    ):
        return DataLoader(
            self._datasets[split],
            self.batch_sizes[split],
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def train_dataloader(self):
        if self.with_clean:
            return {'clean': self.get_dataloader('clean'),
                    'noisy': self.get_dataloader('noisy')}
        return self.get_dataloader('noisy')

    def val_dataloader(self):
        return self.get_dataloader('val', shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()


class NoisyMNIST(NoisyDataModule):
    CLEAN = MNIST
    NOISY = DatasetNoisyMNIST


class NoisyCIFAR10(NoisyDataModule):
    CLEAN = CIFAR10
    NOISY = DatasetNoisyCIFAR10


class NoisyCIFAR100(NoisyDataModule):
    CLEAN = CIFAR100
    NOISY = DatasetNoisyCIFAR100

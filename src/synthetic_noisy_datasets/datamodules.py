import os
import warnings
from math import ceil
from typing import Callable, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import (
    MNIST as C_MNIST,
    CIFAR10 as C_CIFAR10,
    CIFAR100 as C_CIFAR100,
)
from .datasets import (
    NoisyMNIST as N_MNIST,
    NoisyCIFAR10 as N_CIFAR10,
    NoisyCIFAR100 as N_CIFAR100,
)
from .utils import random_select, transition_matrix


__all__ = ['NoisyMNIST', 'NoisyCIFAR10', 'NoisyCIFAR100']
C_CLASS = {'MNIST': C_MNIST, 'CIFAR10': C_CIFAR10, 'CIFAR100': C_CIFAR100}
N_CLASS = {'MNIST': N_MNIST, 'CIFAR10': N_CIFAR10, 'CIFAR100': N_CIFAR100}


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


class NoisyDataModule(LightningDataModule):

    def __init__(
        self,
        name: str,
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
        self.CLEAN = C_CLASS[name]
        self.NOISY = N_CLASS[name]

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

        self.T = transition_matrix(name, noise_type, noise_ratio)

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
        if transform is None:
            transform = self.transforms[split]

        if split == 'clean':
            dataset = self.CLEAN(self.root, transform=transform)
            indices = random_select(dataset.targets,
                                    self.num_clean, self.random_seed)
            return Subset(dataset, indices)
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
        clean = self.get_dataset('clean')
        noisy = self.get_dataset('noisy')
        val = self.get_dataset('val')

        if self.expand_clean:
            m = ((len(noisy) * self.batch_sizes['clean']) /
                 (len(clean) * self.batch_sizes['noisy'] * 2))
            m = max(ceil(m), 1)
            clean = ConcatDataset([clean] * m)

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
        super().__init__(
            'MNIST',
            root,
            num_clean=num_clean,
            noise_type=noise_type,
            noise_ratio=noise_ratio,
            random_seed=random_seed,
            transforms=transforms,
            batch_sizes=batch_sizes,
            expand_clean=expand_clean,
            enumerate_noisy=enumerate_noisy,
        )


class NoisyCIFAR10(NoisyDataModule):
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
        super().__init__(
            'CIFAR10',
            root,
            num_clean=num_clean,
            noise_type=noise_type,
            noise_ratio=noise_ratio,
            random_seed=random_seed,
            transforms=transforms,
            batch_sizes=batch_sizes,
            expand_clean=expand_clean,
            enumerate_noisy=enumerate_noisy,
        )


class NoisyCIFAR100(NoisyDataModule):
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
        super().__init__(
            'CIFAR100',
            root,
            num_clean=num_clean,
            noise_type=noise_type,
            noise_ratio=noise_ratio,
            random_seed=random_seed,
            transforms=transforms,
            batch_sizes=batch_sizes,
            expand_clean=expand_clean,
            enumerate_noisy=enumerate_noisy,
        )

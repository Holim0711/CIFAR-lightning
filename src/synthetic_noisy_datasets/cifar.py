import os
import warnings
from math import ceil
from numpy.random import RandomState
from typing import Callable, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10, CIFAR100
from .utils import *

CIFAR = {10: CIFAR10, 100: CIFAR100}

CLASSES = {
    10: [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ],
    100: [
        'beaver', 'dolphin', 'otter', 'seal', 'whale',
        'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
        'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
        'bottles', 'bowls', 'cans', 'cups', 'plates',
        'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
        'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
        'bed', 'chair', 'couch', 'table', 'wardrobe',
        'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
        'bear', 'leopard', 'lion', 'tiger', 'wolf',
        'bridge', 'castle', 'house', 'road', 'skyscraper',
        'cloud', 'forest', 'mountain', 'plain', 'sea',
        'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
        'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
        'crab', 'lobster', 'snail', 'spider', 'worm',
        'baby', 'boy', 'girl', 'man', 'woman',
        'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
        'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
        'maple', 'oak', 'palm', 'pine', 'willow',
        'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
        'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor',
    ]
}


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


class NoisyCIFAR(LightningDataModule):

    splits = ['clean', 'noisy', 'val']
    num_classes = None

    def __init__(
        self,
        root: str,
        num_clean: int,
        noise_type: str = 'symmetric',
        noise_ratio: float = 0.0,
        transforms: dict[str, Callable] = {},
        batch_sizes: dict[str, int] = {},
        random_seed: Optional[int] = 0,
        enum_noisy: bool = False,
        pure_noisy: bool = False,
        keep_original: bool = False,
    ):
        super().__init__()
        self.CIFAR = CIFAR[self.num_classes]
        self.classes = CLASSES[self.num_classes]

        self.root = root
        self.num_clean = num_clean
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.transforms = [transforms.get(k, ToTensor()) for k in self.splits]
        self.batch_sizes = {k: batch_sizes.get(k, 1) for k in self.splits}
        self.random_seed = random_seed
        self.enum_noisy = enum_noisy
        self.pure_noisy = pure_noisy
        self.keep_original = keep_original

        for k in transforms:
            if k not in self.splits:
                warnings.warn(f"'{k}' in transforms is ignored")
        for k in batch_sizes:
            if k not in self.splits:
                warnings.warn(f"'{k}' in batch_sizes is ignored")

        self.T = transition_matrix(f'CIFAR{self.num_classes}',
                                   noise_type, noise_ratio)

    def prepare_data(self):
        self.CIFAR(self.root, download=True)

    def setup(self, stage=None):
        P = self.CIFAR(self.root, transform=self.transforms[0])
        U = self.CIFAR(self.root, transform=self.transforms[1])
        V = self.CIFAR(self.root, train=False, transform=self.transforms[2])

        random_state = RandomState(self.random_seed)
        indices = random_select(P.targets, self.num_clean, random_state)
        P = Subset(P, indices)

        random_state = RandomState(self.random_seed)
        U.targets = random_noisify(U.targets, self.T, random_state)

        try:
            m = ((len(U) * self.batch_sizes[self.splits[0]]) /
                 (len(P) * self.batch_sizes[self.splits[1]] * 2))
        except ZeroDivisionError:
            m = 1

        P = ConcatDataset([P] * max(ceil(m), 1))

        if self.enum_noisy:
            U = IndexedDataset(U)

        if self.pure_noisy:
            indices = set(indices)
            indices = [x for x in range(len(U)) if x not in indices]
            U = Subset(U, indices)

        self.datasets = dict(zip(self.splits, [P, U, V]))

    def dataloader(
        self, k: str,
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = True
    ):
        return DataLoader(
            self.datasets[k],
            self.batch_sizes[k],
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def train_dataloader(self):
        if self.num_clean and self.batch_sizes[self.splits[0]]:
            return {self.splits[0]: self.dataloader(self.splits[0]),
                    self.splits[1]: self.dataloader(self.splits[1])}
        else:
            return self.dataloader(self.splits[1])

    def val_dataloader(self):
        return self.dataloader(self.splits[2], shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()


class NoisyCIFAR10(NoisyCIFAR):
    num_classes = 10


class NoisyCIFAR100(NoisyCIFAR):
    num_classes = 100

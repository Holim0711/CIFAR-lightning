import os
import math
import numpy
from typing import Callable, Optional
from collections import defaultdict
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10, CIFAR100
from .utils import random_select


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
CIFAR100_CLASSES = [
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


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


def make_subaset(dataset, indices):
    dataset.data = dataset.data[indices]
    dataset.targets = numpy.array(dataset.targets)[indices]


class DeficientCIFAR(LightningDataModule):

    def __init__(
        self, root: str, num_proved: int,
        transforms: dict[str, Callable] = {},
        batch_sizes: dict[str, int] = {},
        random_seed: Optional[int] = 1234,
        show_sample_indices: bool = False,
        validation_set_size: int = 0,
        exclude_proved_from_unproved: bool = False,
    ):
        super().__init__()
        if self.num_classes == 10:
            self.CIFAR = CIFAR10
            self.classes = CIFAR10_CLASSES
        elif self.num_classes == 100:
            self.CIFAR = CIFAR100
            self.classes = CIFAR100_CLASSES

        self.root = root
        self.num_proved = num_proved
        self.num_samples = {
            self.splits[0]: num_proved,
            self.splits[1]: 50000 - validation_set_size,
            'val': validation_set_size or 10000,
            'test': 10000,
        }
        self.transforms = defaultdict(lambda: ToTensor(), transforms)
        self.batch_sizes = defaultdict(lambda: 1, batch_sizes)
        self.random_seed = random_seed
        self.show_sample_indices = show_sample_indices
        self.validation_set_size = validation_set_size
        self.exclude_proved_from_unproved = exclude_proved_from_unproved

        assert all(k in self.splits for k in transforms), "key error"
        assert all(k in self.splits for k in batch_sizes), "key error"

    def prepare_data(self):
        self.CIFAR(self.root, download=True)

    def setup(self, stage=None):
        random_state = numpy.random.RandomState(self.random_seed)

        P = self.CIFAR(self.root, transform=self.transforms[self.splits[0]])
        U = self.CIFAR(self.root, transform=self.transforms[self.splits[1]])
        V = self.CIFAR(self.root, train=(self.validation_set_size != 0), transform=self.transforms['val'])
        T = self.CIFAR(self.root, train=False, transform=self.transforms['val'])

        if self.validation_set_size != 0:
            indices = self.setup_val(V, random_state)
            self.setup_train(P, indices, random_state)
            self.setup_train(U, indices, random_state)

        indices = self.setup_proved(P, random_state)
        self.setup_unproved(U, indices, random_state)

        try:
            m = math.ceil((len(U) * self.batch_sizes[self.splits[0]]) /
                          (len(P) * self.batch_sizes[self.splits[1]] * 2))
        except ZeroDivisionError:
            m = 1

        P = ConcatDataset([P] * m)

        if self.show_sample_indices:
            U = IndexedDataset(U)

        self.datasets = {
            self.splits[0]: P,
            self.splits[1]: U,
            'val': V,
            'test': T,
        }

    def setup_val(self, V, random_state):
        indices = random_select(V.targets, self.validation_set_size, random_state)
        make_subaset(V, indices)
        return indices

    def setup_train(self, trainset, val_indices, random_state):
        val_indices = set(val_indices)
        indices = [x for x in range(len(trainset)) if x not in val_indices]
        make_subaset(trainset, indices)

    def setup_proved(self, P, random_state):
        indices = random_select(P.targets, self.num_proved, random_state)
        make_subaset(P, indices)
        return indices

    def setup_unproved(self, U, proved_indices, random_state):
        if self.exclude_proved_from_unproved:
            proved_indices = set(proved_indices)
            indices = [x for x in range(len(U)) if x not in proved_indices]
            make_subaset(U, indices)

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
        if self.num_proved and self.batch_sizes[self.splits[0]]:
            return {self.splits[0]: self.dataloader(self.splits[0]),
                    self.splits[1]: self.dataloader(self.splits[1])}
        else:
            return self.dataloader(self.splits[1])

    def val_dataloader(self):
        return self.dataloader('val', shuffle=False)

    def test_dataloader(self):
        return self.dataloader('test', shuffle=False)

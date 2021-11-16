import os
import math
import numpy
from typing import Callable, Optional
from collections.abc import Mapping
from collections import defaultdict
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10, CIFAR100
from .utils import random_select, IndexedDataset


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


class DeficientCIFAR(LightningDataModule):

    def __init__(
        self,
        root: str,
        n: int,
        transforms: Mapping[str, Callable] = {},
        batch_sizes: Mapping[str, int] = {},
        random_seed: Optional[int] = 1234,
        show_indices: bool = False,
    ):
        super().__init__()
        if self.num_classes == 10:
            self.CIFAR = CIFAR10
            self.classes = CIFAR10_CLASSES
        elif self.num_classes == 100:
            self.CIFAR = CIFAR100
            self.classes = CIFAR100_CLASSES
        else:
            raise ValueError(f'num_classes error: {self.num_classes}')
        self.root = root
        self.n = n
        self.transforms = defaultdict(lambda: ToTensor(), **transforms)
        self.batch_sizes = defaultdict(lambda: 1, **batch_sizes)
        self.random_seed = random_seed
        self.show_indices = show_indices
        assert all(k in self.splits for k in transforms), "key error"
        assert all(k in self.splits for k in batch_sizes), "key error"

    def prepare_data(self):
        self.CIFAR(self.root, download=True)

    def setup(self, stage=None):
        random_state = numpy.random.RandomState(self.random_seed)

        d0 = self.CIFAR(self.root, transform=self.transforms[self.splits[0]])
        d1 = self.CIFAR(self.root, transform=self.transforms[self.splits[1]])
        d2 = self.CIFAR(self.root, train=False,
                        transform=self.transforms[self.splits[2]])

        self.setup_d0(d0, random_state)  # setup for the clean/labeled
        self.setup_d1(d1, random_state)  # setup for the noisy/unlabeled

        if self.show_indices:
            d0 = IndexedDataset(d0)
            d1 = IndexedDataset(d1)

        try:
            m = math.ceil((len(d1) * self.batch_sizes[self.splits[0]]) /
                          (len(d0) * self.batch_sizes[self.splits[1]] * 2))
        except ZeroDivisionError:
            m = 1

        d0 = ConcatDataset([d0] * m)

        self.datasets = dict(zip(self.splits, [d0, d1, d2]))

    def setup_d0(self, d0, random_state):
        indices = random_select(d0.targets, self.n, random_state)
        d0.data = d0.data[indices]
        d0.targets = numpy.array(d0.targets)[indices]

    def setup_d1(self, d1, random_state):
        pass

    def dataloader(self, k: str):
        return DataLoader(
            self.datasets[k],
            self.batch_sizes[k],
            shuffle=(k != self.splits[2]),
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def train_dataloader(self):
        return {self.splits[0]: self.dataloader(self.splits[0]),
                self.splits[1]: self.dataloader(self.splits[1])}

    def val_dataloader(self):
        return self.dataloader(self.splits[2])

    def test_dataloader(self):
        return self.val_dataloader()

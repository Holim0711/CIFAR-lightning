import os
import math
import numpy
from typing import Callable, Optional
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
        self, root: str, num_proved: int,
        transforms: dict[str, Callable] = {},
        batch_sizes: dict[str, int] = {},
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

        self.root = root
        self.n = dict(zip(self.splits, [num_proved, 50000, 10000]))
        self.transforms = defaultdict(lambda: ToTensor(), **transforms)
        self.batch_sizes = defaultdict(lambda: 1, **batch_sizes)
        self.random_seed = random_seed
        self.show_indices = show_indices

        assert all(k in self.splits for k in transforms), "key error"
        assert all(k in self.splits for k in batch_sizes), "key error"

    def prepare_data(self):
        self.CIFAR(self.root, download=True)

    def setup(self, stage=None):
        rng = numpy.random.RandomState(self.random_seed)

        Dₚ = self.CIFAR(self.root, True, self.transforms[self.splits[0]])
        Dᵤ = self.CIFAR(self.root, True, self.transforms[self.splits[1]])
        Dᵥ = self.CIFAR(self.root, False, self.transforms[self.splits[2]])

        self.setup_proved(Dₚ, rng)
        self.setup_unproved(Dᵤ, rng)

        if self.show_indices:
            Dₚ = IndexedDataset(Dₚ)
            Dᵤ = IndexedDataset(Dᵤ)

        try:
            m = math.ceil((len(Dᵤ) * self.batch_sizes[self.splits[0]]) /
                          (len(Dₚ) * self.batch_sizes[self.splits[1]] * 2))
            Dₚ = ConcatDataset([Dₚ] * m)
        except ZeroDivisionError:
            pass

        self.datasets = dict(zip(self.splits, [Dₚ, Dᵤ, Dᵥ]))

    def setup_proved(self, Dₚ, rng):
        indices = random_select(Dₚ.targets, self.n[self.splits[0]], rng)
        Dₚ.data = Dₚ.data[indices]
        Dₚ.targets = numpy.array(Dₚ.targets)[indices]

    def setup_unproved(self, Dᵤ, rng):
        pass

    def dataloader(
        self, k: str,
        shuffle: Optional[bool] = None,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = True
    ):
        return DataLoader(
            self.datasets[k],
            self.batch_sizes[k],
            shuffle=(k != self.splits[2]) if shuffle is None else shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def train_dataloader(self):
        if self.n[self.splits[0]] and self.batch_sizes[self.splits[0]]:
            return {self.splits[0]: self.dataloader(self.splits[0]),
                    self.splits[1]: self.dataloader(self.splits[1])}
        else:
            return self.dataloader(self.splits[1])

    def val_dataloader(self):
        return self.dataloader(self.splits[2])

    def test_dataloader(self):
        return self.val_dataloader()

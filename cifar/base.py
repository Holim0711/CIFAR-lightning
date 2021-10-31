import os
import math
import numpy
from typing import Callable, Optional
from collections.abc import Mapping
from collections import defaultdict
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import ToTensor
from .utils import random_select


class BaseCIFAR(LightningDataModule):

    def __init__(
        self,
        root: str,
        n: int,
        transforms: Mapping[str, Callable] = {},
        batch_sizes: Mapping[str, int] = {},
        random_seed: Optional[int] = 1234,
    ):
        super().__init__()
        self.root = root
        self.n = n
        self.transforms = defaultdict(lambda: ToTensor(), **transforms)
        self.batch_sizes = defaultdict(lambda: 1, **batch_sizes)
        self.random_seed = random_seed
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

        m = math.ceil((len(d1) * self.batch_sizes[self.splits[0]]) /
                      (len(d0) * self.batch_sizes[self.splits[1]] * 2))
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

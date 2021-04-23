import os
import numpy
import torch
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torch.utils.data import ConcatDataset
from .noisify import noisify_cifar10


class SubsetCIFAR10(CIFAR10):
    def __init__(self, root, indices, **kwargs):
        super().__init__(root, **kwargs)
        assert len(indices) < len(self)
        self.data = self.data[indices]
        self.targets = torch.tensor(self.targets)[indices]


def index_select(labels, n, random_state):
    labels = numpy.array(labels)
    classes = numpy.unique(labels)
    n_per_class = n // len(classes)

    clean_indices = []
    for c in classes:
        class_indices = numpy.where(labels == c)[0]
        chosen_indices = random_state.choice(class_indices, size=n_per_class, replace=False)
        clean_indices.extend(chosen_indices)
    return clean_indices


class NoisyCIFAR10(pl.LightningDataModule):

    def __init__(self, root, num_labeled=4000, seed=None,
                 batch_size=1, valid_batch_size=None,
                 noise_type='asymmetric', noise_ratio=0.2,
                 num_workers=None, pin_memory=True, expand_labeled=True):
        super().__init__()
        self.root = root
        self.num_labeled = num_labeled

        if isinstance(batch_size, dict):
            self.batch_sizeₗ = batch_size['clean']
            self.batch_sizeₙ = batch_size['noisy']
        elif isinstance(batch_size, int):
            self.batch_sizeₗ = batch_size
            self.batch_sizeₙ = batch_size
        else:
            raise ValueError("batch_size should be 'int' or 'dict'")

        if valid_batch_size is not None:
            self.valid_batch_size = valid_batch_size
        else:
            self.valid_batch_size = max(self.batch_sizeₗ, self.batch_sizeₙ) * 2

        self.noise_type = noise_type
        self.noise_ratio = noise_ratio

        self.train_transformₗ = None
        self.train_transformᵤ = None
        self.valid_transform = None

        self.random_state = numpy.random.RandomState(seed)
        self.num_workers = num_workers if num_workers else os.cpu_count()
        self.pin_memory = pin_memory
        self.expand_labeled = expand_labeled

    def prepare_data(self):
        CIFAR10(self.root, train=True, download=True)
        CIFAR10(self.root, train=False, download=True)

    def setup(self, stage=None):
        self.cifar10_valid = CIFAR10(self.root, train=False, transform=self.valid_transform)
        self.cifar10_trainₙ = CIFAR10(self.root, train=True, transform=self.train_transformᵤ)

        indices = index_select(self.cifar10_trainₙ.targets, self.num_labeled, self.random_state)
        self.cifar10_trainₗ = SubsetCIFAR10(self.root, indices, train=True, transform=self.train_transformₗ)

        if self.expand_labeled and self.batch_sizeₗ > 0:
            n = 1 + (len(self.cifar10_trainₙ) - 1) // self.batch_sizeₙ
            m = n * self.batch_sizeₗ // len(indices)
            self.cifar10_trainₗ = ConcatDataset([self.cifar10_trainₗ] * m)

        noisy_targets = noisify_cifar10(self.cifar10_trainₙ.targets,
                                        noise_type=self.noise_type,
                                        noise_ratio=self.noise_ratio,
                                        random_state=self.random_state)
        self.cifar10_trainₙ.targets = list(zip(noisy_targets, self.cifar10_trainₙ.targets))

    def train_dataloader(self):
        loaderₙ = torch.utils.data.DataLoader(
            self.cifar10_trainₙ, self.batch_sizeₙ, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory)
        if self.batch_sizeₗ > 0:
            loaderₗ = torch.utils.data.DataLoader(
                self.cifar10_trainₗ, self.batch_sizeₗ, shuffle=True,
                num_workers=self.num_workers, pin_memory=self.pin_memory)
            return {'clean': loaderₗ, 'noisy': loaderₙ}
        else:
            return {'noisy': loaderₙ}

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar10_valid, self.valid_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    dm = NoisyCIFAR10(**{
        "root": "data/cifar10",
        "num_labeled": 4000,
        "noise_type": 'asymmetric',
        "noise_ratio": 0.2,
        "batch_size": {
            "clean": 64,
            "noisy": 448
        }
    })

    from torchvision import transforms
    transform = transforms.ToTensor()
    dm.train_transformₗ = transform
    dm.train_transformᵤ = transform
    dm.valid_transform = transform

    dm.prepare_data()
    dm.setup()

    print("dataset train-clean:", len(dm.cifar10_trainₗ))
    print("dataset train-noisy:", len(dm.cifar10_trainₙ))
    print("dataset valid:", len(dm.cifar10_valid))
    print("loader train-clean:", len(dm.train_dataloader()['clean']))
    print("loader train-noisy:", len(dm.train_dataloader()['noisy']))
    print("loader valid:", len(dm.val_dataloader()))

    T = numpy.zeros((10, 10), dtype=int)
    for x, y in dm.train_dataloader()['noisy']:
        for flipped, original in zip(*y):
            T[original][flipped] += 1
    T = T / T.sum(axis=1)
    print(T)

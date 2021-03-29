import os
import torch
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from .noisify import noisify_cifar10


class SubsetCIFAR10(CIFAR10):
    def __init__(self, root, indices=None, **kwargs):
        super().__init__(root, **kwargs)
        if indices is not None:
            self.data = self.data[indices]
            self.targets = torch.tensor(self.targets)[indices]


def index_split(labels, num_labeled):
    labels = torch.tensor(labels)
    classes = labels.unique()
    labels_per_class = num_labeled // len(classes)

    idxₗ, idxᵤ = [], []

    for c in classes:
        indices = torch.where(labels == c)[0]
        randperm = torch.randperm(len(indices))
        idxₗ.extend(indices[randperm[:labels_per_class]])
        idxᵤ.extend(indices[randperm[labels_per_class:]])

    return idxₗ, idxᵤ


class NoisyCIFAR10(pl.LightningDataModule):

    def __init__(self, root, num_labeled=4000, batch_size=1,
                 noise_ratio=0.2, noise_type='asymmetric',
                 num_workers=None, pin_memory=True):
        super().__init__()
        self.root = root
        self.num_labeled = num_labeled
        if isinstance(batch_size, dict):
            self.batch_sizeₗ = batch_size['labeled']
            self.batch_sizeᵤ = batch_size['unlabeled']
        else:
            self.batch_sizeₗ = batch_size
            self.batch_sizeᵤ = batch_size
        self.noise_ratio = noise_ratio
        self.noise_type = noise_type
        self.train_transformₗ = None
        self.train_transformᵤ = None
        self.valid_transform = None
        self.num_workers = num_workers if num_workers else os.cpu_count()
        self.pin_memory = pin_memory

    def prepare_data(self):
        CIFAR10(self.root, train=True, download=True)
        CIFAR10(self.root, train=False, download=True)

    def setup(self, stage=None):
        cifar10_train = CIFAR10(self.root, train=True)

        train_idxₗ, train_idxₙ = index_split(cifar10_train.targets, self.num_labeled)

        self.cifar10_trainₗ = SubsetCIFAR10(
            self.root, train_idxₗ, train=True, transform=self.train_transformₗ)
        self.cifar10_trainₙ = SubsetCIFAR10(
            self.root, train_idxₙ, train=True, transform=self.train_transformᵤ)
        self.cifar10_trainₙ.targets = noisify_cifar10(
            self.cifar10_trainₙ.targets, noise_ratio=self.noise_ratio, noise_type=self.noise_type)
        self.cifar10_test = CIFAR10(
            self.root, train=False, transform=self.valid_transform)

    def train_dataloader(self):
        loaderₗ = torch.utils.data.DataLoader(
            self.cifar10_trainₗ, self.batch_sizeₗ, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory)
        loaderₙ = torch.utils.data.DataLoader(
            self.cifar10_trainₙ, self.batch_sizeᵤ, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory)
        return {'clean': loaderₗ, 'noisy': loaderₙ}

    def val_dataloader(self):
        batch_size = max(self.batch_sizeₗ, self.batch_sizeᵤ) * 2
        return torch.utils.data.DataLoader(
            self.cifar10_test, batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    from torchvision import transforms
    trfm = transforms.ToTensor()
    dm = NoisyCIFAR10(**{
        "root": "data/cifar10",
        "num_labeled": 4000,
        "batch_size": {
            "labeled": 64,
            "unlabeled": 448
        }
    })
    dm.train_transformₗ = trfm
    dm.train_transformᵤ = trfm
    dm.valid_transform = trfm
    dm.prepare_data()
    dm.setup()
    print("clean dataset:", len(dm.cifar10_trainₗ))
    print("noisy dataset:", len(dm.cifar10_trainₙ))
    print("test dataset:", len(dm.cifar10_test))
    print("clean loader:", len(dm.train_dataloader()['clean']))
    print("noisy loader:", len(dm.train_dataloader()['noisy']))
    print("test loader:", len(dm.val_dataloader()))
    loader = dm.train_dataloader()
    for c, n in zip(loader['clean'], loader['noisy']):
        print(c)
        print(n)

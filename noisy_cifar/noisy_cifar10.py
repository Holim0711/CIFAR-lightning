import os
import numpy
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torch.utils.data import ConcatDataset, DataLoader

from .index_choice import index_choice
from .noisify import transition_matrix_cifar10, noisify


class NoisyCIFAR10(pl.LightningDataModule):

    def __init__(self, root, num_clean, expand_clean=False,
                 noise_type='asymmetric', noise_ratio=0.2,
                 batch_size={'clean': 0, 'noisy': 1, 'valid': None},
                 dataloader_args={
                     'clean': {'shuffle': True, 'num_workers': None, 'pin_memory': True},
                     'noisy': {'shuffle': True, 'num_workers': None, 'pin_memory': True},
                     'valid': {'shuffle': False, 'num_workers': None, 'pin_memory': True}
                 },
                 random_seed=None):
        super().__init__()
        self.root = root
        self.num_clean = num_clean
        self.expand_clean = expand_clean
        self.random_state = numpy.random.RandomState(random_seed)

        self.T = transition_matrix_cifar10(noise_type, noise_ratio)

        self.batch_size = dict(batch_size)
        train_batch_size = batch_size['clean'] + batch_size['noisy']

        if self.batch_size.get('valid') is None:
            self.batch_size['valid'] = train_batch_size

        self.dataloader_args = dict(dataloader_args)
        if all(('num_workers' in args and args['num_workers'] is None) for args in self.dataloader_args.values()):
            n = os.cpu_count()
            self.dataloader_args['clean']['num_workers'] = int(n * (self.batch_size['clean'] / train_batch_size))
            self.dataloader_args['noisy']['num_workers'] = int(n * (self.batch_size['noisy'] / train_batch_size))
            self.dataloader_args['valid']['num_workers'] = n

        # need to be set
        self.clean_transform = None
        self.noisy_transform = None
        self.valid_transform = None

    def prepare_data(self):
        CIFAR10(self.root, train=True, download=True)
        CIFAR10(self.root, train=False, download=True)

    def setup(self, stage=None):
        self.clean_cifar10 = CIFAR10(self.root, train=True, transform=self.clean_transform)
        clean_indices = index_choice(self.clean_cifar10.targets, self.num_clean, self.random_state)
        self.clean_cifar10.data = self.clean_cifar10.data[clean_indices]
        self.clean_cifar10.targets = numpy.array(self.clean_cifar10.targets)[clean_indices]

        if self.expand_clean:
            train_iter = numpy.ceil(50000 / self.batch_size['noisy'])
            multiplier = train_iter * self.batch_size['clean'] // len(self.clean_cifar10)
            self.clean_cifar10 = ConcatDataset([self.clean_cifar10] * multiplier)

        self.noisy_cifar10 = CIFAR10(self.root, train=True, transform=self.noisy_transform)
        noisy_targets = noisify(self.noisy_cifar10.targets, self.T, self.random_state)
        self.noisy_cifar10.targets = list(zip(noisy_targets, self.noisy_cifar10.targets))

        self.valid_cifar10 = CIFAR10(self.root, train=False, transform=self.valid_transform)

    def train_dataloader(self):
        loader = {}
        if self.batch_size['clean'] > 0:
            loader['clean'] = DataLoader(self.clean_cifar10, self.batch_size['clean'], **self.dataloader_args['clean'])
        if self.batch_size['noisy'] > 0:
            loader['noisy'] = DataLoader(self.noisy_cifar10, self.batch_size['noisy'], **self.dataloader_args['noisy'])
        return loader

    def val_dataloader(self):
        return DataLoader(self.valid_cifar10, self.batch_size['valid'], **self.dataloader_args['valid'])

    def test_dataloader(self):
        return self.val_dataloader()

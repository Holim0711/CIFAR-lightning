import os
import numpy
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import ConcatDataset, DataLoader
from .transition_matrix import transition_matrix_cifar10
from .utils import random_select, random_noisify


class NoisyCIFAR10(pl.LightningDataModule):

    def __init__(self, root, num_clean, noise_type, noise_ratio,
                 exclude_clean=True, multiply_clean=1,
                 transform={'clean': ToTensor(), 'noisy': ToTensor(), 'valid': ToTensor()},
                 batch_size={'clean': 0, 'noisy': 1, 'valid': 1},
                 shuffle={'clean': True, 'noisy': True, 'valid': False},
                 num_workers={'clean': None, 'noisy': None, 'valid': None},
                 pin_memory={'clean': True, 'noisy': True, 'valid': True},
                 random_seed=1234):
        super().__init__()
        self.root = root
        self.num_clean = num_clean              # important!
        self.noise_type = noise_type            # important!
        self.noise_ratio = noise_ratio          # important!
        self.exclude_clean = exclude_clean      # exclude clean data from noisy data
        self.multiply_clean = multiply_clean    # multiply small clean data to reduce reloading
        self.transform = transform              # dataset parameter
        self.batch_size = batch_size            # dataloader parameter
        self.shuffle = shuffle                  # dataloader parameter
        self.num_workers = num_workers          # dataloader parameter
        self.pin_memory = pin_memory            # dataloader parameter
        self.random_state = numpy.random.RandomState(random_seed)

        assert isinstance(num_clean, int) and num_clean >= 0
        assert noise_type in {'symmetric', 'asymmetric'}
        assert 0 <= noise_ratio <= 1

        self.T = transition_matrix_cifar10(noise_type, noise_ratio)

    @property
    def dynamic_num_workers(self):
        n = os.cpu_count()
        train_batch_size = self.batch_size['clean'] + self.batch_size['noisy']
        return {
            'clean': n * self.batch_size['clean'] // train_batch_size,
            'noisy': n * self.batch_size['noisy'] // train_batch_size,
            'valid': n,
        }

    def prepare_data(self):
        CIFAR10(self.root, download=True)

    def setup(self, stage=None):
        dataset = {
            'clean': CIFAR10(self.root, train=True, transform=self.transform['clean']),
            'noisy': CIFAR10(self.root, train=True, transform=self.transform['noisy']),
            'valid': CIFAR10(self.root, train=False, transform=self.transform['valid']),
        }

        # select clean data
        clean_indices = random_select(dataset['clean'].targets, self.num_clean, self.random_state)
        dataset['clean'].data = dataset['clean'].data[clean_indices]
        dataset['clean'].targets = numpy.array(dataset['clean'].targets)[clean_indices]

        # randomly flip to build noisy data
        dataset['noisy'].targets = random_noisify(dataset['noisy'].targets, self.T, self.random_state)

        if self.exclude_clean:
            dataset['noisy'].data = numpy.delete(dataset['noisy'].data, clean_indices, axis=0)
            dataset['noisy'].targets = numpy.delete(dataset['noisy'].targets, clean_indices, axis=0)

        if self.multiply_clean > 1:
            dataset['clean'] = ConcatDataset([dataset['clean']] * self.multiply_clean)

        self.dataset = dataset
        self.clean_indices = clean_indices

    def dataloader(self, split):
        num_workers = self.num_workers[split] or self.dynamic_num_workers[split]
        return DataLoader(self.dataset[split],
                          batch_size=self.batch_size[split],
                          shuffle=self.shuffle[split],
                          num_workers=num_workers,
                          pin_memory=self.pin_memory[split])

    def train_dataloader(self):
        loader = {}
        if self.batch_size['clean'] > 0:
            loader['clean'] = self.dataloader('clean')
        if self.batch_size['noisy'] > 0:
            loader['noisy'] = self.dataloader('noisy')
        return loader

    def val_dataloader(self):
        return self.dataloader('valid')

    def test_dataloader(self):
        return self.val_dataloader()

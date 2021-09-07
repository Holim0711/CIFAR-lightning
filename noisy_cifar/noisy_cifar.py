import os
import numpy
from typing import Callable
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
from torch.utils.data import ConcatDataset, DataLoader
from .transition_matrix import *
from .utils import random_select, random_noisify


__all__ = ['NoisyCIFAR10', 'NoisyCIFAR100']


class NoisyCIFAR(pl.LightningDataModule):

    def __init__(
        self,
        root: str,
        num_clean: int,
        noise_type: str,
        noise_ratio: float,
        exclude_clean: bool = False,
        multiply_clean: int = 1,
        transform_clean: Callable = ToTensor(),
        transform_noisy: Callable = ToTensor(),
        transform_valid: Callable = ToTensor(),
        batch_size_clean: int = 1,
        batch_size_noisy: int = 1,
        batch_size_valid: int = 1,
        dataset_random_seed: int = 1234
    ):
        super().__init__()
        self.root = root
        self.num_clean = num_clean              # important!
        self.noise_type = noise_type            # important!
        self.noise_ratio = noise_ratio          # important!
        self.exclude_clean = exclude_clean      # exclude clean data from noisy data
        self.multiply_clean = multiply_clean    # multiply small clean data to reduce reloading
        self.transform = {
            'clean': transform_clean,
            'noisy': transform_noisy,
            'valid': transform_valid,
        }
        self.batch_size = {
            'clean': batch_size_clean,
            'noisy': batch_size_noisy,
            'valid': batch_size_valid,
        }
        self.dataset_random_seed = dataset_random_seed

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument("--num_clean", type=int, required=True)
        parser.add_argument('--noise_type', type=str, choices=['symmetric', 'asymmetric'], required=True)
        parser.add_argument("--noise_ratio", type=float, choices=[x/10 for x in range(11)], required=True)
        parser.add_argument('--exclude_clean', action='store_true')
        parser.add_argument('--multiply_clean', type=int, default=1)
        parser.add_argument('--dataset_random_seed', type=int, default=1234)
        return parent_parser

    @classmethod
    def from_argparse_args(cls, root, args, **kwargs):
        kwargs['num_clean'] = args.num_clean
        kwargs['noise_type'] = args.noise_type
        kwargs['noise_ratio'] = args.noise_ratio
        kwargs['exclude_clean'] = args.exclude_clean
        kwargs['multiply_clean'] = args.multiply_clean
        kwargs['dataset_random_seed'] = args.dataset_random_seed
        return cls(root, **kwargs)

    def prepare_data(self):
        self.CIFAR(self.root, download=True)

    def setup(self, stage=None):
        random_state = numpy.random.RandomState(self.dataset_random_seed)

        dataset = {
            'clean': self.CIFAR(self.root, train=True, transform=self.transform['clean']),
            'noisy': self.CIFAR(self.root, train=True, transform=self.transform['noisy']),
            'valid': self.CIFAR(self.root, train=False, transform=self.transform['valid']),
        }

        # select clean data
        clean_indices = random_select(dataset['clean'].targets, self.num_clean, random_state)
        dataset['clean'].data = dataset['clean'].data[clean_indices]
        dataset['clean'].targets = numpy.array(dataset['clean'].targets)[clean_indices]

        # randomly flip to build noisy data
        T = self.transition_matrix(self.noise_type, self.noise_ratio)
        dataset['noisy'].targets = random_noisify(dataset['noisy'].targets, T, random_state)

        if self.exclude_clean:
            dataset['noisy'].data = numpy.delete(dataset['noisy'].data, clean_indices, axis=0)
            dataset['noisy'].targets = numpy.delete(dataset['noisy'].targets, clean_indices, axis=0)

        if self.multiply_clean > 1:
            dataset['clean'] = ConcatDataset([dataset['clean']] * self.multiply_clean)

        self.dataset = dataset

    def dataloader(self, split, shuffle=None, num_workers=None, pin_memory=True):
        return DataLoader(
            self.dataset[split],
            self.batch_size[split],
            shuffle=(split != 'valid') if shuffle is None else shuffle,
            num_workers=os.cpu_count() if num_workers is None else num_workers,
            pin_memory=pin_memory,
        )

    def train_dataloader(self):
        if self.batch_size['clean'] == 0:
            return self.dataloader('noisy')
        elif self.batch_size['noisy'] == 0:
            return self.dataloader('clean')
        else:
            return {'clean': self.dataloader('clean'),
                    'noisy': self.dataloader('noisy')}

    def val_dataloader(self):
        return self.dataloader('valid')

    def test_dataloader(self):
        return self.val_dataloader()


class NoisyCIFAR10(NoisyCIFAR):
    num_classes = 10
    CIFAR = CIFAR10
    transition_matrix = staticmethod(transition_matrix_cifar10)


class NoisyCIFAR100(NoisyCIFAR):
    num_classes = 100
    CIFAR = CIFAR100
    transition_matrix = staticmethod(transition_matrix_cifar100)

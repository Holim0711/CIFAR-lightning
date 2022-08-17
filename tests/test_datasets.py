import os
import unittest
import tempfile
import numpy as np
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from synthetic_noisy_datasets.datasets import *


class TestDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('DATASETS')
        if path is None:
            cls.tmpdir = tempfile.TemporaryDirectory()
            cls.root = cls.tmpdir.name
        else:
            cls.root = path

    def setUp(self):
        pass

    def routine(self, cds, nds, path, num_classes, noise_type, noise_ratio):
        clean = cds(os.path.join(self.root, path))
        noisy = nds(
            os.path.join(self.root, path),
            noise_type=noise_type,
            noise_ratio=noise_ratio
        )

        T = np.zeros((num_classes, num_classes), dtype=int)
        for y, ỹ in zip(clean.targets, noisy.targets):
            T[y][ỹ] += 1
        T = T / T.sum(axis=1, keepdims=True)

        err = noisy.T - T
        rmse = np.sqrt((err * err).mean())

        with np.printoptions(edgeitems=10, linewidth=128):
            print(f'[ideal] {path} {noise_type} {noise_ratio}')
            print((noisy.T * 100).round().astype(int))
            print(f'[real] {path} {noise_type} {noise_ratio}')
            print((T * 100).round().astype(int))

        self.assertAlmostEqual(rmse, 0, places=2)

    def test_NoisyMNIST(self):
        for t in ['symmetric', 'asymmetric']:
            for r in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                self.routine(MNIST, NoisyMNIST, 'MNIST', 10, t, r)

    def test_NoisyCIFAR10(self):
        for t in ['symmetric', 'asymmetric']:
            for r in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                self.routine(CIFAR10, NoisyCIFAR10, 'CIFAR10', 10, t, r)

    def test_NoisyCIFAR100(self):
        for t in ['symmetric', 'asymmetric']:
            for r in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                self.routine(CIFAR100, NoisyCIFAR100, 'CIFAR100', 100, t, r)


if __name__ == '__main__':
    unittest.main()

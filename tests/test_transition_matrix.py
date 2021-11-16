import os
import unittest
import tempfile
import numpy as np
from deficient_cifar import NoisyCIFAR10, NoisyCIFAR100


class TestTransitionMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.root = cls.tmpdir.name

    def setUp(self):
        pass

    def routine(self, Class, noise_type, noise_ratio):
        if Class.num_classes == 10:
            path = os.path.join(self.root, 'CIFAR-10')
        elif Class.num_classes == 100:
            path = os.path.join(self.root, 'CIFAR-100')

        dm = Class(path, 0,
                   noise_type=noise_type, noise_ratio=noise_ratio)
        dm.prepare_data()
        dm.setup()

        T = np.zeros((dm.num_classes, dm.num_classes), dtype=int)
        for _, (y1, y2) in dm.datasets[dm.splits[1]]:
            T[y2][y1] += 1
        T = T / T.sum(axis=1, keepdims=True)

        err = dm.T - T
        rmse = np.sqrt((err * err).mean())

        with np.printoptions(edgeitems=10, linewidth=128):
            print(f'[ideal] {noise_type} {noise_ratio}')
            print((dm.T * 100).round().astype(int))
            print(f'[real] {noise_type} {noise_ratio}')
            print((T * 100).round().astype(int))

        self.assertAlmostEqual(rmse, 0, places=2)

    def test_noisy_cifar_10(self):
        for noise_type in ['symmetric', 'asymmetric']:
            for noise_ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                self.routine(NoisyCIFAR10, noise_type, noise_ratio)

    def test_noisy_cifar_100(self):
        for noise_type in ['symmetric', 'asymmetric']:
            for noise_ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                self.routine(NoisyCIFAR100, noise_type, noise_ratio)


if __name__ == '__main__':
    unittest.main()

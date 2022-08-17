import os
import unittest
import tempfile
from synthetic_noisy_datasets.datamodules import *
from torchvision.transforms import Compose, Resize, ToTensor, Lambda


class TestDataModules(unittest.TestCase):
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

    def routine(self, cls, path, num_clean):
        dm = cls(
            os.path.join(self.root, path),
            num_clean=num_clean,
            batch_sizes={
                'clean': 64,
                'noisy': 448,
                'val': 512,
            },
            transforms={
                'clean': Compose([Resize(32), ToTensor()]),
                'noisy': Compose([Resize(28), ToTensor()]),
                'val': Compose([Resize(64), ToTensor()]),
            },
            expand_clean=True,
            enumerate_noisy=True,
        )
        dm.prepare_data()
        dm.setup()

        train = dm.train_dataloader()
        val = dm.val_dataloader()

        self.assertTrue(len(train['noisy']) <= len(train['clean']) * 2)

        x, y = next(iter(train['clean']))
        self.assertEqual(x.shape[-2:], (32, 32))
        self.assertEqual(y.shape, (64,))

        i, (x, y) = next(iter(train['noisy']))
        self.assertEqual(x.shape[-2:], (28, 28))
        self.assertEqual(y.shape, (448,))
        self.assertEqual(i.shape, (448,))

        x, y = next(iter(val))
        self.assertEqual(x.shape[-2:], (64, 64))
        self.assertEqual(y.shape, (512,))

    def test_NoisyMNIST(self):
        self.routine(NoisyMNIST, 'MNIST', 40)
        self.routine(NoisyMNIST, 'MNIST', 100)
        self.routine(NoisyMNIST, 'MNIST', 250)

    def test_NoisyCIFAR10(self):
        self.routine(NoisyCIFAR10, 'CIFAR10', 40)
        self.routine(NoisyCIFAR10, 'CIFAR10', 100)
        self.routine(NoisyCIFAR10, 'CIFAR10', 250)

    def test_NoisyCIFAR100(self):
        self.routine(NoisyCIFAR100, 'CIFAR100', 400)
        self.routine(NoisyCIFAR100, 'CIFAR100', 1000)
        self.routine(NoisyCIFAR100, 'CIFAR100', 2500)


if __name__ == '__main__':
    unittest.main()

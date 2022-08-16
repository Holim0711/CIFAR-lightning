import os
import unittest
import tempfile
from synthetic_noisy_datasets.datamodules import *
from torchvision.transforms import Compose, ToTensor, Lambda


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

    def routine(self, Class, n):
        dm = Class(
            os.path.join(self.root, f'CIFAR{Class.num_classes}'),
            n,
            batch_sizes={'clean': 64, 'noisy': 448},
            transforms={
                'clean': Compose([ToTensor(), Lambda(lambda x: x[0])]),
                'noisy': Compose([ToTensor(), Lambda(lambda x: x[:2])]),
            }
        )
        dm.prepare_data()
        dm.setup()

        for x in dm.datasets['clean'].datasets:
            self.assertEqual(len(x), n)
        self.assertEqual(len(dm.datasets['noisy']), 50000)
        self.assertEqual(len(dm.datasets['val']), 10000)

        dl = dm.train_dataloader()

        self.assertTrue(len(dl['noisy']) <= len(dl['clean']) * 2)

        x, y = next(iter(dl['clean']))
        self.assertEqual(x.shape, (64, 32, 32))
        self.assertEqual(y.shape, (64,))

        x, y = next(iter(dl['noisy']))
        self.assertEqual(x.shape, (448, 2, 32, 32))
        self.assertEqual(y.shape, (448,))

        x, y = next(iter(dm.val_dataloader()))
        self.assertEqual(x.shape, (1, 3, 32, 32))
        self.assertEqual(y.shape, (1,))

    def test_noisy_cifar_10(self):
        self.routine(NoisyCIFAR10, 40)
        self.routine(NoisyCIFAR10, 250)
        self.routine(NoisyCIFAR10, 1000)
        self.routine(NoisyCIFAR10, 4000)

    def test_noisy_cifar_100(self):
        self.routine(NoisyCIFAR100, 400)
        self.routine(NoisyCIFAR100, 2500)
        self.routine(NoisyCIFAR100, 10000)


if __name__ == '__main__':
    unittest.main()

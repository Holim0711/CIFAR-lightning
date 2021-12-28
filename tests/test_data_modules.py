import os
import unittest
import tempfile
from deficient_cifar import *
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
        s = Class.splits

        dm = Class(
            os.path.join(self.root, f'CIFAR-{Class.num_classes}'),
            n,
            batch_sizes={
                s[0]: 64,
                s[1]: 448,
            },
            transforms={
                s[0]: Compose([ToTensor(), Lambda(lambda x: x[0])]),
                s[1]: Compose([ToTensor(), Lambda(lambda x: x[:2])]),
            }
        )
        dm.prepare_data()
        dm.setup()

        for x in dm.datasets[s[0]].datasets:
            self.assertEqual(len(x), n)
        self.assertEqual(len(dm.datasets[s[1]]), 50000)
        self.assertEqual(len(dm.datasets[s[2]]), 10000)

        dl = dm.train_dataloader()

        self.assertTrue(len(dl[s[1]]) <= len(dl[s[0]]) * 2)

        x, y = next(iter(dl[s[0]]))
        self.assertEqual(x.shape, (64, 32, 32))
        self.assertEqual(y.shape, (64,))

        x, y = next(iter(dl[s[1]]))
        self.assertEqual(x.shape, (448, 2, 32, 32))
        self.assertEqual(y.shape, (448,))

        x, y = next(iter(dm.val_dataloader()))
        self.assertEqual(x.shape, (1, 3, 32, 32))
        self.assertEqual(y.shape, (1,))

    def test_noisy_cifar_10(self):
        self.assertEqual(NoisyCIFAR10.splits[0], 'clean')
        self.assertEqual(NoisyCIFAR10.splits[1], 'noisy')
        self.assertEqual(NoisyCIFAR10.splits[2], 'val')
        self.routine(NoisyCIFAR10, 40)
        self.routine(NoisyCIFAR10, 250)
        self.routine(NoisyCIFAR10, 1000)
        self.routine(NoisyCIFAR10, 4000)

    def test_noisy_cifar_100(self):
        self.assertEqual(NoisyCIFAR100.splits[0], 'clean')
        self.assertEqual(NoisyCIFAR100.splits[1], 'noisy')
        self.assertEqual(NoisyCIFAR100.splits[2], 'val')
        self.routine(NoisyCIFAR100, 400)
        self.routine(NoisyCIFAR100, 2500)
        self.routine(NoisyCIFAR100, 10000)

    def test_semi_cifar_10(self):
        self.assertEqual(SemiCIFAR10.splits[0], 'labeled')
        self.assertEqual(SemiCIFAR10.splits[1], 'unlabeled')
        self.assertEqual(SemiCIFAR10.splits[2], 'val')
        self.routine(SemiCIFAR10, 40)
        self.routine(SemiCIFAR10, 250)
        self.routine(SemiCIFAR10, 1000)
        self.routine(SemiCIFAR10, 4000)

    def test_semi_cifar_100(self):
        self.assertEqual(SemiCIFAR100.splits[0], 'labeled')
        self.assertEqual(SemiCIFAR100.splits[1], 'unlabeled')
        self.assertEqual(SemiCIFAR100.splits[2], 'val')
        self.routine(SemiCIFAR100, 400)
        self.routine(SemiCIFAR100, 2500)
        self.routine(SemiCIFAR100, 10000)


if __name__ == '__main__':
    unittest.main()

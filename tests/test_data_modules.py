import unittest
import tempfile
from dual_cifar import *


class TestDataModules(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()

    def routine(self, Class, n):
        s = Class.splits

        dm = Class(self.root.name, n,
                   batch_sizes={s[0]: 64, s[1]: 448, s[2]: 512})
        dm.prepare_data()
        dm.setup()

        for x in dm.datasets[s[0]].datasets:
            self.assertEqual(len(x), n)
        self.assertEqual(len(dm.datasets[s[1]]), 50000)
        self.assertEqual(len(dm.datasets[s[2]]), 10000)

        dl = dm.train_dataloader()

        self.assertTrue(len(dl[s[1]]) <= len(dl[s[0]]) * 2)

        for x, y in dl[s[0]]:
            self.assertEqual(len(x), len(y))

        if s[1] == 'unlabeled':
            for x, y in dl[s[1]]:
                self.assertEqual(len(x), len(y))
        elif s[1] == 'noisy':
            for x, (y1, y2) in dl[s[1]]:
                self.assertTrue(len(x) == len(y1) == len(y2))
        else:
            raise ValueError(f'Unknown split name: {s[1]}')

        for x, y in dm.val_dataloader():
            self.assertEqual(len(x), len(y))

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

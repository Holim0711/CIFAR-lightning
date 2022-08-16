import unittest
from synthetic_noisy_datasets.labels import read_labels


class TestLabels(unittest.TestCase):

    def test_MNIST(self):
        labels = read_labels('MNIST')
        self.assertEqual(labels[0], '0')
        self.assertEqual(labels[-1], '9')
        self.assertEqual(len(labels), 10)

    def test_CIFAR10(self):
        labels = read_labels('CIFAR10')
        self.assertEqual(labels[0], 'airplane')
        self.assertEqual(labels[-1], 'truck')
        self.assertEqual(len(labels), 10)

    def test_CIFAR100(self):
        labels = read_labels('CIFAR100')
        self.assertEqual(labels[0], 'beaver')
        self.assertEqual(labels[-1], 'tractor')
        self.assertEqual(len(labels), 100)


if __name__ == '__main__':
    unittest.main()

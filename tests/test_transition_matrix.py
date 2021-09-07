import os
import numpy
from noisy_cifar import NoisyCIFAR10


def test_stat_cifar10(num_clean, noise_type, noise_ratio):
    dm = NoisyCIFAR10(
        root=os.path.join(os.path.dirname(__file__), 'data'),
        num_clean=num_clean, noise_type=noise_type, noise_ratio=noise_ratio)
    dm.prepare_data()
    dm.setup()

    print("----- dataset length -----")
    print("clean:", len(dm.dataset['clean']))
    print("noisy:", len(dm.dataset['noisy']))
    print("valid:", len(dm.dataset['valid']))

    print("----- dataloader length -----")
    train_loader = dm.train_dataloader()
    print("clean:", len(train_loader['clean']) if 'clean' in train_loader else 0)
    print("noisy:", len(train_loader['noisy']) if 'noisy' in train_loader else 0)
    print("valid:", len(dm.val_dataloader()))

    print("----- transition matrix -----")
    clean = NoisyCIFAR10(
        root=os.path.join(os.path.dirname(__file__), 'data'),
        num_clean=num_clean, noise_type='symmetric', noise_ratio=0.)
    clean.prepare_data()
    clean.setup()

    T = numpy.zeros((10, 10), dtype=int)
    for x, y in zip(clean.dataset['noisy'].targets, dm.dataset['noisy'].targets):
        T[x][y] += 1
    T = T / T.sum(axis=1)

    print(T)


if __name__ == "__main__":
    test_stat_cifar10(10, 'asymmetric', 0.2)

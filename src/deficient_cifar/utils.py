import numpy as np
from torch.utils.data import Dataset

__all__ = [
    'random_select',
    'random_noisify',
    'transition_matrix',
    'IndexedDataset',
]


def random_select(y, N, rng):
    """ Select a total of 'N' indices equally for each class """
    y = np.array(y)
    C = np.unique(y)
    n = N // len(C)

    random_I = []
    for c in C:
        Iᶜ = np.where(y == c)[0]
        random_Iᶜ = rng.choice(Iᶜ, n, replace=False)
        random_I.extend(random_Iᶜ)
    return random_I


def random_noisify(y, T, rng):
    """ Noisify according to the transition matrix 'T' """
    z = rng.random((len(y), 1))
    return (T[y].cumsum(axis=1) > z).argmax(axis=1)


def uniform_transition(size, noise_ratio):
    P = noise_ratio / (size - 1) * np.ones((size, size))
    np.fill_diagonal(P, 1 - noise_ratio)
    return P


def transition_matrix_cifar10(noise_type, noise_ratio):
    if noise_type == 'symmetric':
        P = uniform_transition(10, noise_ratio)
    elif noise_type == 'asymmetric':
        P = np.eye(10)
        P[9, 9], P[9, 1] = 1 - noise_ratio, noise_ratio    # truck → automobile
        P[2, 2], P[2, 0] = 1 - noise_ratio, noise_ratio    # bird → airplane
        P[3, 3], P[3, 5] = 1 - noise_ratio, noise_ratio    # cat → dog
        P[5, 5], P[5, 3] = 1 - noise_ratio, noise_ratio    # dog → cat
        P[4, 4], P[4, 7] = 1 - noise_ratio, noise_ratio    # deer -> horse
    return P


def transition_matrix_cifar100(noise_type, noise_ratio):
    if noise_type == 'symmetric':
        P = uniform_transition(100, noise_ratio)
    elif noise_type == 'asymmetric':
        # flip within the same superclass ({0..4}, ..., {95..99})
        P = (1 - noise_ratio) * np.eye(100)
        for i in range(20):
            for j in range(4):
                P[5 * i + j, 5 * i + j + 1] = noise_ratio
            P[5 * i + 4, 5 * i] = noise_ratio
    return P


def transition_matrix(num_classes, noise_type, noise_ratio):
    if num_classes == 10:
        return transition_matrix_cifar10(noise_type, noise_ratio)
    if num_classes == 100:
        return transition_matrix_cifar100(noise_type, noise_ratio)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, super().__getitem__(index)

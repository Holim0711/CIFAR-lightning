# original: https://github.com/giorgiop/loss-correction
import numpy as np


def uniform(size, noise_ratio):
    P = noise_ratio / (size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (1 - noise_ratio) * np.ones(size))
    return P


def noisify(y, P, random_state=None):
    rand = np.random.RandomState(random_state)
    return [rand.multinomial(1, P[c]).argmax() for c in y]


def noisify_cifar10(y_train, noise_ratio, noise_type, random_state=None):
    if noise_type == 'symmetric':
        P = uniform(10, noise_ratio)

    elif noise_type == 'asymmetric':
        P = np.eye(10)
        P[9, 9], P[9, 1] = 1 - noise_ratio, noise_ratio    # truck → automobile
        P[2, 2], P[2, 0] = 1 - noise_ratio, noise_ratio    # bird → airplane
        P[3, 3], P[3, 5] = 1 - noise_ratio, noise_ratio    # cat → dog
        P[5, 5], P[5, 3] = 1 - noise_ratio, noise_ratio    # dog → cat
        P[4, 4], P[4, 7] = 1 - noise_ratio, noise_ratio    # deer -> horse

    return noisify(y_train, P, random_state=random_state)


def noisify_cifar100(y_train, noise_ratio, noise_type, random_state=None):
    if noise_type == 'symmetric':
        P = uniform(100, noise_ratio)

    elif noise_type == 'asymmetric':
        # flipped within the same CIFAR100 superclass {0..4}, ..., {95..99}
        P = (1 - noise_ratio) * np.eye(100)
        for i in range(20):
            for j in range(4):
                P[5 * i + j, 5 * i + j + 1] = noise_ratio
            P[5 * i + 4, 5 * i] = noise_ratio

    return noisify(y_train, P=P, random_state=random_state)

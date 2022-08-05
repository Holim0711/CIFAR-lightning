import numpy as np

__all__ = [
    'random_select',
    'random_noisify',
    'transition_matrix',
]


def random_select(y, N, random_state):
    """ Select a total of 'N' indices equally for each class """
    y = np.array(y)
    C = np.unique(y)
    n = N // len(C)

    random_I = []
    for c in C:
        Iᶜ = np.where(y == c)[0]
        random_Iᶜ = random_state.choice(Iᶜ, n, replace=False)
        random_I.extend(random_Iᶜ)
    return random_I


def random_noisify(y, T, random_state):
    """ Noisify according to the transition matrix 'T' """
    z = random_state.random((len(y), 1))
    return (T[y].cumsum(axis=1) > z).argmax(axis=1)


def uniform_transition(size, noise_ratio):
    return (1 - noise_ratio) * np.eye(size) + noise_ratio / size


def mnist_transition(noise_ratio):
    P = np.eye(10)
    P[2, 2], P[2, 7] = 1 - noise_ratio, noise_ratio
    P[3, 3], P[3, 8] = 1 - noise_ratio, noise_ratio
    P[5, 5], P[5, 6] = 1 - noise_ratio, noise_ratio
    P[6, 6], P[6, 5] = 1 - noise_ratio, noise_ratio
    P[7, 7], P[7, 1] = 1 - noise_ratio, noise_ratio
    return P


def cifar10_transition(noise_ratio):
    P = np.eye(10)
    P[9, 9], P[9, 1] = 1 - noise_ratio, noise_ratio    # truck → automobile
    P[2, 2], P[2, 0] = 1 - noise_ratio, noise_ratio    # bird → airplane
    P[3, 3], P[3, 5] = 1 - noise_ratio, noise_ratio    # cat → dog
    P[5, 5], P[5, 3] = 1 - noise_ratio, noise_ratio    # dog → cat
    P[4, 4], P[4, 7] = 1 - noise_ratio, noise_ratio    # deer → horse
    return P


def cifar100_transition(noise_ratio):
    P = np.zeros((100, 100))
    Q = np.eye(5)
    Q[0, 0], Q[0, 1] = 1 - noise_ratio, noise_ratio
    Q[1, 1], Q[1, 2] = 1 - noise_ratio, noise_ratio
    Q[2, 2], Q[2, 3] = 1 - noise_ratio, noise_ratio
    Q[3, 3], Q[3, 4] = 1 - noise_ratio, noise_ratio
    Q[4, 4], Q[4, 0] = 1 - noise_ratio, noise_ratio
    for i in range(0, 100, 5):
        P[i:i+5, i:i+5] = Q
    return P


def transition_matrix(dataset, noise_type, noise_ratio):
    num_classes = {
        'MNIST': 10,
        'CIFAR10': 10,
        'CIFAR100': 100,
    }[dataset]

    if noise_type == 'symmetric':
        return uniform_transition(num_classes, noise_ratio)

    if noise_type == 'asymmetric':
        if dataset == 'MNIST':
            return mnist_transition(noise_ratio)
        if dataset == 'CIFAR10':
            return cifar10_transition(noise_ratio)
        if dataset == 'CIFAR100':
            return cifar100_transition(noise_ratio)

    raise ValueError((
        "Unknown dataset configuration: "
        f"{dataset}, {noise_type}, {noise_ratio}"
    ))

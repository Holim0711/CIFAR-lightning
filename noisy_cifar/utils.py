import numpy as np


__all__ = [
    'random_select',
    'random_noisify',
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

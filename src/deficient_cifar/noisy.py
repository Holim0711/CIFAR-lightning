import numpy
from .base import DeficientCIFAR
from .utils import transition_matrix, random_noisify

__all__ = ['NoisyCIFAR10', 'NoisyCIFAR100']


class NoisyCIFAR(DeficientCIFAR):
    splits = ['clean', 'noisy', 'val']

    def __init__(
        self, root: str, num_clean: int,
        noise_type: str = 'symmetric',
        noise_ratio: float = 0.0,
        **kwargs
    ):
        super().__init__(root, num_clean, **kwargs)
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.T = transition_matrix(self.num_classes, noise_type, noise_ratio)

    def setup_unproved(self, U, proved_indices, random_state):
        noisy_targets = random_noisify(U.targets, self.T, random_state)
        U.targets = list(zip(noisy_targets, U.targets))
        super().setup_unproved(U, proved_indices, random_state)
        U.targets = [(noisy, clean) for noisy, clean in U.targets]


class NoisyCIFAR10(NoisyCIFAR):
    num_classes = 10


class NoisyCIFAR100(NoisyCIFAR):
    num_classes = 100

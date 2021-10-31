from torchvision.datasets import CIFAR10, CIFAR100
from .utils import *
from .base import BaseCIFAR

__all__ = ['NoisyCIFAR10', 'NoisyCIFAR100']


class NoisyCIFAR(BaseCIFAR):
    splits = ['clean', 'noisy', 'val']

    def __init__(
        self,
        *args,
        noise_type: str = 'symmetric',
        noise_ratio: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        assert noise_type in {'symmetric', 'asymmetric'}, "noise type error"
        assert 0 <= noise_ratio <= 1, "noise ratio error"

    def setup_d1(self, d1, random_state):
        self.T = self.transition_matrix(self.noise_type, self.noise_ratio)
        noisy_targets = random_noisify(d1.targets, self.T, random_state)
        d1.targets = list(zip(noisy_targets, d1.targets))


class NoisyCIFAR10(NoisyCIFAR):
    num_classes = 10
    CIFAR = CIFAR10
    transition_matrix = staticmethod(transition_matrix_cifar10)


class NoisyCIFAR100(NoisyCIFAR):
    num_classes = 100
    CIFAR = CIFAR100
    transition_matrix = staticmethod(transition_matrix_cifar100)

from .base import DeficientCIFAR

__all__ = ['SemiCIFAR10', 'SemiCIFAR100']


class SemiCIFAR(DeficientCIFAR):
    splits = ['labeled', 'unlabeled', 'val']

    def __init__(self, root: str, num_labeled: int, **kwargs):
        super().__init__(root, num_labeled, **kwargs)


class SemiCIFAR10(SemiCIFAR):
    num_classes = 10


class SemiCIFAR100(SemiCIFAR):
    num_classes = 100

import numpy
from torchvision import transforms
from noisy_cifar import NoisyCIFAR10


if __name__ == "__main__":
    dm = NoisyCIFAR10(**{
        "root": "aaa",
        "num_clean": 4000,
        "batch_size": {
            "clean": 64,
            "noisy": 448
        }
    })

    transform = transforms.ToTensor()
    dm.clean_transform = transform
    dm.noisy_transform = transform
    dm.valid_transform = transform

    dm.prepare_data()
    dm.setup()

    print("----- dataset statistics -----")
    print("train-clean:", len(dm.clean_cifar10))
    print("train-noisy:", len(dm.noisy_cifar10))
    print("valid:", len(dm.valid_cifar10))

    print("----- loader statistics -----")
    train_loader = dm.train_dataloader()
    if 'clean' in train_loader:
        print("train-clean:", len(train_loader['clean']))
    if 'noisy' in train_loader:
        print("train-noisy:", len(train_loader['noisy']))
    print("valid:", len(dm.val_dataloader()))

    print("----- transition matrix -----")
    print(dm.T)
    T = numpy.zeros((10, 10), dtype=int)
    for noisy, original in dm.noisy_cifar10.targets:
        T[original][noisy] += 1
    T = T / T.sum(axis=1)
    print(T)

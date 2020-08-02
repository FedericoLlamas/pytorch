import torch
import torchvision
import torch.nn as nn


class MnistDataset():

    def __init__(self):
        self.batch_size_train = 64

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            './mnist_dataset', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist_dataset', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
batch_size=batch_size_test, shuffle=True)
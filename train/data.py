import torch
import torchvision
from config import DATA_PATH
from models.model import transform


def loadTestData():
    testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    return testset, testloader


def loadTrainData():
    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    return trainset, trainloader
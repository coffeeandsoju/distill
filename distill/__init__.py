import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from loguru import logger


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.debug(f"Using device: {device}")

    # set up datasets and dataloaders
    transforms_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.CIFAR10(root="/data/torchvision", train=True, download=True, transform=transforms_cifar)
    test_dataset = datasets.CIFAR10(root="/data/torchvision", train=False, download=True, transform=transforms_cifar)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)


    pass
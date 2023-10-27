import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

def get_fashionmnist(cfg: dict) :
    DIR = os.getcwd() # todo update
    
    # Load FashionMNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
    )
    print(type(train_loader))
    return train_loader, valid_loader
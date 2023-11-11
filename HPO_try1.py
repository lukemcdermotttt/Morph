import numpy as np
from tqdm import tqdm
import optuna
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from functools import partial
import torch.nn as nn
import os
from datetime import datetime











def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-9, 1e-3, log=True)
    # batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])

    # Initialize the model
    model = BraggNN_D(imgsz=IMG_SIZE, fcsz=FC_LAYER_SIZES).to(device)


    #Don't need to change optimizers
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    if optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.0, 1.0)
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = create_scheduler(optimizer, trial)

    # Setup the data loaders
    # train_loader, valid_loader = setup_data_loaders(batch_size)

    train_loader, valid_loader = setup_data_loaders(batch_size_temp, IMG_SIZE, aug=aug, num_workers=4, pin_memory=False, prefetch_factor=2)

    criterion = torch.nn.MSELoss()


    validation_loss = train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader, device, trial, num_epochs)

    return validation_loss
    

if __name__ == '__main__':
    batch_size_temp=256
    IMG_SIZE = 11
    FC_LAYER_SIZES = (64, 32, 16, 8)  # example sizes of the fully connected layers
    aug=1
    num_epochs=150
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    n_trials=100
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials)


    
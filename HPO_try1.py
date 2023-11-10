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


from BragnnDataset import setup_data_loaders
from model import BraggNN_D

def train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader, device, trial, num_epochs, save=True, patience=10):
    curr_patience = patience
    previous_epoch_loss = float('inf')
    progress_bar = tqdm(range(num_epochs), disable=True)
    
    for epoch in progress_bar:
        # Training phase
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                validation_loss += loss.item()

        validation_loss /= len(valid_loader)
        
        trial.report(validation_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if scheduler:
            scheduler.step()

        if save and validation_loss < previous_epoch_loss:
            curr_patience=patience
            best_model = model.state_dict()

        else:
            curr_patience -= 1
            if curr_patience <= 0:
                date_str = datetime.now().strftime("%Y%m%d")
                model_filename = f'trial{trial.number}Epoch{epoch}_{date_str}.pth'
                model_path = os.path.join('./saved_models', model_filename)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(best_model, model_path)
                break
        progress_bar.set_postfix(prev_loss=f'{previous_epoch_loss:.4e}')
        previous_epoch_loss = validation_loss
    
    progress_bar.close()
    return previous_epoch_loss



def create_scheduler(optimizer, trial):
    scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR', 'ExponentialLR'])
    if scheduler_name == 'StepLR':
        step_size = trial.suggest_int('step_size', 1, 100)
        gamma = trial.suggest_float('gamma', 0.1, 1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'CosineAnnealingLR':
        T_max = trial.suggest_int('T_max', 1, 100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name == 'ExponentialLR':
        gamma = trial.suggest_float('gamma', 0.1, 1.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    return scheduler


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


    
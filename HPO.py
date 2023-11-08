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
import torch.nn.utils.prune as prune

def train_model(model, optimizer, scheduler, train_loader, valid_loader, device, trial, num_epochs, save=True, patience=10):
    criterion = torch.nn.MSELoss()

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
            if curr_patience <= 0: break
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



#Run vanilla HPO to find relatively good search space
def vanilla_objective(trial):
    #Sample Hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-9, 1e-3, log=True)
    momentum = trial.suggest_float('momentum', 0.0, 1.0)

    # Initialize the model
    FC_LAYER_SIZES = (64, 32, 16, 8)  # example sizes of the fully connected layers
    model = BraggNN_D(imgsz=IMG_SIZE, fcsz=FC_LAYER_SIZES).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = create_scheduler(optimizer, trial)

    validation_loss = train_model(model, optimizer, scheduler, train_loader, val_loader, device, trial, num_epochs)

    return validation_loss

#Helper function for pruning
def get_parameters_to_prune(model, bias = False):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            if bias and module.bias != None:
                parameters_to_prune.append((module, 'bias'))
        
    return tuple(parameters_to_prune)

#Search for best model with QAT and Pruning, multi-objective search
#IN PROGRESS, NOT TESTED
def compression_objective(trial):
    #Sample Hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-9, 1e-3, log=True)
    momentum = trial.suggest_float('momentum', 0.0, 1.0)
    amount = trial.suggest_float('momentum', 0.1, .25) #amount to prune each iteration
    prune_iters = 15 #this can be fixed, we dont need HPO on this parameter

    # Initialize the model
    FC_LAYER_SIZES = (64, 32, 16, 8)  # example sizes of the fully connected layers
    model = BraggNN_D(imgsz=IMG_SIZE, fcsz=FC_LAYER_SIZES).to(device)

    #Pretrain model
    validation_loss = train_model(model, optimizer, scheduler, train_loader, val_loader, device, trial, num_epochs)
    bops = None #TODO: Estimate BOPS / FLOPS, make sure to remove pruning parameterization before calculating
    trial.report([validation_loss, bops], 0)
    #Run IMP w/ Finetuning (Learning-Rate-Rewinding Pruning)
    for i in range(1, prune_iters):
        prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=amount)
    
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = create_scheduler(optimizer, trial)

        validation_loss = train_model(model, optimizer, scheduler, train_loader, val_loader, device, trial, num_epochs)
        bops = None #TODO: Estimate BOPS / FLOPS, make sure to remove pruning parameterization before calculating
        trial.report([validation_loss, bops], i) #report the metrics for this specific pruning iteration

    return validation_loss, bops
    

if __name__ == '__main__':
    #Load Dataset
    batch_size_temp=256
    IMG_SIZE = 11
    aug=1
    num_epochs=150
    train_loader, val_loader = setup_data_loaders(batch_size_temp, IMG_SIZE, aug=aug, num_workers=4, pin_memory=False, prefetch_factor=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    n_trials=100
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials)


    
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

def train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader, device, num_epochs, patience=10, trial=None, save=True):
    curr_patience = patience
    previous_epoch_loss = float('inf')
    
    for epoch in range(num_epochs):
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
        
        if scheduler:
            scheduler.step()

        if trial is not None:
            trial.report(validation_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        if validation_loss < previous_epoch_loss:
            curr_patience= patience
            if save:
                modelpath= f"./models/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pth"
                model.save(modelpath)
                trial.set_user_attr("model_path", modelpath)
        else:
            curr_patience -= 1
            if curr_patience <= 0: break
        previous_epoch_loss = validation_loss

    return previous_epoch_loss

    


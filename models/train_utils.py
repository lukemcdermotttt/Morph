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

def train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader, device, num_epochs, save=True, patience=10):
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

    


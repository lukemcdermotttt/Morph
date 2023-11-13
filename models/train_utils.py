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

def train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader, device, num_epochs, save=True, patience=5):
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

        if validation_loss < previous_epoch_loss:
            curr_patience=patience
        else:
            curr_patience -= 1
            if curr_patience <= 0: break
        previous_epoch_loss = validation_loss

    return previous_epoch_loss

    


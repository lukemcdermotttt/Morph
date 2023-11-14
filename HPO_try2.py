import numpy as np
from tqdm import tqdm
import optuna
import torch
import csv
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from functools import partial
import torch.nn as nn
import os
from datetime import datetime
from models.train_utils import train_model

# import sys
from data.BraggnnDataset import setup_data_loaders, BraggNNDataset

class NLB(torch.nn.Module):
    def __init__(self, in_ch, relu_a=0.01):
        self.inter_ch = torch.div(in_ch, 2, rounding_mode='floor').item()
        super().__init__()
        self.theta_layer = torch.nn.Conv2d(in_channels=in_ch, out_channels=self.inter_ch, \
                            kernel_size=1, padding=0)
        self.phi_layer   = torch.nn.Conv2d(in_channels=in_ch, out_channels=self.inter_ch, \
                            kernel_size=1, padding=0)
        self.g_layer     = torch.nn.Conv2d(in_channels=in_ch, out_channels=self.inter_ch, \
                            kernel_size=1, padding=0)
        self.atten_act   = torch.nn.Softmax(dim=-1)
        self.out_cnn     = torch.nn.Conv2d(in_channels=self.inter_ch, out_channels=in_ch, \
                            kernel_size=1, padding=0)
        
    def forward(self, x):
        mbsz, _, h, w = x.size()
        
        theta = self.theta_layer(x).view(mbsz, self.inter_ch, -1).permute(0, 2, 1)
        phi   = self.phi_layer(x).view(mbsz, self.inter_ch, -1)
        g     = self.g_layer(x).view(mbsz, self.inter_ch, -1).permute(0, 2, 1)
        
        theta_phi = self.atten_act(torch.matmul(theta, phi))
        
        theta_phi_g = torch.matmul(theta_phi, g).permute(0, 2, 1).view(mbsz, self.inter_ch, h, w)
        
        _out_tmp = self.out_cnn(theta_phi_g)
        _out_tmp = torch.add(_out_tmp, x)
   
        return _out_tmp
    
class BraggNN(nn.Module):
    def __init__(self, imgsz, cnn_channels=(64, 32, 8), fcsz=(64, 32, 16, 8)):
        super(BraggNN, self).__init__()
        self.cnn_ops = nn.ModuleList()
        cnn_in_chs = (1, ) + cnn_channels[:-1]

        fsz = imgsz
        for ic, oc in zip(cnn_in_chs, cnn_channels):
            self.cnn_ops.append(nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=3, stride=1, padding=0))
            self.cnn_ops.append(nn.LeakyReLU(negative_slope=0.01))
            fsz -= 2  # adjust the size due to convolution without padding

        self.nlb = NLB(in_ch=cnn_channels[0])

        self.dense_ops = nn.ModuleList()
        dense_in_chs = (fsz * fsz * cnn_channels[-1], ) + fcsz[:-1]
        for ic, oc in zip(dense_in_chs, fcsz):
            self.dense_ops.append(nn.Linear(ic, oc))
            self.dense_ops.append(nn.LeakyReLU(negative_slope=0.01))

        # Output layer
        self.dense_ops.append(nn.Linear(fcsz[-1], 2))

    def forward(self, x):
        _out = x
        for layer in self.cnn_ops[:1]:
            _out = layer(_out)

        _out = self.nlb(_out)

        for layer in self.cnn_ops[1:]:
            _out = layer(_out)

        # _out = _out.view(_out.size(0), -1)  # Flatten the tensor for the dense layer
        _out = _out.reshape(_out.size(0), -1)

        for layer in self.dense_ops:
            _out = layer(_out)

        return _out



def create_scheduler(optimizer, trial):
    scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR', 'ExponentialLR'])
    if scheduler_name == 'StepLR':
        step_size = trial.suggest_int('step_size', 1, 100)
        gamma = trial.suggest_float('gamma', 0.1, 1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'CosineAnnealingLR':
        T_max = trial.suggest_int('T_max', 1, 100)
        eta_min = trial.suggest_float('eta_min', 1e-5, 1e-2, log=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name == 'ExponentialLR':
        gamma = trial.suggest_float('gamma', 0.1, 1.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    return scheduler




def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-9, 1e-3, log=True)
    # batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])

    # Initialize the model
    model = BraggNN(imgsz=IMG_SIZE, fcsz=FC_LAYER_SIZES).to(device)


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


    validation_loss = train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader, device, num_epochs, trial=trial, save = True)
    # train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader, device, num_epochs, patience=10, trial=None, save=True):
    return validation_loss
    

if __name__ == '__main__':
    batch_size_temp=256
    IMG_SIZE = 11
    FC_LAYER_SIZES = (64, 32, 16, 8)  # example sizes of the fully connected layers
    aug=1
    num_epochs=250
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    n_trials=100
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials)

    param_names = set()
    for trial in study.trials:
        param_names.update(trial.params.keys())

    # Sort the parameter names for consistent column ordering
    param_names = sorted(list(param_names))

    # Write the trial data to a CSV file
    with open('trial_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        headers = ['Trial Number', 'Validation Loss', 'Model Path'] + param_names
        writer.writerow(headers)

        for trial in study.trials:
            # Collect the necessary information
            trial_number = trial.number
            val_loss = trial.value
            model_path = trial.user_attrs.get("model_path", "N/A")  # Default to "N/A" if no model was saved
            trial_params = [trial.params.get(name, "N/A") for name in param_names]

            # Write to the CSV
            writer.writerow([trial_number, val_loss, model_path] + trial_params)
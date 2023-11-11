from data import BraggnnDataset
from data.BraggnnDataset import setup_data_loaders
import torch
import torch.nn as nn
import optuna
from models.blocks import *
from models.train_utils import *
import time

def objective(trial):
    num_blocks = 3
    
    channel_space = (8,16,24)
    block_channels = [ channel_space[trial.suggest_int('Proj_outchannel', 0, len(channel_space) - 1) ] ] #the channel dimensions before/after each block

    #Sample Block Types
    b = [trial.suggest_categorical('b' + str(i), ['Conv', 'ConvAttn', 'None']) for i in range(num_blocks)]
    Blocks = [] #Store the nn.module blocks

    #Build Blocks
    for i, block_type in enumerate(b):
        if block_type == 'Conv':
            channels, kernels, acts, norms = sample_ConvBlock(trial, 'b' + str(i) + '_Conv', block_channels[-1])
            Blocks.append(ConvBlock(channels, kernels, acts, norms))
            block_channels.append(channels[-1]) #save the final out dimension so b2 knows what to expect
        elif block_type == 'ConvAttn':
            hidden_channels = sample_ConvAttn(trial, 'b' + str(i) + '_ConvAttn')
            Blocks.append(ConvAttn(block_channels[-1], hidden_channels))
            #ConvAttn doesnt change channels bc skip connect
    Blocks = nn.Sequential(*Blocks)

    #Build MLP
    spatial_dims = (9,9) #spatial dims after blocks, this is not (11,11) due to the 3x3 kernel size from first projection conv
    in_dim = block_channels[-1] * spatial_dims[0] * spatial_dims[1] #this assumes spatial dim stays same with padding trick
    widths, acts, norms = sample_MLP(trial, in_dim)
    mlp = MLP(widths, acts, norms)

    #Build Model
    model = CandidateArchitecture(Blocks, mlp, block_channels[0])

    return evaluate(model)

def get_performance(model, dataloader, device):
    distances = []
    with torch.no_grad():
        for features, true_locs in dataloader:
            features = features.to(device)
            preds = model(features)  # assuming model outputs normalized [px, py]
            preds = preds.cpu().numpy()

            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((preds - true_locs.numpy()) ** 2, axis=1))
            distances.extend(distance)  # Changed from append to extend

    mean_distance = np.mean(distances)
    return mean_distance

def get_efficiency(model,device):
    #inference time
    x = torch.randn((256,1,11,11)).to(device)
    start = time.time()
    for _ in range(50):
        y = model(x)
    end = time.time()

    #param count
    count = 0
    count += sum(p.numel() for p in model.Blocks.parameters())
    count += sum(p.numel() for p in model.MLP.parameters())
    count += sum(p.numel() for p in model.conv.parameters())

    return end-start

def evaluate(model):
    num_epochs = 10

    device = torch.device('cuda:0')
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    validation_loss = train_model(model, optimizer, scheduler, criterion, train_loader, val_loader, device, num_epochs)
    
    #Evaluate Performance
    mean_distance = get_performance(model, val_loader, device)

    #Count Params
    inference_time = get_efficiency(model, device)

    print('Mean Distance: ', mean_distance, ', Inference time: ', inference_time, ', Validation Loss: ', validation_loss)
    return mean_distance

#main    
batch_size=256
train_loader, val_loader = setup_data_loaders(batch_size, IMG_SIZE = 11, aug=1, num_workers=4, pin_memory=False, prefetch_factor=2)
print('Loaded Dataset...')

study = optuna.create_study(direction='minimize')
#Run BraggNN first
study.enqueue_trial({
    'b1': 'ConvAttn',
    'b2': 'Conv',
    'b3': 'None',
    'Proj_outchannel': 1,
    'b1_ConvAttn_hiddenchannel' : 3,
    'b2_Conv_channels_0' : 2,
    'b2_Conv_channels_1' : 0,
    'b2_Conv_kernels_0' : 1,
    'b2_Conv_kernels_1' : 1,
    'b2_Conv_acts_0' : 0,
    'b2_Conv_acts_1' : 0,
    'b2_Conv_norms_1' : None,
    'b2_Conv_norms_1' : None,
    'MLP_width_0' : 3,
    'MLP_width_1' : 1,
    'MLP_width_2' : 0,
    'MLP_acts_0' : 0,
    'MLP_acts_1' : 0,
    'MLP_acts_2' : 0,
    'MLP_acts_3' : 0,
    'MLP_norms_0' : None,
    'MLP_norms_1' : None,
    'MLP_norms_2' : None,
    'MLP_norms_3' : None,
    })
study.optimize(objective, n_trials=100)

# Print the best trial
print('Best trial:')
trial = study.best_trial
print(f'Value: {trial.value}')
print(f'Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

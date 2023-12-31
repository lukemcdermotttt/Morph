from data import BraggnnDataset
from data.BraggnnDataset import setup_data_loaders
import torch
import torch.nn as nn
import optuna
from models.blocks import *
from models.train_utils import *
import time
from utils.bops import *


#Optuna Hyperparameters to recreate the OpenHLS BraggNN model
Example1_params = {
    'Proj_outchannel': 3, 'b0': 'None', 'b1': 'Conv', 'b2': 'Conv', 'b1_Conv_channels_0': 4, 'b1_Conv_channels_1': 5, 'b1_Conv_kernels_0': 1, 'b1_Conv_kernels_1': 3, 'b1_Conv_norms_0': 'layer', 'b1_Conv_norms_1': None, 'b1_Conv_acts_0': 0, 'b1_Conv_acts_1': 1, 'b2_Conv_channels_0': 1, 'b2_Conv_channels_1': 1, 'b2_Conv_kernels_0': 3, 'b2_Conv_kernels_1': 3, 'b2_Conv_norms_0': None, 'b2_Conv_norms_1': None, 'b2_Conv_acts_0': 1, 'b2_Conv_acts_1': 1, 'MLP_width_0': 1, 'MLP_width_1': 2, 'MLP_width_2': 4, 'MLP_acts_0': 3, 'MLP_acts_1': 3, 'MLP_acts_2': 1, 'MLP_acts_3': 0, 'MLP_norms_0': 'layer', 'MLP_norms_1': 'batch', 'MLP_norms_2': 'batch', 'MLP_norms_3': None
}
Example2_params = {
    'Proj_outchannel': 0, 'b0': 'Conv', 'b1': 'Conv', 'b2': 'Conv', 'b0_Conv_channels_0': 5, 'b0_Conv_channels_1': 4, 'b0_Conv_kernels_0': 3, 'b0_Conv_kernels_1': 3, 'b0_Conv_norms_0': 'batch', 'b0_Conv_norms_1': 'layer', 'b0_Conv_acts_0': 0, 'b0_Conv_acts_1': 0, 'b1_Conv_channels_0': 2, 'b1_Conv_channels_1': 2, 'b1_Conv_kernels_0': 3, 'b1_Conv_kernels_1': 3, 'b1_Conv_norms_0': 'layer', 'b1_Conv_norms_1': 'batch', 'b1_Conv_acts_0': 1, 'b1_Conv_acts_1': 1, 'b2_Conv_channels_0': 2, 'b2_Conv_channels_1': 2, 'b2_Conv_kernels_0': 1, 'b2_Conv_kernels_1': 3, 'b2_Conv_norms_0': 'batch', 'b2_Conv_norms_1': 'batch', 'b2_Conv_acts_0': 2, 'b2_Conv_acts_1': 0, 'MLP_width_0': 1, 'MLP_width_1': 4, 'MLP_width_2': 3, 'MLP_acts_0': 0, 'MLP_acts_1': 2, 'MLP_acts_2': 0, 'MLP_acts_3': 0, 'MLP_norms_0': None, 'MLP_norms_1': 'layer', 'MLP_norms_2': 'batch', 'MLP_norms_3': None
}
Example3_params = {
    'Proj_outchannel': 2, 'b0': 'Conv', 'b1': 'None', 'b2': 'Conv', 'b0_Conv_channels_0': 2, 'b0_Conv_channels_1': 2, 'b0_Conv_kernels_0': 1, 'b0_Conv_kernels_1': 1, 'b0_Conv_norms_0': None, 'b0_Conv_norms_1': None, 'b0_Conv_acts_0': 3, 'b0_Conv_acts_1': 3, 'b2_Conv_channels_0': 1, 'b2_Conv_channels_1': 4, 'b2_Conv_kernels_0': 1, 'b2_Conv_kernels_1': 1, 'b2_Conv_norms_0': 'layer', 'b2_Conv_norms_1': 'layer', 'b2_Conv_acts_0': 0, 'b2_Conv_acts_1': 0, 'MLP_width_0': 1, 'MLP_width_1': 2, 'MLP_width_2': 3, 'MLP_acts_0': 2, 'MLP_acts_1': 0, 'MLP_acts_2': 2, 'MLP_acts_3': 0, 'MLP_norms_0': 'layer', 'MLP_norms_1': 'batch', 'MLP_norms_2': 'batch', 'MLP_norms_3': None
}

OpenHLS_params = {
    'b0': 'ConvAttn',
    'b1': 'Conv',
    'b2': 'None',
    'Proj_outchannel': 1,
    'b0_ConvAttn_hiddenchannel' : 3,
    'b0_ConvAttn_act' : 0,
    'b1_Conv_channels_0' : 2,
    'b1_Conv_channels_1' : 0,
    'b1_Conv_kernels_0' : 3,
    'b1_Conv_kernels_1' : 3,
    'b1_Conv_acts_0' : 0,
    'b1_Conv_acts_1' : 0,
    'b1_Conv_norms_0' : None, 
    'b1_Conv_norms_1' : None,
    'MLP_width_0' : 2,
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
    }

#Optuna Hyperparameters for original BraggNN model
BraggNN_params = {
    'b0': 'ConvAttn',
    'b1': 'Conv',
    'b2': 'None',
    'Proj_outchannel': 3,
    'b0_ConvAttn_hiddenchannel' : 5,
    'b0_ConvAttn_act' : 2,
    'b1_Conv_channels_0' : 4,
    'b1_Conv_channels_1' : 2,
    'b1_Conv_kernels_0' : 3,
    'b1_Conv_kernels_1' : 3,
    'b1_Conv_acts_0' : 2,
    'b1_Conv_acts_1' : 2,
    'b1_Conv_norms_0' : None,
    'b1_Conv_norms_1' : None,
    'MLP_width_0' : 4,
    'MLP_width_1' : 3,
    'MLP_width_2' : 2,
    'MLP_acts_0' : 2,
    'MLP_acts_1' : 2,
    'MLP_acts_2' : 2,
    'MLP_acts_3' : 3,
    'MLP_norms_0' : None,
    'MLP_norms_1' : None,
    'MLP_norms_2' : None,
    'MLP_norms_3' : None,
    }

def objective(trial):
    #Build Model
    num_blocks = 3
    channel_space = (8,16,32,64)
    block_channels = [ channel_space[trial.suggest_int('Proj_outchannel', 0, len(channel_space) - 1) ] ] #the channel dimensions before/after each block

    #Sample Block Types
    b = [trial.suggest_categorical('b' + str(i), ['Conv', 'ConvAttn', 'None']) for i in range(num_blocks)]
    Blocks = [] #Store the nn.module blocks

    img_size = 9 #size after First conv layer
    bops = 0 #Record Estimated BOPs
    #Build Blocks
    for i, block_type in enumerate(b):
        if block_type == 'Conv':
            channels, kernels, acts, norms = sample_ConvBlock(trial, 'b' + str(i) + '_Conv', block_channels[-1])
            reduce_img_size = 2*sum([1 if k == 3 else 0 for k in kernels]) #amount the image size will be reduced by kernel size, assuming no padding
            while img_size - reduce_img_size <= 0:
                kernels[kernels.index(3)] = 1
                reduce_img_size = 2*sum([1 if k == 3 else 0 for k in kernels])
            Blocks.append(ConvBlock(channels, kernels, acts, norms, img_size))
            bops += calculate_convblock_bops(Blocks[-1], sparsity_dict=None, weight_bit_width=32, activation_bit_width=32)
            img_size -= reduce_img_size
            block_channels.append(channels[-1]) #save the final out dimension so next block knows what to expect
        elif block_type == 'ConvAttn':
            hidden_channels, act = sample_ConvAttn(trial, 'b' + str(i) + '_ConvAttn')
            Blocks.append(ConvAttn(block_channels[-1], hidden_channels, act))
            bops += calculate_convattn_bops(Blocks[-1], input_shape = [batch_size, block_channels[-1], img_size, img_size], bit_width=32)
            #ConvAttn doesnt change channels bc skip connect

    Blocks = nn.Sequential(*Blocks)

    #Build MLP
    in_dim = block_channels[-1] * img_size**2 #this assumes spatial dim stays same with padding trick
    widths, acts, norms = sample_MLP(trial, in_dim)
    mlp = MLP(widths, acts, norms)
    bops +=  calculate_mlpblock_bops(mlp, sparsity_dict=None, weight_bit_width=32, activation_bit_width=32)
    
    #Initialize Model
    model = CandidateArchitecture(Blocks, mlp, block_channels[0])
    print(model)
    print('BOPs:', bops)
    #Evaluate Model
    print('Trial ', trial.number,' begins evaluation.')
    mean_distance, inference_time, validation_loss, param_count = evaluate(model)
    with open("./optuna_trials.txt", "a") as file:
        file.write(f"Trial {trial.number}, Mean Distance: {mean_distance}, BOPs: {bops}, Inference time: {inference_time}, Validation Loss: {validation_loss}, Param Count: {param_count}, Hyperparams: {trial.params}\n")
    return mean_distance, bops



def evaluate(model):
    num_epochs = 50
    device = torch.device('cuda:1')
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=.0015, weight_decay=2.2e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    validation_loss = train_model(model, optimizer, scheduler, criterion, train_loader, val_loader, device, num_epochs)
    
    #Evaluate Performance
    mean_distance = get_performance(model, val_loader, device)
    #Evaluate Efficiency
    param_count = get_param_count(model)
    inference_time = get_inference_time(model, device)

    print('Mean Distance: ', mean_distance, ', Inference time: ', inference_time, ', Validation Loss: ', validation_loss, ', Param Count: ', param_count)
    return mean_distance, inference_time, validation_loss, param_count

def main():
    study = optuna.create_study(sampler=optuna.samplers.NSGAIISampler(population_size = 20), directions=['minimize', 'minimize']) #min mean_distance and inference time
    study.enqueue_trial(OpenHLS_params)
    study.enqueue_trial(BraggNN_params)
    study.enqueue_trial(Example1_params)
    study.enqueue_trial(Example2_params)
    study.enqueue_trial(Example3_params)
    study.optimize(objective, n_trials=1000)

    # Print the best trial
    print('Best trial:')
    trial = study.best_trial
    print(f'Value: {trial.value}')
    print(f'Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

if __name__ == "__main__":
    batch_size=256
    train_loader, val_loader, test_loader = setup_data_loaders(batch_size, IMG_SIZE = 11, aug=1, num_workers=4, pin_memory=False, prefetch_factor=2)
    print('Loaded Dataset...')
    main()

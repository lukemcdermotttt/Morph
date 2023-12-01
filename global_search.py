from data import BraggnnDataset
from data.BraggnnDataset import setup_data_loaders
import torch
import torch.nn as nn
import optuna
from models.blocks import *
from models.train_utils import *
import time
from utils.bops import *
from utils.hyperparam_examples import *

"""
Optuna Objective to evaluate a trial
1) Samples architecture from hierarchical search space
2) Trains Model
3) Evaluates Mean Distance, bops, param count, inference time, and val loss
Saves all information in global_search.txt
"""
def objective(trial):
    #Build Model
    num_blocks = 3
    channel_space = (8,16,32,64)
    block_channels = [ channel_space[trial.suggest_int('Proj_outchannel', 0, len(channel_space) - 1) ] ] #sample the first channel dimension, save future dimensions here

    #Sample Block Types
    b = [trial.suggest_categorical('b' + str(i), ['Conv', 'ConvAttn', 'None']) for i in range(num_blocks)]

    Blocks = [] #Save list of blocks
    img_size = 9 #Size after first conv patch embedding
    bops = 0 #Record Estimated BOPs

    #Build Blocks
    for i, block_type in enumerate(b):
        if block_type == 'Conv':
            #Create block and add to Blocks
            channels, kernels, acts, norms = sample_ConvBlock(trial, 'b' + str(i) + '_Conv', block_channels[-1])
            reduce_img_size = 2*sum([1 if k == 3 else 0 for k in kernels]) #amount the image size will be reduced by kernel size, assuming no padding
            while img_size - reduce_img_size <= 0:
                kernels[kernels.index(3)] = 1
                reduce_img_size = 2*sum([1 if k == 3 else 0 for k in kernels])
            Blocks.append(ConvBlock(channels, kernels, acts, norms, img_size))

            #Calculate bops for this block
            bops += get_conv_bops(Blocks[-1], input_shape = [batch_size, channels[0], img_size, img_size], bit_width=32)
            img_size -= reduce_img_size
            block_channels.append(channels[-1]) #save the final out dimension so next block knows what to expect

        elif block_type == 'ConvAttn':
            #Create block and add to Blocks
            hidden_channels, act = sample_ConvAttn(trial, 'b' + str(i) + '_ConvAttn')
            Blocks.append(ConvAttn(block_channels[-1], hidden_channels, act))

            #Calculate bops for this block
            bops += get_convattn_bops(Blocks[-1], input_shape = [batch_size, block_channels[-1], img_size, img_size], bit_width=32)
            #Note: ConvAttn does not change the input shape because we use a skip connection
    

    #Build MLP
    in_dim = block_channels[-1] * img_size**2 #this assumes spatial dim stays same with padding trick
    widths, acts, norms = sample_MLP(trial, in_dim)
    mlp = MLP(widths, acts, norms)

    #Calculate bops for the mlp
    bops +=  get_mlp_bops(mlp, bit_width=32)
    
    #Initialize Model
    Blocks = nn.Sequential(*Blocks)
    model = CandidateArchitecture(Blocks, mlp, block_channels[0])
    bops += get_conv2d_bops(model.conv, input_shape = [batch_size, 1, 11, 11], bit_width=32) #Calculate bops for the patch embedding

    #Evaluate Model
    print(model)
    print('BOPs:', bops)
    print('Trial ', trial.number,' begins evaluation...')
    mean_distance, inference_time, validation_loss, param_count = evaluate(model, train_loader, valid_loader, device)
    with open("./global_search.txt", "a") as file:
        file.write(f"Trial {trial.number}, Mean Distance: {mean_distance}, BOPs: {bops}, Inference time: {inference_time}, Validation Loss: {validation_loss}, Param Count: {param_count}, Hyperparams: {trial.params}\n")
    return mean_distance, bops


if __name__ == "__main__":
    device = torch.device('cuda:0') #TODO: Change to fit anyones device
    batch_size=256
    train_loader, val_loader, test_loader = setup_data_loaders(batch_size, IMG_SIZE = 11, aug=1, num_workers=4, pin_memory=False, prefetch_factor=2)
    print('Loaded Dataset...')

    study = optuna.create_study(sampler=optuna.samplers.NSGAIISampler(population_size = 20), directions=['minimize', 'minimize']) #min mean_distance and inference time
    #Queue OpenHLS & BraggNN architectures to show the search strategy what to beat.
    study.enqueue_trial(OpenHLS_params)
    study.enqueue_trial(BraggNN_params)
    study.enqueue_trial(Example1_params)
    study.enqueue_trial(Example2_params)
    study.enqueue_trial(Example3_params)
    study.optimize(objective, n_trials=1000)

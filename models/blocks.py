import torch
import torch.nn as nn
import optuna

INPUT_SIZE = (4,1,11,11)

#Pretty much ready to go
#Convolution Attention, [No MLP/FeedForward after], no need for sampling other than in/out channels
#TODO: Eventually we add projection layer? or give it the option to use one.
#TODO: BraggNN does not divide by d. Should we? lets give it the option
class ConvAttn(torch.nn.Module):
    def __init__(self, in_channels = 16, hidden_channels = 8):
        super().__init__()
        self.Wq = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.Wk = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.Wv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)
        self.proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        query = self.Wq(x)
        key = self.Wk(x)
        value = self.Wv(x)
        z = self.softmax(query * key) * value 
        return x + self.proj(z)

class ConvBlock(torch.nn.Module):
    def __init__(self, channels, kernels, acts, norms, input_size = [4,16,9,9]):
        super().__init__()
        self.layers = []
        for i in range(len(kernels)):
            self.layers.append( nn.Conv2d(channels[i], channels[i+1], 
                                          kernel_size=kernels[i], stride=1, 
                                          padding = (kernels[i] - 1) // 2) )
            if norms[i] == 'batch':
                self.layers.append( nn.BatchNorm2d(channels[i+1]) )
            elif norms[i] == 'layer':
                self.layers.append( nn.LayerNorm([channels[i+1]]+input_size[2:]) )

            self.layers.append(acts[i])
        self.layers = nn.Sequential(*self.layers)
      
    def forward(self, x):
        return self.layers(x)

#TODO: Add variable length with pass through layers ie lambda x: x
class MLP(torch.nn.Module):
    def __init__(self, widths, acts, norms):
        super().__init__()

        self.layers = []
        for i in range(len(acts)): 
            self.layers.append( nn.Linear(widths[i], widths[i+1]) )
            if norms[i] == 'batch':
                self.layers.append( nn.BatchNorm1d(widths[i+1]) )
            elif norms[i] == 'layer':
                self.layers.append( nn.LayerNorm(widths[i+1]) )
            #elif None, skip
            self.layers.append( acts[i] )
        self.layers = nn.Sequential(*self.layers)
        

    def forward(self, x):
        return self.layers(x)

def sample_ConvAttn(trial, prefix):
    channel_space = (1,2,4,8,12,16)
    hidden_channels = channel_space[trial.suggest_int(prefix + '_hiddenchannel', 0, len(channel_space) - 1)]
    return hidden_channels

#prefix of the name you suggest variables for, a prefix needs to be mapped to a unique block location.
def sample_ConvBlock(trial, prefix, in_channels, num_layers = 2):
    #Search space to sample from
    channel_space = (2,4,8,16)
    kernel_space = (1,3)
    act_space = (nn.ReLU(), nn.GELU(), lambda x: x)
    norm_space = (None, 'layer', 'batch')

    channels = [in_channels] + [channel_space[ trial.suggest_int(prefix + '_channels_' + str(i), 0, len(channel_space) - 1) ]
                                    for i in range(num_layers)] #Picks an integer an index of channel_space for easier sampling
    kernels = [trial.suggest_categorical(prefix + '_kernels_' + str(i), kernel_space) for i in range(num_layers)]
    norms = [trial.suggest_categorical(prefix + '_norms_' + str(i), norm_space) for i in range(num_layers)]
    acts = [act_space[trial.suggest_categorical(prefix + '_acts_' + str(i), torch.arange(0, len(act_space) - 1))] for i in range(num_layers)]

    return channels, kernels, acts, norms 

def sample_MLP(trial, in_dim, prefix = 'MLP', num_layers = 4):
    width_space = (4,8,12,16,24,32,64)
    act_space = (nn.ReLU(), nn.LeakyReLU(), nn.GELU(), lambda x: x)
    norm_space = (None, 'layer', 'batch')

    widths = [in_dim] + [width_space[trial.suggest_int(prefix + '_width_' + str(i+1), 0, len(width_space) - 1)] for i in range(num_layers-1)] + [2]
    acts = [act_space[trial.suggest_categorical(prefix + '_acts_' + str(i), torch.arange(0, len(act_space) - 1))] for i in range(num_layers)]
    norms = [trial.suggest_categorical(prefix + '_norms_' + str(i), norm_space) for i in range(num_layers)]

    return widths, acts, norms

#This function is pretty much done.
#Requires all blocks/mlp to be created to limit hyperparams
class CandidateArchitecture(torch.nn.Module):
    def __init__(self, Blocks, MLP, hidden_channels, input_channels = 1):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=(3, 3), stride=(1, 1)) #Initial Projection Layer
        self.Blocks = Blocks
        self.MLP = MLP

    def forward(self, x):
        x = self.conv(x)
        x = self.Blocks(x)
        x = torch.flatten(x, 1)
        x = self.MLP(x)
        return x








#Leaving this in for later, currently not used/working.
class TransformerBlock(torch.nn.Module):
    def __init__(self, input_size = (4, 16, 9, 9)):
        super().__init__()
        embed_dim = input_size[-2] * input_size[-1]
               
        #Sample parameters for LinearAttention, rewrite with optuna samplers.
        trial = {
            'num_heads' : 1, #pick from [1,2,4,6,8]
            'norm' : nn.BatchNorm1d(input_size[1]), #pick from [nn.BatchNorm1d(input_size[1]), nn.LayerNorm((4,16,81))]
            'hidden_dim_scale' : 2, #pick from [1,2,4]
            'dropout' : .1, #float
            'bias' : True, #[True, False]
            'num_layers': 1, #[1,2,3]
        }

        self.layers = [ nn.TransformerEncoderLayer(d_model=embed_dim,
                                  nhead=trial['num_heads'],
                                  dim_feedforward=embed_dim * trial['hidden_dim_scale'],
                                  dropout=trial['dropout'],
                                  bias=trial['bias']) for i in range(trial['num_layers'])]

    def forward(self, x):
        x = torch.flatten(x, 2) #now (4, 16, 81)
        for l in self.layers:
          x = l(x)

#Leaving this in for later, currently not used/working.
class SkipBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #Sample hyperparameters
        self.input_size = (4,16,11,11)
        self.channels = [16,4,4,16]
        self.kernels = [1,3,1] #[1,3,5]
        self.act = [nn.ReLU(), nn.ReLU(), lambda x: x, nn.ReLU()] #pick from [nn.ReLU(), nn.LeakyReLU(), nn.GeLU(), lambda x: x]
        self.norm = ['batch', 'batch', 'batch'] #pick from ['identity', 'layer', 'batch']

        self.layers = []
        for i in range(len(self.kernels)):
            self.layers.append( nn.Conv2d(self.channels[i], self.channels[i+1], 
                                          kernel_size=self.kernels[i], stride=1, 
                                          padding = (self.kernels[i] - 1) // 2) )
            if self.norm[i] == 'batch':
                self.layers.append( nn.BatchNorm2d(self.channels[i+1]) )
            elif self.norm[i] == 'layer':
                self.layers.append( nn.LayerNorm(self.input_size) )

            self.layers.append( self.act[i] )
      

    def forward(self, x):
        z = x 
        for l in self.layers:
            z = l(z)
        x += z
        x = self.act[-1](x) #Activation after skip
        return x
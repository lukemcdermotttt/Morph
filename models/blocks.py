import torch
import torch.nn as nn

INPUT_SIZE = (4,1,11,11)

#Convolution Attention, [No MLP/FeedForward after], no need for sampling other than in/out channels
#TODO: Eventually we add projection layer? or give it the option to use one.
#TODO: BraggNN does not divide by d. Should we? lets give it the option
class ConvAttn(torch.nn.Module):
    def __init__(self):
        super().__init__(in_channels = 16, out_channels = 8)
        self.Wq = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.Wk = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.Wv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        query = self.Wq(x)
        key = self.Wk(x)
        value = self.Wv(x)
        z = self.softmax(query * key) * value 
        return x


#This needs a ton of work.
class ConvBlock(torch.nn.Module):
    def __init__(self):
        super().__init__(input_channels = 16)
        #Sample hyperparameters
        self.input_size = (4,16,11,11)
        self.channels = [16,8,2]
        self.kernels = [3,3] #[1,3,5]
        self.act = [nn.ReLU(), nn.ReLU()] #pick from [nn.ReLU(), nn.LeakyReLU(), nn.GeLU(), lambda x: x]
        self.norm = ['identity', 'identity'] #pick from ['identity', 'layer', 'batch']
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
        for l in self.layers:
            x = l(x)
        return x


#This function is ready to go
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__(width, act, norm)

        self.layers = []
        for i in range(len(act)):
            self.layers.append( nn.Linear(width[i], width[i+1]) )
            if norm[i] == 'batch':
                self.layers.append( nn.BatchNorm1d(width[i+1]) )
            elif norm[i] == 'layer':
                self.layers.append( nn.LayerNorm(width[i+1]) )
            #elif None, skip
            self.layers.append( self.act[i] )

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

#This function is pretty much done.
#Requires all blocks/mlp to be created to limit hyperparams
class CandidateArchitecture(torch.nn.Module):
    def __init__(self):
        super().__init__(Block1, Block2, Block3, MLP, hidden_channels, input_channels = 1)
        self.conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=(3, 3), stride=(1, 1)) #Initial Projection Layer
        self.Block1, self.Block2, self.Block3 = Block1, Block2, Block3
        self.MLP = MLP

    def forward(self, x):
        x = self.conv(x)
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = torch.flatten(x, 1)
        x = self.MLP(x)
        return x

"""
#Psuedo-code for hierarchical sampling in optuna
def objective(trial):

    hidden_channels = trial.suggest() #this is for the very first conv projection layer found in CandidateArchitecture
    b1 = trial.suggest_categorical('b1', ['Conv', 'ConvAttn', 'None'])
    b2 = trial.suggest_categorical('b2', ['Conv', 'ConvAttn', 'None'])
    b3 = trial.suggest_categorical('b3', ['Conv', 'ConvAttn', 'None'])

    if b1 == 'Conv':
        channels = trial.suggest('b1_Conv_channels', ) #-> since first block, the input_channels to Conv here is decided before so this is fixed. Sample the other 2 channel params tho
        kernel_size = trial.suggest('b1_Conv_kernelsize', )
        b1_out_channels = channels[-1] #save the final out dimension so b2 knows what to expect
        ...
        Block1 = Conv(channels, kernel_size) #Create block here, then pass it to CandidateArchitecture Later
    elif b1 == 'ConvAttn':
        b1_in_channels = hidden_channels #out dimension of conv projection
        b1_out_channels = trial.suggest()
        Block1 = ConvAttn(hidden_channels, b1_out_channels)
    ...
    if b2 == 'Conv':
        channels = trial.suggest('b2_Conv_channels', )
        ...
        Block2 = Conv(..)
    ...

    #Pick MLP parameters for 4 linear layers
    widths = [162, 64, 32, 16, 2] #the final dim 2 is fixed, the initial dim is dependent on block2, pick the 3 intermediate widths
    act = [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()] #pick from [nn.ReLU(), nn.LeakyReLU(), nn.GeLU(), lambda x: x]
    norm = [None, None, None, None] #pick from [None, 'layer', 'batch'], identity means do nothing or we can make this None
    MLP = MLP(param1, param2, ...)

    #build model
    model = CandidateArchitecture(Block1, Block2, Block3, MLP, hidden_channels)

    return evaluate(model)
"""


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
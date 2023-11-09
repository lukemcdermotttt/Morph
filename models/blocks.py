import torch
import torch.nn as nn


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

#Convolution Attention, [No MLP/FeedForward after], no need for sampling other than in/out channels
#TODO: Should we add projection layer? should this be sampled from optuna?
class ConvAttention(torch.nn.Module):
    def __init__(self):
        super().__init__(in_channels = 16, out_channels = 8)
        #Attention Block
        self.Wq = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.Wk = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.Wv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        query = self.Wq(x)
        key = self.Wk(x)
        value = self.Wv(x)
        z = self.softmax(query * key) * value #BraggNN does not divide by d. Should we?
        return x


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

class ConvBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
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

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #Sample hyperparameters
        self.widths = [162, 64, 32, 16, 2]
        self.input_size = (4,16,11,11)
        self.act = [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()] #pick from [nn.ReLU(), nn.LeakyReLU(), nn.GeLU(), lambda x: x]
        self.norm = [None, None, None, None] #pick from [None, 'layer', 'batch'], identity means do nothing or we can make this None

        self.layers = []
        for i in range(len(self.act)):
            self.layers.append( nn.Linear(self.widths[i], self.widths[i+1]) )
            if self.norm[i] == 'batch':
                self.layers.append( nn.BatchNorm2d(self.channels[i+1]) )
            elif self.norm[i] == 'layer':
                self.layers.append( nn.LayerNorm(self.input_size) )

            self.layers.append( self.act[i] )
      

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

class CandidateArchitecture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        channels = 16
        self.conv = nn.Conv2d(1, channels, kernel_size=(3, 3), stride=(1, 1))
        self.feature_extractor = [ ConvAttention(), SkipBlock(), ConvBlock() ]
        self.MLP = MLP()
      

    def forward(self, x):
        x = self.conv(x) #for attention, this acts as the tokenizer
        for l in self.feature_extractor:
            x = l(x)
        x = torch.flatten(x, 1)
        x = self.MLP(x)
        return x

"""
model = CandidateArchitecture()
x = torch.randn((4,1,11,11))
x = model(x)
print(x.size())
"""
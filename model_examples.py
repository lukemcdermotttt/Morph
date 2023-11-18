from data import BraggnnDataset
from data.BraggnnDataset import setup_data_loaders
import torch
import torch.nn as nn
import optuna
from models.blocks import *
from models.train_utils import *
import time

x = torch.randn((256,1,11,11))

#High performing model found by search, Mean Distance: 0.203, BOPs: 17.3 Million, Param Count: 13922
Blocks = nn.Sequential(
    ConvBlock([16,4,2], [1,3], [nn.GELU(), nn.GELU()], [None, 'batch'],img_size=9)
)
mlp = MLP(widths=[98, 64, 8, 4, 2], acts=[nn.ReLU(), nn.GELU(), nn.GELU(), None], norms=['layer', 'batch','layer',None])
model = CandidateArchitecture(Blocks,mlp,16)
y = model(x)
print('Example Model:')
print(model)

#OpenHLS Model
Blocks = nn.Sequential(
    ConvAttn(16,8, norm=None, act=nn.ReLU()),
    ConvBlock([16,8,2], [3,3], [nn.ReLU(), nn.ReLU()], [None, None],img_size=9)
)
mlp = MLP(widths=[50, 16, 8, 4, 2], acts=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()], norms=[None, None, None, None])
model = CandidateArchitecture(Blocks,mlp,16)
y = model(x)
print('OpenHLS Model:')
print(model)


#BraggNN model
Blocks = nn.Sequential(
    ConvAttn(64,32, norm=None, act=nn.LeakyReLU(negative_slope=0.01)),
    ConvBlock([64,32,8], [3,3], [nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01)], [None, None],img_size=9)
)
mlp = MLP(widths=[200,64,32,16,8,2], acts=[nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01), None], norms=[None, None, None, None, None])
model = CandidateArchitecture(Blocks,mlp,64)
y = model(x)
print('Original BraggNN Model:')
print(model)

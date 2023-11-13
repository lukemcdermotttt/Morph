from data import BraggnnDataset
from data.BraggnnDataset import setup_data_loaders
import torch
import torch.nn as nn
import optuna
from models.blocks import *
from models.train_utils import *
import time


x = torch.randn((256,1,11,11))


#High performing model found by search
Blocks = nn.Sequential(
    ConvBlock([24,16,16], [3,1], [nn.GELU(), nn.ReLU()], [None, 'batch']),
    ConvAttn(16,1),
    ConvBlock([16,4,16], [3,3], [nn.GELU(), nn.GELU()], [None, 'batch'])
)
mlp = MLP(widths=[1296, 64, 16, 4, 2], acts=[nn.GELU(), None, None, None], norms=['layer', 'layer','layer',None])
model = CandidateArchitecture(Blocks, mlp,24)
y = model(x)

#OpenHLS Model
Blocks = nn.Sequential(
    ConvAttn(16,8, norm=None, act=nn.ReLU()),
    ConvBlock([16,8,2], [3,3], [nn.ReLU(), nn.ReLU()], [None, None])
)
mlp = MLP(widths=[2*9*9, 16, 8, 4, 2], acts=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()], norms=[None, None, None, None])
model = CandidateArchitecture(Blocks, mlp,16)
y = model(x)
print('OpenHLS Model:')
print(model)

#BraggNN model
Blocks = nn.Sequential(
    ConvAttn(64,32, norm=None, act=nn.LeakyReLU(negative_slope=0.01)),
    ConvBlock([64,32,8], [3,3], [nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01)], [None, None])
)
mlp = MLP(widths=[8*9*9,64,32,16,8,2], acts=[nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01), None], norms=[None, None, None, None, None])
model = CandidateArchitecture(Blocks, mlp,64)
y = model(x)
print('Original BraggNN Model:')
print(model)

"""
for h in [1,8,2,4,6,10]:
    Blocks = nn.Sequential(
        ConvBlock([24,16,16], [3,1], [nn.GELU(), nn.ReLU()], [None, 'batch']),
        ConvAttn(16,h),
        ConvBlock([16,4,16], [3,3], [nn.GELU(), nn.GELU()], [None, 'batch'])
    )
    mlp = MLP(widths=[1296, 64, 16, 4, 2], acts=[nn.GELU(), None, None, None], norms=['layer', 'layer','layer',None])

    model = CandidateArchitecture(Blocks, mlp,24)

    x = torch.randn((256,1,11,11))
    start = time.time()
    for _ in range(10000):
        model(x)
    end = time.time()
    print('Inference time w/ Hidden Dim in Attn = ' + str(h) + ': ', end-start)
"""
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
    ConvBlock([24,16,16], [3,1], [nn.GELU(), nn.ReLU()], [None, 'batch'],img_size=9),
    ConvAttn(16,1),
    ConvBlock([16,4,16], [3,3], [nn.GELU(), nn.GELU()], [None, 'batch'],img_size=7)
)
mlp = MLP(widths=[144, 64, 16, 4, 2], acts=[nn.GELU(), None, None, None], norms=['layer', 'layer','layer',None])
model = CandidateArchitecture(Blocks,mlp,24)
y = model(x)
print('Our Model:')
print(model)
device = torch.device("cuda:1")
model.to(device)
dummy_input = torch.randn((256,1,11,11), dtype=torch.float).to(device)

# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 100000
warmup_repetitions = 5000
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(warmup_repetitions):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn, std_syn)

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
model.to(device)
dummy_input = torch.randn((256,1,11,11), dtype=torch.float).to(device)

# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(warmup_repetitions):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn, std_syn)

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
model.to(device)
dummy_input = torch.randn((256,1,11,11), dtype=torch.float).to(device)

# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(warmup_repetitions):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn, std_syn)




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
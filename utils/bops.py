import torch
import torch.nn as nn
import math 

#Returns sparsity or percentage of zeros in a tensor
def get_sparsity(tensor):
    num_zeros = torch.sum(tensor == 0)
    total_params = tensor.numel()
    return num_zeros / total_params #sparsity

"""
Calculate the bops in the matrix multiplication in attention
Let a,b be the shape of matrices A & B
For the QK matmul, we perform a dot product across the embedding dimension, for each row and column,
so there are seq_len^2 many dot products. Each dot product uses embed_dim many multiplications and
embed_dim - 1 additions. 

DISCLAIMER: This only calculates bops for dense matmuls
"""
def get_matmul_bops(a, b, bit_width=32):
    if a[0] != b[0] or a[1] != b[2] or :
        raise ValueError("Inner dimensions of arrays do not match for matrix multiplication.")
    batch_size = a[0]
    embed_dim = a[1]
    seq_len = a[2]

    bops_per_mult = bit_width**2
    bops_per_add = bit_width

    mult_bops = seq_len * seq_len * embed_dim * bops_per_mult
    add_bops = seq_len * seq_len * (embed_dim - 1) * bops_per_add
    bops = mult_bops + add_bops
    return bops

def get_linear_bops(layer, bit_width=32):
    sparsity = get_sparsity(layer.weight.data)
    return layer.output_features * layer.input_features * ( (1-sparsity) * bit_width**2 + 2*bit_width + math.log2(layer.input_features))

def get_conv2d_bops(layer, input_shape, bit_width=32):
    output_spatial_dim = input_shape[-1] if layer.kernel_size == 1 else input_shape[-1] - 2
    output_shape = (input_shape[0], layer.out_channels, output_spatial_dim, output_spatial_dim)

    input_numel = torch.prod(input_shape[1:])
    output_numel = torch.prod(output_shape[1:])
    
    sparsity = get_sparsity(layer.weight.data)
    return output_numel * input_numel * layer.kernel_size**2 * ((1-sparsity) * bit_width**2 + 2*bit_width + math.log2(input_numel * layer.kernel_size**2))

def get_Conv_bops(block, input_shape, bit_width=32):
    bops = 0
    for i, layer in enumerate(block.layers):
        if isinstance(layer, nn.Conv2d):
            sparsity = get_sparsity(layer.weight.data)
            bops += get_conv2d_bops(layer, input_shape, bit_width)
            #Update input_shape for future Conv2D layers
            output_spatial_dim = input_shape[-1] if layer.kernel_size == 1 else input_shape[-1] - 2
            input_shape = (input_shape[0], layer.out_channels, output_spatial_dim, output_spatial_dim)
    return bops

def get_ConvAttn_bops(module, input_shape=(64,1,9,9), bit_width):
    bops = 0
    #Add bops for each Wk, Wq, Wv, Proj
    for i, layer in enumerate(block.layers):
        if isinstance(layer, nn.Conv2d):
            sparsity = get_sparsity(layer.weight.data)
            bops += get_conv2d_bops(layer, sparsity, bit_width)
        elif isinstance(layer, nn.Linear):
            sparsity = get_sparsity(layer.weight.data)
            bops += get_linear_bops(layer, sparsity, bit_width)
    
    #Get Input Shape and Reshaped dims
    batch_size, seq_len, h, w = input_shape
    embed_dim = h*w

    #Add softmax bops
    bops += batch_size * embed_dim**2 * 1.5 * (bit_width-1) + batch_size * embed_dim * (embed_dim-1) + batch_size*(embed_dim)**2

    #Add QK MatMul bops
    Q_shape = (batch_size, embed_dim, seq_len)
    K_shape = (batch_size, seq_len, embed_dim)
    bops += get_matmul_bops(Q_shape, K_shape, bit_width=32)

    #Add SV Matmul bops
    S_dims=(batch_size, seq_len, seq_len) #S is the output scores from softmax
    V_dims=(batch_size, embed_dim, seq_len)
    bops += get_matmul_bops(S_shape, V_shape, bit_width=32)

    return bops

def get_MLP_bops(block, bit_width=32):
    bops = 0
    for i, layer in enumerate(block.layers):
        if isinstance(layer, nn.Linear):
            sparsity = get_sparsity(layer.weight.data)
            bops += get_linear_bops(layer, sparsity, bit_width)
    return bops

#BatchNorm, LayerNorm, etc. are extremely small relative to the Conv2D, linear, and matmul operations.
import numpy as np
import torch
import torch.nn as nn
import math 

def countNonZero(tensor, name):
    nz_count = np.count_nonzero(tensor)
    total_params = np.prod(tensor.shape)
    sparsity = 1-(nz_count / total_params)
    # print(sparsity)
    # print(f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}")
    return nz_count, total_params, sparsity

def forwardHook(module, input, output):
    if len(output) == 2:
        output, _ = output
    # print(f"[{module.__class__.__name__}] Forward hook: Input shape: {input[0].shape}, Output shape: {output.shape}")
    module.input_shape = input[0].shape
    module.output_shape = output.shape

def registerHooks(model):
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Sequential):
            continue
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.MaxPool2d, nn.Flatten,
                                 nn.ReLU, nn.GELU, nn.LeakyReLU, nn.BatchNorm2d, nn.LayerNorm)):
            hooks.append(module.register_forward_hook(forwardHook))
        # elif isinstance(module, (QuantLinear, QuantConv2d)):
        #     hooks.append(module.register_forward_hook(forwardHook))
    return hooks

def removeHooks(hooks):
    for hook in hooks:
        hook.remove()

def getInputShapes(model, input_shape):
    hooks = registerHooks(model)
    model(torch.randn(input_shape))
    removeHooks(hooks)

def calculate_layer_sparsity(model):
    layer_sparsity = {}
    for name, module in model.named_modules():

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            tensor = module.weight.data.cpu().numpy()
            _, _, sparsity = countNonZero(tensor, f"{name}.weight")
            layer_sparsity[name] = sparsity

        elif isinstance(module, ConvAttn):
            sparsities = []
            for conv in [module.Wq, module.Wk, module.Wv, module.proj]:
                tensor = conv.weight.data.cpu().numpy()
                _, _, sparsity = countNonZero(tensor, f"{name}.{conv.__class__.__name__}.weight")
                # print(sparsity)
                sparsities.append(sparsity)
            layer_sparsity[name] = tuple(sparsities)

        elif isinstance(module, ConvBlock):
            for i, sub_module in enumerate(module.layers):
                if isinstance(sub_module, nn.Conv2d):
                    tensor = sub_module.weight.data.cpu().numpy()
                    _, _, sparsity = countNonZero(tensor, f"{name}.layers[{i}].weight")
                    # print(sparsity)
                    layer_sparsity[f"{name}.layers[{i}]"] = sparsity
    return layer_sparsity

def old_matmul_3darray_bops(shape1, shape2, bit_width):
    """
    Calculate the Bit Operations (BOP) for multiplying two 3D arrays.
    """

    b, hw, hc1 = shape1
    _, hc2, h_w = shape2

    if hc1 != hc2:
        raise ValueError("Inner dimensions of arrays do not match for matrix multiplication.")

    total_multiplications = b * hw * h_w * hc1
    
    total_additions = b * hw * h_w * (hc1 - 1)

    bop_per_multiplication = bit_width ** 2
    bop_per_addition = bit_width

    batch, n, d = shape1

    print('my guess:',  )
    print('tot mult:', total_multiplications, bop_per_multiplication, total_multiplications * bop_per_multiplication)
    total_bop = (total_multiplications * bop_per_multiplication) + (total_additions * bop_per_addition)

    return total_bop

def matmul_3darray_bops(matrix_a_dims, matrix_b_dims,sparsity,bit_width=32):
    a_batch_size, a_embed_dim, seq_len  = matrix_a_dims
    b_batch_size, seq_len, b_embed_dim  = matrix_b_dims
    if a_batch_size != b_batch_size or b_embed_dim != b_embed_dim:
        raise ValueError("Inner dimensions of arrays do not match for matrix multiplication.")
    batch_size = a_batch_size
    embed_dim = a_embed_dim

    mult_bops = bit_width**2
    add_bops = bit_width
    total_mult_bops = seq_len * seq_len * embed_dim * mult_bops * (1 - sparsity)
    total_add_bops = seq_len * seq_len * (embed_dim - 1) * add_bops
    total_bops = total_mult_bops + total_add_bops
    return total_bops

def calculate_linear_bops( layer, sparsity, weight_bit_width=32, activation_bit_width=32):
    input_features, output_features = layer.in_features, layer.out_features
    # Multiplications + Additions, adjusted for bit widths and sparsity
    # print(f'{output_features} * {input_features} * ({sparsity} * {activation_bit_width} * {weight_bit_width} + {activation_bit_width} + {weight_bit_width} + {math.log2(input_features)})')
    # num_nonzero_weights = output_features * input_features * (1-sparsity)
    return output_features * input_features * ( (1-sparsity) * activation_bit_width * weight_bit_width + activation_bit_width + weight_bit_width + math.log2(input_features))
    # return 2 * input_features * output_features * sparsity * weight_bit_width * activation_bit_width

def calculate_conv2d_bops(layer, sparsity, activation_bit_width, weight_bit_width):
    in_shape = np.prod(layer.in_channels)
    out_shape = np.prod(layer.out_channels)
    kernel_size = np.prod(layer.kernel_size)
    # Multiplications + Additions, adjusted for bit widths and sparsity
    # print("conv2d ",sparsity)
    return out_shape * in_shape * kernel_size * kernel_size * ((1-sparsity) * activation_bit_width * weight_bit_width + activation_bit_width + weight_bit_width + math.log2(in_shape * kernel_size * kernel_size))
    # return in_shape * out_shape * kernel_size * kernel_size * (sparsity * self.activation_bit_width * self.weight_bit_width + self.activation_bit_width + self.weight_bit_width + math.log2(in_shape * kernel_size * kernel_size))

def calculate_batchnorm_bops(layer):
    # BatchNorm involves normalization and scaling which are linear operations
    features = layer.num_features
    return 2 * features

def calculate_layernorm_bops(self, layer):
    # Similar to BatchNorm, but normalization is across a different dimension
    normalized_shape = layer.normalized_shape[0]
    return 2 * normalized_shape

def calculate_maxpool2d(layer):
    B, C, H, W = layer.input_shape  # Assuming input_shape is a tuple (Batch, Channel, Height, Width)
    kH = layer.kernel_size
    kW = layer.kernel_size

    # Approximate number of windows
    num_windows = (H // kH) * (W // kW)

    # Operations per window (ignoring the -1 for simplicity)
    ops_per_window = kH * kW

    # Total operations
    total_ops = B * C * num_windows * ops_per_window

    return total_ops

def calculate_convblock_bops(conv_block, sparsity_dict=None, weight_bit_width=32, activation_bit_width=32):
    total_bops = 0
    for i, layer in enumerate(conv_block.layers):
        layer_name = f"layers[{i}]"

        if isinstance(layer, nn.Conv2d):
            if sparsity_dict:
                sparsity = sparsity_dict.get(layer_name, 0)  # Default sparsity is 0 (no sparsity) if not found
            else:
                sparsity = 0

            layer_bops = calculate_conv2d_bops(layer, sparsity, activation_bit_width, weight_bit_width)
            total_bops += layer_bops
        if isinstance(layer, nn.BatchNorm2d):
          total_bops += calculate_batchnorm_bops(layer)


    return total_bops

def old_calculate_convattn_bops(module, sparsity, input_shape, weight_bit_width=32, activation_bit_width=32):
    total_bops = 0

    for i, conv in enumerate([module.Wq, module.Wk, module.Wv, module.proj]):
        conv_bops = calculate_conv2d_bops(conv, sparsity[i], activation_bit_width, weight_bit_width)
        total_bops += conv_bops

    # Calculate BOPs for matrix multiplications in attention mechanism
    print(f" total conv bops: {total_bops}")
    b_w=weight_bit_width

    b, c, h, w = input_shape
    hw = h * w

    # Softmax
    hidden_channels=module.hidden_channels
    softmax_bops = b*hw**2 * 1.5 * (b_w-1) + b* hw * (hw-1) + b*(h*w)**2
    print(f"softmax bops: {softmax_bops}")
    total_bops += softmax_bops

    #MatMul
    query_shape = (b, hw, hidden_channels)
    keyshape = (b, hidden_channels, hw)
    matmul1_bops = matmul_3darray_bops(query_shape,keyshape, b_w)

    z_dims=(b,hidden_channels, hidden_channels)
    value_dims=(b, hw, hidden_channels)
    matmul2_bops= matmul_3darray_bops(z_dims, value_dims, b_w)
    print(matmul1_bops, matmul2_bops)
    # Add the matmul BOPS to the total
    total_bops += matmul1_bops + matmul2_bops
    print('Matmul bops:', matmul1_bops + matmul2_bops,  matmul1_bops, matmul2_bops)
    return total_bops

def calculate_convattn_bops(module, input_shape, bit_width):
    total_bops = 0

    for i, conv in enumerate([module.Wq, module.Wk, module.Wv, module.proj]):
        conv_bops = calculate_conv2d_bops(conv, 0, bit_width, bit_width) #TODO: Change to calculate sparsity
        total_bops += conv_bops
    print(f"Total conv bops: {total_bops}")
    
    #Input is reshaped
    batch_size, seq_len, h, w = input_shape
    embed_dim = h*w

    # Softmax
    softmax_bops = batch_size * embed_dim**2 * 1.5 * (bit_width-1) + batch_size * embed_dim * (embed_dim-1) + batch_size*(embed_dim)**2
    print(f"Softmax bops: {softmax_bops}")
    total_bops += softmax_bops

    #MatMul
    Q_shape = (batch_size, embed_dim, seq_len)
    K_shape = (batch_size, seq_len, embed_dim)
    QK_bops = matmul_3darray_bops(Q_shape,K_shape, 0, bit_width) #TODO: Change for sparsity

    S_dims=(batch_size, seq_len, seq_len) #S is the output scores from softmax
    V_dims=(batch_size, embed_dim, seq_len)
    SV_bops= matmul_3darray_bops(S_dims, V_dims, 0, bit_width) #TODO: Change for sparsity
    print(QK_bops, SV_bops)
    total_bops += QK_bops + SV_bops
    return total_bops

def calculate_mlpblock_bops(mlpblock, sparsity_dict=None, weight_bit_width=32, activation_bit_width=32):
    total_bops=0
    for i, layer in enumerate(mlpblock.layers):
          layer_name = f"layers[{i}]"

          if isinstance(layer, nn.Linear):
            if sparsity_dict:
                sparsity = sparsity_dict.get(layer_name, 0)
            else:
                sparsity = 0

            #  layer, sparsity, weight_bit_width=32, activation_bit_width=32
            layer_bops = calculate_linear_bops(layer, sparsity, weight_bit_width,activation_bit_width)
            total_bops +=layer_bops


    return total_bops

def BOP_counter(model, input_shape, weight_bit_width=32, activation_bit_width=32):
  getInputShapes(model, input_shape)

  full_bops=0
  layer_sparsity = calculate_layer_sparsity(model)

  for name, module in model.named_modules():

    if isinstance(module, ConvAttn):
            full_bops += calculate_convattn_bops(module,layer_sparsity[name], input_shape, weight_bit_width, activation_bit_width)

    if isinstance(module, ConvBlock):
      full_bops += calculate_convblock_bops(module, layer_sparsity, weight_bit_width, activation_bit_width)

    if isinstance(module, MLP):
      full_bops += calculate_mlpblock_bops(module, layer_sparsity, weight_bit_width, activation_bit_width)

  return full_bops


print(matmul_3darray_bops((1,81,8), (1, 8, 81), 0, 32))
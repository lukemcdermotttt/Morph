
import torch
import torch.nn as nn

class NLB(torch.nn.Module):
    def __init__(self, in_ch, relu_a=0.01):
        self.inter_ch = torch.div(in_ch, 2, rounding_mode='floor').item()
        super().__init__()
        self.theta_layer = torch.nn.Conv2d(in_channels=in_ch, out_channels=self.inter_ch, \
                            kernel_size=1, padding=0)
        self.phi_layer   = torch.nn.Conv2d(in_channels=in_ch, out_channels=self.inter_ch, \
                            kernel_size=1, padding=0)
        self.g_layer     = torch.nn.Conv2d(in_channels=in_ch, out_channels=self.inter_ch, \
                            kernel_size=1, padding=0)
        self.atten_act   = torch.nn.Softmax(dim=-1)
        self.out_cnn     = torch.nn.Conv2d(in_channels=self.inter_ch, out_channels=in_ch, \
                            kernel_size=1, padding=0)
        
    def forward(self, x):
        mbsz, _, h, w = x.size()
        
        theta = self.theta_layer(x).view(mbsz, self.inter_ch, -1).permute(0, 2, 1)
        phi   = self.phi_layer(x).view(mbsz, self.inter_ch, -1)
        g     = self.g_layer(x).view(mbsz, self.inter_ch, -1).permute(0, 2, 1)
        
        theta_phi = self.atten_act(torch.matmul(theta, phi))
        
        theta_phi_g = torch.matmul(theta_phi, g).permute(0, 2, 1).view(mbsz, self.inter_ch, h, w)
        
        _out_tmp = self.out_cnn(theta_phi_g)
        _out_tmp = torch.add(_out_tmp, x)
   
        return _out_tmp
    


class BraggNN_D(nn.Module):
    def __init__(self, imgsz, cnn_channels=(64, 32, 8), fcsz=(64, 32, 16, 8)):
        super(BraggNN_D, self).__init__()
        self.cnn_ops = nn.ModuleList()
        cnn_in_chs = (1, ) + cnn_channels[:-1]

        fsz = imgsz
        for ic, oc in zip(cnn_in_chs, cnn_channels):
            self.cnn_ops.append(nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=3, stride=1, padding=0))
            self.cnn_ops.append(nn.LeakyReLU(negative_slope=0.01))
            fsz -= 2  # adjust the size due to convolution without padding

        self.nlb = NLB(in_ch=cnn_channels[0])

        self.dense_ops = nn.ModuleList()
        dense_in_chs = (fsz * fsz * cnn_channels[-1], ) + fcsz[:-1]
        for ic, oc in zip(dense_in_chs, fcsz):
            self.dense_ops.append(nn.Linear(ic, oc))
            self.dense_ops.append(nn.LeakyReLU(negative_slope=0.01))

        # Output layer
        self.dense_ops.append(nn.Linear(fcsz[-1], 2))

    def forward(self, x):
        _out = x
        for layer in self.cnn_ops[:1]:
            _out = layer(_out)

        _out = self.nlb(_out)

        for layer in self.cnn_ops[1:]:
            _out = layer(_out)

        # _out = _out.view(_out.size(0), -1)  # Flatten the tensor for the dense layer
        _out = _out.reshape(_out.size(0), -1)

        for layer in self.dense_ops:
            _out = layer(_out)

        return _out
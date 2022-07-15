import math

import torch.nn.init as init
from torch import nn, Tensor

"""
assisting default layers
"""


class ContainerLinear(nn.Linear):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 bias: bool = True,
                 gain=1,
                 **kwargs):
        self.gain = gain
        super(ContainerLinear, self).__init__(input_dim,
                                              output_dim,
                                              bias=bias)

    def forward(self, input: Tensor, **kwargs):
        return super().forward(input)

    def reset_parameters(self):
        init.orthogonal_(self.weight, self.gain)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def update_input_dim(self, dim):
        self.in_features = dim

    def update_output_dim(self, dim):
        self.out_features = dim


class ContainerConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 gain=1,
                 **kwargs):
        self.gain = gain

        super(ContainerConv2d, self).__init__(in_channels,
                                              out_channels,
                                              kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              groups=groups,
                                              bias=bias,
                                              padding_mode=padding_mode)

    def forward(self, input: Tensor, **kwargs):
        return super().forward(input)

    def reset_parameters(self):
        init.orthogonal_(self.weight, self.gain)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def update_input_dim(self, dim):
        self.in_channels = dim

    def update_output_dim(self, dim):
        self.out_channels = dim

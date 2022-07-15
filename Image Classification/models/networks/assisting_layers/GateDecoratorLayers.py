import torch
from torch import nn as nn


class GatedBatchNorm(nn.Module):

    """
    Our interpretation/implementation of GatedBatchNorm of the paper
    Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks
    https://arxiv.org/abs/1909.08174
    """

    def __init__(self, bn: nn.BatchNorm1d, device="cuda"):
        super(GatedBatchNorm, self).__init__()
        self.device = device
        self.bn = bn.to(device)
        self.gate = torch.nn.Parameter(torch.randn(bn.weight.shape).to(device), requires_grad=True)

    def forward(self, x):
        dims = len(x.shape)
        if dims == 4:
            return (self.gate * (self.bn.forward(x)).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        elif dims == 2:
            return self.gate * self.bn.forward(x)
        else:
            raise ValueError

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    def buffers(self, *args, **kwargs):
        return self.bn.buffers(*args, **kwargs)
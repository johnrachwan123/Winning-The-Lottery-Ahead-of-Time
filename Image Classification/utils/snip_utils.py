import torch.nn.functional as F

"""
assisting maskeable functions for the methods we introduced.
"""

def snip_forward_conv2d(self, x):
    return F.conv2d(x,
                    self.weight * self.weight_mask,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups)


def snip_forward_linear(self, x):
    return F.linear(x.float(),
                    self.weight.float() * self.weight_mask.float(),
                    bias=self.bias)


def group_snip_forward_linear(self, x):
    return F.linear(x.float(),
                    self.weight,
                    # gov_out are new
                    bias=self.bias.float()) * self.gov_out.float()


def group_snip_conv2d_forward(self, x):
    # 0 is batch size, 1 is the channel dimension (# of channels)
    if self.weight.is_cuda:
        return (F.conv2d(x,
                         self.weight,
                         self.bias,
                         self.stride,
                         self.padding,
                         self.dilation,
                         self.groups).permute(0, 3, 2, 1) * self.gov_out.float()).permute(0, 3, 2, 1)
    else:
        return (F.conv2d(x.cpu(),
                         self.weight,
                         self.bias,
                         self.stride,
                         self.padding,
                         self.dilation,
                         self.groups).permute(0, 3, 2, 1) * self.gov_out.float()).permute(0, 3, 2, 1)

import torch.nn.functional as F
import torch

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


def group_snip_forward_embedded(self, input):
    # breakpoint()
    return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse) * self.gov_out.float()


def group_snip_forward_lstm(self, input, state):
    hx, cx = state
    gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
             torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy * self.gov_out.float(), cy

def group_snip_conv2d_forward(self, x):
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

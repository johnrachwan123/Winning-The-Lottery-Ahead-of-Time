import torch.nn as nn

"""
assisting layers
"""


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 input_dim=1,
                 output_dim=1,
                 downsample=False,
                 groups=1,
                 base_width=64,
                 padding=1,
                 norm_layer=nn.BatchNorm2d,
                 conv_layer=nn.Conv2d):
        super(BasicBlock, self).__init__()

        self._check_input(base_width, groups, padding)

        leak = 0.05
        gain = nn.init.calculate_gain('leaky_relu', leak)

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.pruneinput1 = conv_layer(input_dim, output_dim,
                                kernel_size=3,
                                stride=2 if downsample else 1,
                                padding=padding,
                                groups=groups,
                                bias=True,
                                dilation=padding,
                                gain=gain)
        self.bn1 = norm_layer(output_dim,
                              eps=1e-05,
                              momentum=0.1,
                              affine=True,
                              track_running_stats=True)
        self.relu = nn.LeakyReLU(leak, inplace=True)
        self.conv2 = conv_layer(output_dim, output_dim,
                                kernel_size=3,
                                stride=1,
                                padding=padding,
                                groups=groups,
                                bias=True,
                                dilation=padding,
                                gain=gain)

        self.bn2 = norm_layer(output_dim,
                              eps=1e-05,
                              momentum=0.1,
                              affine=True,
                              track_running_stats=True)
        if downsample:
            downsample = nn.Sequential(
                conv_layer(input_dim, output_dim,
                           kernel_size=1,
                           stride=2,
                           bias=False,
                           gain=gain),
                norm_layer(output_dim,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True)
            )
        self.downsample = downsample
        self.stride = 2 if downsample else 1

    def _check_input(self, base_width, groups, padding):
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if padding > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

    def forward(self, x):
        identity = x

        out = self.pruneinput1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not False:
            identity = self.downsample(x)
        try:
            out += identity
        except:
            breakpoint()
        out = self.relu(out)

        return out


class BasicBlock_downsample(nn.Module):
    expansion = 1

    def __init__(self,
                 input_dim=1,
                 output_dim=1,
                 downsample=True,
                 groups=1,
                 base_width=64,
                 padding=1,
                 norm_layer=nn.BatchNorm2d,
                 conv_layer=nn.Conv2d):
        super(BasicBlock_downsample, self).__init__()

        self._check_input(base_width, groups, padding)
        leak = 0.05
        gain = nn.init.calculate_gain('leaky_relu', leak)

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.dontpruneinput1 = conv_layer(input_dim, output_dim,
                                kernel_size=3,
                                stride=2 if downsample else 1,
                                padding=padding,
                                groups=groups,
                                bias=True,
                                dilation=padding,
                                gain=gain)

        self.bn1 = norm_layer(output_dim,
                              eps=1e-05,
                              momentum=0.1,
                              affine=True,
                              track_running_stats=True)
        self.relu = nn.LeakyReLU(leak, inplace=True)
        self.conv2 = conv_layer(output_dim, output_dim,
                                kernel_size=3,
                                stride=1,
                                padding=padding,
                                groups=groups,
                                bias=True,
                                dilation=padding,
                                gain=gain)

        self.bn2 = norm_layer(output_dim,
                              eps=1e-05,
                              momentum=0.1,
                              affine=True,
                              track_running_stats=True)
        if downsample:
            downsample = nn.Sequential(
                conv_layer(input_dim, output_dim,
                           kernel_size=1,
                           stride=2,
                           bias=False,
                           gain=gain),
                norm_layer(output_dim,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True)
            )
        self.downsample = downsample
        self.stride = 2 if downsample else 1

    def _check_input(self, base_width, groups, padding):
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if padding > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

    def forward(self, x):
        identity = x

        out = self.dontpruneinput1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not False:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

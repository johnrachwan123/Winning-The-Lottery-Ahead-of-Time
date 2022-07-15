import torch
import torch.nn as nn

from models.Pruneable import Pruneable
from models.networks.assisting_layers.ResNetLayers import BasicBlock, BasicBlock_downsample
from utils.constants import PROD_SMALL_POOL, SMALL_POOL


class ResNet18(Pruneable):

    def __init__(self, device="cuda", output_dim=2, input_dim=(1, 1, 1), **kwargs):
        super(ResNet18, self).__init__(device=device, output_dim=output_dim, input_dim=input_dim, **kwargs)

        channels, _, _ = input_dim

        leak = 0.05
        gain = nn.init.calculate_gain('leaky_relu', leak)
        self.hooks = {}
        self.conv1 = self.Conv2d(channels, 64,
                                 kernel_size=7,
                                 stride=2,
                                 padding=3,
                                 bias=False,
                                 gain=gain).to(device)
        self.bn1 = nn.BatchNorm2d(64).to(device)
        self.relu = nn.LeakyReLU(leak, inplace=True).to(device)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1).to(device)

        self.layer1 = nn.Sequential(
            BasicBlock(input_dim=64, output_dim=64, conv_layer=self.Conv2d),
            BasicBlock(input_dim=64, output_dim=64, conv_layer=self.Conv2d)
        ).to(device)

        self.layer2 = nn.Sequential(
            BasicBlock_downsample(input_dim=64, output_dim=128, downsample=True, conv_layer=self.Conv2d),
            BasicBlock(input_dim=128, output_dim=128, conv_layer=self.Conv2d)
        ).to(device)

        self.layer3 = nn.Sequential(
            BasicBlock_downsample(input_dim=128, output_dim=256, downsample=True, conv_layer=self.Conv2d),
            BasicBlock(input_dim=256, output_dim=256, conv_layer=self.Conv2d)
        ).to(device)

        self.layer4 = nn.Sequential(
            BasicBlock_downsample(input_dim=256, output_dim=512, downsample=True, conv_layer=self.Conv2d),
            BasicBlock(input_dim=512, output_dim=512, conv_layer=self.Conv2d)
        ).to(device)

        self.avgpool = nn.AdaptiveAvgPool2d(SMALL_POOL).to(device)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            self.Linear(input_dim=512 * PROD_SMALL_POOL, output_dim=256, bias=True, gain=gain),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            nn.Dropout(p=0.3, inplace=False),
            self.Linear(input_dim=256, output_dim=output_dim, bias=True),
        ).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    device = "cpu"

    mnist = torch.randn((21, 1, 28, 28)).to(device)
    print(ResNet18(output_dim=10, input_dim=mnist.shape[1:], device=device))
    cifar = torch.randn((21, 3, 32, 32)).to(device)
    imagenet = torch.randn((2, 4, 244, 244)).to(device)

    for test_batch in [mnist, cifar, imagenet]:
        conv = ResNet18(output_dim=10, input_dim=test_batch.shape[1:], device=device)

        print(conv.forward(test_batch).shape)

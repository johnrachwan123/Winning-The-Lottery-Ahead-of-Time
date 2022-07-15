import torch
import torch.nn as nn

from models.Pruneable import Pruneable
from utils.constants import PROD_SMALL_POOL, SMALL_POOL


class VGG16(Pruneable):
    def __init__(self, device="cuda", output_dim=2, input_dim=(1, 1, 1), **kwargs):
        super(VGG16, self).__init__(device=device, output_dim=output_dim, input_dim=input_dim, **kwargs)

        channels, _, _ = input_dim

        self.features1 = nn.Sequential(
            # in to 64
            VGGBlock(device=device, features_in=channels, features_out=64, conv_layer=self.Conv2d),
            VGGBlock(device=device, features_in=64, features_out=64, conv_layer=self.Conv2d),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        ).to(device)
        self.features2 = nn.Sequential(

            # 64 to 128
            VGGBlock(device=device, features_in=64, features_out=128, conv_layer=self.Conv2d),
            VGGBlock(device=device, features_in=128, features_out=128, conv_layer=self.Conv2d),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        ).to(device)
        self.features3 = nn.Sequential(

            # 128 to 256
            VGGBlock(device=device, features_in=128, features_out=256, conv_layer=self.Conv2d),
            VGGBlock(device=device, features_in=256, features_out=256, conv_layer=self.Conv2d),
            VGGBlock(device=device, features_in=256, features_out=256, conv_layer=self.Conv2d),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        ).to(device)
        self.features4 = nn.Sequential(

            # 256 to 512
            VGGBlock(device=device, features_in=256, features_out=512, conv_layer=self.Conv2d),
            VGGBlock(device=device, features_in=512, features_out=512, conv_layer=self.Conv2d),
            VGGBlock(device=device, features_in=512, features_out=512, conv_layer=self.Conv2d),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        ).to(device)
        self.features5 = nn.Sequential(

            # 512 to 1024
            VGGBlock(device=device, features_in=512, features_out=512, conv_layer=self.Conv2d),
            VGGBlock(device=device, features_in=512, features_out=512, conv_layer=self.Conv2d),
            VGGBlock(device=device, features_in=512, features_out=512 * 2, conv_layer=self.Conv2d),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        ).to(device)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=SMALL_POOL).to(device)

        leak = 0.05
        gain = nn.init.calculate_gain('leaky_relu', leak)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            self.Linear(input_dim=512 * PROD_SMALL_POOL * 2, output_dim=4096, bias=True, gain=gain),
            nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak, inplace=True),

            nn.Dropout(p=0.3, inplace=False),
            self.Linear(input_dim=4096, output_dim=4096, bias=True, gain=gain),
            nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak, inplace=True),

            nn.Dropout(p=0.3, inplace=False),
            self.Linear(input_dim=4096, output_dim=output_dim, bias=True),
        ).to(device)

    def forward(self, x):
        for feat in [self.features1, self.features2, self.features3, self.features4, self.features5]:
            x = feat.forward(x)

        # x = self.features.forward(x)
        x = self.avg_pool.forward(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier.forward(x)
        return x


class VGGBlock(nn.Module):

    def __init__(self, device="cuda", features_in=0, features_out=0, conv_layer=nn.Conv2d):
        super(VGGBlock, self).__init__()

        leak = 0.05
        gain = nn.init.calculate_gain('leaky_relu', leak)

        self.layers = nn.Sequential(
            conv_layer(features_in, features_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), gain=gain),
            nn.BatchNorm2d(features_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak, inplace=True)
        ).to(device)

    def forward(self, x, **kwargs):
        return self.layers.forward(x)


if __name__ == '__main__':
    device = "cpu"

    cifar = torch.randn((21, 3, 32, 32)).to(device)
    imagenet = torch.randn((2, 4, 244, 244)).to(device)

    for test_batch in [cifar, imagenet]:
        conv = VGG16(output_dim=10, input_dim=test_batch.shape[1:], device=device)

        print(conv.forward(test_batch).shape)

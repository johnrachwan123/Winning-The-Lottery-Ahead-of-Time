import torch
import torch.nn as nn

from models.Pruneable import Pruneable
from utils.constants import MIDDLE_POOL, PROD_MIDDLE_POOL, PROD_SMALL_POOL, SMALL_POOL


class AlexNet(Pruneable):

    def __init__(self, device="cuda", output_dim=2, input_dim=(1, 1, 1), **kwargs):
        super(AlexNet, self).__init__(device=device, output_dim=output_dim, input_dim=input_dim, **kwargs)

        channels, _, _ = input_dim

        leak = 0.05
        gain = nn.init.calculate_gain('leaky_relu', leak)

        self.features = nn.Sequential(
            self.Conv2d(channels, 64, kernel_size=5, stride=1, padding=2, gain=gain),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            nn.MaxPool2d(kernel_size=3, stride=2),

            self.Conv2d(64, 192, kernel_size=5, padding=2, gain=gain),
            nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            nn.MaxPool2d(kernel_size=3, stride=2),

            self.Conv2d(192, 384, kernel_size=3, padding=1, gain=gain),
            nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),

            self.Conv2d(384, 256, kernel_size=3, padding=1, gain=gain),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),

            self.Conv2d(256, 512, kernel_size=3, padding=1, gain=gain),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            # nn.MaxPool2d(kernel_size=3, stride=2),

        ).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d(SMALL_POOL).to(device)

        self.classifier = nn.Sequential(

            nn.Dropout(p=0.3, inplace=False),
            self.Linear(512*PROD_SMALL_POOL, 4096, bias=True, gain=gain),
            nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),

            nn.Dropout(p=0.3, inplace=False),
            self.Linear(4096, 4096, bias=True, gain=gain),
            nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),

            nn.Dropout(p=0.3, inplace=False),
            self.Linear(4096, output_dim, bias=True),
        ).to(device)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    device = "cpu"

    mnist = torch.randn((21, 1, 28, 28)).to(device)
    cifar = torch.randn((21, 3, 32, 32)).to(device)
    imagenet = torch.randn((2, 4, 244, 244)).to(device)

    for test_batch in [mnist, cifar, imagenet]:
        conv = AlexNet(output_dim=10, input_dim=test_batch.shape[1:], device=device)

        print(conv.forward(test_batch).shape)
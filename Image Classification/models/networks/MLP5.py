import torch
import torch.nn as nn

from models.Pruneable import Pruneable

import numpy as np


class MLP5(Pruneable):

    def __init__(self, device="cuda", hidden_dim=(10,), output_dim=2, input_dim=(1,), **kwargs):
        super(MLP5, self).__init__(device=device, output_dim=output_dim, input_dim=input_dim, **kwargs)

        hidden_dim = hidden_dim[0]
        input_dim = int(np.prod(input_dim))

        leak = 0.05
        gain = nn.init.calculate_gain('leaky_relu', leak)

        self.layers = nn.Sequential(
            self.Linear(input_dim=input_dim, output_dim=hidden_dim, bias=True, gain=gain),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            nn.Dropout(p=0.3, inplace=False),
            self.Linear(input_dim=hidden_dim, output_dim=hidden_dim, bias=True, gain=gain),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            nn.Dropout(p=0.3, inplace=False),
            self.Linear(input_dim=hidden_dim, output_dim=hidden_dim, bias=True, gain=gain),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            nn.Dropout(p=0.3, inplace=False),
            self.Linear(input_dim=hidden_dim, output_dim=hidden_dim, bias=True, gain=gain),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            nn.Dropout(p=0.3, inplace=False),
            self.Linear(input_dim=hidden_dim, output_dim=output_dim, bias=True)
        ).to(device)

    def forward(self, x: torch.Tensor, **kwargs):
        x = x.view(x.shape[0], -1)
        return self.layers.forward(x, **kwargs)


if __name__ == '__main__':
    device = "cuda"

    mnist = torch.randn((21, 1, 28, 28)).to(device)
    cifar = torch.randn((21, 3, 32, 32)).to(device)
    imagenet = torch.randn((2, 4, 244, 244)).to(device)

    for test_batch in [mnist, cifar, imagenet]:
        conv = MLP5(output_dim=10, input_dim=test_batch.shape[1:], device=device)

        print(conv.forward(test_batch).shape)
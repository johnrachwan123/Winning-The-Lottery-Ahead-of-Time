import torch
import torch.nn as nn

from models.Pruneable import Pruneable
import numpy as np

class LeNet300(Pruneable):

    def __init__(self, device="cuda", output_dim=2, input_dim=(1,), **kwargs):
        super(LeNet300, self).__init__(device=device, output_dim=output_dim, input_dim=input_dim, **kwargs)

        input_dim = int(np.prod(input_dim))

        leak = 0.05
        gain = nn.init.calculate_gain('leaky_relu', leak)
        # breakpoint()
        self.layers = nn.Sequential(
            self.Linear(input_dim=input_dim, output_dim=300, bias=True, gain=gain),
            nn.BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            self.Linear(input_dim=300, output_dim=100, bias=True, gain=gain),
            nn.BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            self.Linear(input_dim=100, output_dim=output_dim, bias=True)
        ).to(device)

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        return self.layers.forward(x)

if __name__ == '__main__':
    device = "cuda"

    mnist = torch.randn((21, 1, 28, 28)).to(device)
    cifar = torch.randn((21, 3, 32, 32)).to(device)
    imagenet = torch.randn((2, 4, 244, 244)).to(device)

    for test_batch in [mnist, cifar, imagenet]:
        conv = LeNet300(output_dim=10, input_dim=test_batch.shape[1:], device=device)

        print(conv.forward(test_batch).shape)
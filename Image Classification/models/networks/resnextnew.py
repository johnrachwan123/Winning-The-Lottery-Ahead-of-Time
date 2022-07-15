import torch
import torch.nn as nn

from models.Pruneable import Pruneable
# from models.networks.assisting_layers.ResNetLayers import BasicBlock
from utils.constants import SMALL_POOL, PROD_SMALL_POOL
import torchvision.models as models
from utils import resnextto

class resnextnew(Pruneable):

    def __init__(self, device="cuda", output_dim=2, input_dim=(1, 1, 1), **kwargs):
        super(resnextnew, self).__init__(device=device, output_dim=output_dim, input_dim=input_dim, **kwargs)

        self.m = resnextto.resnext101_32x8d(num_classes=10, pretrained=False)

    def forward(self, x):
        x = self.m(x)

        return x



if __name__ == '__main__':
    device = "cuda"

    mnist = torch.randn((21, 1, 28, 28)).to(device)
    cifar = torch.randn((21, 3, 32, 32)).to(device)
    imagenet = torch.randn((2, 4, 244, 244)).to(device)

    for test_batch in [mnist, cifar, imagenet]:
        conv = ResNext(output_dim=10, input_dim=test_batch.shape[1:], device=device)
        print(conv.forward(test_batch).shape)

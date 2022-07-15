import torch
import torch.nn as nn

from models.Pruneable import Pruneable
# from models.networks.assisting_layers.ResNetLayers import BasicBlock
from utils.constants import SMALL_POOL, PROD_SMALL_POOL
import torchvision.models as models

def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)

class ResNext_16(Pruneable):

    def __init__(self, device="cuda", output_dim=2, input_dim=(1, 1, 1), **kwargs):
        super(ResNext_16, self).__init__(device=device, output_dim=output_dim, input_dim=input_dim, **kwargs)

        self.m = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
        reset_all_weights(self.m)

    def forward(self, x):
        x = self.m(x)

        return x



if __name__ == '__main__':
    device = "cuda"

    mnist = torch.randn((21, 1, 28, 28)).to(device)
    cifar = torch.randn((21, 3, 32, 32)).to(device)
    imagenet = torch.randn((2, 4, 244, 244)).to(device)

    for test_batch in [mnist, cifar, imagenet]:
        conv = ResNext_16(output_dim=10, input_dim=test_batch.shape[1:], device=device)
        print(conv.forward(test_batch).shape)

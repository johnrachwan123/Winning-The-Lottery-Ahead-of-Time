import torch

from models.criterions.General import General


class EmptyCrit(General):

    """
    Placeholder class for when you don't want to prune
    """

    def __init__(self, *args, device="cuda", **kwargs):
        super(EmptyCrit, self).__init__(device=device, **kwargs)

    @staticmethod
    def get_prune_indices(*args, **kwargs):
        return torch.empty((0))

    @staticmethod
    def get_grow_indices(*args, **kwargs):
        return torch.empty((0))

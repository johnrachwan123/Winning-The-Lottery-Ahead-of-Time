from random import randint

import torch

from models.criterions.StructuredRandom import StructuredRandom


class EfficientConvNets(StructuredRandom):

    """
    Our interpretation/implementation of the pruning algorithm described in the paper:
    Pruning filters for efficient ConvNets
    https://arxiv.org/abs/1608.08710
    """

    def __init__(self, *args, limit=0.0, **kwargs):
        super(EfficientConvNets, self).__init__(*args, **kwargs)
        self.limit = limit
        self.pruned = False
        self.steps = None

    def prune(self, *args, **kwargs):

        # prune only once
        if self.pruned:
            return
        else:
            self.pruned = True
            super().prune(*args, **kwargs)

    def get_out_vector(self, param, percentage):
        """ returns a vector which determines which nodes from the output dimension to keep """

        prune_dim = [1] + ([] if len(param.shape) <= 2 else [2, 3])

        filter_weights = param.abs().sum(dim=tuple(prune_dim))

        count = len(filter_weights)
        amount = int(count * self.limit)

        limit = torch.topk(filter_weights, amount, largest=False).values[-1]

        return filter_weights > limit

    def get_in_vector(self, param, percentage):
        """ returns a vector which determines which nodes from the input dimension to keep """


        prune_dim = [0] + ([] if len(param.shape) <= 2 else [2, 3])
        filter_weights = param.abs().sum(dim=tuple(prune_dim))

        count = len(filter_weights)
        amount = int(count * self.limit)
        limit = torch.topk(filter_weights, amount, largest=False).values[-1]

        return filter_weights > limit

    def get_inclusion_vectors(self, i, in_indices, last_is_conv, module, modules, out_indices, percentage):
        """ returns a vectors which determine which nodes to keep """

        param = module.weight
        dims = param.shape[:2]  # out, in
        if in_indices is None:
            in_indices = torch.ones(dims[1])
            if self.model._outer_layer_pruning:
                in_indices = self.get_in_vector(param, percentage)
        else:
            in_indices = out_indices
        out_indices = self.get_out_vector(param, percentage)
        is_last = (len(modules) - 1) == i
        if is_last and not self.model._outer_layer_pruning:
            out_indices = torch.ones(dims[0])
        now_is_fc = isinstance(module, torch.nn.Linear)
        now_is_conv = isinstance(module, torch.nn.Conv2d)
        if last_is_conv and now_is_fc:
            ratio = param.shape[1] // in_indices.shape[0]
            in_indices = torch.repeat_interleave(in_indices, ratio)
        last_is_conv = now_is_conv
        if in_indices.sum() == 0:
            in_indices[randint(0, len(in_indices) - 1)] = 1
        if out_indices.sum() == 0:
            out_indices[randint(0, len(out_indices) - 1)] = 1
        return in_indices.bool(), last_is_conv, now_is_conv, out_indices.bool()

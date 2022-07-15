import torch

from models.criterions.General import General


class UnstructuredRandom(General):

    """
    Implements Random (unstructured before training)
    """

    def __init__(self, *args, limit=0.0, **kwargs):
        self.limit = limit
        super(UnstructuredRandom, self).__init__(*args, **kwargs)

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):

        for (name, weights) in self.model.named_parameters():

            if name in self.model.mask:
                mask = torch.bernoulli(weights.clone().detach(), 1.0-percentage)
                self.model.mask[name] = mask

        self.model.apply_weight_mask()
        print("sparsity after pruning", self.model.pruned_percentage)

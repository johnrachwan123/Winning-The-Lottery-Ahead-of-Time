import torch

from models.criterions.General import General


class IMP(General):

    """
    Our interpretation/implementation from the (global) magnitude pruning part of the IMP algorithm from the papers
    https://arxiv.org/abs/1803.03635
    https://arxiv.org/abs/1903.01611
    """

    def __init__(self, *args, limit=0.0, steps=5, **kwargs):
        self.limit = limit
        super(IMP, self).__init__(*args, **kwargs)

        # define the k_i steps
        if limit > 0.5:
            self.steps = [limit - (limit - 0.5) * (0.5 ** i) for i in range(steps + 1)] + [limit]
        else:
            self.steps = [limit - (limit - 0.25) * (0.5 ** i) for i in range(steps + 1)] + [limit]
        self.global_pruning = True

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):
        if len(self.steps) > 0:
            percentage = self.steps.pop(0)
            limit = 0


            if self.global_pruning:

                # get threshold
                all_weights = torch.cat(
                    [torch.flatten(x) for name, x in self.model.named_parameters() if name in self.model.mask])
                count = len(all_weights)
                amount = int(count * percentage)
                limit = torch.topk(all_weights.abs(), amount, largest=False).values[-1]

            for (name, weights) in self.model.named_parameters():

                if name in self.model.mask:
                    if not self.global_pruning:

                        # get threshold
                        flattened = weights.flatten()
                        count = len(flattened)
                        amount = int(count * percentage)
                        limit = torch.topk(flattened.abs(), amount, largest=False).values[-1]

                    # prune on l1
                    mask = weights.abs() > limit

                    self.model.mask[name] = mask

        self.model.apply_weight_mask()
        print("sparsity after pruning", self.model.pruned_percentage)

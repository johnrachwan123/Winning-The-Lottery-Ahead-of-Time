from models.criterions.CroP import CroP


class EarlyCroP(CroP):
    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNIP-it (before training)
    """

    def __init__(self, *args, limit=0.0, steps=100, **kwargs):
        self.limit = limit
        super(EarlyCroP, self).__init__(*args, **kwargs)
        if limit > 0.5:
            self.steps = [limit - (limit - 0.5) * (0.5 ** i) for i in range(steps + 1)] + [limit]
        else:
            self.steps = [limit - (limit - 0.25) * (0.5 ** i) for i in range(steps + 1)] + [limit]

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):
        if len(self.steps) != 0 and self.model.structural_sparsity != 0:
            old = self.model.weight_count
            new = self.model._get_weight_count()
            desired_num_weights = old * (1-self.steps[-1])
            self.limit = 1 - desired_num_weights / new
            self.steps = [self.limit * 0.25, self.limit * 0.5, self.limit * 0.75, self.limit]
            while len(self.steps) > 0:
                percentage = self.steps.pop(0)
                super().prune(percentage=percentage, *args, **kwargs)
        else:
            while len(self.steps) > 0:

                if len(self.steps) != 0:
                    while self.model.pruned_percentage > self.steps[0]:
                        self.steps.pop(0)
                        if len(self.steps) == 0:
                            break
                    # determine k_i
                    percentage = self.steps.pop(0)

                    # prune
                    super().prune(percentage=percentage, *args, **kwargs)

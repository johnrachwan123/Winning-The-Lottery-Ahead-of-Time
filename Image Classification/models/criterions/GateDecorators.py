from collections import OrderedDict

import torch
import torch.nn.functional as F

from models.criterions.SNAP import SNAP
from utils.constants import SNIP_BATCH_ITERATIONS


class GateDecorators(SNAP):

    """
    Our interpretation/implementation of GateDecorators pruning of the paper
    Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks
    https://arxiv.org/abs/1909.08174
    """

    def __init__(self, *args, limit=0.0, **kwargs):
        super(GateDecorators, self).__init__(*args, limit=0.0, **kwargs)

        # define steps of k_i
        self.steps = [x / 100.0 for x in range(1, int(limit * 100), 5)] + [limit]
        self.limit = limit
        self.left = 1.0
        self.pruned = 0.0

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):
        if len(self.steps) == 0:
            print("finished all pruning events already")
            return

        # fetch k_i
        percentage = self.steps.pop(0)
        prune_now = (percentage - self.pruned) / (self.left + 1e-8)

        # prune
        kwargs["percentage"] = prune_now
        SNAP.prune(self, **kwargs)

        # adjust
        self.pruned = self.model.structural_sparsity  # percentage
        self.left = 1.0 - self.pruned

    def get_weight_saliencies(self, train_loader):
        """ collect sensitivity statistics from GatedBatchNorms """

        iterations = SNIP_BATCH_ITERATIONS

        self.model.zero_grad()
        loss_sum = torch.zeros([1]).to(self.device)
        for i, (x, y) in enumerate(train_loader):

            if i == iterations: break

            inputs = x.to(self.model.device)
            targets = y.to(self.model.device)
            outputs = self.model.forward(inputs)
            loss = F.nll_loss(outputs, targets) / iterations
            loss.backward()
            loss_sum += loss.item()

        grad_abs = OrderedDict()
        last_shape = (None,)
        saliencies = None
        for name, module in self.model.named_modules():
            name_ = f"{name}.weight"
            if hasattr(module, "gate"):
                current_number = name_.split(".")[-2]
                final_bit = ".".join(name_.split(".")[-3:])
                try:
                    # breakpoint()
                    name_ = name_.replace(final_bit, final_bit.replace(current_number, str(int(current_number) - 1)))
                except:
                    # breakpoint()
                    # name_ = name_.replace(final_bit, final_bit.replace(current_number, current_number[:-1] + str(int(current_number[-1]))))
                    name_ = '.'.join(name_.split('.')[:-1]) + '.bn.' + name_.split('.')[-1]
                    # name_ = 'conv1.weight'
                out_dim, in_dim = last_shape

                if len(grad_abs) == 0:
                    grad_abs[(0, name_)] = torch.zeros(in_dim).to(self.device)
                else:
                    grad_abs[id(saliencies), name_] = saliencies

                saliencies = torch.abs(module.gate.grad.data * module.gate.data)
                grad_abs[id(saliencies), name_] = saliencies

            try:
                last_shape = module.weight.shape[:2]
            except:
                continue

        grad_abs[id(saliencies), name_] = saliencies

        out_dim, in_dim = last_shape
        grad_abs[2, name_] = torch.zeros(out_dim).to(self.device)

        all_scores = torch.cat([torch.flatten(x) for _, x in grad_abs.items()])
        norm_factor = torch.sum(all_scores.abs())
        all_scores.div_(norm_factor)

        log10 = all_scores.sort().values.log10()

        return all_scores, grad_abs, log10, norm_factor, [x.shape[0] for x in grad_abs.values()]

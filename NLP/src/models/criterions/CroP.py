import copy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from models.criterions.SNIP import SNIP
from constants import SNIP_BATCH_ITERATIONS
from torch.autograd import Variable


class CroP(SNIP):
    """
    Adapted implementation of GraSP from the paper:
    Picking Winning Tickets Before Training by Preserving Gradient Flow
    https://arxiv.org/abs/2002.07376
    from the authors' github:
    https://github.com/alecwangcq/GraSP
    """

    def __init__(self, *args, **kwargs):
        super(CroP, self).__init__(*args, **kwargs)

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, percentage, all_scores, grads_abs):
        # don't prune more or less than possible
        pass

    def get_weight_saliencies(self, train_loader, batch=None):

        device = self.model.device

        iterations = SNIP_BATCH_ITERATIONS

        net = self.model.eval()

        self.their_implementation(device, iterations, net, train_loader)

        # collect gradients
        grads = {}
        for name, layer in net.named_modules():
            if "Norm" in str(layer): continue
            if name + ".weight" in self.model.mask:
                grads[name + ".weight"] = torch.abs(layer.weight.data * layer.weight.grad)
            elif name + ".weight_ih" in self.model.mask:
                grads[name + ".weight_ih"] = torch.abs(layer.weight_ih.data * layer.weight_ih.grad)
                grads[name + ".weight_hh"] = torch.abs(layer.weight_hh.data * layer.weight_hh.grad)

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for _, x in grads.items()])

        so = all_scores.sort().values

        norm_factor = 1
        log10 = so.log10()
        # all_scores.div_(norm_factor)

        self.model = self.model.train()
        self.model.zero_grad()

        return all_scores, grads, log10, norm_factor

    def their_implementation(self, device, iterations, net, train_loader):
        net.zero_grad()
        weights = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight)
        inputs_one = []
        targets_one = []
        grad_w = None
        grad_f = None
        for w in weights:
            w.requires_grad_(True)
        dataloader_iter = iter(train_loader)
        for idx, (data, label) in enumerate(train_loader, 1):

            if idx == iterations: break
            inputs = data
            targets = label
            N = inputs.shape[0]
            din = copy.deepcopy(inputs)
            dtarget = copy.deepcopy(targets)

            start = 0
            intv = N

            while start < N:
                end = min(start + intv, N)
                inputs_one.append(din[start:end])
                targets_one.append(Variable(dtarget[start:end].view(-1)))
                outputs = net.forward(inputs[start:end])  # divide by temperature to make it uniform
                loss = F.nll_loss(outputs, Variable(targets[start:end].view(-1)).to(device))
                grad_w_p = autograd.grad(loss, weights, create_graph=False)
                # grad_w_p = autograd.grad(outputs, weights, grad_outputs=torch.ones_like(outputs), create_graph=False)
                if grad_w is None:
                    grad_w = list(grad_w_p)
                else:
                    for idx in range(len(grad_w)):
                        grad_w[idx] += grad_w_p[idx]
                start = end
        for it in range(len(inputs_one)):
            inputs = inputs_one.pop(0)
            targets = targets_one.pop(0).to(device)
            outputs = net.forward(inputs)  # divide by temperature to make it uniform
            loss = F.nll_loss(outputs, targets)
            grad_f = autograd.grad(loss, weights, create_graph=True)
            # grad_f = autograd.grad(outputs, weights, grad_outputs=torch.ones_like(outputs), create_graph=True)
            z = 0
            count = 0
            for name, layer in net.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    z += (grad_w[count] * grad_f[count] * self.model.mask[name + ".weight"]).sum()
                    count += 1
            z.backward()

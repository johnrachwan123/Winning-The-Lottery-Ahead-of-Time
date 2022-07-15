import copy

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from models.criterions.SNAP_Res import SNAP_Res
from utils.constants import SNIP_BATCH_ITERATIONS
from collections import OrderedDict
from tqdm import tqdm
from models.networks.ResNext import ResNext
class CroPStructuredRes(SNAP_Res):

    def __init__(self, *args, **kwargs):
        super(CroPStructuredRes, self).__init__(*args, **kwargs)


    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_weight_saliencies(self, train_loader):
        # copy network
        self.model = self.model.cpu()
        net = copy.deepcopy(self.model)
        net = net.to(self.device)
        net = net.eval()

        # insert c to gather elasticities
        self.insert_governing_variables(net)

        iterations = SNIP_BATCH_ITERATIONS
        if isinstance(self.model, ResNext):
            iterations = 1

        device = self.device

        self.their_implementation(device, iterations, net, train_loader)

        # gather elasticities
        grads_abs = OrderedDict()
        grads_abs2 = OrderedDict()
        for name, layer in net.named_modules():
            if "Norm" in str(layer): continue
            name_ = f"{name}.weight"
            if hasattr(layer, "gov_in"):
                for (identification, param) in [(id(param), param) for param in [layer.gov_in, layer.gov_out] if
                                                param.requires_grad]:
                    try:
                        grad_ab = torch.abs(param.grad.data)
                    except:
                        grad_ab = torch.zeros_like(param.data)
                    grads_abs2[(identification, name_)] = grad_ab
                    if identification not in grads_abs:
                        grads_abs[identification] = grad_ab

        # reset model
        net = net.cpu()
        del net
        self.model = self.model.to(self.device)
        self.model = self.model.train()

        all_scores = torch.cat([torch.flatten(x) for _, x in grads_abs.items()])
        norm_factor = 1
        all_scores.div_(norm_factor)

        log10 = all_scores.sort().values.log10()
        return all_scores, grads_abs2, log10, norm_factor, [x.shape[0] for x in grads_abs.values()]

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
        for it in tqdm(range(iterations)):
            inputs, targets = next(dataloader_iter)
            N = inputs.shape[0]
            din = copy.deepcopy(inputs)
            dtarget = copy.deepcopy(targets)
            start = 0
            intv = 20
            while start < N:
                end = min(start + intv, N)
                inputs_one.append(din[start:end])
                targets_one.append(dtarget[start:end])
                outputs = net.forward(inputs[start:end].to(device)) # divide by temperature to make it uniform
                loss = F.cross_entropy(outputs, targets[start:end].to(device))
                grad_w_p = autograd.grad(loss, weights, create_graph=False)
                if grad_w is None:
                    grad_w = list(grad_w_p)
                else:
                    for idx in range(len(grad_w)):
                        grad_w[idx] += grad_w_p[idx]
                start = end
        for it in tqdm(range(len(inputs_one))):
            inputs = inputs_one.pop(0).to(device)
            targets = targets_one.pop(0).to(device)
            outputs = net.forward(inputs)  # divide by temperature to make it uniform
            loss = F.cross_entropy(outputs, targets)
            grad_f = autograd.grad(loss, weights, create_graph=True)
            z = 0
            count = 0
            for name, layer in net.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if grad_w[count].is_cuda:
                        z += (grad_w[count] * grad_f[count] * self.model.mask[name + ".weight"]).sum()
                    else:
                        z += (grad_w[count] * grad_f[count] * self.model.mask[name + ".weight"].cpu()).sum()
                    count += 1
            z.backward()

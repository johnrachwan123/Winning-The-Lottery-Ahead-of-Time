import numpy as np
import torch
import torch.nn as nn

from models.GeneralModel import GeneralModel
from models.networks.assisting_layers.ContainerLayers import ContainerLinear, ContainerConv2d
from utils.constants import ZERO_SIGMA


class Pruneable(GeneralModel):
    """

    Defines and manages a pruneable model and gathers statistics

    """

    def __init__(self,
                 criterion=None,
                 is_maskable=True,
                 is_tracking_weights=False,
                 is_rewindable=True,
                 is_growable=False,
                 outer_layer_pruning=True,
                 device="cuda",
                 l0=False,
                 N=0,
                 beta_ema=0,
                 maintain_outer_mask_anyway=False,
                 l0_reg=1.0,
                 l2_reg=0.0,
                 **kwargs):
        self.hooks = {}
        self.l2_reg = l2_reg
        self.l0_reg = l0_reg
        self.maintain_outer_mask_anyway = maintain_outer_mask_anyway
        self.beta_ema = beta_ema
        self.N = N
        self._outer_layer_pruning = outer_layer_pruning
        self.is_growable = is_growable
        self.device = device
        self.is_maskable = is_maskable
        self.is_tracking_weights = is_tracking_weights
        self.is_rewindable = is_rewindable
        self.weight_count = 0
        self.deductable_weightcount = 0
        self.l0 = l0
        self._set_class_references()
        super(Pruneable, self).__init__(device=device, **kwargs)
        self.criterion = criterion

    def get_num_nodes(self, init=False):
        counter = 0
        addition = 0
        for i, (name, module) in enumerate(self.named_modules()):
            if (hasattr(module, "weight") or hasattr(module, "weights")) and not ("Norm" in str(module.__class__)):
                if self.l0:
                    if init:
                        addition = module.sample_z(1).shape[1]
                    else:
                        addition = (module.sample_z(100) != 0).squeeze().float().mean(dim=(0)).sum().item()
                else:
                    addition = module.weight.shape[0]
                counter += addition
        if self.l0:
            return counter
        else:
            return counter - addition

    def _set_class_references(self):

        self.Linear = ContainerLinear
        self.Conv2d = ContainerConv2d

    def add_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                # if self.hooks[name] is not None:
                #     self.hooks[name].append(output.detach())
                # else:
                if len(output.detach().shape) == 4:
                    temp = torch.mean(output.detach(), (2, 3)).T.detach().cpu().numpy()
                else:
                    temp = output.detach().T.detach().cpu().numpy()
                if name in self.hooks.keys():
                    self.hooks[name].append(temp)
                else:
                    self.hooks[name] = [temp]

            return hook

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(get_activation(name))

    def post_init_implementation(self):

        with torch.no_grad():
            self._num_nodes_start = self.get_num_nodes(init=True)
            self.weight_count = self._get_weight_count()

            if self.is_maskable:
                self.mask = {name + ".weight": torch.ones_like(module.weight.data).to(self.device) for name, module in
                             self.named_modules() if isinstance(module, (nn.Linear, nn.Conv2d))
                             }

                if not self._outer_layer_pruning:
                    names = list(self.mask.keys())
                    self.first_layer_name = names[0]
                    self.last_layer_name = names[-1]
                    deductable = self.mask[names[0]].flatten().size()[0] + self.mask[names[-1]].flatten().size()[0]
                    self.percentage_fraction = self.weight_count / (1 + self.weight_count - deductable)

                    self.deductable_weightcount = deductable
                    if not self.maintain_outer_mask_anyway:
                        del self.mask[names[0]]
                        del self.mask[names[-1]]

            if self.is_rewindable:
                self.save_rewind_weights()

            if self.is_tracking_weights:
                self.prev_weights = self._clone_weights(self.named_parameters())
                self.sign_flips_counts = {name: torch.zeros_like(tens) for name, tens in self.prev_weights.items()}
                self.moving_average = self._clone_weights(self.sign_flips_counts.items())
                self.moving_variance = self._clone_weights(self.sign_flips_counts.items())
                self.batch_num = 0

            if self.l0:
                if self.beta_ema > 0.:
                    self.steps_ema = 0
                    self.avg_param = self._clone_weights(self.named_parameters(), bias=True).values()

                for module in self.modules():
                    if hasattr(module, "lamba"):
                        module.lamba = self.l0_reg
                    if hasattr(module, "prior_prec"):
                        module.prior_prec = self.l2_reg * self.N

    def _get_weight_count(self):
        return sum([tens.flatten().size()[0] for name, tens in self.named_parameters() if
                    'weight' in name])

    def _clone_weights(self, weight_list_reference, bias=False):

        return {name: tens.data.detach().clone().to(self.device) for name, tens in
                weight_list_reference if
                ('weight' in name) or bias}

    def save_rewind_weights(self):
        """ Saves the weights used to rewind to"""

        if not self.is_rewindable:
            raise Exception("rewind weights is off")

        self.rewind_weights = self._clone_weights(self.named_parameters())

    def save_prev_weights(self):
        """ Saves weights of the last update used for monitoring sign flips """

        if not self.is_tracking_weights:
            raise Exception("track_weights is off")

        self.prev_weights = self._clone_weights(self.named_parameters())

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("please inherit child-class")

    def update_ema(self):

        if not self.l0:
            raise Exception("l_0 is off")

        with torch.no_grad():

            self.steps_ema += 1
            for p, avg_p in zip(self.parameters(), self.avg_param):
                avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):

        if not self.l0:
            raise Exception("l_0 is off")

        with torch.no_grad():
            for p, avg_p in zip(self.parameters(), self.avg_param):
                p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):

        if not self.l0:
            raise Exception("l_0 is off")

        with torch.no_grad():
            for p, avg_p in zip(self.parameters(), params):
                p.data.copy_(avg_p)

    def norm(self, p=2):

        regularisation = torch.zeros([1])
        for name, param in self.named_parameters():

            if "rho" in name:
                regularisation -= torch.norm((param == ZERO_SIGMA) * ZERO_SIGMA, p=p)
            regularisation += torch.norm(param, p=p)

        return regularisation.log()

    def apply_mask(self):
        self.apply_grad_mask()
        self.apply_weight_mask()

    def apply_weight_mask(self):

        if not self.is_maskable:
            raise Exception("mask is off")

        with torch.no_grad():
            for name, tensor in self.named_parameters():
                if name in self.mask:
                    if tensor.is_cuda:
                        tensor.data *= self.mask[name]
                    else:
                        tensor.data *= self.mask[name].cpu()
                    if "rho" in name:
                        tensor.data += (self.mask[name] == 0) * ZERO_SIGMA

    def apply_grad_mask(self):

        if not self.is_maskable:
            raise Exception("mask is off")

        for name, tensor in self.named_parameters():
            if name in self.mask:
                tensor.grad.data *= self.mask[name]

    def do_rewind(self):

        if not self.is_rewindable:
            raise Exception("rewind_weights is off")

        if not self.is_maskable:
            raise Exception("mask is off")

        with torch.no_grad():
            for name, tensor in self.named_parameters():
                if name in self.mask:
                    tensor.data = self.rewind_weights[name].detach().clone() * self.mask[name]
                    tensor.requires_grad = True

    def insert_noise_for_gradient(self, noise):
        if noise == 0:  return

        with torch.no_grad():
            for name, tensor in self.named_parameters():
                tensor.grad.data += noise * torch.randn_like(tensor.grad.data)

    def update_tracked_weights(self, batch_number: int):

        if not self.is_tracking_weights:
            raise Exception("track_weights is off")

        with torch.no_grad():
            for name, new_weight in self.named_parameters():

                if 'weight' in name:
                    # unpack
                    old_weights = self.prev_weights[name]
                    flip_counts = self.sign_flips_counts[name]
                    avg_old = self.moving_average[name]
                    var_old = self.moving_variance[name]

                    # flips increment
                    flip_counts += ((old_weights * new_weight) < 0).bool()

                    # moving statistics
                    self.batch_num = batch_number
                    count = batch_number
                    delta = new_weight - avg_old
                    avg_old += delta / count
                    delta2 = new_weight - avg_old
                    var_old += delta * delta2

                    # clean
                    del self.prev_weights[name]

    @property
    def l2_norm(self):
        norm = self.norm(p=2).item()
        return norm

    @property
    def l1_norm(self):
        norm = self.norm(p=1).item()
        return norm

    @property
    def number_of_pruned_weights(self):

        if (not self.is_maskable) and (not self.l0):
            return 0

        total = 0

        if self.l0:
            for name, module in self.named_modules():
                if ("L0" in str(type(module)) and self.l0):
                    tensor = module.sample_weights()
                    total += torch.sum(tensor == 0).item()
            return int(total)
        else:

            for name, tensor in self.named_parameters():
                if 'weight' in name:

                    if 'rho' in name:
                        total += torch.sum(tensor == ZERO_SIGMA).item()
                    else:
                        total += torch.sum(tensor == 0).item()
            return int(total)

    @property
    def pruned_percentage(self):
        return (self.number_of_pruned_weights + (self.weight_count - self._get_weight_count())) / (
                self.weight_count + 1e-6)

    def get_weight_counts(self):
        return self.weight_count, self.get_weight_counts()

    def reset_weight_counts(self):
        self.weight_count = self._get_weight_count()

    @property
    def structural_sparsity(self):
        return 1.0 - ((self.get_num_nodes(init=False) + 1e-6) / (self._num_nodes_start + 1e-6))

    @property
    def get_params(self):
        params = self._clone_weights(self.named_parameters(), bias=True)
        return params

    @property
    def l0_regularisation(self):

        if not self.l0:
            raise Exception("l_0 is off")

        total = 0

        for name, module in self.named_modules():
            if "L0" in str(type(module)):
                total += - (1 / self.N) * module.l0_regularisation

        return total

    @property
    def expected_l0(self):

        if not self.l0:
            raise Exception("l_0 is off")

        total = 0

        for name, module in self.named_modules():
            if "L0" in str(type(module)):
                module: L0Linear
                total += module.count_expected_flops_and_l0()[1]

        return total

    @property
    def pruned_percentage_of_prunable(self):
        return self.number_of_pruned_weights / (self.weight_count - self.deductable_weightcount + 1e-6)

    @property
    def compressed_size(self):

        if not self.is_maskable and not self.l0:
            return np.nan
            # raise Exception("mask is off")

        size = 0
        with torch.no_grad():
            if self.l0:
                for module in self.modules():
                    module: L0Linear
                    weight = 0
                    try:
                        weight = module.sample_weights()
                    except:
                        continue
                    nonzero = torch.sum(weight != 0).item()
                    size += nonzero * 34

            else:
                for name, tensor in self.named_parameters():
                    if 'weight' in name:
                        nonzero = 0
                        if 'rho' in name:
                            nonzero = torch.sum(tensor != ZERO_SIGMA).item()
                        else:
                            nonzero = torch.sum(tensor != 0).item()
                        temp = tensor.view(tensor.shape[0], -1).detach()
                        m, n = temp.shape[0], temp.shape[1]
                        smallest = min(m, n)
                        size += nonzero * 34 + 2 * (smallest + 1)
        return size

    @property
    def variance(self):
        if not self.is_tracking_weights:
            raise Exception("track_weights is off")
        return {name: tens / self.batch_num for name, tens in self.moving_variance.items()}

    @property
    def mean(self):
        if not self.is_tracking_weights:
            raise Exception("track_weights is off")
        return self.moving_average

    @property
    def flips(self):
        if not self.is_tracking_weights:
            raise Exception("track_weights is off")
        return self.sign_flips_counts

    @property
    def flips_numbers_log(self):
        if not self.is_tracking_weights:
            raise Exception("track_weights is off")

        with torch.no_grad():
            total = 0
            for x in self.sign_flips_counts.values():
                total += x.sum()
            return total.log().item()

    @property
    def flips_unique_log(self):
        if not self.is_tracking_weights:
            raise Exception("track_weights is off")

        with torch.no_grad():
            total = 0
            for x in self.sign_flips_counts.values():
                total += (x > 0).sum()

            return total.float().log().item()

    @property
    def variance_log(self):
        if not self.is_tracking_weights:
            raise Exception("track_weights is off")

        with torch.no_grad():
            total = 0
            for x in self.variance.values():
                total += x.sum()
            return total.log().item()

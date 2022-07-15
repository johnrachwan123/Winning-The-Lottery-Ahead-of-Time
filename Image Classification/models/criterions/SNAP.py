import copy
import os
import types
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.criterions.General import General
from models.networks.assisting_layers.GateDecoratorLayers import GatedBatchNorm
from utils.constants import SNIP_BATCH_ITERATIONS, RESULTS_DIR, OUTPUT_DIR
from utils.data_utils import lookahead_type, lookahead_finished
from utils.snip_utils import group_snip_forward_linear, group_snip_conv2d_forward


class SNAP(General):

    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNAP (structured), which is one of the steps from the algorithm SNAP-it
    Additionally, this class contains most of the code the actually reduce pytorch tensors, in order to obtain speedup
    """

    def __init__(self, *args, **kwargs):
        super(SNAP, self).__init__(*args, **kwargs)

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage, train_loader=None, manager=None, **kwargs):

        all_scores, grads_abs, log10, norm_factor, vec_shapes = self.get_weight_saliencies(train_loader)

        # manager.save_python_obj(all_scores.cpu().numpy(),
        #                         os.path.join(RESULTS_DIR, manager.stamp, OUTPUT_DIR, f"scores"))

        self.handle_pruning(all_scores, grads_abs, norm_factor, percentage)

    def handle_pruning(self, all_scores, grads_abs, norm_factor, percentage):
        summed_weights = sum([np.prod(x.shape) for name, x in self.model.named_parameters() if "weight" in name])
        num_nodes_to_keep = int(len(all_scores) * (1 - percentage))

        # handle outer layers
        if not self.model._outer_layer_pruning:
            offsets = [len(x[0][1]) for x in lookahead_finished(grads_abs.items()) if x[1][0] or x[1][1]]
            all_scores = all_scores[offsets[0]:-offsets[1]]
            num_nodes_to_keep = int(len(all_scores) * (1 - percentage))

        # dont prune more or less than is available
        if num_nodes_to_keep > len(all_scores):
            num_nodes_to_keep = len(all_scores)
        elif num_nodes_to_keep == 0:
            num_nodes_to_keep = 1

        # threshold
        threshold, _ = torch.topk(all_scores, num_nodes_to_keep, sorted=True)
        del _
        acceptable_score = threshold[-1]
        # prune
        summed_pruned = 0
        toggle_row_column = True
        cutoff = 0
        length_nonzero = 0
        for ((identification, name), grad), (first, last) in lookahead_finished(grads_abs.items()):
            binary_keep_neuron_vector = ((grad / norm_factor) >= acceptable_score).float().to(self.device)
            corresponding_weight_parameter = [val for key, val in self.model.named_parameters() if key == name][0]
            is_conv = len(corresponding_weight_parameter.shape) > 2
            corresponding_module: nn.Module = \
                [val for key, val in self.model.named_modules() if key == name.split(".weight")[0]][0]

            # ensure not disconnecting
            if binary_keep_neuron_vector.sum() == 0:
                best_index = torch.argmax(grad)
                binary_keep_neuron_vector[best_index] = 1

            if first or last:
                # noinspection PyTypeChecker
                length_nonzero = self.handle_outer_layers(binary_keep_neuron_vector,
                                                          first,
                                                          is_conv,
                                                          last,
                                                          length_nonzero,
                                                          corresponding_module,
                                                          name,
                                                          corresponding_weight_parameter)
            else:

                cutoff, length_nonzero = self.handle_middle_layers(binary_keep_neuron_vector,
                                                                   cutoff,
                                                                   is_conv,
                                                                   length_nonzero,
                                                                   corresponding_module,
                                                                   name,
                                                                   toggle_row_column,
                                                                   corresponding_weight_parameter)

            cutoff, summed_pruned = self.print_layer_progress(cutoff,
                                                              grads_abs,
                                                              length_nonzero,
                                                              name,
                                                              summed_pruned,
                                                              toggle_row_column,
                                                              corresponding_weight_parameter)
            toggle_row_column = not toggle_row_column
        for line in str(self.model).split("\n"):
            if "BatchNorm" in line or "Conv" in line or "Linear" in line or "AdaptiveAvg" in line or "Sequential" in line:
                print(line)
        print("final percentage after snap:", summed_pruned / summed_weights)

        self.model.apply_weight_mask()
        self.cut_lonely_connections()

    def handle_middle_layers(self,
                             binary_vector,
                             cutoff,
                             is_conv,
                             length_nonzero,
                             module,
                             name,
                             toggle_row_column,
                             weight):



        indices = binary_vector.bool()
        length_nonzero_before = int(np.prod(weight.shape))
        n_remaining = binary_vector.sum().item()
        if not toggle_row_column:
            self.handle_output(indices,
                               is_conv,
                               module,
                               n_remaining,
                               name,
                               weight)

        else:
            cutoff, length_nonzero = self.handle_input(cutoff,
                                                       indices,
                                                       is_conv,
                                                       length_nonzero,
                                                       module,
                                                       n_remaining,
                                                       name,
                                                       weight)

        cutoff += (length_nonzero_before - int(np.prod(weight.shape)))
        return cutoff, length_nonzero

    def handle_input(self, cutoff, indices, is_conv, length_nonzero, module, n_remaining, name, weight):
        """ shrinks a input dimension """
        module.update_input_dim(n_remaining)
        length_nonzero = int(np.prod(weight.shape))
        cutoff = 0
        if is_conv:
            weight.data = weight[:, indices, :, :]
            try:
                weight.grad.data = weight.grad.data[:, indices, :, :]
            except AttributeError:
                pass
            if name in self.model.mask:
                self.model.mask[name] = self.model.mask[name][:, indices, :, :]
        else:
            if ((weight.shape[1] % indices.shape[0]) == 0) and not (weight.shape[1] == indices.shape[0]):
                ratio = weight.shape[1] // indices.shape[0]
                module.update_input_dim(n_remaining * ratio)
                new_indices = torch.repeat_interleave(indices, ratio)
                weight.data = weight[:, new_indices]
                if name in self.model.mask:
                    self.model.mask[name] = self.model.mask[name][:, new_indices]
                try:
                    weight.grad.data = weight.grad.data[:, new_indices]
                except AttributeError:
                    pass
            else:
                weight.data = weight[:, indices]
                try:
                    weight.grad.data = weight.grad.data[:, indices]
                except AttributeError:
                    pass
                if name in self.model.mask:
                    self.model.mask[name] = self.model.mask[name][:, indices]
        if self.model.is_tracking_weights:
            raise NotImplementedError
        return cutoff, length_nonzero

    def handle_output(self, indices, is_conv, module, n_remaining, name, weight):
        """ shrinks a output dimension """
        module.update_output_dim(n_remaining)
        self.handle_batch_norm(indices, n_remaining, name)
        if is_conv:
            weight.data = weight[indices, :, :, :]
            try:
                weight.grad.data = weight.grad.data[indices, :, :, :]
            except AttributeError:
                pass
            if name in self.model.mask:
                self.model.mask[name] = self.model.mask[name][indices, :, :, :]
        else:
            weight.data = weight[indices, :]
            try:
                weight.grad.data = weight.grad.data[indices, :]
            except AttributeError:
                pass
            if name in self.model.mask:
                self.model.mask[name] = self.model.mask[name][indices, :]
        self.handle_bias(indices, name)
        if self.model.is_tracking_weights:
            raise NotImplementedError

    def handle_bias(self, indices, name):
        """ shrinks a bias """
        bias = [val for key, val in self.model.named_parameters() if key == name.split("weight")[0] + "bias"][0]
        bias.data = bias[indices]
        try:
            bias.grad.data = bias.grad.data[indices]
        except AttributeError:
            pass

    def handle_batch_norm(self, indices, n_remaining, name):
        """ shrinks a batchnorm layer """

        batchnorm = [val for key, val in self.model.named_modules() if
                     key == name.split(".weight")[0][:-1] + str(int(name.split(".weight")[0][-1]) + 1)][0]
        if isinstance(batchnorm, (nn.BatchNorm2d, nn.BatchNorm1d, GatedBatchNorm)):
            batchnorm.num_features = n_remaining
            from_size = len(batchnorm.bias.data)
            batchnorm.bias.data = batchnorm.bias[indices]
            batchnorm.weight.data = batchnorm.weight[indices]
            try:
                batchnorm.bias.grad.data = batchnorm.bias.grad[indices]
                batchnorm.weight.grad.data = batchnorm.weight.grad[indices]
            except TypeError:
                pass
            if hasattr(batchnorm, "gate"):
                batchnorm.gate.data = batchnorm.gate.data[indices]
                batchnorm.gate.grad.data = batchnorm.gate.grad.data[indices]
                batchnorm.bn.num_features = n_remaining
            for buffer in batchnorm.buffers():
                if buffer.data.shape == indices.shape:
                    buffer.data = buffer.data[indices]
            print(f"trimming nodes in layer {name} from {from_size} to {len(batchnorm.bias.data)}")

    def handle_outer_layers(self,
                            binary_vector,
                            first,
                            is_conv,
                            last,
                            length_nonzero,
                            module,
                            name,
                            param):

        n_remaining = binary_vector.sum().item()
        if first:
            length_nonzero = int(np.prod(param.shape))
            if self.model._outer_layer_pruning:
                module.update_input_dim(n_remaining)
                if is_conv:
                    permutation = (0, 3, 2, 1)
                    self.model.mask[name] = (self.model.mask[name].permute(permutation) * binary_vector).permute(
                        permutation)
                else:
                    self.model.mask[name] *= binary_vector
        elif last and self.model._outer_layer_pruning:
            module.update_output_dim(n_remaining)
            if is_conv:
                permutation = (3, 1, 2, 0)
                self.model.mask[name] = (self.model.mask[name].permute(permutation) * binary_vector).permute(
                    permutation)
            else:
                self.model.mask[name] = (binary_vector * self.model.mask[name].t()).t()
        if self.model._outer_layer_pruning:
            number_removed = (self.model.mask[name] == 0).sum().item()
            print("set to zero but not removed because of input-output compatibility:", number_removed,
                  f"({len(binary_vector) - n_remaining} features)")
        return length_nonzero

    def print_layer_progress(self, cutoff, grads_abs, length_nonzero, name, summed_pruned, toggle, weight):
        if not toggle:
            if len(grads_abs) == 2:
                cutoff /= 2
            summed_pruned += cutoff
            print("pruning", cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)
        return cutoff, summed_pruned

    def get_weight_saliencies(self, train_loader):

        # copy network
        self.model = self.model.cpu()
        net = copy.deepcopy(self.model)
        net = net.to(self.device)
        net = net.eval()

        # insert c to gather elasticities
        self.insert_governing_variables(net)

        iterations = SNIP_BATCH_ITERATIONS

        # accumalate gradients with multiple batches
        net.zero_grad()
        loss_sum = torch.zeros([1]).to(self.device)
        for i, (x, y) in enumerate(train_loader):

            if i == iterations: break

            inputs = x.to(self.model.device)
            targets = y.to(self.model.device)
            outputs = net.forward(inputs)
            loss = F.nll_loss(outputs, targets) / iterations
            loss.backward()
            loss_sum += loss.item()

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
        norm_factor = torch.abs(loss_sum)
        all_scores.div_(norm_factor)

        log10 = all_scores.sort().values.log10()
        return all_scores, grads_abs2, log10, norm_factor, [x.shape[0] for x in grads_abs.values()]

    def insert_governing_variables(self, net):
        """ inserts c vectors in all parameters """

        govs = []
        gov_in = None
        gov_out = None
        do_avg_pool = 0
        for layer, (is_conv, next_is_conv) in lookahead_type(net.modules()):

            is_conv = isinstance(layer, nn.Conv2d)
            is_fc = isinstance(layer, nn.Linear)
            is_avgpool = isinstance(layer, nn.AdaptiveAvgPool2d)

            if is_avgpool:
                do_avg_pool = int(np.prod(layer.output_size))


            elif is_conv or is_fc:

                out_dim, in_dim = layer.weight.shape[:2]

                if gov_in is None:

                    gov_in = nn.Parameter(torch.ones(in_dim).to(self.device), requires_grad=True)
                    govs.append(gov_in)

                else:
                    gov_in = gov_out

                gov_out = nn.Parameter(torch.ones(out_dim).to(self.device), requires_grad=True)
                govs.append(gov_out)

                # insert variables
                layer.gov_out = gov_out
                layer.gov_in = gov_in

                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

            # substitute activation function
            if is_fc:
                if do_avg_pool > 0:
                    layer.do_avg_pool = do_avg_pool
                    do_avg_pool = 0
                layer.forward = types.MethodType(group_snip_forward_linear, layer)
            if is_conv:
                layer.forward = types.MethodType(group_snip_conv2d_forward, layer)

        return govs
import copy
import types
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from constants import SNIP_BATCH_ITERATIONS
from collections import OrderedDict
from tqdm import tqdm
from torch.autograd import Variable
from model import PSMM

from General import General
from GateDecoratorLayers import GatedBatchNorm
from constants import SNIP_BATCH_ITERATIONS, RESULTS_DIR, OUTPUT_DIR
from data_utils import lookahead_type, lookahead_finished
from snip_utils import group_snip_forward_linear, group_snip_conv2d_forward, group_snip_forward_embedded, group_snip_forward_lstm
from torch.autograd import *

class CroPStructured(General):

    def __init__(self, *args, **kwargs):
        super(CroPStructured, self).__init__(*args, **kwargs)

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage, train_loader=None, manager=None, **kwargs):
        all_scores, grads_abs, log10, norm_factor, vec_shapes = self.get_weight_saliencies(train_loader)
        self.handle_pruning(all_scores, grads_abs, norm_factor, percentage)

    def get_weight_saliencies(self, train_loader):
        self.model = self.model.cpu()
        net = PSMM(128, 10000, 300, True)
        net = net.to('cuda')
        net = net.eval()

        # insert c to gather elasticities
        self.insert_governing_variables(net)
        iterations = SNIP_BATCH_ITERATIONS
        device = self.device

        self.get_scores(device, iterations, net, train_loader)

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

    def get_scores(self, device, iterations, net, train_loader):
        net.zero_grad()
        weights = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.Embedding) or isinstance(layer, nn.LSTMCell):
                try:
                    weights.append(layer.weight)
                except:
                    weights.append(layer.weight_hh)
                    weights.append(layer.weight_ih)
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
                if grad_w is None:
                    grad_w = list(grad_w_p)
                else:
                    for idx in range(len(grad_w)):
                        grad_w[idx] += grad_w_p[idx]
                start = end
        for it in tqdm(range(len(inputs_one))):
            inputs = inputs_one.pop(0)
            targets = targets_one.pop(0).to(device)
            outputs = net.forward(inputs)  # divide by temperature to make it uniform
            loss = F.nll_loss(outputs, targets)
            grad_f = autograd.grad(loss, weights, create_graph=True)
            z = 0
            count = 0
            for name, layer in net.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.Embedding):
                    if grad_w[count].is_cuda:
                        z += (grad_w[count] * grad_f[count]).sum()
                    else:
                        z += (grad_w[count] * grad_f[count]).sum()
                    count += 1
                elif isinstance(layer, nn.LSTMCell):
                    z += (grad_w[count] * grad_f[count]).sum()
                    count+=1
            z.backward()

    def semi_prune(self, tensor, num):
        try:
            idx = (tensor >= torch.topk(tensor, num).values[-1]).int()
            if idx.sum() != num:
                for i in range(0, len(tensor) - num, 1):
                    index = torch.kthvalue(tensor, len(tensor) - num - i).indices
                    tensor[index] = torch.max(tensor)
                    idx[index] = 0
            return idx
        except:
            breakpoint()

    def prep_indices(self, idx, indices, scores):
        indices[idx[0]] = torch.ones_like(indices[idx[0]])
        indices[idx[len(idx) - 1]] = torch.ones_like(indices[idx[len(idx) - 1]])
        num = max(indices[idx[3]].sum(), indices[idx[5]].sum(), indices[idx[1]].sum())
        indices[idx[3]] = self.semi_prune(scores[idx[3]], num)
        indices[idx[5]] = self.semi_prune(scores[idx[5]], num)
        indices[idx[1]] = self.semi_prune(scores[idx[1]], num)
        indices[idx[6]] = indices[idx[3]]
        indices[idx[4]] = indices[idx[3]]
        indices[idx[2]] = indices[idx[1]]
        self.model.hidden_size = indices[idx[1]].sum()
        self.model.rnn.input_size = indices[idx[1]].sum()
        self.model.rnn.hidden_size = indices[idx[1]].sum()
        self.model.sentinel_vector = Variable(torch.rand(self.model.rnn.hidden_size, 1), requires_grad=True)
        if self.device == 'cuda':
            self.model.sentinel_vector = self.model.sentinel_vector.cuda()
        return indices

    def handle_pruning(self, all_scores, grads_abs, norm_factor, percentage):
        summed_weights = sum([np.prod(x.shape) for name, x in self.model.named_parameters() if "weight" in name])
        num_nodes_to_keep = int(len(all_scores) * (1 - percentage))
        # handle outer layers
        self.model._outer_layer_pruning = False
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
        indices = {}
        idx = {}
        scores = {}
        i = 0
        for ((identification, name), grad), (first, last) in lookahead_finished(grads_abs.items()):
            indices[(i, name)] = ((grad / norm_factor) >= acceptable_score).int()
            idx[i] = (i, name)
            scores[(i, name)] = grad / norm_factor
            i += 1
        indices = self.prep_indices(idx, indices, scores)
        for ((identification, name), grad), (first, last) in lookahead_finished(indices.items()):
            binary_keep_neuron_vector = grad.float().to(self.device)
            corresponding_weight_parameter_list = [val for key, val in self.model.named_parameters() if key == name]
            if len(corresponding_weight_parameter_list) == 0:
                corresponding_weight_parameter_list = [val for key, val in self.model.named_parameters() if
                                                       name in key]
                # breakpoint()
            for corresponding_weight_parameter in corresponding_weight_parameter_list:
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
            if "BatchNorm" in line or "Conv" in line or "Linear" in line or "AdaptiveAvg" in line or "Sequential" in line or 'Emb' in line or 'LSTM' in line:
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
        module.in_channels = n_remaining
        length_nonzero = int(np.prod(weight.shape))
        cutoff = 0
        if is_conv:
            try:
                weight.data = weight[:, indices, :, :]
            except:
                return cutoff, length_nonzero
            try:
                weight.grad.data = weight.grad.data[:, indices, :, :]
            except AttributeError:
                pass
            if name in self.model.mask:
                self.model.mask[name] = self.model.mask[name][:, indices, :, :]
        else:
            if ((indices.shape[0] % weight.shape[0]) == 0) and not (weight.shape[1] == indices.shape[0]):
                ratio = weight.shape[1] // indices.shape[0]
                module.in_channels = n_remaining * ratio
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
                elif 'rnn.weight_ih' in self.model.mask:
                    name = 'rnn.weight_ih'
                    try:
                        self.model.mask[name] = self.model.mask[name][:, indices]
                    except:
                        name = 'rnn.weight_hh'
                        self.model.mask[name] = self.model.mask[name][:, indices]

        if self.model.is_tracking_weights:
            raise NotImplementedError
        return cutoff, length_nonzero

    def handle_output(self, indices, is_conv, module, n_remaining, name, weight):
        """ shrinks a output dimension """
        # if 'conv2' in name:
        #     return
        module.out_channels = n_remaining

        # self.handle_batch_norm(indices, n_remaining, name)
        if is_conv:
            weight.data = weight[indices, :, :, :]
            try:
                weight.grad.data = weight.grad.data[indices, :, :, :]
            except AttributeError:
                pass
            if name in self.model.mask:
                self.model.mask[name] = self.model.mask[name][indices, :, :, :]
        else:
            new_indices = None
            try:
                weight.data = weight[indices, :]
            except IndexError:
                try:
                    weight.data = weight[:, indices]
                except IndexError:
                    new_indices = torch.repeat_interleave(indices, 4)
                    weight.data = weight[new_indices, :]
            try:
                weight.grad.data = weight.grad.data[indices, :]
            except AttributeError:
                pass
            except IndexError:
                try:
                    weight.grad.data = weight.grad.data[:, indices]
                except IndexError:
                    weight.grad.data = weight.grad.data[new_indices, :]
            if name in self.model.mask:
                try:
                    self.model.mask[name] = self.model.mask[name][indices, :]
                except IndexError:
                    self.model.mask[name] = self.model.mask[name][:, indices]
            elif 'rnn.weight_ih' in self.model.mask:
                name = 'rnn.weight_ih'
                try:
                    self.model.mask[name] = self.model.mask[name][new_indices, :]
                except:
                    name = 'rnn.weight_hh'
                    self.model.mask[name] = self.model.mask[name][new_indices, :]
        if new_indices is None:
            self.handle_bias(indices, name)
        else:
            self.handle_bias(new_indices, name)
        if self.model.is_tracking_weights:
            raise NotImplementedError

    def handle_bias(self, indices, name):
        """ shrinks a bias """
        bias = [val for key, val in self.model.named_parameters() if
                key == name.split("weight")[0] + "bias" or key == 'bn' + name.split("weight")[0][-2] + ".bias"]

        if len(bias) == 1:
            bias = bias[0]
        elif len(bias) > 1:
            bias = bias[-1]
        elif 'embed' in name:
            return
        elif 'rnn' in name:
            if 'ih' in name:
                bias = [val for key, val in self.model.named_parameters() if key == 'rnn.bias_ih'][0]
            elif 'hh' in name:
                bias = [val for key, val in self.model.named_parameters() if key == 'rnn.bias_hh'][0]
        # bias = bias[0] if len(bias)==0 else bias[1]
        try:
            bias.data = bias[indices]
        except:
            breakpoint()
        try:
            bias.grad.data = bias.grad.data[indices]
        except:
            pass

    def handle_batch_norm(self, indices, n_remaining, name):
        """ shrinks a batchnorm layer """
        if name == 'm.conv1.weight':
            batchnorm = [val for key, val in self.model.named_modules() if
                         key == name.split(".")[0] + ".bn" + str(int(name.split(".weight")[0][-1]))][0]
        elif 'layer' in name:

            if 'downsample' in name:
                batchnorm = [val for key, val in self.model.named_modules() if
                             key == name.split(".weight")[0][:-1] + str(int(name.split(".weight")[0][-1]) + 1)][0]
            else:
                try:
                    if name == 'm.conv1.weight':
                        batchnorm = [val for key, val in self.model.named_modules() if
                                     key == name.split(".")[0] + ".bn" + str(int(name.split(".weight")[0][-1]))][0]
                    else:
                        batchnorm = [val for key, val in self.model.named_modules() if
                                     key == name.split(".")[0] + "." + name.split(".")[1] + ".bn" + str(
                                         int(name.split(".weight")[0][-1]))][0]
                except:
                    batchnorm = [val for key, val in self.model.named_modules() if
                                 key == name.split(".")[0] + "." + name.split(".")[1] + '.' + name.split(".")[
                                     2] + ".bn" + str(int(name.split(".weight")[0][-1]))][0]

        else:

            batchnorm = [val for key, val in self.model.named_modules() if
                         key == name.split(".weight")[0][:-1] + str(
                             int(name.split(".weight")[0][-1]) + 1) or key == 'bn' + str(
                             int(name.split(".weight")[0][-1]))]
            if len(batchnorm) == 1:
                batchnorm = batchnorm[0]
            elif len(batchnorm) > 1:
                batchnorm = batchnorm[-1]
            else:
                return
        if isinstance(batchnorm, (nn.BatchNorm2d, nn.BatchNorm1d, GatedBatchNorm)):
            batchnorm.num_features = n_remaining
            from_size = len(batchnorm.bias.data)
            try:
                batchnorm.bias.data = batchnorm.bias[indices]
            except:
                breakpoint()
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
                module.in_channels = n_remaining
                if is_conv:
                    permutation = (0, 3, 2, 1)
                    self.model.mask[name] = (self.model.mask[name].permute(permutation) * binary_vector).permute(
                        permutation)
                else:
                    self.model.mask[name] *= binary_vector
        elif last and self.model._outer_layer_pruning:
            module.out_channels = n_remaining
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
            is_lstm = isinstance(layer, nn.LSTMCell)
            is_embedding = isinstance(layer, nn.Embedding)
            if is_avgpool:
                do_avg_pool = int(np.prod(layer.output_size))


            elif is_conv or is_fc or is_embedding:
                out_dim, in_dim = layer.weight.shape[:2]

                if gov_in is None:
                    gov_in = nn.Parameter(torch.ones(in_dim).to(self.device), requires_grad=True)
                    govs.append(gov_in)
                else:
                    gov_in = gov_out
                if is_embedding:
                    gov_out = nn.Parameter(torch.ones(in_dim).to(self.device), requires_grad=True)
                else:
                    gov_out = nn.Parameter(torch.ones(out_dim).to(self.device), requires_grad=True)
                govs.append(gov_out)

                # insert variables
                layer.gov_out = gov_out
                layer.gov_in = gov_in

                layer.weight.requires_grad = False
                try:
                    layer.bias.requires_grad = False
                except:
                    pass

            elif is_lstm:
                out_dim_ih, in_dim_ih = layer.weight_ih.shape[:2]
                out_dim_hh, in_dim_hh = layer.weight_hh.shape[:2]
                if gov_in is None:
                    gov_in = nn.Parameter(torch.ones(in_dim_hh).to(self.device), requires_grad=True)
                    govs.append(gov_in)
                else:
                    gov_in = gov_out

                # gov_out_ih = nn.Parameter(torch.ones(out_dim_ih).to(self.device), requires_grad=True)
                # govs.append(gov_out_ih)
                # gov_out_hh = nn.Parameter(torch.ones(out_dim_hh).to(self.device), requires_grad=True)
                # govs.append(gov_out_hh)
                # # insert variables
                # layer.gov_out_ih = gov_out_ih
                # layer.gov_out_hh = gov_out_hh
                gov_out = nn.Parameter(torch.ones(in_dim_hh).to(self.device), requires_grad=True)
                govs.append(gov_out)
                layer.gov_in = gov_in
                layer.gov_out = gov_out

                layer.weight_hh.requires_grad = False
                layer.weight_ih.requires_grad = False
                try:
                    layer.bias_hh.requires_grad = False
                    layer.bias_ih.requires_grad = False
                except:
                    pass
            # substitute activation function
            if is_fc:
                if do_avg_pool > 0:
                    layer.do_avg_pool = do_avg_pool
                    do_avg_pool = 0
                layer.forward = types.MethodType(group_snip_forward_linear, layer)
            if is_conv:
                layer.forward = types.MethodType(group_snip_conv2d_forward, layer)
            if is_embedding:
                layer.forward = types.MethodType(group_snip_forward_embedded, layer)
            if is_lstm:
                layer.forward = types.MethodType(group_snip_forward_lstm, layer)

        return govs
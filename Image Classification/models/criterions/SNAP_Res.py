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
from models.networks.ResNet18 import ResNet18


class SNAP_Res(General):
    """
    Modified version from:  https://arxiv.org/abs/2006.00896
    Implements SNAP (structured)
    Additionally, this class contains most of the code the actually reduce pytorch tensors, in order to obtain speedup
    """

    def __init__(self, *args, **kwargs):
        super(SNAP_Res, self).__init__(*args, **kwargs)

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def semi_prune(self, tensor, num):
        try:
            if num == 0:
                num = 1
            idx = (tensor >= torch.topk(tensor, num).values[-1]).int()
            if idx.sum() != num:
                for i in range(0, len(tensor) - num, 1):
                    index = torch.kthvalue(tensor, len(tensor) - num - i).indices
                    tensor[index] = torch.max(tensor)
                    idx[index] = 0
                if idx.sum() != num:
                    breakpoint()
            return idx
        except:
            breakpoint()

    def prep_indices_ResNet18(self, idx, indices, scores):
        """
            Reimplementation of method described in: https://openaccess.thecvf.com/content_CVPR_2020/papers/Luo_Neural_Network_Pruning_With_Residual-Connections_and_Limited-Data_CVPR_2020_paper.pdf
        """
        # 1
        num = max([
                    indices[idx[40]].sum(),
                   indices[idx[39]].sum(),
                   indices[idx[36]].sum(),
                   indices[idx[35]].sum(),
                   indices[idx[33]].sum()])
        indices[idx[39]] = self.semi_prune(scores[idx[39]], num)
        indices[idx[40]] = indices[idx[39]]
        indices[idx[36]] = self.semi_prune(scores[idx[36]], num)
        indices[idx[35]] = self.semi_prune(scores[idx[35]],
                                           num)
        indices[idx[33]] = self.semi_prune(scores[idx[33]], num)
        # 2
        num = max([indices[idx[30]].sum(),
                   indices[idx[29]].sum(),
                   indices[idx[26]].sum(),
                   indices[idx[25]].sum(),
                   indices[idx[23]].sum()])
        indices[idx[29]] = self.semi_prune(scores[idx[29]], num)
        indices[idx[30]] = indices[idx[29]]
        indices[idx[26]] = self.semi_prune(scores[idx[26]], num)
        indices[idx[25]] = self.semi_prune(scores[idx[25]],
                                           num)
        indices[idx[23]] = self.semi_prune(scores[idx[23]], num)

        # 3
        num = max([indices[idx[20]].sum(),
                   indices[idx[19]].sum(),
                   indices[idx[16]].sum(),
                   indices[idx[15]].sum(),
                   indices[idx[13]].sum()])
        indices[idx[19]] = self.semi_prune(scores[idx[19]], num)
        indices[idx[20]] = indices[idx[19]]
        indices[idx[16]] = self.semi_prune(scores[idx[16]], num)
        indices[idx[15]] = self.semi_prune(scores[idx[15]],
                                           num)
        indices[idx[13]] = self.semi_prune(scores[idx[13]], num)

        # 4
        num = max([indices[idx[10]].sum(),
                   indices[idx[9]].sum(), indices[idx[6]].sum(),
                   indices[idx[5]].sum(), indices[idx[2]].sum(),
                   indices[idx[1]].sum()])
        indices[idx[9]] = self.semi_prune(scores[idx[9]], num)
        indices[idx[10]] = indices[idx[9]]
        indices[idx[6]] = self.semi_prune(scores[idx[6]], num)
        indices[idx[5]] = self.semi_prune(scores[idx[5]], num)
        indices[idx[2]] = self.semi_prune(scores[idx[2]], num)
        indices[idx[1]] = self.semi_prune(scores[idx[1]], num)

        # Downsample
        indices[idx[14]] = indices[idx[9]]
        indices[idx[24]] = indices[idx[19]]
        indices[idx[34]] = indices[idx[29]]

        return indices

    def prep_indices(self, idx, indices, scores):
        groups = [i for i in range(0, 16384, 32)]
        for i in range(len(indices) - 2, -1, -20):
            num = indices[idx[i]].sum()
            indices[idx[i - 1]] = self.semi_prune(scores[idx[i - 1]], num)
            indices[idx[i - 6]] = self.semi_prune(scores[idx[i - 6]], num)
            indices[idx[i - 7]] = self.semi_prune(scores[idx[i - 7]], num)
            indices[idx[i - 12]] = self.semi_prune(scores[idx[i - 12]], num)
            indices[idx[i - 13]] = self.semi_prune(scores[idx[i - 13]], num)
            indices[idx[i - 15]] = self.semi_prune(scores[idx[i - 15]], num)

            indices[idx[i - 14]] = indices[idx[i - 21]]
            break

        # Layer 3
        num = indices[idx[len(indices) - 22]].sum()
        for i in range(len(indices) - 22, 56, -6):
            indices[idx[i - 1]] = self.semi_prune(scores[idx[i - 1]], num)
            indices[idx[i - 6]] = self.semi_prune(scores[idx[i - 6]], num)
        i -= 6
        indices[idx[i - 1]] = self.semi_prune(scores[idx[i - 1]], num)
        indices[idx[i - 3]] = self.semi_prune(scores[idx[i - 3]], num)

        indices[idx[i - 2]] = indices[idx[i - 9]]

        # Layer 2
        num = indices[idx[48]].sum()
        for i in range(48, 30, -6):
            indices[idx[i - 1]] = self.semi_prune(scores[idx[i - 1]], num)
            indices[idx[i - 6]] = self.semi_prune(scores[idx[i - 6]], num)
        i -= 6
        indices[idx[i - 1]] = self.semi_prune(scores[idx[i - 1]], num)
        indices[idx[i - 3]] = self.semi_prune(scores[idx[i - 3]], num)

        indices[idx[i - 2]] = indices[idx[i - 9]]

        # Layer 1
        # breakpoint()
        num = indices[idx[22]].sum()
        for i in range(22, 10, -6):
            indices[idx[i - 1]] = self.semi_prune(scores[idx[i - 1]], num)
            indices[idx[i - 6]] = self.semi_prune(scores[idx[i - 6]], num)
        i -= 6
        indices[idx[i - 1]] = self.semi_prune(scores[idx[i - 1]], num)
        indices[idx[i - 3]] = self.semi_prune(scores[idx[i - 3]], num)
        # breakpoint()
        indices[idx[2]] = torch.ones_like(indices[idx[2]]).bool()
        indices[idx[1]] = torch.ones_like(indices[idx[1]]).bool()
        indices[idx[0]] = torch.ones_like(indices[idx[0]]).bool()
        input = False
        for i in range(3, len(indices)):
            if 'conv1' in idx[i][1] and not input or 'conv2' in idx[i][1] and input:
                # breakpoint()
                for j in range(len(groups)):
                    if indices[idx[i]].sum() <= groups[j]:
                        indices[idx[i]] = self.semi_prune(scores[idx[i]], groups[j])
                        break

            elif 'conv3' in idx[i][1] and input or 'conv2' in idx[i][1] and not input:
                for j in range(len(groups)):
                    if indices[idx[i]].sum() <= groups[j]:
                        indices[idx[i]] = self.semi_prune(scores[idx[i]], groups[j])
                        break
            input = not input
        return indices

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
        indices = {}
        idx = {}
        scores = {}
        i = 0
        for ((identification, name), grad), (first, last) in lookahead_finished(grads_abs.items()):
            indices[(i, name)] = ((grad / norm_factor) >= acceptable_score).int()
            idx[i] = (i, name)
            scores[(i, name)] = grad / norm_factor
            i += 1
        gate_dec = {}
        i=0
        is_gate = False
        for key, val in self.model.named_parameters():
            gate_dec[key] = i
            if 'gate' in key:
                is_gate=True
            i+=1
        if isinstance(self.model, ResNet18):
            indices = self.prep_indices_ResNet18(idx, indices, scores)
        else:
            indices = self.prep_indices(idx, indices, scores)
        for ((identification, name), grad), (first, last) in lookahead_finished(indices.items()):
            binary_keep_neuron_vector = grad.float().to(self.device)
            if not is_gate:
                corresponding_weight_parameter = [val for key, val in self.model.named_parameters() if key == name][0]
            else:
                # breakpoint()
                if name == 'bn1.bn.weight':
                    name = 'conv1.weight'
                    corresponding_weight_parameter = [val for key, val in self.model.named_parameters() if key == name][
                        0]
                else:
                    # breakpoint()
                    if 'downsample' in name or 'fc' in name:
                        corresponding_weight_parameter = \
                        [val for key, val in self.model.named_parameters() if key == name][0]
                    else:
                        name, corresponding_weight_parameter = [(key, val) for key, val in self.model.named_parameters()][gate_dec[name] - 3]

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
        module.in_channels = n_remaining
        length_nonzero = int(np.prod(weight.shape))
        cutoff = 0
        if 'conv2' in name and not isinstance(self.model, ResNet18):
            assert indices.sum() % 32 == 0
            size = indices.sum()
            indices = torch.zeros(weight.shape[1])
            for i in range(size / 32):
                indices[i] = 1
            indices = indices.bool().cuda()
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
            if ((weight.shape[1] % indices.shape[0]) == 0) and not (weight.shape[1] == indices.shape[0]):
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
                try:
                    weight.data = weight[:, indices]
                except:
                    breakpoint()
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

        module.out_channels = n_remaining

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
        bias = [val for key, val in self.model.named_parameters() if
                key == name.split("weight")[0] + "bias" or key == 'bn' + name.split("weight")[0][-2] + ".bias"]
        if len(bias) == 1:
            bias = bias[0]
        elif len(bias) > 1:
            bias = bias[-1]
        else:
            return
        # bias = bias[0] if len(bias)==0 else bias[1]
        try:
            bias.data = bias[indices]
        except:
            pass
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
            try:
                batchnorm = [val for key, val in self.model.named_modules() if key == name.split(".weight")[0][:-1] + str(
                    int(name.split(".weight")[0][-1]) + 1) or key == 'bn' + str(int(name.split(".weight")[0][-1]))]
            except:
                batchnorm = [val for key, val in self.model.named_modules() if
                             key == name.split(".weight")[0][:-1] + str(
                                 int(name.split(".")[0][-1]) + 1) or key == 'bn' + str(
                                 int(name.split(".")[0][-1]))]
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
                    self.model.mask[name] = (self.model.mask[name].permute(permutation) * binary_vector.cuda()).permute(
                        permutation)
                else:
                    self.model.mask[name] *= binary_vector.cuda()
        elif last and self.model._outer_layer_pruning:
            module.out_channels = n_remaining
            if is_conv:
                permutation = (3, 1, 2, 0)
                self.model.mask[name] = (self.model.mask[name].permute(permutation) * binary_vector.cuda()).permute(
                    permutation)
            else:
                self.model.mask[name] = (binary_vector.cuda() * self.model.mask[name].t()).t()
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
                try:
                    layer.bias.requires_grad = False
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

        return govs

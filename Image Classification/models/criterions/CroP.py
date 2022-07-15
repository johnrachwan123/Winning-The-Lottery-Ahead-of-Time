import copy
import types
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from models.networks.assisting_layers.GateDecoratorLayers import GatedBatchNorm

from models.criterions.SNIP import SNIP
from utils.constants import SNIP_BATCH_ITERATIONS
from utils.data_utils import lookahead_type, lookahead_finished
import numpy as np
from utils.snip_utils import group_snip_forward_linear, group_snip_conv2d_forward


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

    def cut_lonely_connections(self):
        indices = {}
        idx = 0
        empty_weights = []
        for id, layer in self.model.mask.items():
            if 'conv' in id or 'downsample' in id:
                # breakpoint()
                # input
                input = []
                for i in range(layer.shape[1]):
                    empty_weights.append(torch.sum(layer[:, i, :, :] == 0).cpu().numpy() / layer[:, i, :, :].numel())
                    if len(torch.nonzero(layer[:, i, :, :])) == 0:
                        input.append(0)
                    else:
                        input.append(1)
                # output
                output = []
                for i in range(layer.shape[0]):
                    # empty_weights.append(torch.sum(layer[i, :, :, :] == 0).cpu().numpy()/layer[i, :, :, :].numel())
                    if len(torch.nonzero(layer[i, :, :, :])) == 0:
                        output.append(0)
                    else:
                        output.append(1)
            else:
                # breakpoint()
                # input
                input = []
                for i in range(layer.shape[1]):
                    empty_weights.append(torch.sum(layer[:, i] == 0).cpu().numpy() / layer[:, i].numel())
                    if len(torch.nonzero(layer[:, i])) == 0:
                        input.append(0)
                    else:
                        input.append(1)
                # output
                output = []
                for i in range(layer.shape[0]):
                    # empty_weights.append(torch.sum(layer[i, :] == 0).cpu().numpy()/layer[i, :].numel())
                    if len(torch.nonzero(layer[i, :])) == 0:
                        output.append(0)
                    else:
                        output.append(1)
            # indices
            indices[(idx, id)] = torch.tensor(input)
            idx += 1
            indices[(idx, id)] = torch.tensor(output)
            idx += 1
        return self.prep_indices(indices)

    def prep_indices(self, indices):
        # Step 1
        old_key = (-1, "nothing")
        res_key = ()
        res_length = 0
        res_key_beginning = None
        old_length = 0
        input = True
        for key, value in indices.items():
            length = len(value)
            if 'downsample' in key[1] and input:
                # breakpoint()
                indices[key] = value.__or__(indices[res_key])
                indices[res_key] = value.__or__(indices[res_key])
                res_key = key
                res_length = length
            if 'layer' in key[1].split('.')[0] + key[1].split('.')[1] or (
                    'layer' not in key[1] and 'downsample' not in key[1] and 'layer' in old_key[1]):
                if res_key_beginning is not None and res_key_beginning != key[1].split('.')[0] + key[1].split('.')[1]:
                    # breakpoint()
                    if input == True:
                        if length == res_length:
                            indices[key] = value.__or__(indices[res_key])
                            indices[res_key] = value.__or__(indices[res_key])
                        elif res_length != 0 and length % res_length == 0:
                            ratio = length // res_length
                            new_indices = torch.repeat_interleave(indices[res_key], ratio)
                            for i in range(res_length):
                                if sum(new_indices[i * ratio:ratio * (i + 1)].__or__(
                                        value[i * ratio:ratio * (i + 1)])) == ratio:
                                    indices[res_key][i] = 1
                                else:
                                    indices[res_key][i] = 0
                            indices[key] = torch.repeat_interleave(indices[res_key], ratio)
                    res_key = key
                    res_length = length
                    res_key_beginning = key[1].split('.')[0] + key[1].split('.')[1]
                elif res_key_beginning is None:
                    res_key = key
                    res_length = length
                    res_key_beginning = key[1].split('.')[0] + key[1].split('.')[1]
            old_length = length
            old_key = key
            input = not input

        # Step 2
        old_key = ()
        res_key = ()
        res_length = 0
        res_key_beginning = None
        old_length = 0
        input = True
        for key, value in indices.items():
            length = len(value)
            if input == True:
                if length == old_length:
                    # indices[old_key] = value.__or__(indices[old_key])
                    # indices[key] = value.__or__(indices[old_key])
                    indices[key] = value
                    indices[old_key] = value
                elif old_length != 0 and length % old_length == 0 and ('fc' in key[1] or 'classifier' in key[1]):
                    ratio = length // old_length
                    # new_indices = torch.repeat_interleave(indices[old_key], ratio)
                    for i in range(old_length):
                        if sum((value[i * ratio:ratio * (i + 1)])) == ratio:
                            # if sum(new_indices[i*ratio:ratio*(i+1)].__or__(value[i*ratio:ratio*(i+1)])) == ratio:
                            indices[old_key][i] = 1
                        else:
                            indices[old_key][i] = 0
                    indices[key] = torch.repeat_interleave(indices[old_key], ratio)
                elif length != 0 and old_length % length == 0 and 'downsample' in key[1]:
                    ratio = old_length // length
                    new_indices = torch.repeat_interleave(value, ratio)
                    for i in range(length):
                        if sum((new_indices[i * ratio:ratio * (i + 1)])) == ratio:
                            indices[key][i] = 1
                        else:
                            indices[key][i] = 0
                    indices[old_key] = torch.repeat_interleave(indices[key], ratio)

            old_length = length
            old_key = key
            input = not input
        self.structured_prune(indices)
        return indices

    def grow(self, percentage, train_loader):
        device = self.model.device
        iterations = SNIP_BATCH_ITERATIONS
        net = self.model.eval()

        # accumalate gradients of multiple batches
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
        # get elasticities
        grads_abs = {}
        for name, layer in net.named_modules():
            if "Norm" in str(layer): continue
            if name + ".weight" in self.model.mask:
                grads_abs[name + ".weight"] = torch.abs(layer.weight.grad)
        all_scores = torch.cat([torch.flatten(x) for _, x in grads_abs.items()])
        # percentage + the number of elements that are already there
        num_params_to_grow = int(
            len(all_scores) * (percentage) + sum([len(torch.nonzero(t)) for t in self.model.mask.values()]))
        if num_params_to_grow < 1:
            num_params_to_grow += 1
        elif num_params_to_grow > len(all_scores):
            num_params_to_grow = len(all_scores)

        # threshold
        threshold, _ = torch.topk(all_scores, num_params_to_grow, sorted=True)
        acceptable_score = threshold[-1]
        # grow
        for name, grad in grads_abs.items():
            self.model.mask[name] = ((grad) > acceptable_score).__or__(
                self.model.mask[name].bool()).float().to(self.device)

        self.model.apply_weight_mask()

    def get_weight_saliencies(self, train_loader, batch=None):

        device = self.model.device

        iterations = SNIP_BATCH_ITERATIONS

        net = self.model.eval()

        self.get_scores(device, iterations, net, train_loader)

        # collect gradients
        grads = {}
        for name, layer in net.named_modules():
            if "Norm" in str(layer): continue
            if name + ".weight" in self.model.mask:
                grads[name + ".weight"] = torch.abs(layer.weight.data * layer.weight.grad)

        # # accumalate gradients of multiple batches
        # net.zero_grad()
        # loss_sum = torch.zeros([1]).to(self.device)
        # for i, (x, y) in enumerate(train_loader):
        #
        #     if i == iterations: break
        #
        #     inputs = x.to(self.model.device)
        #     targets = y.to(self.model.device)
        #     outputs = net.forward(inputs)
        #     loss = F.nll_loss(outputs, targets) / iterations
        #     loss.backward()
        #     loss_sum += loss.item()
        #
        # # get elasticities
        # grads_abs = {}
        # for name, layer in net.named_modules():
        #     if "Norm" in str(layer): continue
        #     if name + ".weight" in self.model.mask:
        #         grads[name + ".weight"] += torch.abs(
        #             layer.weight.grad * (layer.weight.data / (1e-8 + loss_sum.item())))

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for _, x in grads.items()])

        so = all_scores.sort().values

        norm_factor = 1
        log10 = so.log10()
        # all_scores.div_(norm_factor)

        self.model = self.model.train()
        self.model.zero_grad()

        return all_scores, grads, log10, norm_factor

    def get_scores(self, device, iterations, net, train_loader):
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
        for it in range(iterations):
            inputs, targets = next(dataloader_iter)
            N = inputs.shape[0]
            din = copy.deepcopy(inputs)
            dtarget = copy.deepcopy(targets)

            start = 0
            intv = N

            while start < N:
                end = min(start + intv, N)
                inputs_one.append(din[start:end])
                targets_one.append(dtarget[start:end])
                outputs = net.forward(inputs[start:end].to(device))  # divide by temperature to make it uniform
                loss = F.cross_entropy(outputs, targets[start:end].to(device))
                grad_w_p = autograd.grad(loss, weights, create_graph=False)
                # grad_w_p = autograd.grad(outputs, weights, grad_outputs=torch.ones_like(outputs), create_graph=False)
                if grad_w is None:
                    grad_w = list(grad_w_p)
                else:
                    for idx in range(len(grad_w)):
                        grad_w[idx] += grad_w_p[idx]
                start = end
        for it in range(len(inputs_one)):
            inputs = inputs_one.pop(0).to(device)
            targets = targets_one.pop(0).to(device)
            outputs = net.forward(inputs)  # divide by temperature to make it uniform
            loss = F.cross_entropy(outputs, targets)
            grad_f = autograd.grad(loss, weights, create_graph=True)
            # grad_f = autograd.grad(outputs, weights, grad_outputs=torch.ones_like(outputs), create_graph=True)
            z = 0
            count = 0
            for name, layer in net.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    z += (grad_w[count] * grad_f[count] * self.model.mask[name + ".weight"]).sum()
                    count += 1
            z.backward()

    def structured_prune(self, indices):
        summed_weights = sum([np.prod(x.shape) for name, x in self.model.named_parameters() if "weight" in name])
        # handle outer layers
        if not self.model._outer_layer_pruning:
            offsets = [len(x[0][1]) for x in lookahead_finished(indices.items()) if x[1][0] or x[1][1]]
            # breakpoint()
        #     all_scores = all_scores[offsets[0]:-offsets[1]]
        # prune
        summed_pruned = 0
        toggle_row_column = True
        cutoff = 0
        length_nonzero = 0
        for ((identification, name), grad), (first, last) in lookahead_finished(indices.items()):
            # breakpoint()
            binary_keep_neuron_vector = ((grad) > 0).float().to(self.device)
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
                                                              indices,
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
            if ((indices.shape[0] % weight.shape[0]) == 0) and not (weight.shape[1] == indices.shape[0]):
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
        if 'layer' in name:
            if 'downsample' in name:
                batchnorm = [val for key, val in self.model.named_modules() if
                             key == name.split(".weight")[0][:-1] + str(int(name.split(".weight")[0][-1]) + 1)][0]
            else:
                batchnorm = [val for key, val in self.model.named_modules() if
                             key == name.split(".")[0] + "." + name.split(".")[1] + ".bn" + str(
                                 int(name.split(".weight")[0][-1]))][0]
        else:
            batchnorm = [val for key, val in self.model.named_modules()
                         if key == name.split(".weight")[0][:-1] + str(int(name.split(".weight")[0][-1]) + 1)
                         or key == 'bn' + str(int(name.split(".weight")[0][-1]))]
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

from random import randint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from models.criterions.General import General
from models.networks.ResNet18 import ResNet18
from models.networks.assisting_layers.GateDecoratorLayers import GatedBatchNorm

class StructuredRandom(General):

    """
    Original creation from paper:  https://arxiv.org/abs/2006.00896
    Implements Random (structured before training), a surprisingly strong baseline.
    Additionally, this class contains most of the code the actually reduce pytorch tensors, in order to obtain speedup
    """

    def __init__(self, *args, **kwargs):
        super(StructuredRandom, self).__init__(*args, **kwargs)

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def semi_prune(self, tensor, num):
        try:
            tensor = torch.rand(tensor.shape)
            idx = (tensor >= torch.topk(tensor, num).values[-1]).int()
            if idx.sum() != num:
                for i in range(0, len(tensor) - num, 1):
                    index = torch.kthvalue(tensor, len(tensor) - num - i).indices
                    tensor[index] = torch.max(tensor)
                    idx[index] = 0
            return idx
        except:
            breakpoint()


    def prep_indices_ResNet18(self, indices):

        input = True
        prev_key = None
        step2_key = None
        # breakpoint()
        # 1
        idx = []
        num = max([indices[(39, 'layer4.1.conv2')].sum(),
                        indices[(36, 'layer4.1.pruneinput1')].sum(),
                        indices[(35, 'layer4.0.downsample.0')].sum(), indices[(33, 'layer4.0.conv2')].sum()])
        indices[(39, 'layer4.1.conv2')] = self.semi_prune(indices[(39, 'layer4.1.conv2')], num)
        indices[(40, 'fc.1')] = self.semi_prune(indices[(40, 'fc.1')], num*4)
        # breakpoint()
        indices[(36, 'layer4.1.pruneinput1')] = self.semi_prune(indices[(36, 'layer4.1.pruneinput1')], num)
        indices[(35, 'layer4.0.downsample.0')] = self.semi_prune(indices[(35, 'layer4.0.downsample.0')],
                                                                        num)
        indices[(33, 'layer4.0.conv2')] = self.semi_prune(indices[(33, 'layer4.0.conv2')], num)


        # 2
        num = max([indices[(30, 'layer4.0.dontpruneinput1')].sum(),
                        indices[(29, 'layer3.1.conv2')].sum(),
                        indices[(26, 'layer3.1.pruneinput1')].sum(),
                        indices[(25, 'layer3.0.downsample.0')].sum(),
                        indices[(23, 'layer3.0.conv2')].sum()])
        indices[(29, 'layer3.1.conv2')] = self.semi_prune(indices[(29, 'layer3.1.conv2')], num)
        indices[(30, 'layer4.0.dontpruneinput1')] = indices[(29, 'layer3.1.conv2')]
        indices[(26, 'layer3.1.pruneinput1')] = self.semi_prune(indices[(26, 'layer3.1.pruneinput1')], num)
        indices[(25, 'layer3.0.downsample.0')] = self.semi_prune(indices[(25, 'layer3.0.downsample.0')],
                                                                        num)
        indices[(23, 'layer3.0.conv2')] = self.semi_prune(indices[(23, 'layer3.0.conv2')], num)


        # 3
        num = max([indices[(20, 'layer3.0.dontpruneinput1')].sum(),
                        indices[(19, 'layer2.1.conv2')].sum(),
                        indices[(16, 'layer2.1.pruneinput1')].sum(),
                        indices[(15, 'layer2.0.downsample.0')].sum(),
                        indices[(13, 'layer2.0.conv2')].sum()])
        indices[(19, 'layer2.1.conv2')] = self.semi_prune(indices[(19, 'layer2.1.conv2')], num)
        indices[(20, 'layer3.0.dontpruneinput1')] = indices[(19, 'layer2.1.conv2')]
        indices[(16, 'layer2.1.pruneinput1')] = self.semi_prune(indices[(16, 'layer2.1.pruneinput1')], num)
        indices[(15, 'layer2.0.downsample.0')] = self.semi_prune(indices[(15, 'layer2.0.downsample.0')],
                                                                        num)
        indices[(13, 'layer2.0.conv2')] = self.semi_prune(indices[(13, 'layer2.0.conv2')], num)


        # 4
        num = max([indices[(10, 'layer2.0.dontpruneinput1')].sum(),
                        indices[(9, 'layer1.1.conv2')].sum(), indices[(6, 'layer1.1.pruneinput1')].sum(),
                        indices[(5, 'layer1.0.conv2')].sum(), indices[(2, 'layer1.0.pruneinput1')].sum(),
                        indices[(1, 'conv1')].sum()])
        indices[(9, 'layer1.1.conv2')] = self.semi_prune(indices[(9, 'layer1.1.conv2')], num)
        indices[(10, 'layer2.0.dontpruneinput1')] = indices[(9, 'layer1.1.conv2')]
        indices[(6, 'layer1.1.pruneinput1')] = self.semi_prune(indices[(6, 'layer1.1.pruneinput1')], num)
        indices[(5, 'layer1.0.conv2')] = self.semi_prune(indices[(5, 'layer1.0.conv2')], num)
        indices[(2, 'layer1.0.pruneinput1')] = self.semi_prune(indices[(2, 'layer1.0.pruneinput1')], num)
        indices[(1, 'conv1')] = self.semi_prune(indices[(1, 'conv1')], num)

        # Downsample
        indices[(14, 'layer2.0.downsample.0')] = indices[(9, 'layer1.1.conv2')]
        indices[(24, 'layer3.0.downsample.0')] = indices[(19, 'layer2.1.conv2')]
        indices[(34, 'layer4.0.downsample.0')] = indices[(29, 'layer3.1.conv2')]
        # breakpoint()
        for index in indices.values():
            idx.append(index.bool())
        return idx

    def prune(self, percentage, **kwargs):

        in_indices, out_indices = None, None

        modules = [(name, elem)
                   for name, elem in self.model.named_modules()
                   if isinstance(elem, (torch.nn.Linear, torch.nn.Conv2d))]
        last_is_conv = False
        indices = {}
        modu = []
        idx = 0
        if isinstance(self.model, ResNet18):
            for i, (name, module) in enumerate(modules):
                num_params = np.prod(module.weight.shape)
                in_indices, last_is_conv, now_is_conv, out_indices = self.get_inclusion_vectors(i,
                                                                                                in_indices,
                                                                                                last_is_conv,
                                                                                                module,
                                                                                                modules,
                                                                                                out_indices,
                                                                                                percentage)
                indices[(idx, name)] = in_indices
                idx+=1
                indices[(idx, name)] = out_indices
                idx+=1
                modu.append([in_indices, last_is_conv, now_is_conv, out_indices, module, name, num_params])

            idx = self.prep_indices_ResNet18(indices)
            for module in modu:
                in_indices, last_is_conv, now_is_conv, out_indices, mod, name, num_params = module
                # in_indices = indices.pop(0)
                in_indices = idx.pop(0)
                self.handle_input(in_indices, mod, now_is_conv)
                # out_indices = indices.pop(0)
                out_indices = idx.pop(0)
                self.handle_output(out_indices, now_is_conv, mod, name)
                params_left = np.prod(mod.weight.shape)
                pruned = num_params - params_left
                print("pruning", pruned, "percentage", (pruned) / num_params, "length_nonzero", num_params)
        else:
            for i, (name, module) in enumerate(modules):
                num_params = np.prod(module.weight.shape)
                in_indices, last_is_conv, now_is_conv, out_indices = self.get_inclusion_vectors(i,
                                                                                                in_indices,
                                                                                                last_is_conv,
                                                                                                module,
                                                                                                modules,
                                                                                                out_indices,
                                                                                                percentage)
                self.handle_input(in_indices, module, now_is_conv)
                self.handle_output(out_indices, now_is_conv, module, name)
                params_left = np.prod(module.weight.shape)
                pruned = num_params - params_left
                print("pruning", pruned, "percentage", (pruned) / num_params, "length_nonzero", num_params)
        self.model.mask = {name + ".weight": torch.ones_like(module.weight.data).to(self.device)
                           for name, module in self.model.named_modules()
                           if isinstance(module, (nn.Linear, nn.Conv2d))
                           }

        print(self.model)
        print("Final percentage: ", self.model.pruned_percentage)

    def handle_output(self, indices, is_conv, module, name):
        weight = module.weight
        module.update_output_dim(indices.sum())
        self.handle_batch_norm(indices, indices.sum(), name)
        if is_conv:
            weight.data = weight[indices, :, :, :]
            try:
                weight.grad.data = weight.grad.data[indices, :, :, :]
            except:
                pass
        else:
            weight.data = weight[indices, :]
            try:
                weight.grad.data = weight.grad.data[indices, :]
            except:
                pass
        self.handle_bias(indices, module)

    def handle_bias(self, indices, module):
        bias = module.bias
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
        # breakpoint()
        if not isinstance(self.model, ResNet18):
            batchnorm = [val for key, val in self.model.named_modules() if key == name[:-1] + str(int(name[-1]) + 1)]
            if len(batchnorm) == 1:
                batchnorm = batchnorm[0]
            elif len(batchnorm) > 1:
                batchnorm = batchnorm[-1]
            else:
                return
        elif name == 'm.conv1.weight':
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
            batchnorm = [val for key, val in self.model.named_modules() if key == name.split(".weight")[0][:-1] + str(
                int(name.split(".weight")[0][-1]) + 1) or key == 'bn' + str(int(name.split(".weight")[0][-1]))]
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

    def handle_input(self, indices, module, now_is_conv):
        module.update_input_dim(indices.sum())
        if now_is_conv:
            module.weight.data = module.weight.data[:, indices, :, :]
            try:
                module.weight.grad.data = module.weight.grad.data[:, indices, :, :]
            except:
                pass
        else:
            module.weight.data = module.weight.data[:, indices]
            try:
                module.weight.grad.data = module.weight.grad.data[:, indices]
            except:
                pass

    def get_inclusion_vectors(self, i, in_indices, last_is_conv, module, modules, out_indices, percentage):
        param = module.weight
        dims = param.shape[:2]  # out, in
        if in_indices is None:
            in_indices = torch.ones(dims[1])
            if self.model._outer_layer_pruning:
                in_indices = self.get_in_vector(dims, percentage)
        else:
            in_indices = out_indices
        out_indices = self.get_out_vector(dims, percentage)
        is_last = (len(modules) - 1) == i
        if is_last and not self.model._outer_layer_pruning:
            out_indices = torch.ones(dims[0])
        now_is_fc = isinstance(module, torch.nn.Linear)
        now_is_conv = isinstance(module, torch.nn.Conv2d)
        if last_is_conv and now_is_fc:
            ratio = param.shape[1] // in_indices.shape[0]
            in_indices = torch.repeat_interleave(in_indices, ratio)
        last_is_conv = now_is_conv
        if in_indices.sum() == 0:
            in_indices[randint(0, len(in_indices) - 1)] = 1
        if out_indices.sum() == 0:
            out_indices[randint(0, len(out_indices) - 1)] = 1
        return in_indices.bool(), last_is_conv, now_is_conv, out_indices.bool()

    def get_out_vector(self, dims, percentage):
        return (init.sparse(torch.empty(dims), percentage)[:, 0] != 0).long()

    def get_in_vector(self, dims, percentage):
        return (init.sparse(torch.empty(dims), percentage)[0, :] != 0).long()

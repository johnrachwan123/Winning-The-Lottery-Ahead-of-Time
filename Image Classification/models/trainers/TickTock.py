import torch
import torch.nn as nn

from models.networks.assisting_layers.GateDecoratorLayers import GatedBatchNorm
from models.trainers.DefaultTrainer import DefaultTrainer
from utils.constants import OPTIMS
from utils.model_utils import find_right_model


class TickTock(DefaultTrainer):

    """
    Own interpretation of TickTock schedule of the paper
    Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks
    https://arxiv.org/abs/1909.08174
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ticks_done = 0
        self.tocks_done = 0
        self.lam = 1

    def _forward_pass(self,
                      *args,
                      train=True):
        """ implementation of a forward pass """

        if train:
            self._optimizer_gates.zero_grad()
        accuracy, loss, out = super()._forward_pass(*args, train=train)
        _, __, at_tocks = self.get_state()
        if at_tocks:
            accumulated = torch.zeros([1], device=self._device)
            for name, module in self._model.named_modules():
                if isinstance(module, GatedBatchNorm):
                    accumulated += self.lam * module.gate.data.sum()
            loss += accumulated
        return accuracy, loss, out

    def _epoch_iteration(self,
                         *args):
        """ implementation of an epoch """

        self._optimizer_gates = find_right_model(OPTIMS, self._arguments.optimizer,
                                                 params=[param for name, param in self._model.named_parameters() if
                                                         "gate" in name],
                                                 lr=self._arguments.learning_rate,
                                                 weight_decay=self._arguments.l2_reg)

        at_ft, at_ticks, at_tocks = self.get_state()

        # noinspection PyArgumentList
        assert torch.BoolTensor([at_ft, at_ticks, at_tocks]).sum() == 1

        super()._epoch_iteration(*args)

        if at_ticks:
            self.ticks_done += 1
            at_ft, at_ticks, at_tocks = self.get_state()
            if at_ft:
                self.ticks_done, self.tocks_done = -1, -1

        if at_tocks:
            self.tocks_done += 1
            if self.tocks_done == 10:
                self.ticks_done, self.tocks_done = 0, 0

    def get_state(self):
        at_ticks = self.ticks_done < 10 and self.tocks_done == 0
        at_tocks = self.ticks_done == 10 and self.tocks_done < 10
        at_ft = self._arguments.pruning_limit <= self._model.structural_sparsity
        return at_ft, at_ticks, at_tocks

    def _is_pruning_time(self, epoch):

        return self.get_state()[1]

    def _backward_pass(self, loss):
        """ implementation of a backward pass """

        loss.backward()
        self._model.insert_noise_for_gradient(self._arguments.grad_noise)
        if self._arguments.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._arguments.grad_clip)
        at_ft, at_ticks, at_tocks = self.get_state()
        opt = self._optimizer_gates if at_ticks else self._optimizer
        opt.step()

    def train(self):

        # convert to gated batchnorms
        for name, module in self._model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                new_module = GatedBatchNorm(module, device=self._device)
                names = name.split(".")
                last_name = names.pop(-1)
                parent = self._model
                for left in names:
                    parent = parent._modules[left]
                parent._modules[last_name] = new_module

        self._optimizer = find_right_model(OPTIMS, self._arguments.optimizer,
                                           params=[param for name, param in self._model.named_parameters() if
                                                   "gate" not in name],
                                           lr=self._arguments.learning_rate,
                                           weight_decay=self._arguments.l2_reg)

        self._optimizer_gates = find_right_model(OPTIMS, self._arguments.optimizer,
                                                 params=[param for name, param in self._model.named_parameters() if
                                                         "gate" in name],
                                                 lr=self._arguments.learning_rate,
                                                 weight_decay=self._arguments.l2_reg)

        super().train()

import torch

from models.GeneralModel import GeneralModel


class General(GeneralModel):

    """
    parent class
    > deprecated
    """

    def __init__(self, *args, model=None, device="cuda", before_training=False, structured=False, **kwargs):
        super(General, self).__init__(device=device, **kwargs)
        self.structured = structured
        self.before_training = before_training
        self.model = model

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def grow(self, percentage, train_loader):

        if not self.model.is_growable:
            raise Exception("growing is off")
        elif not self.model.is_maskable:
            raise Exception("mask is off")

        if not self.model._outer_layer_pruning:
            percentage *= self.percentage_fraction

        with torch.no_grad():
            for name, tensor in self.model.named_parameters():
                if name in self.model.mask:
                    # flatten tensors
                    flattened = tensor.flatten()
                    grad = tensor.grad.flatten()
                    old_mask = self.model.mask[name].flatten()

                    growable_indices = self.get_grow_indices(percentage=percentage,
                                                             flattened_weights=flattened,
                                                             flattened_grad=grad,
                                                             mask=old_mask)

                    # get new mask
                    old_mask[growable_indices] = 1

                    # multiply new mask with old mask
                    new_mask = old_mask.view(tensor.size())

                    # save mask
                    self.model.mask[name] = new_mask

        self.model.apply_weight_mask()

    def prune(self, percentage, **kwargs):

        if not self.model.is_maskable:
            raise Exception("mask is off")

        if not self.model._outer_layer_pruning:
            percentage *= self.model.percentage_fraction
            if (percentage > 0.99):
                percentage = 0.99

        with torch.no_grad():

            for name, tensor in self.model.named_parameters():

                if name in self.model.mask:
                    old_mask = self.model.mask[name].flatten()

                    prunable_indices = self.get_prune_indices(
                        tensor=tensor,
                        name=name,
                        percentage=percentage,
                        mask=old_mask)

                    if prunable_indices.shape[0] == 0: continue

                    # get new mask
                    old_mask[prunable_indices] = 0

                    # multiply new mask with old mask
                    new_mask = old_mask.view(tensor.size())

                    # save mask
                    self.model.mask[name] = new_mask

        self.cut_lonely_connections()

        self.model.apply_weight_mask()

    def cut_lonely_connections(self):

        # ignoring lonely connections for now

        return

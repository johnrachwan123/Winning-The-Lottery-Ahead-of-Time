import torch.nn as nn


class Meta(type):

    """ forces the post init function after init """

    def __call__(self, *args, **kwargs):
        ins = super().__call__(*args, **kwargs)
        ins: GeneralModel
        ins.post_init()
        return ins


class GeneralModel(nn.Module, metaclass=Meta):

    """ defines a general model (e.g. networks, trainers, criterions) """

    def __init__(self, device, input_dim=None, output_dim=None, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.initialised = False
        self.device = device
        super(GeneralModel, self).__init__()

        if len(kwargs) > 0:
            print(f"Ignored arguments in {self.__class__.__name__}: {kwargs}")

    def post_init(self):

        if self.initialised:
            raise Exception("illegal post-init")

        self.post_init_implementation()

        self.initialised = True

    def post_init_implementation(self):
        print(f"No post init specified in {self.__class__.__name__}")

import argparse
import os
import random

import numpy as np
import torch

from utils.constants import TIMEOUT
from utils.data_manager import DataManager


"""
Handles loading config and autoconfig
"""

def configure_seeds(arguments, device):
    seed = arguments.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def configure_device(arguments):
    device = arguments.device
    assert "cpu" in device or torch.cuda.is_available(), f"DEVICE {device} UNAVAILABLE"
    return torch.device(device)


def parse() -> argparse.Namespace:
    """ does argument parsing """

    parser = argparse.ArgumentParser()

    """ int """

    # technical specifications
    parser.add_argument('--eval_freq', default=1000, type=int, help='evaluate every n batches')
    parser.add_argument('--save_freq', default=1e6, type=int,
                        help='save model every n epochs, besides before and after training')
    parser.add_argument('--batch_size', default=512, type=int, help='size of batches')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument('--max_training_minutes', default=TIMEOUT, type=int,
                        help="process killed after n minutes (after finish of epoch)")
    parser.add_argument('--plot_weights_freq', default=50, type=int,
                        help="plot pictures to tensorboard every n epochs")
    # train hyperparams
    parser.add_argument('--prune_freq', default=1, type=int,
                        help="if pruning during training: how long to wait before starting")
    parser.add_argument('--prune_delay', default=0, type=int,
                        help="if pruning during training: 't' from algorithm box, itnterval between pruning events")
    parser.add_argument('--epochs', default=80, type=int, help='number of epochs')
    parser.add_argument('--rewind_to', default=0, type=int, help="rewind to this epoch if rewinding is done")
    # model hyperparams
    parser.add_argument('--hidden_dim', default=None, type=int, help='size of hidden_dim')
    parser.add_argument('--input_dim', default=None, type=tuple, help='size of input_dim')
    parser.add_argument('--output_dim', default=None, type=int, help='size of output_dim')
    # data
    parser.add_argument('--N', default=1, type=int, help='size of dataset (used for l0)')
    # snip
    parser.add_argument('--snip_steps', default=5, type=int,
                        help="'s' in algorithm box, number of pruning steps for 'rule of thumb' ")  # todo

    """ float """
    parser.add_argument('--pruning_rate', default=0.0, type=float,
                        help="pruning rate passed to criterion at pruning event. however, most override this")
    parser.add_argument('--growing_rate', default=0.0000, type=float,
                        help="grow back so much every epoch (for future criterions)")
    parser.add_argument('--pruning_limit', default=0.5, type=float,
                        help="Prune until here, if structured in nodes, if unstructured in weights. most criterions use this instead of the pruning_rate")
    parser.add_argument('--learning_rate', default=2e-3, type=float, help='learning rate')
    parser.add_argument('--grad_clip', default=10, type=float, help='max norm gradients')
    parser.add_argument('--prune_to', default=100, type=int, help='early pruning point')
    parser.add_argument('--structured_prior', default='0', type=int, help='structured prune before performing the rest')
    parser.add_argument('--grad_noise', default=0, type=float, help='added gaussian noise to gradients')
    parser.add_argument('--l2_reg', default=5e-5, type=float, help='weight decay')
    parser.add_argument('--l1_reg', default=0, type=float, help='l1-norm regularisation')
    parser.add_argument('--lp_reg', default=0, type=float, help='lp regularisation with p < 1')
    parser.add_argument('--l0_reg', default=1.0, type=float, help='l0 reg lambda hyperparam')
    parser.add_argument('--hoyer_reg', default=1.0, type=float, help='hoyer reg lambda hyperparam')
    parser.add_argument('--beta_ema', type=float, default=0.999, help="l0 reg beta ema hyperparam")
    parser.add_argument('--momentum', type=float, default=0.0, help="momentum of SGD")
    """ str """
    # model-names
    parser.add_argument('--loss', default="CrossEntropy", type=str, help='loss-function model name')
    parser.add_argument('--optimizer', default="ADAM", type=str, help='optimizer model name')
    parser.add_argument('--model', default="MLP5", type=str, help='network model name')
    parser.add_argument('--data_set', default='MNIST', type=str, help="dataset name")
    parser.add_argument('--prune_criterion', default='SNAP', type=str, help="prune criterion name")
    parser.add_argument('--train_scheme', default='DefaultTrainer', type=str, help="training schedule name")
    parser.add_argument('--test_scheme', default='AdversarialEvaluation', type=str, help="testing schedule name")
    parser.add_argument('--attack', default="CarliniWagner", type=str,
                        help="name of adversarial attack if ran in that setting (=eval with the right testscheme)")

    # technical specifications
    parser.add_argument("--device", type=str, default="cuda", help="cpu or gpu")
    parser.add_argument("--pruning_device", type=str, default="cuda", help="cpu or gpu")
    parser.add_argument('--run_name', default="_example_runname", type=str, help='extra identification for run')

    # checkpoints
    parser.add_argument('--checkpoint_name', default=None
                        , type=str, help='load from this identification for run (foldernames in results folder). If None, then nothing is loaded')
    parser.add_argument('--checkpoint_model', default="ResNet18_finished", type=str,
                        help='extra identification for model to load within run, usually: [MODELNAME]_finished')

    """ bool """

    # technical specifications
    parser.add_argument("--disable_cuda_benchmark", action="store_false",
                        help="speedup (disable) vs reproducibility (leave it)")
    parser.add_argument("--eval", action="store_true", help="run in test mode or train mode")
    parser.add_argument("--disable_autoconfig", action="store_false", help="for the brave only")
    parser.add_argument("--preload_all_data", action="store_true", help="load all data into ram memory for speedups")
    parser.add_argument("--tuning", action="store_true",
                        help="splits trainset into train and validationset, omits test set")

    # train hyperparams
    parser.add_argument('--track_weights', action='store_true', help="keep statistics on the weights through training")
    parser.add_argument('--disable_masking', action='store_false', help="disable the ability to prune unstructured")
    parser.add_argument('--enable_rewinding', action='store_true',
                        help="enable the ability to rewind to previous weights")
    parser.add_argument('--outer_layer_pruning', action='store_true',
                        help="allow to prune outer layers (unstructured) or not (structured)")
    parser.add_argument('--random_shuffle_labels', action='store_true',
                        help="run with random-label experiment from zhang et al")
    parser.add_argument('--l0', action='store_true', help="run with l0 criterion, might overwrite some other arguments")
    parser.add_argument('--hoyer_square', action='store_true',
                        help="run in unstructured DeephoyerSquare criterion, might overwrite some other arguments")
    parser.add_argument('--group_hoyer_square', action='store_true',
                        help="run in unstructured Group-DeephoyerSquare criterion, might overwrite some other arguments")

    # plotting
    parser.add_argument('--disable_histograms', action='store_false', help="regards plotting, changing not recommended")
    parser.add_argument('--disable_saliency', action='store_false', help="regards plotting, changing not recommended")
    parser.add_argument('--disable_confusion', action='store_false', help="regards plotting, changing not recommended")
    parser.add_argument('--disable_weightplot', action='store_true', help="regards plotting, changing not recommended")
    parser.add_argument('--disable_netplot', action='store_true', help="regards plotting, changing not recommended")
    parser.add_argument('--skip_first_plot', action='store_false', help="regards plotting, changing not recommended")

    return parser.parse_args()


def check_incompatible_props(properties, *names):
    if properties.count(True) > 1:
        raise Exception(f"Properties {names} are incompatible")


def autoconfig(config):
    print("setting autoconfig")
    temp_loader = DataManager(os.path.join(".", "utils"))
    auto_configuration = temp_loader.load_json("autoconfig")
    if config.data_set in auto_configuration["dataset"]:
        for key, value in auto_configuration["dataset"][config.data_set].items():
            setattr(config, key, value)
    if config.model in auto_configuration["model"]:
        for key, value in auto_configuration["model"][config.model].items():
            setattr(config, key, value)
    if config.l0:
        for key, value in auto_configuration["l0"].items():
            setattr(config, key, value)
    if config.prune_criterion in auto_configuration:
        for key, value in auto_configuration[config.prune_criterion].items():
            setattr(config, key, value)
    if config.hoyer_square:
        for key, value in auto_configuration["hoyer_square"].items():
            setattr(config, key, value)
    if config.group_hoyer_square:
        for key, value in auto_configuration["group_hoyer_square"].items():
            setattr(config, key, value)

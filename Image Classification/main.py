import sys
import warnings

from models import GeneralModel
from models.statistics.Metrics import Metrics
from utils.config_utils import *
from utils.model_utils import *
from utils.system_utils import *
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from codecarbon import EmissionsTracker

warnings.filterwarnings("ignore")


def main(
        arguments: argparse.Namespace,
        metrics: Metrics
):
    if arguments.disable_autoconfig:
        autoconfig(arguments)

    global out
    out = metrics.log_line
    out(f"starting at {get_date_stamp()}")

    # hardware
    device = configure_device(arguments)

    if arguments.disable_cuda_benchmark:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # for reproducibility
    configure_seeds(arguments, device)

    # filter for incompatible properties
    assert_compatibilities(arguments)
    # get model
    model: GeneralModel = find_right_model(
        NETWORKS_DIR, arguments.model,
        device=device,
        hidden_dim=arguments.hidden_dim,
        input_dim=arguments.input_dim,
        output_dim=arguments.output_dim,
        is_maskable=arguments.disable_masking,
        is_tracking_weights=arguments.track_weights,
        is_rewindable=arguments.enable_rewinding,
        is_growable=arguments.growing_rate > 0,
        outer_layer_pruning=arguments.outer_layer_pruning,
        maintain_outer_mask_anyway=(
                                       not arguments.outer_layer_pruning) and (
                                           "Structured" in arguments.prune_criterion),
        l0=arguments.l0,
        l0_reg=arguments.l0_reg,
        N=arguments.N,
        beta_ema=arguments.beta_ema,
        l2_reg=arguments.l2_reg
    ).to(device)

    # get criterion
    criterion = find_right_model(
        CRITERION_DIR, arguments.prune_criterion,
        model=model,
        limit=arguments.pruning_limit,
        start=0.5,
        steps=arguments.snip_steps,
        device=arguments.pruning_device
    )

    # load pre-trained weights if specified
    load_checkpoint(arguments, metrics, model)

    # load data
    train_loader, test_loader = find_right_model(
        DATASETS, arguments.data_set,
        arguments=arguments
    )

    # get loss function
    loss = find_right_model(
        LOSS_DIR, arguments.loss,
        device=device,
        l1_reg=arguments.l1_reg,
        lp_reg=arguments.lp_reg,
        l0_reg=arguments.l0_reg,
        hoyer_reg=arguments.hoyer_reg
    )

    # get optimizer
    optimizer = find_right_model(
        OPTIMS, arguments.optimizer,
        params=model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.l2_reg,
    )
    scheduler = StepLR(optimizer, step_size=30000, gamma=0.2)

    if not arguments.eval:

        # build trainer
        trainer = find_right_model(
            TRAINERS_DIR, arguments.train_scheme,
            model=model,
            loss=loss,
            optimizer=optimizer,
            device=device,
            arguments=arguments,
            train_loader=train_loader,
            test_loader=test_loader,
            metrics=metrics,
            criterion=criterion,
            scheduler=scheduler,
        )

        tracker = EmissionsTracker()
        tracker.start()
        trainer.train()
        emissions: float = tracker.stop()

    else:

        tester = find_right_model(
            TESTERS_DIR, arguments.test_scheme,
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            loss=loss,
            optimizer=optimizer,
            device=device,
            arguments=arguments,
        )

        return tester.evaluate()

    out(f"finishing at {get_date_stamp()}")


def assert_compatibilities(arguments):
    check_incompatible_props([arguments.loss != "L0CrossEntropy", arguments.l0], "l0", arguments.loss)
    check_incompatible_props([arguments.train_scheme != "L0Trainer", arguments.l0], "l0", arguments.train_scheme)
    check_incompatible_props([arguments.l0, arguments.group_hoyer_square, arguments.hoyer_square],
                             "Choose one mode, not multiple")


def load_checkpoint(arguments, metrics, model):
    if (not (arguments.checkpoint_name is None)) and (not (arguments.checkpoint_model is None)):
        path = os.path.join(RESULTS_DIR, arguments.checkpoint_name, MODELS_DIR, arguments.checkpoint_model)
        state = DATA_MANAGER.load_python_obj(path)
        try:
            model.load_state_dict(state)
        except KeyError as e:
            print(list(state.keys()))
            raise e
        out(f"Loaded checkpoint {arguments.checkpoint_name} from {arguments.checkpoint_model}")


def log_start_run():
    arguments.PyTorch_version = torch.__version__
    arguments.PyThon_version = sys.version
    arguments.pwd = os.getcwd()
    out("PyTorch version:", torch.__version__, "Python version:", sys.version)
    out("Working directory: ", os.getcwd())
    out("CUDA avalability:", torch.cuda.is_available(), "CUDA version:", torch.version.cuda)
    out(arguments)


def get_arguments():
    global arguments
    arguments = parse()
    if arguments.disable_autoconfig:
        autoconfig(arguments)
    return arguments

if __name__ == '__main__':
    metrics = Metrics()
    out = metrics.log_line
    print = out
    ensure_current_directory()
    get_arguments()
    log_start_run()
    out("\n\n")
    metrics._batch_size = arguments.batch_size
    metrics._eval_freq = arguments.eval_freq
    main(arguments, metrics)

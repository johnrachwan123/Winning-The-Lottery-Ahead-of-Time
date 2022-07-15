import argparse
import sys
import time
from scipy import stats

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from models import GeneralModel
from models.statistics import Metrics
from utils.model_utils import find_right_model, linear_CKA, kernel_CKA, batch_CKA, cka, cka_batch
from utils.system_utils import *
from torch.optim.lr_scheduler import StepLR


class DefaultTrainer:
    """
    Implements generalised computer vision classification with pruning
    """

    def __init__(self,
                 model: GeneralModel,
                 loss: GeneralModel,
                 optimizer: Optimizer,
                 device,
                 arguments: argparse.Namespace,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 metrics: Metrics,
                 criterion: GeneralModel,
                 scheduler: StepLR,
                 run_name: str = 'test_delete'
                 # pruner: RigLScheduler
                 ):
        self.epoch = 0
        self.diff = 0
        self._test_loader = test_loader
        self._train_loader = train_loader
        self._test_model = None
        self._fim_loader = None
        self.gradient_adtest = []
        self.loss_test = []
        self._stable = False
        self._overlap_queue = []
        self._loss_function = loss
        self._model = model
        self._arguments = arguments
        self._optimizer = optimizer
        self._device = device
        self._global_steps = 0
        self.out = metrics.log_line
        self.patience = 0
        DATA_MANAGER.set_date_stamp(addition=run_name)
        self._writer = SummaryWriter(os.path.join(DATA_MANAGER.directory, RESULTS_DIR, DATA_MANAGER.stamp, SUMMARY_DIR))
        self._metrics: Metrics = metrics
        self._metrics.init_training(self._writer)
        self._acc_buffer = []
        self._loss_buffer = []
        self._elapsed_buffer = []
        self._criterion = criterion
        self._scheduler = scheduler
        self.gt = []
        self.pred = []
        # self._pruner = pruner
        self.ts = None
        self.old_score = None
        self.old_grads = None
        self.gradient_flow = 0
        self.weights = None
        self._variance = 0
        self.mask1 = self._model.mask.copy()
        self.mask2 = None
        self.newgrad = None
        self.newweight = None
        self.all_scores = None
        self.scores = []
        self.count = 0
        self._step = 0.97
        self._percentage = 0.999
        self._metrics.write_arguments(arguments)
        self._metrics.model_to_tensorboard(model, timestep=-1)
        self.threshold = None

        ## Metrics for SEML ##
        self.test_acc = None
        self.train_acc = None
        self.test_loss = None
        self.train_loss = None
        self.sparse_weight = None
        self.sparse_node = None
        self.sparse_hm = None
        self.sparse_log_disk_size = None
        self.time_gpu = None
        self.flops_per_sample = None
        self.flops_log_cum = None
        self.gpu_ram = None
        self.max_gpu_ram = None
        self.batch_time = None

    def weight_reset(self):
        reset_parameters = getattr(self._model, "reset_parameters", None)
        if callable(reset_parameters):
            self._model.reset_parameters()

    def _batch_iteration(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         train: bool = True):
        """ one iteration of forward-backward """

        # unpack
        x, y = x.to(self._device).float(), y.to(self._device)

        # update metrics
        self._metrics.update_batch(train)

        # record time
        if "cuda" in str(self._device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        # forward pass
        accuracy, loss, out = self._forward_pass(x, y, train=train)
        # backward pass
        if train:
            self._backward_pass(loss)

        # record time
        if "cuda" in str(self._device):
            end.record()
            torch.cuda.synchronize(self._device)
            time = start.elapsed_time(end)
        else:
            time = 0

        # free memory
        for tens in [out, y, x, loss]:
            tens.detach()

        return accuracy, loss.item(), time

    def _forward_pass(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      train: bool = True):
        """ implementation of a forward pass """

        if train:
            self._optimizer.zero_grad()
            if self._model.is_maskable:
                self._model.apply_weight_mask()

        out = self._model(x).squeeze()
        loss = self._loss_function(
            output=out,
            target=y,
            weight_generator=self._model.parameters(),
            model=self._model,
            criterion=self._criterion
        )
        accuracy = self._get_accuracy(out, y)
        return accuracy, loss, out

    def _backward_pass(self, loss):
        """ implementation of a backward pass """

        loss.backward()
        self._model.insert_noise_for_gradient(self._arguments.grad_noise)
        if self._arguments.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._arguments.grad_clip)
        # if self._arguments.prune_criterion == "RigL" and self._pruner():
        self._optimizer.step()
        if self._model.is_maskable:
            self._model.apply_weight_mask()

    def smooth(self, scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        return smoothed

    def ntk(self, model, inp):
        """Calculate the neural tangent kernel of the model on the inputs.
        Returns the gradient feature map along with the tangent kernel.
        """
        out = model(inp.to(self._device).float())
        p_vec = torch.nn.utils.parameters_to_vector(model.parameters())
        p, = p_vec.shape
        n, outdim = out.shape
        # assert outdim == 1, "cant handle output dim higher than 1 for now"

        # this is the transpose jacobian (grad y(w))^T)
        features = torch.zeros(n, p, requires_grad=False)

        for i in range(outdim):  # for loop over data points
            model.zero_grad()
            out[0][i].backward(retain_graph=True)
        p_grad = torch.tensor([], requires_grad=False).to(self._device)
        for p in model.parameters():
            p_grad = torch.cat((p_grad, p.grad.reshape(-1)))
        features[0, :] = p_grad

        tk = features @ features.t()  # compute the tangent kernel
        return features, tk

    def _epoch_iteration(self):
        """ implementation of an epoch """
        self._model.train()
        self.out("\n")
        div = []
        self._acc_buffer, self._loss_buffer = self._metrics.update_epoch()
        mean_abs_mag_grad = 0
        gradient_norm = []
        gradient_adtest = []
        loss_test = []
        for batch_num, batch in enumerate(self._train_loader):
            self.out(f"\rTraining... {batch_num}/{len(self._train_loader)}", end='')

            if self._model.is_tracking_weights:
                self._model.save_prev_weights()
            # Perform one batch iteration
            acc, loss, elapsed = self._batch_iteration(*batch, self._model.training)

            if self._model.is_tracking_weights:
                self._model.update_tracked_weights(self._metrics.batch_train)

            self._acc_buffer.append(acc)
            self._loss_buffer.append(loss)

            loss_test.append(loss)
            self._metrics.add(loss, key="loss/step")

            self._elapsed_buffer.append(elapsed)

            self._log(batch_num)

            self._check_exit_conditions_epoch_iteration()
            self._scheduler.step()

            self._optimizer.zero_grad()

        self._model.eval()

        if not self._stable:
            weights = torch.cat([torch.flatten(k) for k in self._model.parameters()])
            weights = weights.cpu().detach()
            if self.weights is not None:
                if self.threshold is None:
                    if 'Structured' in self._arguments.prune_criterion:
                        self.threshold = (torch.norm(self.weights - weights, p=2) / torch.norm(self.weights, p=2)) * (
                            min(1 - self._arguments.pruning_limit - 0.1, 0.99))
                    else:
                        self.threshold = (torch.norm(self.weights - weights, p=2) / torch.norm(self.weights, p=2)) * (
                                1 - self._arguments.pruning_limit)
                print(torch.norm(self.weights - weights, p=2) / torch.norm(self.weights,p=2) - self.diff)
                print(self.threshold)
                if self.threshold is not None and torch.norm(self.weights - weights, p=2) / torch.norm(self.weights,
                                                                                                       p=2) - self.diff < self.threshold:
                    self._stable = True
                self.diff = torch.norm(self.weights - weights, p=2) / torch.norm(self.weights, p=2)
            if self.weights is None:
                self.weights = weights

    def _log(self, batch_num: int):
        """ logs to terminal and tensorboard if the time is right"""

        if (batch_num % self._arguments.eval_freq) == 0 and batch_num != 0:
            # validate on test and train set
            train_acc, train_loss = np.mean(self._acc_buffer), np.mean(self._loss_buffer)
            test_acc, test_loss, test_elapsed = self.validate()
            self._elapsed_buffer += test_elapsed

            # log metrics
            self._add_metrics(test_acc, test_loss, train_acc, train_loss)

            # reset for next log
            self._acc_buffer, self._loss_buffer, self._elapsed_buffer = [], [], []

            # print to terminal
            self.out(self._metrics.printable_last)

    def validate(self):
        """ validates the model on test set """

        self.out("\n")

        # init test mode
        self._model.eval()
        cum_acc, cum_loss, cum_elapsed = [], [], []

        with torch.no_grad():
            for batch_num, batch in enumerate(self._test_loader):
                acc, loss, elapsed = self._batch_iteration(*batch, self._model.training)
                cum_acc.append(acc)
                cum_loss.append(loss),
                cum_elapsed.append(elapsed)
                self.out(f"\rEvaluating... {batch_num}/{len(self._test_loader)}", end='')
        self.out("\n")

        # put back into train mode
        self._model.train()

        return float(np.mean(cum_acc)), float(np.mean(cum_loss)), cum_elapsed

    def _add_metrics(self, test_acc, test_loss, train_acc, train_loss):
        """
        save metrics
        """

        sparsity = self._model.pruned_percentage
        spasity_index = 2 * ((sparsity * test_acc) / (1e-8 + sparsity + test_acc))

        if self.train_acc is not None:
            if self.train_acc < train_acc:
                self.train_acc = train_acc
        else:
            self.train_acc = train_acc

        if self.test_acc is not None:
            if self._arguments.prune_criterion == "EmptyCrit":
                if self.test_acc < test_acc:
                    self.test_acc = test_acc
            elif self._arguments.prune_criterion == 'EfficientConvNets':
                if sparsity > 0.1:
                    if self.test_acc < test_acc:
                        self.test_acc = test_acc
            else:
                if not self._is_not_finished_pruning():
                    if self.test_acc < test_acc:
                        self.test_acc = test_acc
                else:
                    self.test_acc = 0
        else:
            self.test_acc = test_acc
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.sparse_weight = sparsity
        self.sparse_node = self._model.structural_sparsity
        self.sparse_hm = spasity_index
        self.sparse_log_disk_size = np.log(self._model.compressed_size)
        self.time_gpu = np.mean(self._elapsed_buffer)
        if torch.cuda.is_available():
            self.gpu_ram = torch.cuda.memory_allocated(0)
            self.max_gpu_ram = torch.cuda.max_memory_allocated(0)
        self._metrics.add(train_acc, key="acc/train")
        self._metrics.add(train_loss, key="loss/train")
        self._metrics.add(test_loss, key="loss/test")
        self._metrics.add(test_acc, key="acc/test")
        self._metrics.add(sparsity, key="sparse/weight")
        self._metrics.add(self._model.structural_sparsity, key="sparse/node")
        self._metrics.add(spasity_index, key="sparse/hm")
        self._metrics.add(np.log(self._model.compressed_size), key="sparse/log_disk_size")
        self._metrics.add(np.mean(self._elapsed_buffer), key="time/gpu_time")
        if torch.cuda.is_available():
            self._metrics.add(torch.cuda.memory_allocated(0), key="cuda/ram_footprint")
            self._metrics.add(torch.cuda.max_memory_allocated(0), key="cuda/max_ram_footprint")
        batch_time = self._metrics.timeit()
        self.batch_time = batch_time

    def train(self):
        """ main training function """

        # setup data output directories:
        setup_directories()
        save_codebase_of_run(self._arguments)
        DATA_MANAGER.write_to_file(
            os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "calling_command.txt"), str(" ".join(sys.argv)))

        # data gathering
        self._fim_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(self._test_loader.dataset, [i for i in range(40)]), batch_size=8, shuffle=False)

        # self._fim_loader = self._test_loader
        epoch = self._metrics._epoch
        self._model.train()
        if self._arguments.structured_prior == 1:
            # get structured criterion
            from models.criterions.CroPitStructured import StructuredEFGit
            from models.criterions.CroPitStructured import StructuredEFG
            criterion = StructuredEFG(limit=0.5, model=self._model)
            # criterion = StructuredEFGit(limit=self._arguments.pruning_limit - 0.2, model=self._model)
            criterion.prune(train_loader=self._train_loader, manager=DATA_MANAGER, percentage=0.5)
            self._optimizer = find_right_model(OPTIMS, self._arguments.optimizer,
                                               params=self._model.parameters(),
                                               lr=self._arguments.learning_rate,
                                               weight_decay=self._arguments.l2_reg)
            self._metrics.model_to_tensorboard(self._model, timestep=epoch)

        try:

            self.out(
                f"{PRINTCOLOR_BOLD}Started training{PRINTCOLOR_END}"
            )

            # if self._arguments.skip_first_plot:
            #     self._metrics.handle_weight_plotting(0, trainer_ns=self)

            if "Early" in self._arguments.prune_criterion:

                while self._stable == False:
                    self.out("Network has not transitioned into Lazy Kernel Regime")
                    self.out(f"\n\n{PRINTCOLOR_BOLD}EPOCH {epoch} {PRINTCOLOR_END} \n\n")

                    # Check for loop exit
                    if epoch == self._arguments.prune_to or self._stable == True:
                        self._stable = True
                        # Train the pruned model using OCCLR
                        self._scheduler = OneCycleLR(self._optimizer, max_lr=self._arguments.learning_rate,
                                                     steps_per_epoch=len(self._train_loader),
                                                     epochs=self._arguments.epochs)
                        break
                    # do epoch
                    self._epoch_iteration()

                    epoch += 1
                    # self._metrics.handle_weight_plotting(epoch, trainer_ns=self)
            else:
                self._stable = True
            # if snip we prune before training
            if self._arguments.prune_criterion in SINGLE_SHOT:
                self._criterion.prune(self._arguments.pruning_limit,
                                      train_loader=self._test_loader,
                                      manager=DATA_MANAGER)
                self._scheduler = OneCycleLR(self._optimizer, max_lr=self._arguments.learning_rate,
                                             steps_per_epoch=len(self._train_loader), epochs=self._arguments.epochs)
                if self._arguments.prune_criterion in STRUCTURED_SINGLE_SHOT:
                    self._optimizer = find_right_model(OPTIMS, self._arguments.optimizer,
                                                       params=self._model.parameters(),
                                                       lr=self._arguments.learning_rate,
                                                       weight_decay=self._arguments.l2_reg)
                    self._scheduler = OneCycleLR(self._optimizer, max_lr=self._arguments.learning_rate,
                                                 steps_per_epoch=len(self._train_loader), epochs=self._arguments.epochs)
                    self._metrics.model_to_tensorboard(self._model, timestep=epoch)
                if next(self._model.parameters()).is_cuda:
                    pass
                else:
                    self._model.cuda()

            # do training
            self.epoch = epoch
            for epoch in range(epoch, self._arguments.epochs + epoch):
                self.out(f"\n\n{PRINTCOLOR_BOLD}EPOCH {epoch} {PRINTCOLOR_END} \n\n")
                self._handle_pruning(epoch)

                # do epoch
                self._epoch_iteration()

                # save what needs to be saved
                self._handle_backing_up(epoch)

            if self._arguments.skip_first_plot:
                self._metrics.handle_weight_plotting(epoch + 1, trainer_ns=self)

            # example last save
            save_models([self._model, self._metrics], "finished")

        except KeyboardInterrupt as e:
            self.out(f"Killed by user: {e} at {time.time()}")
            save_models([self._model, self._metrics], f"KILLED_at_epoch_{epoch}")
            sys.stdout.flush()
            DATA_MANAGER.write_to_file(
                os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "log.txt"), self._metrics.log)
            self._writer.close()
            exit(69)
        except Exception as e:
            self._writer.close()
            report_error(e, self._model, epoch, self._metrics)
        # flush prints
        sys.stdout.flush()
        DATA_MANAGER.write_to_file(
            os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "log.txt"), self._metrics.log)
        self._writer.close()

    def _handle_backing_up(self, epoch):
        if (epoch % self._arguments.save_freq) == 0 and epoch > 0:
            self.out("\nSAVING...\n")
            save_models(
                [self._model, self._metrics],
                f"save_at_epoch_{epoch}"
            )
        sys.stdout.flush()
        DATA_MANAGER.write_to_file(
            os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "log.txt"),
            self._metrics.log
        )

    def _handle_pruning(self, epoch):
        if self._is_pruning_time(epoch):
            if self._is_not_finished_pruning():
                self.out("\nPRUNING...\n")
                # Here we call SNIP-it
                self._criterion.prune(
                    percentage=self._arguments.pruning_rate,
                    train_loader=self._train_loader,
                    manager=DATA_MANAGER
                )
                if self._arguments.prune_criterion in DURING_TRAINING:
                    self._optimizer = find_right_model(
                        OPTIMS, self._arguments.optimizer,
                        params=self._model.parameters(),
                        lr=self._arguments.learning_rate,
                        weight_decay=self._arguments.l2_reg
                    )
                    self._metrics.model_to_tensorboard(self._model, timestep=epoch)
                # Only set the CLR when we get the final pruned model
                if not self._is_not_finished_pruning():
                    self._scheduler = OneCycleLR(self._optimizer, max_lr=self._arguments.learning_rate,
                                                 steps_per_epoch=len(self._train_loader),
                                                 epochs=self._arguments.epochs - epoch)

                if self._model.is_rewindable:
                    self.out("rewinding weights to checkpoint...\n")
                    self._model.do_rewind()

                if next(self._model.parameters()).is_cuda:
                    pass
                else:
                    self._model.cuda()

            if self._model.is_growable:
                self.out("growing too...\n")
                self._criterion.grow(self._arguments.growing_rate)

        if self._is_checkpoint_time(epoch):
            self.out(f"\nCreating weights checkpoint at epoch {epoch}\n")
            self._model.save_rewind_weights()

    def _is_not_finished_pruning(self):
        if self._arguments.prune_criterion not in DURING_TRAINING and self._model.pruned_percentage > self._arguments.pruning_limit - 0.005:
            return False
        if self._criterion.steps is not None and len(self._criterion.steps) == 0:
            return False
        return self._arguments.pruning_limit > self._model.pruned_percentage \
               or \
               (
                       self._arguments.prune_criterion in DURING_TRAINING
                       and
                       self._arguments.pruning_limit > self._model.structural_sparsity
               )

    @staticmethod
    def _get_accuracy(output, y):
        # predictions = torch.round(output)
        predictions = output.argmax(dim=-1, keepdim=True).view_as(y)
        correct = y.eq(predictions).sum().item()
        return correct / output.shape[0]

    def _is_checkpoint_time(self, epoch: int):
        return epoch == self._arguments.rewind_to and self._model.is_rewindable

    def _is_pruning_time(self, epoch: int):
        if self._arguments.prune_criterion == "EmptyCrit":
            return False
        # Ma bet ballech abel ma ye2ta3 prune_delay epochs
        epoch -= self._arguments.prune_delay
        print(self._arguments.prune_freq)
        if self._arguments.prune_freq != 0:
            return (epoch % self._arguments.prune_freq) == 0 and \
                   epoch >= 0 and \
                   self._model.is_maskable and \
                   self._arguments.prune_criterion not in SINGLE_SHOT
        else:
            return (epoch % self.epoch) == 0 and \
                   epoch >= 0 and \
                   self._model.is_maskable and \
                   self._arguments.prune_criterion not in SINGLE_SHOT

    def _check_exit_conditions_epoch_iteration(self, patience=1):

        time_passed = datetime.now() - DATA_MANAGER.actual_date
        # check if runtime is expired
        if (time_passed.total_seconds() > (self._arguments.max_training_minutes * 60)) \
                and \
                self._arguments.max_training_minutes > 0:
            raise KeyboardInterrupt(
                f"Process killed because {self._arguments.max_training_minutes} minutes passed "
                f"since {DATA_MANAGER.actual_date}. Time now is {datetime.now()}")
        if patience == 0:
            raise NotImplementedError("feature to implement",
                                      KeyboardInterrupt("Process killed because patience is zero"))

import inspect
import re
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from markdown import markdown
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from models.criterions.SNIP import SNIP

"""
Util classes regarding data collection, management, printing and storing
"""

class Metric(list):
    def __init__(self, key, first_value):
        super().__init__()
        self.key = key
        self.type = type(first_value)
        self.precision = f"{{:^{len(key) + 2}.{7}{'s' if isinstance(self.type, str) else 'f'}}}"
        self.append(first_value)

    def append(self, item):
        if not isinstance(item, self.type):
            raise TypeError(f'item of type {type(item)} is not of expected type {self.type}')
        super(Metric, self).append(item)

    @property
    def last(self):
        return self.__getitem__(len(self) - 1)


class Metrics:

    def __init__(self):
        # redirect sys.stdout to a buffer
        self._data = defaultdict(Metric)
        self.log = ""
        self._writer: SummaryWriter = None
        self._epoch = 0
        self.batch_train = 0
        self._batch_test = 0
        self._batch_size = 0
        self._start_time = 0
        self._last_time = time.time()
        self._eval_freq = 0

    def write_arguments(self, args):

        txt = '<table> <thead> <tr> <td> <strong> Argument </strong> </td> <td> <strong> Value </strong> </td> </tr> </thead>'
        txt += ' <tbody> '
        for name, var in vars(args).items():
            txt += '<tr> <td> <code>' + str(name) + ' </code> </td> ' + '<td> <code> ' + str(
                var) + ' </code> </td> ' + '<tr> '
        txt += '</tbody> </table>'
        self._writer.add_text('args', markdown(txt))

    def init_training(self, writer: SummaryWriter):
        self._writer = writer
        self._time = time.time()

    def timeit(self):
        now = time.time()
        # self.add(now - self._start_time, key="sparse/cum_time")
        diff = now - self._last_time / self._eval_freq
        self.add(diff, key="time/batch_time")
        self._last_time = now

        return diff


    def update_batch(self, train):
        if train:
            self.batch_train += 1
            if self._batch_test != 0:
                self._batch_test = 0
        else:
            self._batch_test += 1

    def update_epoch(self):
        self._epoch += 1
        return [], []

    def log_line(self, *line, **kwargs):
        self.log += "\n"
        for elem in line:
            self.log += f"{str(elem)} "
        print(*line, **kwargs)

    def __repr__(self):
        return str(self.log) + "\n\n" + str(self._data)

    def add(self, value, key="", log=True):
        if key == "":
            get_rid_of = ['add(', ',', ')', '\n']
            calling_code = inspect.getouterframes(inspect.currentframe())[1][4][0]
            calling_code = calling_code[calling_code.index('add'):]
            for garbage in get_rid_of:
                calling_code = calling_code.replace(garbage, '')
            var_names = calling_code.split()
            key = var_names[0]
        if not (key in self._data):
            self._data[key] = Metric(key, value)
        else:
            self._data[key].append(value)

        if not (self._writer is None) and log:
            self._writer.add_scalar(key, value, self.batch_train * self._batch_size)

    @property
    def printable_last(self):

        header = "$  "
        data = "$"
        for i, (key, metric) in enumerate(self._data.items()):
            header += str(key) + "  |  "
            value = metric.last
            precision = metric.precision
            if (len(str(value).split(".")[0]) > 6):
                precision = precision.replace("7", "1")
            data += " {} |".format(precision).format(value)

        return self._wrap_printable(header, data)

    def _wrap_printable(self, header, data):
        if (len(header) > 200):

            block_indices = [m.start() for m in re.finditer(' \| ', header)]
            cutoff = block_indices[int(np.argmin([abs(b - 200) for b in block_indices]))]

            return "{}\n{}\n{}".format(header[:cutoff], data[:cutoff],
                                       self._wrap_printable(header[cutoff:], data[cutoff:]))
        else:
            return "${}\n${}".format(header, data)

    @property
    def json(self):
        return {key: dict(value) for key, value in self._data.items()}

    def state_dict(self):
        return self.saveable

    @property
    def saveable(self):
        return {"_data": self._data,
                "log": self.log,
                "_epoch": self._epoch,
                "_batch_train": self.batch_train,
                "_batch_test": self._batch_test,
                "_batch_size": self._batch_size,
                "_start_time": self._start_time,
                "_last_time": self._last_time,
                "_eval_freq": self._eval_freq}

    def load(self,
             _data=None,
             log=None,
             _epoch=None,
             _batch_train=None,
             _batch_test=None,
             _batch_size=None,
             _start_time=None,
             _last_time=None,
             _eval_freq=None
             ):
        self._data = _data
        self.log = log
        self._epoch = _epoch
        self.batch_train = _batch_train
        self._batch_test = _batch_test
        self._batch_size = _batch_size
        self._start_time = _start_time
        self._last_time = _last_time
        self._eval_freq = _eval_freq

    def model_to_tensorboard(self, model, timestep=0):
        txt = str(model).replace("\n", "<br>&emsp;").replace("    ", "&emsp;&emsp;")
        self._writer.add_text('architecture', markdown(txt), global_step=timestep)

    def handle_weight_plotting(self, epoch: int, trainer_ns):

        try:

            self.log_line("plotting..")

            plt.clf()
            plt.close()

            canvas, fig, g, histograms, layer_active, layer_nodes, sparsities = self.extract_from_layers(epoch,
                                                                                                         trainer_ns)

            if trainer_ns._arguments.disable_netplot:
                print("doing netplot")
                self._build_network_representation(canvas, epoch, fig, g, layer_active, layer_nodes)

            if trainer_ns._arguments.disable_histograms:
                print("doing histograms")
                self._writer.add_histogram("weight/magnitude", histograms, epoch)
                self._writer.add_histogram("layer/sparsity", sparsities, epoch)

            if trainer_ns._arguments.disable_confusion:
                print("doing confusion matrix")
                self._write_confusion_matrix(epoch, trainer_ns)

            if trainer_ns._arguments.disable_saliency:
                print("doing saliencies")
                self._write_saliency(epoch, trainer_ns)
                if not trainer_ns._arguments.l0:
                    self._write_snip(epoch, trainer_ns)

            self._writer.flush()
        except Exception as e:
            print(f"ERROR IN PLOTTING {e}")
            raise e
        print("finished plotting")

    def extract_from_layers(self, epoch, trainer_ns):

        # setup
        fig = plt.figure()
        canvas = FigureCanvasAgg(fig)
        g = nx.Graph()
        layer_nodes = {}
        layer_active = {}
        weight_histograms = torch.zeros([0]).float()
        sparsities = torch.zeros([0]).float()

        i = 0  # have to count manually becauseof l0
        for name, element in (
                trainer_ns._model.named_modules() if trainer_ns._arguments.l0 else trainer_ns._model.mask.items()):

            # is it a leaf node?
            if trainer_ns._arguments.l0 and (not "L0" in str(type(element))):
                continue
            i += 1

            # get flattend weight layer
            flattend = self._get_flattend(element, name, trainer_ns)

            # append to histogram data
            if trainer_ns._arguments.disable_histograms:
                sparsities, weight_histograms = self._get_histogram_data(flattend, sparsities, weight_histograms)

            # handle weight matrix
            graph, plottable = self._write_plottable_weight_matrix(element, epoch, name, trainer_ns)

            # handle graph
            self.handle_graph_building(graph, i, layer_active, layer_nodes, plottable)

        return canvas, fig, g, weight_histograms, layer_active, layer_nodes, sparsities

    def _get_flattend(self, element, name, trainer_ns):
        if trainer_ns._arguments.l0:
            flattend = element.sample_weights().flatten()
        else:
            # all_scores, grads, log10, norm_factor = trainer_ns._criterion.get_weight_saliencies(trainer_ns._train_loader)
            # breakpoint()
            # flattend = grads[name].flatten().log()
            flattend = trainer_ns._model.state_dict()[name].flatten()
        return flattend

    def handle_graph_building(self, graph, i, layer_active, layer_nodes, plottable):
        if graph:
            # extract nodes
            nodes = [str(i) + "_" + str(x) for x in range(plottable.shape[-1])]
            layer_nodes[i] = nodes
            nodes = [str(i + 1) + "_" + str(x) for x in range(plottable.shape[0])]
            layer_nodes[i + 1] = nodes

            # extract edges
            layer_active[i] = plottable.nonzero().numpy()

    def _write_plottable_weight_matrix(self, element, epoch, name, trainer_ns):

        # get matrix from right location
        if trainer_ns._arguments.l0:
            plottable = element.sample_weights().cpu()
            if len(plottable.shape) == 2:
                plottable = plottable.t()
        else:
            plottable = trainer_ns._model.state_dict()[name].cpu()

        # get matrix in right sizes
        graph = trainer_ns._arguments.disable_netplot
        first_dim = plottable.shape[0]
        if len(plottable.shape) == 4:
            graph = False
            first_dim = int(max(element.shape))

        plottable_ = plottable.view(first_dim, -1)

        if trainer_ns._arguments.disable_weightplot:
            # reformat image
            plottable = (plottable_ / 6) * 255
            plottable = torch.stack(
                [(plottable * x).long() for x in [plottable < 0, plottable > 0]] + [
                    ((plottable == 0).long() * 255)])
            plottable = plottable.abs().numpy()
            trainer_ns._writer.add_image("weight/" + name, plottable, epoch)

        return graph, plottable_

    def _get_histogram_data(self, flattend, sparsities, weight_histograms):

        # add to sparsities
        sparsities = torch.cat([sparsities,
                                torch.zeros([1]) + 100 * (
                                        flattend.nonzero().flatten().shape[0] / flattend.shape[0])])

        # add to weight histograms
        weight_histograms = torch.cat([weight_histograms, 100 * flattend.cpu()])
        return sparsities, weight_histograms

    def _write_confusion_matrix(self, epoch, trainer_ns):

        # test
        x, y = next(iter(trainer_ns._test_loader))
        _, _, out = trainer_ns._forward_pass(x.to(trainer_ns._device).float(), y.to(trainer_ns._device), train=False)

        # build matrix
        cm = confusion_matrix(y.detach()
                              .cpu()
                              .numpy(),
                              out.detach()
                              .argmax(dim=-1, keepdim=True)
                              .view_as(y)
                              .cpu()
                              .numpy())
        cm = 1 - (cm / (trainer_ns._arguments.batch_size / trainer_ns._arguments.output_dim))

        # write
        self._writer.add_image("network/confusion", cm.reshape(1, *cm.shape), epoch)

    def _write_saliency(self, epoch, trainer_ns):
        img = trainer_ns.saliency.get_grad()
        self._writer.add_image("network/saliency", img, epoch)

    def _write_snip(self, epoch, trainer_ns):

        all_scores, grads_abs, log10, norm_factor = SNIP(model=trainer_ns._model,
                                                         device=trainer_ns._device).get_weight_saliencies(
            trainer_ns._train_loader)

        fig = plt.figure()
        canvas = FigureCanvasAgg(fig)

        scores = log10

        if len(scores) > 5e6:
            indices = np.random.rand(len(scores)) > (1 - (5e5 / len(scores)))
            indices[-int(len(scores) / 150):] = 1
            scores = scores[indices]

        plt.plot(scores.cpu().numpy(), label="sorted_weight_relevance")
        plt.ylim((-11, 0))
        plt.xticks([i * (len(scores) // 8) for i in range(0, 9)],
                   [str(int(100 * (i * (len(scores) // 8)) / len(scores))) + "%" for i in range(0, 9)])
        plt.grid()

        picture = self.plt_to_tensor(canvas, fig)

        self._writer.add_image("track/weight_saliency", picture, epoch)

    def _build_network_representation(self, canvas, epoch, fig, g, layer_active, layer_nodes):

        # build nodes
        max_length = max([len(layer) for layer in layer_nodes.values()])
        for i, layer in layer_nodes.items():
            len_layer = len(layer)
            for j, node in enumerate(layer):
                g.add_node(node, pos=(i, j * (max_length / len_layer)))

        # build edges
        for i, active in layer_active.items():
            edges = [(str(i) + "_" + str(int(connection[1])), str(i + 1) + "_" + str(int(connection[0]))) for
                     connection in active]
            for edge in edges:
                g.add_edge(*edge, weight=0.01)

        # draw
        pos = nx.get_node_attributes(g, 'pos')
        nx.draw(g, pos)  # with_labels=True
        picture = self.plt_to_tensor(canvas, fig)

        # write
        self._writer.add_image("network/visualisation", picture, epoch)

    def plt_to_tensor(self, canvas, fig):
        plt.draw()
        canvas.draw()
        _, (width, height) = canvas.print_to_buffer()
        s = canvas.tostring_rgb()
        plt.close(fig)
        picture = np.fromstring(s, dtype='uint8').reshape((height, width, 3))
        picture = torch.from_numpy(np.moveaxis(picture, -1, 0))
        return picture

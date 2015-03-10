from __future__ import division, print_function
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from datetime import datetime, timedelta
from numpy.random import rand
from time import time
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import (InputLayer, LSTMLayer, ReshapeLayer, Layer,
                            ConcatLayer, ElemwiseSumLayer, DenseLayer,
                            get_all_layers, Conv1DLayer, FeaturePoolLayer)
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.utils import floatX
from lasagne.updates import nesterov_momentum
from .source import quantize
theano.config.compute_test_value = 'raise'

"""
rsync command: 
rsync -uvzr --progress --exclude '.git' --exclude '.ropeproject' --exclude 'ipynb_checkpoints' /home/jack/workspace/python/neuralnilm/ /mnt/sshfs/imperial/workspace/python/neuralnilm/
"""

class ansi:
    # from dnouri/nolearn/nolearn/lasagne.py
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'


class TrainingError(Exception):
    pass


######################## Neural network class ########################
class Net(object):
    # Much of this code is adapted from craffel/nntools/examples/lstm.py

    def __init__(self, source, layers_config, 
                 updates=partial(nesterov_momentum, learning_rate=0.1),
                 experiment_name="", 
                 validation_interval=10, 
                 save_plot_interval=100,
                 loss_function=lasagne.objectives.mse,
                 X_processing_func=lambda X: X,
                 layer_changes=None,
                 seed=42,
                 epoch_callbacks=None
    ):
        """
        Parameters
        ----------
        layers_config : list of dicts.  Keys are:
            'type' : BLSTMLayer or a subclass of lasagne.layers.Layer
            'num_units' : int
        """
        print("Initialising network...")
        if seed is not None:
            np.random.seed(seed)
        self.source = source
        self.updates = updates
        self.experiment_name = experiment_name
        self.validation_interval = validation_interval
        self.save_plot_interval = save_plot_interval
        self.loss_function = loss_function
        self.X_processing_func = X_processing_func
        self.layer_changes = {} if layer_changes is None else layer_changes
        self.epoch_callbacks = {} if epoch_callbacks is None else epoch_callbacks

        self.generate_validation_data_and_set_shapes()

        self.validation_costs = []
        self.training_costs = []
        self.layers = []

        # Shape is (number of examples per batch,
        #           maximum number of time steps per example,
        #           number of features per example)
        self.layers.append(InputLayer(shape=self.input_shape))
        self.add_layers(layers_config)
        print("Done initialising network for", self.experiment_name)

    def generate_validation_data_and_set_shapes(self):
        # Generate a "validation" sequence whose cost we will compute
        self.X_val, self.y_val = self.source.validation_data()
        self.input_shape = self.X_val.shape
        self.output_shape = self.y_val.shape
        self.n_seq_per_batch = self.input_shape[0]

    def add_layers(self, layers_config):
        for layer_config in layers_config:
            layer_type = layer_config.pop('type')

            # Reshape if necessary
            prev_layer_output_shape = self.layers[-1].get_output_shape()
            n_dims = len(prev_layer_output_shape)
            n_features = prev_layer_output_shape[-1]
            if layer_type in [LSTMLayer, BLSTMLayer, DimshuffleLayer]:
                if n_dims == 2:
                    seq_length = int(prev_layer_output_shape[0] / 
                                     self.source.n_seq_per_batch)
                    shape = (self.source.n_seq_per_batch, 
                             seq_length,
                             n_features)
                    self.layers.append(ReshapeLayer(self.layers[-1], shape))
            elif layer_type in [DenseLayer]:
                if n_dims == 3:
                    # The prev layer_config was a time-aware layer_config, so reshape to 2-dims.
                    seq_length = prev_layer_output_shape[1]
                    shape = (self.source.n_seq_per_batch * seq_length,
                             n_features)
                    self.layers.append(ReshapeLayer(self.layers[-1], shape))

            # Init new layer_config
            print('Initialising layer_config :', layer_type)
            self.layers.append(layer_type(self.layers[-1], **layer_config))

        # Reshape output if necessary...
        if self.layers[-1].get_output_shape() != self.output_shape:
            self.layers.append(ReshapeLayer(self.layers[-1], self.output_shape))

    def print_net(self):
        for layer in self.layers:
            print(layer)
            try:
                print(" Input shape: ", layer.input_shape)
            except:
                pass
            print("Output shape: ", layer.get_output_shape())
            print()

    def compile(self):
        input = T.tensor3('input')
        target_output = T.tensor3('target_output')

        # add test values
        input.tag.test_value = floatX(rand(*self.input_shape))
        target_output.tag.test_value = floatX(rand(*self.output_shape))

        print("Compiling Theano functions...")
        loss = self.loss_function(
            self.layers[-1].get_output(input), target_output)

        # Use NAG for training
        all_params = lasagne.layers.get_all_params(self.layers[-1])
        updates = self.updates(loss, all_params)

        # Theano functions for training, getting output, and computing loss
        self.train = theano.function(
            [input, target_output],
            loss, updates=updates, on_unused_input='warn',
            allow_input_downcast=True)

        self.y_pred = theano.function(
            [input], self.layers[-1].get_output(input), on_unused_input='warn',
            allow_input_downcast=True)

        self.compute_cost = theano.function(
            [input, target_output], loss, on_unused_input='warn',
            allow_input_downcast=True)

        print("Done compiling Theano functions.")

    def fit(self, n_iterations=None):
        # Training loop. Need to wrap this in a try-except loop so
        # we can always call self.source.stop()
        self.source.start()
        try:
            self._training_loop(n_iterations)
        except:
            raise
        finally:
            self.source.stop()      

    def _change_layers(self, epoch):
        self.source.stop()
        self.source.empty_queue()
        print("Changing layers...\nOld architecture:")
        self.print_net()        
        layer_changes = self.layer_changes[epoch]
        for layer_to_remove in range(layer_changes['remove_from'], 0):
            print("Removed", self.layers.pop(layer_to_remove))
        if 'callback' in layer_changes:
            layer_changes['callback'](self, epoch)
        self.add_layers(layer_changes['new_layers'])
        print("New architecture:")
        self.print_net()
        self.compile()
        self.source.start()

    def _training_loop(self, n_iterations):
        # Adapted from dnouri/nolearn/nolearn/lasagne.py
        print("""
 Epoch  |  Train cost  |  Valid cost  |  Train / Val  | Sec per epoch
--------|--------------|--------------|---------------|---------------\
""")
        validation_cost = (self.validation_costs[-1] if self.validation_costs 
                           else None)

        epoch = len(self.training_costs)
        while epoch != n_iterations:
            epoch = len(self.training_costs)
            if epoch in self.layer_changes:
                self._change_layers(epoch)
            if epoch in self.epoch_callbacks:
                self.epoch_callbacks[epoch](self, epoch)
            t0 = time() # for calculating training duration
            X, y = self.source.queue.get(timeout=30)
            train_cost = self.train(X, y).flatten()[0]
            self.training_costs.append(train_cost)
            if not epoch % self.validation_interval:
                validation_cost = self.compute_cost(self.X_val, self.y_val).flatten()[0]
                self.validation_costs.append(validation_cost)
            if not epoch % self.save_plot_interval:
                self.plot_costs(save=True)
                self.plot_estimates(save=True)
                self.save_params()
                self.save_activations()
            # Print progress
            duration = time() - t0
            is_best_train = train_cost == min(self.training_costs)
            is_best_valid = validation_cost == min(self.validation_costs)
            print("  {:>5} |  {}{:>10.6f}{}  |  {}{:>10.6f}{}  |"
                  "  {:>11.6f}  |  {:>3.1f}s".format(
                      epoch,
                      ansi.BLUE if is_best_train else "",
                      train_cost,
                      ansi.ENDC if is_best_train else "",
                      ansi.GREEN if is_best_valid else "",
                      validation_cost,
                      ansi.ENDC if is_best_valid else "",
                      train_cost / validation_cost,
                      duration
            ))
            if np.isnan(train_cost):
                raise TrainingError("training cost is NaN!")

    def plot_costs(self, save=False):
        fig, ax = plt.subplots(1)
        ax.plot(self.training_costs, label='Training')
        validation_x = range(0, len(self.training_costs), self.validation_interval)
        n_validations = min(len(validation_x), len(self.validation_costs))
        ax.plot(validation_x[:n_validations], 
                self.validation_costs[:n_validations],
                label='Validation')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.legend()
        fig.tight_layout()
        if save:
            filename = self._plot_filename('costs', include_epochs=False)
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        return ax

    def plot_estimates(self, sequences=None, **kwargs):
        if sequences is None:
            sequences = range(min(self.n_seq_per_batch, 5))
        for seq_i in sequences:
            self._plot_estimates(seq_i=seq_i, **kwargs)

    def _plot_estimates(self, save=False, seq_i=0, use_validation_data=True, 
                        X=None, y=None, linewidth=0.2):
        fig, axes = plt.subplots(3)
        if X is None or y is None:
            if use_validation_data:
                X, y = self.X_val, self.y_val
            else:
                X, y = self.source.queue.get(timeout=30)
        y_predictions = self.y_pred(X)

        n = len(y_predictions[seq_i, :, :])
        axes[0].set_title('Network output')
        axes[0].plot(y_predictions[seq_i, :, :], linewidth=linewidth)
        axes[0].set_xlim([0, n])
        axes[1].set_title('Target')
        axes[1].plot(y[seq_i, :, :], linewidth=linewidth)
        # alpha: lower = more transparent
        axes[1].legend(self.source.get_labels(), fancybox=True, framealpha=0.5,
                       prop={'size': 6})
        axes[1].set_xlim([0, n])
        axes[2].set_title('Network input')
        start, end = self.source.inside_padding()
        axes[2].plot(X[seq_i, start:end, :], linewidth=linewidth)
        axes[2].set_xlim([0, self.source.seq_length])
        for ax in axes:
            ax.grid(True)
        fig.tight_layout()
        if save:
            filename = self._plot_filename('estimates', end_string=seq_i)
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        return axes

    def _plot_filename(self, string, include_epochs=True, end_string=""):
        end_string = str(end_string)
        return (
            self.experiment_name + ("_" if self.experiment_name else "") + 
            string +
            ("_{:d}epochs".format(self.n_epochs()) if include_epochs else "") +
            ("_" if end_string else "") + end_string +
            ".pdf")

    def n_epochs(self):
        return max(len(self.training_costs) - 1, 0)

    def save_params(self, filename=None):
        """
        Save it to HDF in the following format:
            /epoch<N>/L<I>_<type>/P<I>_<name>

        Parameters
        ----------
        layers : list of ints
        """
        # Process function parameters
        if filename is None:
            filename = self.experiment_name + ".hdf5"

        mode = 'w' if self.n_epochs() == 0 else 'a'
        f = h5py.File(filename, mode=mode)
        epoch_name = 'epoch{:06d}'.format(self.n_epochs())
        try:
            epoch_group = f.create_group(epoch_name)
        except ValueError as exception:
            print("Not saving params because", exception)
            f.close()
            return
            
        layers = get_all_layers(self.layers[-1])
        layers.reverse()
        for layer_i, layer in enumerate(layers):
            params = layer.get_params()
            if not params:
                continue
            layer_name = 'L{:02d}_{}'.format(layer_i, layer.__class__.__name__)
            layer_group = epoch_group.create_group(layer_name)
            for param_i, param in enumerate(params):
                param_name = 'P{:02d}'.format(param_i)
                if param.name:
                    param_name += "_" + param.name
                data = param.get_value()
                layer_group.create_dataset(param_name, data=data, compression="gzip")
            
        f.close()

    def save_activations(self):
        filename = self.experiment_name + "_activations.hdf5"
        mode = 'w' if self.n_epochs() == 0 else 'a'
        f = h5py.File(filename, mode=mode)
        epoch_name = 'epoch{:06d}'.format(self.n_epochs())
        try:
            epoch_group = f.create_group(epoch_name)
        except ValueError as exception:
            print("Not saving params because", exception)
            f.close()
            return

        layers = get_all_layers(self.layers[-1])
        layers.reverse()
        for layer_i, layer in enumerate(layers):
            # We only care about layers with params
            if not (layer.get_params() or isinstance(layer, FeaturePoolLayer)):
                continue

            output = layer.get_output(self.X_val).eval()
            n_features = output.shape[-1]
            seq_length = int(output.shape[0] / self.source.n_seq_per_batch)

            if isinstance(layer, DenseLayer):
                shape = (self.source.n_seq_per_batch, seq_length, n_features)
                output = output.reshape(shape)
            elif isinstance(layer, Conv1DLayer):
                output = output.transpose(0, 2, 1)

            layer_name = 'L{:02d}_{}'.format(layer_i, layer.__class__.__name__)
            epoch_group.create_dataset(layer_name, data=output, compression="gzip")

        # save validation data
        if self.n_epochs() == 0:
            f.create_dataset('validation_data', data=self.X_val, compression="gzip")

        f.close()

            
def BLSTMLayer(l_previous, num_units, **kwargs):
    # setup forward and backwards LSTM layers.  Note that
    # LSTMLayer takes a backwards flag. The backwards flag tells
    # scan to go backwards before it returns the output from
    # backwards layers.  It is reversed again such that the output
    # from the layer is always from x_1 to x_n.

    # If learn_init=True then you can't have multiple
    # layers of LSTM cells.
    l_fwd = LSTMLayer(l_previous, num_units, backwards=False, **kwargs)
    l_bck = LSTMLayer(l_previous, num_units, backwards=True, **kwargs)
    return ElemwiseSumLayer([l_fwd, l_bck])


class DimshuffleLayer(Layer):
    def __init__(self, input_layer, pattern):
        super(DimshuffleLayer, self).__init__(input_layer)
        self.pattern = pattern

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[i] for i in self.pattern])

    def get_output_for(self, input, *args, **kwargs):
        return input.dimshuffle(self.pattern)

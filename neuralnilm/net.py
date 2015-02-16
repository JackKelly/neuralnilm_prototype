from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from numpy.random import rand
from time import time
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import (InputLayer, LSTMLayer, ReshapeLayer, Layer,
                            ConcatLayer, ElemwiseSumLayer, DenseLayer)
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.utils import floatX
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

######################## Neural network class ########################
class Net(object):
    # Much of this code is adapted from craffel/nntools/examples/lstm.py

    def __init__(self, source, layers_config, learning_rate=1e-1, 
                 experiment_name="", 
                 validation_interval=10, save_plot_interval=100,
                 loss_function=lasagne.objectives.mse):
        """
        Parameters
        ----------
        layers_config : list of dicts.  Keys are:
            'type' : BLSTMLayer or a subclass of lasagne.layers.Layer
            'num_units' : int
        """
        print("Initialising network...")
        self.source = source
        self.learning_rate = learning_rate
        self.experiment_name = experiment_name
        self.validation_interval = validation_interval
        self.save_plot_interval = save_plot_interval
        self.loss_function = loss_function

        self.input_shape = source.input_shape()
        self.output_shape = source.output_shape()
        self.n_seq_per_batch = self.input_shape[0]
        self.validation_costs = []
        self.training_costs = []
        self.layers = []

        # Shape is (number of examples per batch,
        #           maximum number of time steps per example,
        #           number of features per example)
        self.layers.append(InputLayer(shape=self.input_shape))

        for layer_config in layers_config:
            layer_type = layer_config.pop('type')

            # Reshape if necessary
            prev_layer_output_shape = self.layers[-1].get_output_shape()
            n_dims = len(prev_layer_output_shape)
            n_features = prev_layer_output_shape[-1]
            if layer_type in [LSTMLayer, BLSTMLayer, 
                              SubsampleLayer, DimshuffleLayer]:
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

        # Generate a "validation" sequence whose cost we will compute
        self.X_val, self.y_val = self.source.validation_data()
        print("Done initialising network.")

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
        cost = self.loss_function(
            self.layers[-1].get_output(input), target_output)

        # Use NAG for training
        all_params = lasagne.layers.get_all_params(self.layers[-1])
        updates = lasagne.updates.nesterov_momentum(
            cost, all_params, self.learning_rate)

        # Theano functions for training, getting output, and computing cost
        self.train = theano.function(
            [input, target_output],
            cost, updates=updates, on_unused_input='warn',
            allow_input_downcast=True)

        self.y_pred = theano.function(
            [input], self.layers[-1].get_output(input), on_unused_input='warn',
            allow_input_downcast=True)

        self.compute_cost = theano.function(
            [input, target_output], cost, on_unused_input='warn',
            allow_input_downcast=True)

        print("Done compiling Theano functions.")

    def fit(self, n_iterations=None):
        # Training loop
        self.source.start()
        try:
            self._training_loop(n_iterations)
        except:
            raise
        finally:
            self.source.stop()      

    def _training_loop(self, n_iterations):
        # Adapted from dnouri/nolearn/nolearn/lasagne.py
        print("""
 Epoch  |  Train cost  |  Valid cost  |  Train / Val  | Sec per epoch
--------|--------------|--------------|---------------|---------------\
""")
        validation_cost = (self.validation_costs[-1] if self.validation_costs 
                           else None)
        i = 0
        while i != n_iterations:
            t0 = time() # for calculating training duration
            X, y = self.source.queue.get(timeout=30)
            train_cost = self.train(X, y).flatten()[0]
            self.training_costs.append(train_cost)
            epoch = len(self.training_costs) - 1
            if not epoch % self.validation_interval:
                validation_cost = self.compute_cost(self.X_val, self.y_val).flatten()[0]
                self.validation_costs.append(validation_cost)
            if not epoch % self.save_plot_interval:
                self.plot_costs(save=True)
                for seq_i in range(self.n_seq_per_batch):
                    self.plot_estimates(save=True, seq_i=seq_i)
            # Print progress
            duration = time() - t0
            is_best_train = train_cost == min(self.training_costs)
            is_best_valid = validation_cost == min(self.validation_costs)
            print("  {:>5} |  {}{:>10.6f}{}  |  {}{:>10.6f}{}  |"
                  "  {:>11.6f}  |  {:>3.1f}s".format(
                      len(self.training_costs),
                      ansi.BLUE if is_best_train else "",
                      train_cost,
                      ansi.ENDC if is_best_train else "",
                      ansi.GREEN if is_best_valid else "",
                      validation_cost,
                      ansi.ENDC if is_best_valid else "",
                      train_cost / validation_cost,
                      duration
            ))
            i += 1

    def plot_costs(self, ax=None, save=False):
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, sharex=True)
        ax.plot(self.training_costs, label='Training')
        validation_x = range(0, len(self.training_costs), self.validation_interval)
        ax.plot(validation_x, self.validation_costs, label='Validation')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.legend()
        if save:
            filename = self._plot_filename('costs', include_epochs=False)
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        return ax

    def plot_estimates(self, axes=None, save=False, seq_i=0):
        fig = None
        if axes is None:
            fig, axes = plt.subplots(3, sharex=False)
        X, y = self.source.validation_data()
        y_predictions = self.y_pred(X)
        axes[0].set_title('Appliance estimates')
        axes[0].plot(y_predictions[seq_i,:,:])
        axes[1].set_title('Appliance ground truth')
        axes[1].plot(y[seq_i,:,:])
        axes[2].set_title('Aggregate')
        axes[2].plot(X[seq_i,:,:])
        if save:
            filename = self._plot_filename('estimates', end_string=seq_i)
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        return axes

    def _plot_filename(self, string, include_epochs=True, end_string=""):
        n_epochs = len(self.training_costs)
        end_string = str(end_string)
        return (
            self.experiment_name + ("_" if self.experiment_name else "") + 
            string +
            ("_{:d}epochs".format(n_epochs) if include_epochs else "") + 
            ("_" if end_string else "") + end_string +
            ".eps")


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


class SubsampleLayer(Layer):
    def __init__(self, input_layer, stride):
        if input_layer is not None:
            super(SubsampleLayer, self).__init__(input_layer)
        self.stride = stride

    def get_output_shape_for(self, input_shape):
        assert len(input_shape) == 3
        if input_shape[1] % self.stride:
            raise RuntimeError("Seq length must be exactly divisible by stride.")
        seq_length = int(input_shape[1] / self.stride)
        return (input_shape[0], seq_length, input_shape[2])

    def get_output_for(self, input, *args, **kwargs):
        shape = tuple(list(self.get_output_shape()) + [-1])
        reshaped = input.reshape(shape)
        return reshaped.sum(axis=-1)


class DimshuffleLayer(Layer):
    def __init__(self, input_layer, pattern):
        super(DimshuffleLayer, self).__init__(input_layer)
        self.pattern = pattern

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[i] for i in self.pattern])

    def get_output_for(self, input, *args, **kwargs):
        return input.dimshuffle(self.pattern)


class QuantizeLayer(Layer):
    def __init__(self, input_layer, n_bins=50, all_hot=False, boolean=False):
        super(QuantizeLayer, self).__init__(input_layer)
        self.n_bins = n_bins
        self.all_hot = all_hot
        self.boolean = boolean

    def get_output_shape_for(self, input_shape):
        assert input_shape[2] == 1
        return (input_shape[0], input_shape[1], self.n_bins)

    def get_output_for(self, input, *args, **kwargs):
        output = np.empty(shape=self.get_output_shape())
        for batch_i in range(self.input_shape[0]):
            for i in range(self.input_shape[1]):
                output[batch_i,i,:] = quantize_scalar(
                    input[batch_i,i,0],
                    n_bins=self.n_bins,
                    all_hot=self.all_hot,
                    boolean=self.boolean
                )
        return output


def quantize_scalar(x, n_bins=10, all_hot=False, boolean=True):
    output = np.empty(n_bins) 
    # bin_i = T.floor(x * n_bins).astype('int32')
    # bin_i = T.min([bin_i, n_bins-1])
    bin_i = int(x * n_bins)
    bin_i = min(bin_i, n_bins-1)
    output[bin_i] = 1 if boolean else ((x * n_bins) - bin_i)
    if all_hot:
        output[:bin_i] = 1
    return output

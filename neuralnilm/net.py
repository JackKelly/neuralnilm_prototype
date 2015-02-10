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
from lasagne.layers import (InputLayer, LSTMLayer, ReshapeLayer, 
                            ConcatLayer, ElemwiseSumLayer, DenseLayer)
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.utils import floatX
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

    def __init__(self, source, layers, learning_rate=1e-1, 
                 output_nonlinearity=None, experiment_name="",
                 validation_interval=10, save_plot_interval=100,
                 loss_function=lasagne.objectives.mse):
        """
        Parameters
        ----------
        layers : list of dicts.  Keys are:
            'type' : BLSTMLayer or a subclass of lasagne.layers.Layer
            'num_units' : int
        """
        print("Initialising network...")
        self.source = source
        input_shape = source.input_shape()
        output_shape = source.output_shape()
        self.validation_interval = validation_interval
        self.save_plot_interval = save_plot_interval
        self.validation_costs = []
        self.training_costs = []
        self.experiment_name = experiment_name
        self.loss_function = loss_function

        # Shape is (number of examples per batch,
        #           maximum number of time steps per example,
        #           number of features per example)
        l_previous = InputLayer(shape=input_shape)

        for layer in layers:
            layer_type = layer.pop('type')

            # Reshape if necessary
            n_dims = len(l_previous.get_output_shape())
            if layer_type in [LSTMLayer, BLSTMLayer]:
                if n_dims == 2:
                    shape = (self.source.n_seq_per_batch, 
                             self.source.seq_length, 
                             l_previous.get_output_shape()[-1])
                    l_previous = ReshapeLayer(l_previous, shape)
            elif n_dims == 3:
                # DenseLayer or similar...
                shape = (self.source.n_seq_per_batch * self.source.seq_length, 
                         self.source.n_inputs)
                l_previous = ReshapeLayer(l_previous, shape)

            # Init new layer
            l_previous = layer_type(l_previous, **layer)

        # Reshape output if necessary...
        if l_previous.get_output_shape() == output_shape:
            l_out = l_previous
        else:
            l_out = ReshapeLayer(l_previous, output_shape)

        input = T.tensor3('input')
        target_output = T.tensor3('target_output')

        # add test values
        input.tag.test_value = floatX(rand(*input_shape))
        target_output.tag.test_value = floatX(rand(*output_shape))

        print("Compiling Theano functions...")
        cost = loss_function(l_out.get_output(input), target_output)

        # Use NAG for training
        all_params = lasagne.layers.get_all_params(l_out)
        updates = lasagne.updates.nesterov_momentum(
            cost, all_params, learning_rate)

        # Theano functions for training, getting output, and computing cost
        self.train = theano.function(
            [input, target_output],
            cost, updates=updates, on_unused_input='warn',
            allow_input_downcast=True)

        self.y_pred = theano.function(
            [input], l_out.get_output(input), on_unused_input='warn',
            allow_input_downcast=True)

        self.compute_cost = theano.function(
            [input, target_output], cost, on_unused_input='warn',
            allow_input_downcast=True)

        # Generate a "validation" sequence whose cost we will compute
        self.X_val, self.y_val = self.source.validation_data()
        print("Done initialising network.")

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
                self.plot_estimates(save=True)
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
        if ax is None:
            fig, ax = plt.subplots(1, sharex=True)
        ax.plot(self.training_costs, label='Training')
        validation_x = range(0, len(self.training_costs), self.validation_interval)
        ax.plot(validation_x, self.validation_costs, label='Validation')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.legend()
        filename = self._plot_filename('costs') if save else None
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()
        return ax

    def plot_estimates(self, axes=None, save=False):
        if axes is None:
            fig, axes = plt.subplots(3, sharex=True)
        X, y = self.source.validation_data()
        y_predictions = self.y_pred(X)
        axes[0].set_title('Appliance estimates')
        axes[0].plot(y_predictions[0,:,:])
        axes[1].set_title('Appliance ground truth')
        axes[1].plot(y[0,:,:])
        axes[2].set_title('Aggregate')
        axes[2].plot(X[0,:,:])#, label='Fdiff')
        #axes[1].plot(np.cumsum(X[0,:,1]), label='Cumsum')
        filename = self._plot_filename('estimates') if save else None
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()
        return axes

    def _plot_filename(self, string):
        return (
            self.experiment_name + ("_" if self.experiment_name else "") + 
            "{}_{:d}epochs_{}.eps".format(
            string, len(self.training_costs),
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S")))


def BLSTMLayer(l_previous, **kwargs):
    # setup forward and backwards LSTM layers.  Note that
    # LSTMLayer takes a backwards flag. The backwards flag tells
    # scan to go backwards before it returns the output from
    # backwards layers.  It is reversed again such that the output
    # from the layer is always from x_1 to x_n.

    # If learn_init=True then you can't have multiple
    # layers of LSTM cells.
    l_fwd = LSTMLayer(l_previous, backwards=False, **kwargs)
    l_bck = LSTMLayer(l_previous, backwards=True, **kwargs)
    return ElemwiseSumLayer([l_fwd, l_bck])

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

    def __init__(self, source, learning_rate=1e-1, 
                 n_cells_per_hidden_layer=None, output_nonlinearity=None,
                 n_dense_cells_per_layer=20, experiment_name="",
                 validation_interval=10, save_plot_interval=100):
        """
        Parameters
        ----------
        n_cells_per_hidden_layer = list of ints
        """
        print("Initialising network...")
        self.source = source
        input_shape = source.input_shape()
        output_shape = source.output_shape()
        if n_cells_per_hidden_layer is None:
            n_cells_per_hidden_layer = [5]
        self.validation_interval = validation_interval
        self.save_plot_interval = save_plot_interval
        self.validation_costs = []
        self.training_costs = []
        self.experiment_name = experiment_name

        # Shape is (number of examples per batch,
        #           maximum number of time steps per example,
        #           number of features per example)
        l_previous = InputLayer(shape=input_shape)

        concat_shape = (self.source.n_seq_per_batch * self.source.seq_length, 
                        self.source.n_inputs)
        l_reshape1 = ReshapeLayer(l_previous, concat_shape)
        l_dense1 = DenseLayer(
            l_reshape1, num_units=n_dense_cells_per_layer, nonlinearity=sigmoid,
            b=np.random.uniform(-25,25,n_dense_cells_per_layer).astype(theano.config.floatX),
            W=np.random.uniform(-25,25,(1,n_dense_cells_per_layer)).astype(theano.config.floatX)
        )
        l_dense2 = DenseLayer(
            l_dense1, num_units=n_dense_cells_per_layer, nonlinearity=sigmoid,
            b=np.random.uniform(-10,10,n_dense_cells_per_layer).astype(theano.config.floatX),
            W=np.random.uniform(-10,10,(n_dense_cells_per_layer,n_dense_cells_per_layer)).astype(theano.config.floatX)
        )

        concat_shape = (self.source.n_seq_per_batch, self.source.seq_length, 
                        n_dense_cells_per_layer)
        l_previous = ReshapeLayer(l_dense2, concat_shape)

        # setup forward and backwards LSTM layers.  Note that
        # LSTMLayer takes a backwards flag. The backwards flag tells
        # scan to go backwards before it returns the output from
        # backwards layers.  It is reversed again such that the output
        # from the layer is always from x_1 to x_n.
        for n_cells in n_cells_per_hidden_layer:
            # l_previous = LSTMLayer(l_previous, n_cells, backwards=False,
            #                        learn_init=True, peepholes=True)
            # If learn_init=True then you can't have multiple
            # layers of LSTM cells.
            l_fwd = LSTMLayer(l_previous, n_cells, backwards=False,
                              learn_init=False, peepholes=True,
                              W_in_to_cell=lasagne.init.Normal(1.0))
            l_bck = LSTMLayer(l_previous, n_cells, backwards=True,
                              learn_init=False, peepholes=True,
                              W_in_to_cell=lasagne.init.Normal(1.0))
            l_previous = ElemwiseSumLayer([l_fwd, l_bck])

        concat_shape = (self.source.n_seq_per_batch * self.source.seq_length, 
                        n_cells_per_hidden_layer[-1])
        # concatenate forward and backward LSTM layers
        l_reshape = ReshapeLayer(l_previous, concat_shape)
        # We need a reshape layer which combines the first (batch
        # size) and second (number of timesteps) dimensions, otherwise
        # the DenseLayer will treat the number of time steps as a
        # feature dimension.  Specifically, LSTMLayer expects a shape
        # of (n_batch, n_time_steps, n_features) but the DenseLayer
        # will flatten that shape to (n_batch,
        # n_time_steps*n_features) by default which is
        # wrong. Dimshuffling is done inside the LSTMLayer. You need
        # to dimshuffle because Theano's scan function iterates over
        # the first dimension, and if the shape is (n_batch,
        # n_time_steps, n_features) then you need to dimshuffle(1, 0,
        # 2) in order to iterate over time steps.

        l_recurrent_out = DenseLayer(l_reshape, num_units=self.source.n_outputs,
                                     nonlinearity=output_nonlinearity)
        l_out = ReshapeLayer(l_recurrent_out, output_shape)
        """
        l_out1 = DenseLayer(l_dense2, num_units=self.source.n_outputs, 
                            nonlinearity=output_nonlinearity)
        l_out = ReshapeLayer(l_out1, output_shape)
        """
        input = T.tensor3('input')
        target_output = T.tensor3('target_output')

        # add test values
        input.tag.test_value = rand(
            *input_shape).astype(theano.config.floatX)
        target_output.tag.test_value = rand(
            *output_shape).astype(theano.config.floatX)

        print("Compiling Theano functions...")
        # Cost = mean squared error
        cost = T.mean((l_out.get_output(input) - target_output)**2)
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
        validation_cost = None
        i = 0
        while i != n_iterations:
            t0 = time() # for calculating training duration
            X, y = self.source.queue.get(timeout=30)
            train_cost = self.train(X, y).flatten()[0]
            self.training_costs.append(train_cost)
            if not i % self.validation_interval:
                validation_cost = self.compute_cost(self.X_val, self.y_val).flatten()[0]
                self.validation_costs.append(validation_cost)
            if not i % self.save_plot_interval:
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
            ax = plt.gca()
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
        show_or_save_plot(filename)
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


def show_or_save_plot(filename):
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta
from numpy.random import rand
from time import time
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import (InputLayer, LSTMLayer, ReshapeLayer, 
                            ConcatLayer, ElemwiseSumLayer, DenseLayer)
from lasagne.nonlinearities import sigmoid
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
                 n_dense_cells_per_layer=20):
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

        # Shape is (number of examples per batch,
        #           maximum number of time steps per example,
        #           number of features per example)
        l_previous = InputLayer(shape=input_shape)

        concat_shape = (self.source.n_seq_per_batch * self.source.seq_length, 
                        self.source.n_inputs)
        l_reshape1 = ReshapeLayer(l_previous, concat_shape)
        l_dense1 = DenseLayer(
            l_reshape1, num_units=n_dense_cells_per_layer, nonlinearity=sigmoid,
            b=np.random.uniform(-1,1,n_dense_cells_per_layer)
        )
        l_dense2 = DenseLayer(
            l_dense1, num_units=n_dense_cells_per_layer, nonlinearity=sigmoid,
            b=np.random.uniform(-1,1,n_dense_cells_per_layer)
        )
        """
        concat_shape = (self.source.n_seq_per_batch, self.source.seq_length, 
                        n_dense_cells_per_layer)
        l_previous = ReshapeLayer(l_dense2, concat_shape)


        # setup forward and backwards LSTM layers.  Note that
        # LSTMLayer takes a backwards flag. The backwards flag tells
        # scan to go backwards before it returns the output from
        # backwards layers.  It is reversed again such that the output
        # from the layer is always from x_1 to x_n.
        for n_cells in n_cells_per_hidden_layer:
            l_previous = LSTMLayer(l_previous, n_cells, backwards=False,
                                   learn_init=True, peepholes=True)
            # l_fwd = LSTMLayer(l_previous, n_cells, backwards=False,
            #                   learn_init=True, peepholes=True)
            # l_bck = LSTMLayer(l_previous, n_cells, backwards=True,
            #                   learn_init=True, peepholes=True)
            # l_previous = ElemwiseSumLayer([l_fwd, l_bck])

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

        print("Done initialising network.")

    def fit(self, n_iterations=100):
        # column 0 = training cost
        # column 1 = validation cost
        self.costs = np.zeros(shape=(n_iterations, 2))
        self.costs[:,:] = np.nan

        # Generate a "validation" sequence whose cost we will compute
        X_val, y_val = self.source.validation_data()

        # Training loop
        self.source.start()
        try:
            self._training_loop(n_iterations, X_val, y_val)
        except:
            raise
        finally:
            self.source.stop()

    def _training_loop(self, n_iterations, X_val, y_val):
        # Adapted from dnouri/nolearn/nolearn/lasagne.py
        print("""
 Epoch  |  Train cost  |  Valid cost  |  Train / Val  | Dur per epoch
--------|--------------|--------------|---------------|---------------\
""")
        for n in range(n_iterations):
            t0 = time() # for calculating training duration
            X, y = self.source.queue.get()
            train_cost = self.train(X, y).flatten()[0]
            validation_cost = self.compute_cost(X_val, y_val).flatten()[0]
            self.costs[n] = train_cost, validation_cost

            # Print progress
            duration = time() - t0
            is_best_train = train_cost == np.nanmin(self.costs[:,0])
            is_best_valid = validation_cost == np.nanmin(self.costs[:,1])
            print("  {:>5} |  {}{:>10.6f}{}  |  {}{:>10.6f}{}  |"
                  "  {:>11.6f}  |  {:>3.1f}s".format(
                      n,
                      ansi.BLUE if is_best_train else "",
                      train_cost,
                      ansi.ENDC if is_best_train else "",
                      ansi.GREEN if is_best_valid else "",
                      validation_cost,
                      ansi.ENDC if is_best_valid else "",
                      train_cost / validation_cost,
                      duration
            ))

    def plot_costs(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.costs[:,0], label='training')
        ax.plot(self.costs[:,1], label='validation')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.legend()
        plt.show()
        return ax

    def plot_estimates(self, axes=None):
        if axes is None:
            _, axes = plt.subplots(2, sharex=True)
        X, y = self.source.validation_data()
        y_predictions = self.y_pred(X)
        axes[0].set_title('Appliance')
        axes[0].plot(y_predictions[0,:,0], label='Estimates')
        axes[0].plot(y[0,:,0], label='Ground truth')
        axes[0].legend()
        axes[1].set_title('Aggregate')
        axes[1].plot(X[0,:,0])#, label='Fdiff')
        #axes[1].plot(np.cumsum(X[0,:,1]), label='Cumsum')
        axes[1].legend()
        plt.show()

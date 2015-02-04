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
                            ConcatLayer, DenseLayer)
theano.config.compute_test_value = 'raise'

"""
rsync command: 
rsync -uvz --progress /home/jack/workspace/python/neuralnilm/scripts/*.py /mnt/sshfs/imperial/workspace/python/neuralnilm/scripts
"""

class ansi:
    # from dnouri/nolearn/nolearn/lasagne.py
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'

######################## Neural network class ########################
class Net(object):
    # Much of this code is adapted from craffel/nntools/examples/lstm.py

    def __init__(self, source, learning_rate=1e-1, n_hidden=5):
        print("Initialising network...")
        self.source = source
        self.input_shape = source.input_shape()
        self.output_shape = source.output_shape()

        # Construct LSTM RNN: One LSTM layer and one dense output layer
        l_in = InputLayer(shape=self.input_shape)

        # setup fwd and bck LSTM layer.
        l_fwd = LSTMLayer(
            l_in, n_hidden, backwards=False, learn_init=True, peepholes=True)
        l_bck = LSTMLayer(
            l_in, n_hidden, backwards=True, learn_init=True, peepholes=True)

        # concatenate forward and backward LSTM layers
        concat_shape = (self.source.N_SEQ_PER_BATCH * self.source.SEQ_LENGTH, 
                        n_hidden)
        l_fwd_reshape = ReshapeLayer(l_fwd, concat_shape)
        l_bck_reshape = ReshapeLayer(l_bck, concat_shape)
        l_concat = ConcatLayer([l_fwd_reshape, l_bck_reshape], axis=1)

        l_recurrent_out = DenseLayer(l_concat, num_units=self.source.N_OUTPUTS,
                                     nonlinearity=None)
        l_out = ReshapeLayer(l_recurrent_out, self.output_shape)

        input = T.tensor3('input')
        target_output = T.tensor3('target_output')

        # add test values
        input.tag.test_value = rand(
            *self.input_shape).astype(theano.config.floatX)
        target_output.tag.test_value = rand(
            *self.output_shape).astype(theano.config.floatX)

        print("Compiling Theano functions...")
        # Cost = mean squared error
        cost = T.mean((l_out.get_output(input) - target_output)**2)

        # Use NAG for training
        all_params = lasagne.layers.get_all_params(l_out)
        updates = lasagne.updates.nesterov_momentum(cost, all_params, learning_rate)

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
        self.costs = np.zeros(shape=(n_interations, 2))
        self.costs[:,:] = np.nan

        # Generate a "validation" sequence whose cost we will compute
        X_val, y_val = self.source.validation_data()

        # Adapted from dnouri/nolearn/nolearn/lasagne.py
        print("""
 Epoch  |  Train cost  |  Valid cost  |  Train / Val  | Dur per epoch
--------|--------------|--------------|---------------|---------------\
""")
        # Training loop
        self.source.start()
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
        X, y = self.source._gen_unquantized_data(validation=True)
        y_predictions = self.y_pred(gen_data(X=X)[0])
        axes[0].set_title('Appliance forward difference')
        axes[0].plot(y_predictions[0,:,0], label='Estimates')
        axes[0].plot(y[0,:,0], label='Appliance ground truth')
        axes[0].legend()
        axes[1].set_title('Aggregate')
        axes[1].plot(X[0,:,1], label='Fdiff')
        axes[1].plot(np.cumsum(X[0,:,1]), label='Cumsum')
        axes[1].legend()
        plt.show()

if __name__ == "__main__":
    net = Net()
    net.training_loop()
    net.plot_costs()
    net.plot_estimates()

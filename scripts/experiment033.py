from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
from time import time

"""
INPUT: quantized mains fdiff
OUTPUT: appliance fdiff

Code taken from Lasagne and nolearn!
"""

SEQ_LENGTH = 400
N_HIDDEN = 5
N_SEQ_PER_BATCH = 30 # Number of sequences per batch
LEARNING_RATE = 1e-1 # SGD learning rate
N_ITERATIONS = 100  # Number of training iterations
N_INPUT_FEATURES = 10
N_OUTPUTS = 1

input_shape  = (N_SEQ_PER_BATCH, SEQ_LENGTH, N_INPUT_FEATURES)
output_shape = (N_SEQ_PER_BATCH, SEQ_LENGTH, N_OUTPUTS)

############### GENERATE DATA ##############################

def quantized(inp, all_hot=True):
    N_BINS = 10
    out = np.zeros(shape=(N_SEQ_PER_BATCH, SEQ_LENGTH, N_BINS))
    for i_batch in range(N_SEQ_PER_BATCH):
        for i_element in range(SEQ_LENGTH):
            hist, _ = np.histogram(inp[i_batch, i_element, 0], bins=N_BINS,
                                   range=(-1, 1))
            if all_hot:
                where = np.where(hist==1)[0][0]
                if where > 5:
                    hist[5:where] = 1
                elif where < 5:
                    hist[where:5] = 1
            out[i_batch,i_element,:] = hist
    return (out * 2) - 1


def gen_single_appliance(power, on_duration, min_off_duration=20, fdiff=True):
    length = SEQ_LENGTH + 1 if fdiff else SEQ_LENGTH
    appliance_power = np.zeros(length)
    i = 0
    while i < length:
        if np.random.binomial(n=1, p=0.2):
            end = min(i + on_duration, length)
            appliance_power[i:end] = power
            i += on_duration + min_off_duration
        else:
            i += 1
    return np.diff(appliance_power) if fdiff else appliance_power


def gen_batches_of_single_appliance(*args, **kwargs):
    batches = np.zeros(shape=(N_SEQ_PER_BATCH, SEQ_LENGTH, 1))
    for i in range(N_SEQ_PER_BATCH):
        batches[i, :, :] = gen_single_appliance(*args, **kwargs).reshape(SEQ_LENGTH, 1)
    return batches


def gen_unquantized_data(n_appliances=2, 
                         appliance_powers=[10,30],
                         appliance_on_durations=[10,2], validation=False):
    '''Generate a simple energy disaggregation data.

    :parameters:

    :returns:
        - X : np.ndarray, shape=(n_batch, length, 1)
            Input sequence
        - y : np.ndarray, shape=(n_batch, length, 1)
            Target sequence, appliance 1
    '''
    y = gen_batches_of_single_appliance(power=appliance_powers[0], 
                                        on_duration=appliance_on_durations[0])
    X = y.copy()
    for power, on_duration in zip(appliance_powers, appliance_on_durations)[1:]:
        X += gen_batches_of_single_appliance(power=power, on_duration=on_duration)

    max_power = np.sum(appliance_powers)
    return X / max_power, y / max_power


def gen_data(*args, **kwargs):
    X, y = gen_unquantized_data(*args, **kwargs)
    return quantized(X), y

class ansi:
    # from dnouri/nolearn/nolearn/lasagne.py
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'

######################## Neural network class ########################
class Net(object):
    # Much of this code is adapted from craffel/nntools/examples/lstm.py

    def __init__(self):
        print("Initialising network...")
        import theano
        import theano.tensor as T
        import lasagne
        from lasagne.layers import (InputLayer, LSTMLayer, ReshapeLayer, 
                                    ConcatLayer, DenseLayer)
        theano.config.compute_test_value = 'raise'

        # Construct LSTM RNN: One LSTM layer and one dense output layer
        l_in = InputLayer(shape=input_shape)

        # setup fwd and bck LSTM layer.
        l_fwd = LSTMLayer(
            l_in, N_HIDDEN, backwards=False, learn_init=True, peepholes=True)
        l_bck = LSTMLayer(
            l_in, N_HIDDEN, backwards=True, learn_init=True, peepholes=True)

        # concatenate forward and backward LSTM layers
        concat_shape = (N_SEQ_PER_BATCH * SEQ_LENGTH, N_HIDDEN)
        l_fwd_reshape = ReshapeLayer(l_fwd, concat_shape)
        l_bck_reshape = ReshapeLayer(l_bck, concat_shape)
        l_concat = ConcatLayer([l_fwd_reshape, l_bck_reshape], axis=1)

        l_recurrent_out = DenseLayer(l_concat, num_units=N_OUTPUTS, 
                                     nonlinearity=None)
        l_out = ReshapeLayer(l_recurrent_out, output_shape)

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
        updates = lasagne.updates.nesterov_momentum(cost, all_params, LEARNING_RATE)

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

    def training_loop(self):
        # column 0 = training cost
        # column 1 = validation cost
        self.costs = np.zeros(shape=(N_ITERATIONS, 2))
        self.costs[:,:] = np.nan

        # Generate a "validation" sequence whose cost we will compute
        X_val, y_val = gen_data(validation=True)
        assert X_val.shape == input_shape
        assert y_val.shape == output_shape

        # Adapted from dnouri/nolearn/nolearn/lasagne.py
        print("""
 Epoch  |  Train cost  |  Valid cost  |  Train / Val  | Dur per epoch
--------|--------------|--------------|---------------|---------------\
""")
        # Training loop
        for n in range(N_ITERATIONS):
            t0 = time() # for calculating training duration
            X, y = gen_data()
            train_cost = self.train(X, y).flatten()[0]
            validation_cost = self.compute_cost(X_val, y_val).flatten()[0]
            self.costs[n] = train_cost, validation_cost

            if n==N_ITERATIONS-1 or not n % 10:
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
        X, y = gen_unquantized_data()
        y_predictions = self.y_pred(quantized(X))
        axes[0].set_title('Appliance forward difference')
        axes[0].plot(y_predictions[0,:,0], label='Estimates')
        axes[0].plot(y[0,:,0], label='Appliance ground truth')
        axes[0].legend()
        axes[1].set_title('Aggregate')
        axes[1].plot(X[0,:,0], label='Fdiff')
        axes[1].plot(np.cumsum(X[0,:,0]), label='Cumsum')
        axes[1].legend()
        plt.show()

if __name__ == "__main__":
    net = Net()
    net.training_loop()
    net.plot_costs()
    net.plot_estimates()

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import lasagne
from gen_data_029 import gen_data, N_BATCH, LENGTH
theano.config.compute_test_value = 'raise'

"""
tanh output
lower learning rate
* does just about learn something sensible, but not especially convincing, 
  even after 2000 iterations.
"""

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 5
# SGD learning rate
LEARNING_RATE = 1e-2
# Number of iterations to train the net
N_ITERATIONS = 2000

# Generate a "validation" sequence whose cost we will periodically compute
X_val, y_val = gen_data()

n_features = X_val.shape[-1]
n_output = y_val.shape[-1]
assert X_val.shape == (N_BATCH, LENGTH, n_features)
assert y_val.shape == (N_BATCH, LENGTH, n_output)

# Construct LSTM RNN: One LSTM layer and one dense output layer
l_in = lasagne.layers.InputLayer(shape=(N_BATCH, LENGTH, n_features))


# setup fwd and bck LSTM layer.
l_fwd = lasagne.layers.LSTMLayer(
    l_in, N_HIDDEN, backwards=False, learn_init=True, peepholes=True)
l_bck = lasagne.layers.LSTMLayer(
    l_in, N_HIDDEN, backwards=True, learn_init=True, peepholes=True)

# concatenate forward and backward LSTM layers
l_fwd_reshape = lasagne.layers.ReshapeLayer(l_fwd, (N_BATCH*LENGTH, N_HIDDEN))
l_bck_reshape = lasagne.layers.ReshapeLayer(l_bck, (N_BATCH*LENGTH, N_HIDDEN))
l_concat = lasagne.layers.ConcatLayer([l_fwd_reshape, l_bck_reshape], axis=1)


l_recurrent_out = lasagne.layers.DenseLayer(
    l_concat, num_units=n_output, nonlinearity=lasagne.nonlinearities.tanh)
l_out = lasagne.layers.ReshapeLayer(
    l_recurrent_out, (N_BATCH, LENGTH, n_output))

input = T.tensor3('input')
target_output = T.tensor3('target_output')

# add test values
input.tag.test_value = np.random.rand(
    *X_val.shape).astype(theano.config.floatX)
target_output.tag.test_value = np.random.rand(
    *y_val.shape).astype(theano.config.floatX)

# Cost = mean squared error
cost = T.mean((l_out.get_output(input) - target_output)**2)

# Use NAG for training
all_params = lasagne.layers.get_all_params(l_out)
updates = lasagne.updates.nesterov_momentum(cost, all_params, LEARNING_RATE)
# Theano functions for training, getting output, and computing cost
train = theano.function([input, target_output],
                        cost, updates=updates, on_unused_input='warn',
                        allow_input_downcast=True)
y_pred = theano.function(
    [input], l_out.get_output(input), on_unused_input='warn',
    allow_input_downcast=True)

compute_cost = theano.function(
    [input, target_output], cost, on_unused_input='warn',
    allow_input_downcast=True)

# Train the net
def run_training():
    costs = np.zeros(N_ITERATIONS)
    for n in range(N_ITERATIONS):
        X, y = gen_data()

        # you should use your own training data mask instead of mask_val
        costs[n] = train(X, y)
        if not n % 10:
            cost_val = compute_cost(X_val, y_val)
            print "Iteration {} validation cost = {}".format(n, cost_val)

    plt.plot(costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()

def plot_estimates():
    X, y = gen_data()
    y_predictions = y_pred(X)
    ax = plt.gca()
    ax.plot(y_predictions[0,:,0], label='estimate')
    ax.plot(y[0,:,0], label='ground truth')
    # ax.plot(X[0,:,0], label='aggregate')
    ax.legend()
    plt.show()

run_training()
plot_estimates()

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import lasagne

"""
INPUT: quantized mains fdiff
OUTPUT: appliance fdiff
"""

theano.config.compute_test_value = 'raise'
# Sequence length
LENGTH = 400
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 5
# Number of training sequences in each batch
N_BATCH = 30
# SGD learning rate
LEARNING_RATE = 1e-1
# Number of iterations to train the net
N_ITERATIONS = 200

def quantized(inp):
    n = 10
    n_batch, length, _ = inp.shape
    out = np.zeros(shape=(n_batch, length, n))
    for i_batch in range(n_batch):
        for i_element in range(length):
            out[i_batch,i_element,:], _ = np.histogram(inp[i_batch, i_element, 0], [-1,-.8,-.6,-.4,-.2,0.0,.2,.4,.6,.8,1])
    return (out * 2) - 1

def gen_single_appliance(length, power, on_duration, min_off_duration=20, 
                         fdiff=True):
    if fdiff:
        length += 1
    appliance_power = np.zeros(shape=(length))
    i = 0
    while i < length:
        if np.random.binomial(n=1, p=0.2):
            end = min(i + on_duration, length)
            appliance_power[i:end] = power
            i += on_duration + min_off_duration
        else:
            i += 1
    return np.diff(appliance_power) if fdiff else appliance_power

def gen_batches_of_single_appliance(length, n_batch, *args, **kwargs):
    batches = np.zeros(shape=(n_batch, length, 1))
    for i in range(n_batch):
        batches[i, :, :] = gen_single_appliance(length, *args, **kwargs).reshape(length, 1)
    return batches

def gen_data(length=LENGTH, n_batch=N_BATCH, n_appliances=2, 
             appliance_powers=[10,20], 
             appliance_on_durations=[10,2]):
    '''Generate a simple energy disaggregation data.

    :parameters:
        - length : int
            Length of sequences to generate
        - n_batch : int
            Number of training sequences per batch

    :returns:
        - X : np.ndarray, shape=(n_batch, length, 1)
            Input sequence
        - y : np.ndarray, shape=(n_batch, length, 1)
            Target sequence, appliance 1
    '''
    y = gen_batches_of_single_appliance(length, n_batch, 
                                        power=appliance_powers[0], 
                                        on_duration=appliance_on_durations[0])
    X = y.copy()
    for power, on_duration in zip(appliance_powers, appliance_on_durations)[1:]:
        X += gen_batches_of_single_appliance(length, n_batch, power=power, on_duration=on_duration)

    max_power = np.sum(appliance_powers)
    
    return quantized(X / max_power), y / max_power

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
    l_concat, num_units=n_output, nonlinearity=None)
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
        if not n % 100:
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

from __future__ import print_function, division
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import lasagne
from lasagne.utils import floatX
from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import tanh
from lasagne.objectives import mse
from lasagne.init import Uniform

from neuralnilm.layers import MixtureDensityLayer
from neuralnilm.objectives import mdn_nll

# Number of units in the hidden (recurrent) layer
N_HIDDEN_LAYERS = 2
N_UNITS_PER_LAYER = 5
N_COMPONENTS = 3
# Number of training sequences in each batch
SHAPE = (100, 1)
# SGD learning rate
LEARNING_RATE = 0.001
# Number of iterations to train the net
N_ITERATIONS = 10000

RNG = np.random.RandomState(42)

def gen_data():
    '''
    Generate toy data, from Bishop p273.

    :returns:
        - X : np.ndarray, shape=(n_batch, 1)
            Input sequence
        - t : np.ndarray, shape=(n_batch, 1)
            Target sequence
    '''
    t = RNG.uniform(low=0.1, high=0.9, size=SHAPE)
    noise = RNG.uniform(low=-0.1, high=0.1, size=SHAPE)
    X = t + (0.3 * np.sin(2 * np.pi * t)) + noise
    return floatX(X), floatX(t)


X_val, t_val = gen_data()

# Configure layers
layers = [InputLayer(shape=SHAPE)]
for i in range(N_HIDDEN_LAYERS):
    layer = DenseLayer(
        layers[-1], N_UNITS_PER_LAYER, nonlinearity=tanh, W=Uniform(0.1))
    layers.append(layer)
layers.append(MixtureDensityLayer(
    layers[-1], n_output_features=1, n_components=N_COMPONENTS))

print("Total parameters: {}".format(
    sum([p.get_value().size 
         for p in lasagne.layers.get_all_params(layers[-1])])))

X = T.matrix('X')
t = T.matrix('t')

# add test values
X.tag.test_value = floatX(np.random.rand(*SHAPE))
t.tag.test_value = floatX(np.random.rand(*SHAPE))

loss_func = mdn_nll
y = layers[-1].get_output(X)
loss = loss_func(y, t)

all_params = lasagne.layers.get_all_params(layers[-1])
updates = lasagne.updates.nesterov_momentum(loss, all_params, LEARNING_RATE)

# Theano functions for training, getting output, and computing loss
print("Compiling Theano functions...")
train = theano.function([X, t], loss, updates=updates)
y_pred = theano.function([X], layers[-1].get_output(X))
compute_loss = theano.function([X, t], loss)
print("Done compiling Theano functions.")

# Train the net
print("Starting training...")
costs = np.zeros(N_ITERATIONS)
for n in range(N_ITERATIONS):
    X, t = gen_data()
    costs[n] = train(X, t)
    if not n % 100:
        cost_val = compute_loss(X_val, t_val)
        print("Iteration {} validation cost = {}".format(n, cost_val))

# Plot means
ax = plt.gca()
y = y_pred(X_val)
for i in range(N_COMPONENTS):
    ax.plot(X_val[:,0], y[0][:,0,i], 'x')
ax.plot(X_val[:,0], t_val[:,0], 'x')
plt.show()


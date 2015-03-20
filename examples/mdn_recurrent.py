from __future__ import print_function, division
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import lasagne
from lasagne.utils import floatX
from lasagne.layers import InputLayer, RecurrentLayer, ReshapeLayer
from lasagne.nonlinearities import tanh

from neuralnilm.layers import MixtureDensityLayer
from neuralnilm.objectives import mdn_nll

# Number of units in the hidden (recurrent) layer
N_HIDDEN_LAYERS = 2
N_UNITS_PER_LAYER = 5
N_COMPONENTS = 3
# Number of training sequences in each batch
N_SEQ_PER_BATCH = 1
SEQ_LENGTH = 200
SHAPE = (N_SEQ_PER_BATCH, SEQ_LENGTH, 1)
# SGD learning rate
LEARNING_RATE = 0.001
# Number of iterations to train the net
N_ITERATIONS = 5000

np.random.seed(42)


def gen_data():
    '''
    Generate toy data.

    :returns:
        - X : np.ndarray, shape=SHAPE
            Input sequence
        - t : np.ndarray, shape=SHAPE
            Target sequence
    '''

    def noise():
        return floatX(np.random.uniform(low=-0.1, high=0.1, size=SHAPE))
    t = np.zeros(shape=SHAPE, dtype=np.float32) - 0.9
    X = np.zeros(shape=SHAPE, dtype=np.float32) - 0.9
    X[:,100:150,:] = 0.9
    for batch_i in range(N_SEQ_PER_BATCH):
        if np.random.binomial(n=1, p=0.5):
            X[batch_i,50:100,0] = np.linspace(-0.9, 0.9, 50)
            t[batch_i,:,0] = X[batch_i,:,0].copy()
    X += noise()
    t += noise()
    return X, t


X_val, t_val = gen_data()

# Configure layers
layers = [InputLayer(shape=SHAPE)]
for i in range(N_HIDDEN_LAYERS):
    layer = RecurrentLayer(layers[-1], N_UNITS_PER_LAYER, nonlinearity=tanh)
    layers.append(layer)
layers.append(ReshapeLayer(layers[-1], (N_SEQ_PER_BATCH * SEQ_LENGTH, N_UNITS_PER_LAYER)))
layers.append(MixtureDensityLayer(
    layers[-1], num_units=t_val.shape[-1], num_components=N_COMPONENTS))
# layers.append(ReshapeLayer(layers[-1], SHAPE))

print("Total parameters: {}".format(
    sum([p.get_value().size 
         for p in lasagne.layers.get_all_params(layers[-1])])))

X = T.tensor3('X')
t = T.matrix('t')

# add test values
X.tag.test_value = floatX(np.random.rand(*SHAPE))
t.tag.test_value = floatX(np.random.rand(N_SEQ_PER_BATCH * SEQ_LENGTH, 1))

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
t_val = t_val.reshape((N_SEQ_PER_BATCH * SEQ_LENGTH, 1))
for n in range(N_ITERATIONS):
    X, t = gen_data()
    t = t.reshape((N_SEQ_PER_BATCH * SEQ_LENGTH, 1))
    costs[n] = train(X, t)
    if not n % 100:
        cost_val = compute_loss(X_val, t_val)
        print("Iteration {} validation cost = {}".format(n, cost_val))

# Plot means
# ax = plt.gca()
# y = y_pred(X_val)
# for i in range(N_COMPONENTS):
#     ax.scatter(X_val[:,0], y[0][:,0,i], s=y[2][:,i].mean() * 50)
# ax.scatter(X_val[:,0], t_val[:,0], color='g')
# plt.show()


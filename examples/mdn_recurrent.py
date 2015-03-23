from __future__ import print_function, division
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from scipy.stats import norm

import lasagne
from lasagne.utils import floatX
from lasagne.layers import InputLayer, RecurrentLayer, ReshapeLayer, DenseLayer
from lasagne.nonlinearities import tanh
from lasagne.init import Normal
from lasagne.objectives import Objective

from neuralnilm.layers import MixtureDensityLayer
from neuralnilm.objectives import mdn_nll

# Number of units in the hidden (recurrent) layer
N_HIDDEN_LAYERS = 2
N_UNITS_PER_LAYER = 25
N_COMPONENTS = 1
# Number of training sequences in each batch
N_SEQ_PER_BATCH = 16
SEQ_LENGTH = 256
SHAPE = (N_SEQ_PER_BATCH, SEQ_LENGTH, 1)
# SGD learning rate
LEARNING_RATE = 0.00005
#LEARNING_RATE = 0.001
# Number of iterations to train the net
N_ITERATIONS = 7000

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
    NOISE_MAGNITUDE = 0.1
    PULSE_WIDTH = 10
    START = 100
    STOP = 250
    ON = 1.0
    OFF = 0.0
    def noise():
        return floatX(np.random.uniform(
            low=-NOISE_MAGNITUDE, high=NOISE_MAGNITUDE, size=SHAPE))

    t = np.zeros(shape=SHAPE, dtype=np.float32) + OFF
    X = np.zeros(shape=SHAPE, dtype=np.float32) + OFF
    X[:,START:STOP,:] = ON

    for batch_i in range(N_SEQ_PER_BATCH):
        if np.random.binomial(n=1, p=0.5):
            for pulse_start in range(START, STOP, PULSE_WIDTH*2):
                pulse_end = pulse_start + PULSE_WIDTH
                X[batch_i, pulse_start:pulse_end, 0] = OFF
            t[batch_i, :, 0] = X[batch_i, :, 0].copy()
    X += noise()
    return X, t


X_val, t_val = gen_data()

# Configure layers
layers = [InputLayer(shape=SHAPE)]
for i in range(N_HIDDEN_LAYERS):
    layer = RecurrentLayer(
        layers[-1], N_UNITS_PER_LAYER, nonlinearity=tanh, 
        W_in_to_hid=Normal(std=1.0/np.sqrt(layers[-1].get_output_shape()[-1])),
        gradient_steps=100)
    layers.append(layer)
layers.append(ReshapeLayer(layers[-1], (N_SEQ_PER_BATCH * SEQ_LENGTH, N_UNITS_PER_LAYER)))
layers.append(MixtureDensityLayer(
    layers[-1], num_units=t_val.shape[-1], num_components=N_COMPONENTS))
# layers.append(DenseLayer(
#     layers[-1], num_units=1, nonlinearity=None, 
#     W=Normal(std=1.0/np.sqrt(N_UNITS_PER_LAYER))))
# layers.append(ReshapeLayer(layers[-1], SHAPE))

print("Total parameters: {}".format(
    sum([p.get_value().size 
         for p in lasagne.layers.get_all_params(layers[-1])])))

X = T.tensor3('X')
t = T.matrix('t')

# add test values
X.tag.test_value = floatX(np.random.rand(*SHAPE))
t.tag.test_value = floatX(np.random.rand(N_SEQ_PER_BATCH * SEQ_LENGTH, 1))

objective = Objective(layers[-1], loss_function=mdn_nll)
loss = objective.get_loss(X, t)

all_params = lasagne.layers.get_all_params(layers[-1])
updates = lasagne.updates.momentum(loss, all_params, LEARNING_RATE)

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


def gmm_pdf(theta, x):
    """
    Parameters
    ----------
    theta : tuple of (mu, sigma, mixing)
    """
    pdf = None
    for mu, sigma, mixing in zip(*theta):
        norm_pdf = norm.pdf(x=x, loc=mu, scale=sigma)
        norm_pdf *= mixing
        if pdf is None:
            pdf = norm_pdf
        else:
            pdf += norm_pdf
    return pdf


def gmm_heatmap(thetas, ax):
    """
    Parameters
    ----------
    thetas : tuple of (array of mus, array of sigmas, array of mixing)
    """
    N_X = 100
    UPPER_LIMIT = 2
    LOWER_LIMIT = -2
    n_y = len(thetas[0])
    x = np.linspace(UPPER_LIMIT, LOWER_LIMIT, N_X)
    img = np.zeros(shape=(N_X, n_y))
    i = 0
    for i, (mu, sigma, mixing) in enumerate(zip(*thetas)):
        img[:, i] = gmm_pdf((mu[0], sigma, mixing), x)
    EXTENT = (0, n_y, LOWER_LIMIT, UPPER_LIMIT) # left, right, bottom, top
    ax.imshow(img, interpolation='none', extent=EXTENT, aspect='auto')
    return ax    


# Plot means
y = y_pred(X_val)
mu, sigma, mixing = y

batch_i = 0
fig, axes = plt.subplots(3, sharex=True)
rng = slice(batch_i*SEQ_LENGTH, (batch_i+1)*SEQ_LENGTH)
gmm_heatmap((mu[rng], sigma[rng], mixing[rng]), axes[0])
axes[1].plot(X_val[batch_i, :, 0])
axes[1].plot(t_val[rng, 0])
x = range(SEQ_LENGTH)
#ax.plot(y[rng, 0])
for i in range(N_COMPONENTS):
    axes[2].scatter(x, y[0][rng, 0, i], s=y[2][:, i] * 5)

plt.show()

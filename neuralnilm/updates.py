from __future__ import print_function, division
import numpy as np

import theano
import theano.tensor as T
from theano.gradient import grad_clip

from neuralnilm.utils import sfloatX


def clipped_nesterov_momentum(loss, all_params, learning_rate, 
                              clip_range, momentum=0.9):
    # Adapted from Lasagne/lasagne/updates.py
    all_grads = theano.grad(grad_clip(loss, clip_range[0], clip_range[1]),
                            all_params)

    updates = []

    for param_i, grad_i in zip(all_params, all_grads):
        mparam_i = theano.shared(np.zeros(param_i.get_value().shape,
                                          dtype=theano.config.floatX))
        v = momentum * mparam_i - learning_rate * grad_i  # new momemtum
        w = param_i + momentum * v - learning_rate * grad_i  # new param values
        updates.append((mparam_i, v))
        updates.append((param_i, w))

    return updates


def anneal_learning_rate(initial_learning_rate, normaliser, iteration):
    return sfloatX(initial_learning_rate / (1 + (iteration / normaliser)))

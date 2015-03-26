from __future__ import print_function, division
import numpy as np

import theano
from theano.ifelse import ifelse
import theano.tensor as T

from lasagne.utils import floatX

from neuralnilm.utils import sfloatX


def scaled_cost(x, t, loss_func=lambda x, t: (x - t) ** 2):
    THRESHOLD = 0
    error = loss_func(x, t)
    def mask_and_mean_error(mask):
        masked_error = error[mask.nonzero()]
        mean = masked_error.mean()
        mean = ifelse(T.isnan(mean), 0.0, mean)
        return mean
    mask = t > THRESHOLD
    above_thresh_mean = mask_and_mean_error(mask)
    below_thresh_mean = mask_and_mean_error(-mask)
    return (above_thresh_mean + below_thresh_mean) / 2.0


TWO_PI = sfloatX(2 * np.pi)

def mdn_nll(theta, t):
    """Computes the mean of negative log likelihood for P(t|theta) for
    Mixture Density Network output layers.

    :parameters:
        - theta : Output of the net. Contains mu, sigma, mixing
        - t : targets. Shape = (minibatch_size, output_size)

    :returns:
        - NLL per output
    """
    # Adapted from NLL() in
    # github.com/aalmah/ift6266amjad/blob/master/experiments/mdn.py

    # mu, sigma, mixing have shapes (batch_size, num_units, num_components)
    mu     = theta[:,:,:,0]
    sigma  = theta[:,:,:,1]
    mixing = theta[:,:,:,2]
    x = t.dimshuffle(0, 1, 'x')
    log_likelihood = normal_log_likelihood_per_component(x, mu, sigma, mixing)
    summed_over_components = log_sum_exp(log_likelihood, axis=2)
    return -summed_over_components.reshape(shape=t.shape)


def log_sum_exp(x, axis=None, keepdims=True):
    """Numerically stable version of log(sum(exp(x)))."""
    # adapted from https://github.com/Theano/Theano/issues/1563
    x_max = T.max(x, axis=axis, keepdims=keepdims)
    x_mod = x - x_max
    summed = T.sum(T.exp(x_mod), axis=axis, keepdims=keepdims)
    return T.log(summed) + x_max


MINUS_HALF_LOG_2PI = sfloatX(- 0.5 * np.log(2 * np.pi))

def normal_log_likelihood_per_component(x, mu, sigma, mixing):
     return (
        MINUS_HALF_LOG_2PI
        - T.log(sigma)
        - 0.5 * T.inv(sigma**2) * (x - mu)**2
        + T.log(mixing)
    )


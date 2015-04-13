from __future__ import print_function, division
import numpy as np

import theano
from theano.ifelse import ifelse
import theano.tensor as T

from lasagne.utils import floatX

from neuralnilm.utils import sfloatX

THRESHOLD = 0
mse = lambda x, t: (x - t) ** 2

def scaled_cost(x, t, loss_func=mse):
    error = loss_func(x, t)
    def mask_and_mean_error(mask):
        masked_error = error[mask.nonzero()]
        mean = masked_error.mean()
        mean = ifelse(T.isnan(mean), 0.0, mean)
        return mean
    mask = t > THRESHOLD
    above_thresh_mean = mask_and_mean_error(mask)
    below_thresh_mean = mask_and_mean_error(-mask)
    cost = (above_thresh_mean + below_thresh_mean) / 2.
    return cost


def ignore_inactive(x, t, loss_func=mse, seq_length=None):
    error = loss_func(x, t)
    if seq_length is not None:
        n_seq_per_batch = t.shape[0] // seq_length
        shape = (n_seq_per_batch, seq_length, t.shape[-1])
        error = error.reshape(shape)
        t = t.reshape(shape)

    active_seqs = (t > THRESHOLD).sum(axis=1) > 0
    active_seqs = active_seqs.nonzero()
    error_only_active = error.dimshuffle(0, 2, 1)[active_seqs]
    return error_only_active.mean()


def scaled_cost_ignore_inactive(x, t, loss_func=mse, seq_length=None):
    error = loss_func(x, t)
    if seq_length is None:
        seq_length = t.shape[1]
    else:
        n_seq_per_batch = t.shape[0] // seq_length
        shape = (n_seq_per_batch, seq_length, t.shape[-1])
        error = error.reshape(shape)
        t = t.reshape(shape)

    error = error.eval()
    n_seq_per_batch = error.shape[1]
#    n_seq_per_batch = 16 # CHANGE THIS!
    for seq_i in range(n_seq_per_batch):
        elements_above_thresh = t[seq_i, :, 0] > THRESHOLD
        n_above_thresh = elements_above_thresh.sum()
        if n_above_thresh == 0:
            error[seq_i, :, :] = 0
        else:
            error[seq_i, elements_above_thresh, 0] *= 0.5 / n_above_thresh
            n_below_thresh = seq_length - n_above_thresh
            error[seq_i, -elements_above_thresh, 0] *= 0.5 / n_below_thresh

    return error.sum()


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


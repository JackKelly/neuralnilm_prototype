from __future__ import print_function, division
from theano.ifelse import ifelse
import theano.tensor as T
import numpy as np
from lasagne.utils import floatX


def scaled_cost(x, t, loss_func=lambda x, t: (x - t) ** 2):
    THRESHOLD = 0
    error = loss_func(x, t)
    def mask_and_mean_error(mask):
        masked_error = error[mask.nonzero()]
        mean = masked_error.mean()
        mean = ifelse(T.isnan(mean), 0.0, mean)
        return mean
    above_thresh_mean = mask_and_mean_error(t > THRESHOLD)
    below_thresh_mean = mask_and_mean_error(t <= THRESHOLD)
    return (above_thresh_mean + below_thresh_mean) / 2.0


def mdn_nll(theta, t):
    """Computes the mean of negative log likelihood for P(t|theta) for
    Mixture Density Network output layers.

    :parameters:
        - theta : mu, sigma, mixing
        - t : T.matrix('t') (minibatch_size, output_size)

    :returns:
        - res : NLL per output
    """
    # Adapted from NLL() in
    # github.com/aalmah/ift6266amjad/blob/master/experiments/mdn.py

    # mu, sigma, mixing have shapes (batch_size, num_units, num_components)
    mu     = theta[:,:,:,0]
    sigma  = theta[:,:,:,1]
    mixing = theta[:,:,:,2]
    mu.name     = 'mu'
    sigma.name  = 'sigma'
    mixing.name = 'mixing'
    pdf = gmm_pdf(t.dimshuffle(0, 1, 'x'), mu, sigma, mixing)
    pdf = pdf.reshape(shape=t.shape)
    log_pdf = T.log(pdf)
    return -log_pdf


SQRT_OF_2PI = floatX(np.sqrt(2 * np.pi))
def normal_pdf(x, mu, sigma):
    exponent = -((x - mu)**2) / (2 * sigma**2)
    normaliser = sigma * SQRT_OF_2PI
    return T.exp(exponent) / normaliser


def gmm_pdf(x, mu, sigma, mixing):
    normal_pdfs = normal_pdf(x, mu, sigma)
    return T.batched_tensordot(normal_pdfs, mixing, axes=1)


def LogSumExp(x, axis=None):
    # from https://github.com/Theano/Theano/issues/1563
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max

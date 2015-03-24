from __future__ import print_function, division
import numpy as np

import theano
from theano.ifelse import ifelse
import theano.tensor as T

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


TWO_PI = getattr(np, theano.config.floatX)(2 * np.pi)
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

    # TODO:
    # 1. see if my previous version is faster.
    # 2. calculate the 1D GMM in log space + log(mixing)
    #    and then do the log_sum_exp trick.
    # http://en.wikipedia.org/wiki/Normal_distribution#Estimation_of_parameters

    num_outputs = t.shape[1]
    sq_error = (t.dimshuffle(0, 1, 'x') - mu) ** 2
    sum_sq_error = T.sum(sq_error, axis=1)
    exponent = -0.5 * T.inv(sigma) * sum_sq_error
    normalizer = TWO_PI * sigma
    exponent = exponent + T.log(mixing) - (num_outputs * .5) * T.log(normalizer)
    log_gauss = log_sum_exp(exponent, axis=1)
    return -T.mean(log_gauss)


def log_sum_exp(x, axis=None):
    """Numerically stable version of log(sum(exp(x)))."""
    # adapted from https://github.com/Theano/Theano/issues/1563
    x_max = T.max(x, axis=axis, keepdims=True)
    x_mod = x - x_max
    summed = T.sum(T.exp(x_mod), axis=axis, keepdims=True)
    return T.log(summed) + x_max


def NLL(mu, sigma, mixing, y):
    """Computes the mean of negative log likelihood for P(y|x)
    
    y = T.matrix('y') # (minibatch_size, output_size)
    mu = T.tensor3('mu') # (minibatch_size, output_size, n_components)
    sigma = T.matrix('sigma') # (minibatch_size, n_components)
    mixing = T.matrix('mixing') # (minibatch_size, n_components)

    """
    
    # multivariate Gaussian
    exponent = -0.5 * T.inv(sigma) * T.sum((y.dimshuffle(0,1,'x') - mu)**2, axis=1)
    normalizer = (2 * np.pi * sigma)
    exponent = exponent + T.log(mixing) - (y.shape[1]*.5)*T.log(normalizer) 
    max_exponent = T.max(exponent ,axis=1, keepdims=True)
    mod_exponent = exponent - max_exponent
    gauss_mix = T.sum(T.exp(mod_exponent),axis=1)
    log_gauss = max_exponent + T.log(gauss_mix) 
    res = -T.mean(log_gauss)
    return res



def old():
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

from theano.ifelse import ifelse
import theano.tensor as T
import numpy as np


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


def mdn_nll(x, t):
    """Computes the mean of negative log likelihood for P(t|x) for
    Mixture Density Network output layers.

    :parameters:
        - x : a list of mu, sigma, mixing:
            - mu : shape = (minibatch_size, output_size, n_components)
            - sigma : shape = (minibatch_size, n_components)
            - mixing : shape = (minibatch_size, n_components)
        - t : T.matrix('t') (minibatch_size, output_size)

    :returns:
        - res : the mean NLL across all dimensions
    """
    # Adapted from NLL() in
    # github.com/aalmah/ift6266amjad/blob/master/experiments/mdn.py

    mu, sigma, mixing = x[0], x[1], x[2]

    # multivariate Gaussian
    exponent = -0.5 * T.inv(sigma) * T.sum((t.dimshuffle(0, 1, 'x') - mu)**2, 
                                           axis=1)
    normalizer = (2 * np.pi * sigma)
    exponent += T.log(mixing) - (t.shape[1] * .5) * T.log(normalizer)
    max_exponent = T.max(exponent, axis=1, keepdims=True)
    mod_exponent = exponent - max_exponent
    gauss_mix = T.sum(T.exp(mod_exponent), axis=1)
    log_gauss = max_exponent + T.log(gauss_mix)
    res = -T.mean(log_gauss)
    return res

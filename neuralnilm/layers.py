from __future__ import print_function, division
import theano
import theano.tensor as T

import numpy as np

from lasagne.layers import Layer, LSTMLayer, RecurrentLayer, ElemwiseSumLayer
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import floatX
            
def BLSTMLayer(*args, **kwargs):
    # setup forward and backwards LSTM layers.  Note that
    # LSTMLayer takes a backwards flag. The backwards flag tells
    # scan to go backwards before it returns the output from
    # backwards layers.  It is reversed again such that the output
    # from the layer is always from x_1 to x_n.

    # If learn_init=True then you can't have multiple
    # layers of LSTM cells.
    return BidirectionalLayer(LSTMLayer, *args, **kwargs)

          
def BidirectionalRecurrentLayer(*args, **kwargs):
    # setup forward and backwards LSTM layers.  Note that
    # LSTMLayer takes a backwards flag. The backwards flag tells
    # scan to go backwards before it returns the output from
    # backwards layers.  It is reversed again such that the output
    # from the layer is always from x_1 to x_n.

    # If learn_init=True then you can't have multiple
    # layers of LSTM cells.
    return BidirectionalLayer(RecurrentLayer, *args, **kwargs)


def BidirectionalLayer(layer_class, *args, **kwargs):
    kwargs.pop('backwards', None)
    l_fwd = layer_class(*args, backwards=False, **kwargs)
    l_bck = layer_class(*args, backwards=True, **kwargs)
    return ElemwiseSumLayer([l_fwd, l_bck])


class DimshuffleLayer(Layer):
    def __init__(self, input_layer, pattern):
        super(DimshuffleLayer, self).__init__(input_layer)
        self.pattern = pattern

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[i] for i in self.pattern])

    def get_output_for(self, input, *args, **kwargs):
        return input.dimshuffle(self.pattern)


class MixtureDensityLayer(Layer):
    """
    Based on:
    * amjadmahayri.wordpress.com/2014/04/30/mixture-density-networks
    * github.com/aalmah/ift6266amjad/blob/master/experiments/mdn.py
    """

    def __init__(self, incomming, num_units, 
                 num_components=2,
                 nonlinearity=None, 
                 W_mu=None, 
                 W_sigma=None, 
                 W_mixing=None):
        """
        - nonlinearity : callable or None
            The nonlinearity that is applied to the layer's mu activations.
            If None is provided, the layer will be linear.

        - num_units : int
            Number of features in the target

        - num_components : int
            Number of Gaussian components per output feature.
        """
        # TODO sanity check parameters
        # TODO: add biases
        super(MixtureDensityLayer, self).__init__(incomming)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        n_input_features = incomming.get_output_shape()[-1]
        self.num_units = num_units
        self.num_components = num_components

        def init_params(shape):
            return floatX(
                np.random.uniform(
                    low=-np.sqrt(6. / (n_input_features + num_units)),
                    high=np.sqrt(6. / (n_input_features + num_units)),
                    size=shape))

        init_range = np.sqrt(6. / (n_input_features + num_units))
        if W_mu is None:
            W_mu = init_params((n_input_features, num_units, num_components))
        if W_sigma is None:
            W_sigma = init_params((n_input_features, num_components))
        if W_mixing is None:
            # Initialising with the same values as W_sigma appears 
            # to help learning.
            W_mixing = W_sigma
    
        self.W_mu = self.create_param(
            W_mu, (n_input_features, num_units, num_components), name='W_mu')
        self.W_sigma = self.create_param(
            W_sigma, (n_input_features, num_components), name='W_sigma')
        self.W_mixing = self.create_param(
            W_mixing, (n_input_features, num_components), name='W_mixing')
    
    def get_output_for(self, input, *args, **kwargs):
        mu_activation = T.tensordot(input, self.W_mu, axes=[[1],[0]])
        mu = self.nonlinearity(mu_activation)
        sigma = T.nnet.softplus(T.dot(input, self.W_sigma))
        mixing = T.nnet.softmax(T.dot(input, self.W_mixing))
        return [mu, sigma, mixing]

    def get_params(self):
        return [self.W_mu, self.W_sigma, self.W_mixing]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], 
                self.num_units * self.num_components * 3)

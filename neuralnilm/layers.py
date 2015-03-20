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
    """Mixture density network output layer [#bishop1994]. 

    MDNs are trained to minimise the negative log likelihood of its parameters
    given the data.  This can be done using, for example, SGD.

    Based on work by Amjad Almahairi:
    * amjadmahayri.wordpress.com/2014/04/30/mixture-density-networks
    * github.com/aalmah/ift6266amjad/blob/master/experiments/mdn.py

    :references:
        .. [#bishop1994] Christopher Bishop. "Mixture density networks". 
           Neural Computing Research Group, Aston University. 
           Tech. Rep. NCRG/94/004. (1994)
    """

    def __init__(self, incomming, num_units, 
                 num_components=2,
                 nonlinearity=None, 
                 W_mu=None, 
                 W_sigma=None, 
                 W_mixing=None,
                 b_mu=init.Constant(0.),
                 b_sigma=init.Constant(0.),
                 b_mixing=init.Constant(0.),
                 **kwargs
             ):
        """
        :parameters:
            - num_units : int
                Number of output features.

            - num_components : int
                Number of Gaussian components per output feature.

            - nonlinearity : callable or None
                The nonlinearity that is applied to the layer's mu activations.
                If None is provided, the layer will be linear.

            - W_mu, W_sigma, W_mixing, b_mu, b_sigma, b_mixing : 
                Theano shared variable, numpy array or callable
        """
        # TODO sanity check parameters
        # TODO: add biases
        super(MixtureDensityLayer, self).__init__(incomming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        n_input_features = incomming.get_output_shape()[-1]
        self.num_units = num_units
        self.num_components = num_components

        init_value = np.sqrt(6. / (n_input_features + num_units))
        if W_mu is None:
            W_mu = init.Uniform(init_value)
        if W_sigma is None:
            W_sigma = init.Uniform(init_value)
        if W_mixing is None:
            W_mixing = init.Uniform(init_value)
    
        # weights
        self.W_mu = self.create_param(
            W_mu, (n_input_features, num_units, num_components), name='W_mu')
        self.W_sigma = self.create_param(
            W_sigma, (n_input_features, num_components), name='W_sigma')
        self.W_mixing = self.create_param(
            W_mixing, (n_input_features, num_components), name='W_mixing')

        # biases
        self.b_mu = self.create_param(
            b_mu, (num_units, num_components), name='b_mu')
        self.b_sigma = self.create_param(
            b_sigma, (num_components, ), name='b_sigma')
        self.b_mixing = self.create_param(
            b_mixing, (num_components, ), name='b_mixing')

    
    def get_output_for(self, input, *args, **kwargs):
        """
        :returns:
            mu : (batch_size, num_units, num_components)
            sigma : (batch_size, num_components)
            mixing : (batch_size, num_components)
        """
        # mu
        mu_activation = T.tensordot(input, self.W_mu, axes=[[1],[0]])
        mu_activation += self.b_mu.dimshuffle('x', 0, 1)
        mu = self.nonlinearity(mu_activation)

        # sigma
        sigma_activation = T.dot(input, self.W_sigma)
        sigma_activation += self.b_sigma.dimshuffle('x', 0)
        sigma = T.nnet.softplus(sigma_activation)

        # mixing
        mixing_activation = T.dot(input, self.W_mixing)
        mixing_activation += self.b_mixing.dimshuffle('x', 0)
        mixing = T.nnet.softmax(mixing_activation)

        return mu, sigma, mixing

    def get_params(self):
        return [self.W_mu, self.W_sigma, self.W_mixing] + self.get_bias_params()

    def get_bias_params(self):
        return [self.b_mu, self.b_sigma, self.b_mixing]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], 
                self.num_units * self.num_components * 3)

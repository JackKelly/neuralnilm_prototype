from __future__ import print_function, division
import theano
import theano.tensor as T

import numpy as np

from lasagne.layers import Layer, LSTMLayer, RecurrentLayer, ElemwiseSumLayer
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import floatX

from neuralnilm.utils import remove_nones

            
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
        super(MixtureDensityLayer, self).__init__(incomming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        self.num_components = num_components

        init_value = np.sqrt(6. / (num_inputs + num_units))
        if W_mu is None:
            W_mu = init.Uniform(init_value)
        if W_sigma is None:
            W_sigma = init.Uniform(init_value)
        if num_components == 1:
            W_mixing = None
            b_mixing = None
        elif W_mixing is None:
            W_mixing = init.Uniform(init_value)

        def create_param(param, *args, **kwargs):
            if param is None:
                return None
            else:
                return self.create_param(param, *args, **kwargs)
    
        # weights
        weight_shape = (num_inputs, num_units * num_components)
        self.W_mu = create_param(W_mu, weight_shape, name='W_mu')
        self.W_sigma = create_param(W_sigma, weight_shape, name='W_sigma')
        self.W_mixing = create_param(W_mixing, weight_shape, name='W_mixing')

        # biases
        bias_shape = (num_units * num_components, )
        self.b_mu = create_param(b_mu, bias_shape, name='b_mu')
        self.b_sigma = create_param(b_sigma, bias_shape, name='b_sigma')
        self.b_mixing = create_param(b_mixing, bias_shape, name='b_mixing')

    def get_output_for(self, input, *args, **kwargs):
        """
        :returns:
            A 3D tensor.  The first two dimensions are batch_size and num_units.
            The third dimension is the number of components.
            The last dimension always has exactly 3 elements: mu, sigma, mixing.
        """
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        param_output_shape = (
            self.input_shape[0], self.num_units, self.num_components)

        def forward_pass(param, nonlinearity):
            W = getattr(self, 'W_' + param)
            b = getattr(self, 'b_' + param)
            activation = T.dot(input, W)
            if b is not None:
                activation += b.dimshuffle('x', 0)
            output = nonlinearity(activation)
            output = output.reshape(shape=param_output_shape)
            output = T.shape_padright(output)
            output.name = param
            return output

        mu = forward_pass('mu', self.nonlinearity)
        sigma = forward_pass('sigma', T.nnet.softplus)
        if self.num_components == 1:
            mixing = np.ones(shape=param_output_shape + (1,), 
                             dtype=theano.config.floatX)
        else:
            mixing = forward_pass('mixing', T.nnet.softmax)
        return T.concatenate((mu, sigma, mixing), axis=3)

    def get_params(self):
        weight_params = remove_nones(self.W_mu, self.W_sigma, self.W_mixing)
        return weight_params + self.get_bias_params()

    def get_bias_params(self):
        return remove_nones(self.b_mu, self.b_sigma, self.b_mixing)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units, self.num_components, 3)

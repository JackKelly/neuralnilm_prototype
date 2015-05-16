from __future__ import print_function, division
import theano
import theano.tensor as T

import numpy as np

from lasagne.layers import (Layer, LSTMLayer, RecurrentLayer, ElemwiseSumLayer,
                            Conv1DLayer, DenseLayer)
from lasagne.layers.conv import conv_output_length
from lasagne import nonlinearities
from lasagne import init

from neuralnilm.utils import remove_nones


def BLSTMLayer(*args, **kwargs):
    """Configures forward and backwards LSTM layers to create a
    bidirectional LSTM.

    If learn_init=True then you can't have multiple
    layers of LSTM cells.
    See https://github.com/craffel/nntools/issues/11
    """
    return BidirectionalLayer(LSTMLayer, *args, **kwargs)


def BidirectionalRecurrentLayer(*args, **kwargs):
    """Configures forward and backwards RecurrentLayers to create a
    bidirectional recurrent layer."""
    return BidirectionalLayer(RecurrentLayer, *args, **kwargs)


def BidirectionalLayer(layer_class, *args, **kwargs):
    kwargs.pop('backwards', None)
    l_fwd = layer_class(*args, backwards=False, **kwargs)
    l_bck = layer_class(*args, backwards=True, **kwargs)
    return ElemwiseSumLayer([l_fwd, l_bck])


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
                 W_mu=None,
                 W_sigma=None,
                 W_mixing=None,
                 b_mu=init.Constant(0.),
                 b_sigma=init.Constant(0.),
                 b_mixing=init.Constant(0.),
                 min_sigma=0.0,
                 nonlinearity_mu=nonlinearities.identity,
                 nonlinearity_sigma=T.nnet.softplus,
                 nonlinearity_mixing=T.nnet.softmax,
                 **kwargs):
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

        self.nonlinearity_mu = nonlinearity_mu
        self.nonlinearity_sigma = nonlinearity_sigma
        self.nonlinearity_mixing = nonlinearity_mixing

        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        self.num_components = num_components
        self.min_sigma = min_sigma
        self.param_output_shape = (
            self.input_shape[0], self.num_units, self.num_components, 1)

        init_value = np.sqrt(6. / (num_inputs + num_units))
        if W_mu is None:
            W_mu = init.Uniform(init_value)
        if W_sigma is None:
            W_sigma = init.Uniform(init_value)
        if num_components == 1:
            W_mixing = None
            b_mixing = None
            self.mixing_all_ones = T.constant(
                np.ones(shape=self.param_output_shape,
                        dtype=theano.config.floatX))
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
            A tensor.  The dimensions are:
            1. batch_size
            2. num_units
            3. number of mixture components
            4. The last dimension always has exactly 3 elements:
               1) mu, 2) sigma, 3) mixing.
        """
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        def forward_pass(param):
            W = getattr(self, 'W_' + param)
            b = getattr(self, 'b_' + param)
            nonlinearity = getattr(self, 'nonlinearity_' + param)
            activation = T.dot(input, W)
            if b is not None:
                activation += b.dimshuffle('x', 0)
            output = nonlinearity(activation)
            output = output.reshape(shape=self.param_output_shape)
            output.name = param
            return output

        # mu
        mu = forward_pass('mu')

        # sigma
        sigma = forward_pass('sigma')
        if self.min_sigma:
            sigma += self.min_sigma

        # mixing
        if self.num_components == 1:
            mixing = self.mixing_all_ones
        else:
            mixing = forward_pass('mixing')

        return T.concatenate((mu, sigma, mixing), axis=3)

    def get_params(self):
        weight_params = remove_nones(self.W_mu, self.W_sigma, self.W_mixing)
        return weight_params + self.get_bias_params()

    def get_bias_params(self):
        return remove_nones(self.b_mu, self.b_sigma, self.b_mixing)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units, self.num_components, 3)


class DeConv1DLayer(Conv1DLayer):
    def __init__(self, incomming, filter_length, num_output_channels,
                 border_mode='full', shared_weights=False, shared_biases=False,
                 **kwargs):
        """border_mode needs to be 'full' to get back to original shape."""
        self.num_input_filters = incomming.get_output_shape()[1]
        self.num_output_channels = num_output_channels
        self.filter_length = filter_length
        self.shared_weights = shared_weights
        self.shared_biases = shared_biases

        # if W is not None:
        #     try:
        #         W_shape = W.shape.eval()
        #     except AttributeError:
        #         W_shape = W.shape

        #     TRANSPOSE = (1, 0, 2)
        #     if W_shape.transpose(TRANSPOSE) == self.get_W_shape():
        #         try:
        #             W = W.dimshuffle(TRANSPOSE)
        #         except AttributeError:
        #             W = W.transpose(TRANSPOSE)

        #         self.shared_weights = True

        #     kwargs['W'] = W

        # self.shared_weights = (False if shared_weights is None
        #                        else shared_weights)

        super(DeConv1DLayer, self).__init__(
            incomming, num_filters=num_output_channels,
            filter_length=filter_length, border_mode=border_mode, **kwargs)

    def get_W_shape(self):
        return (self.num_output_channels,
                self.num_input_filters,
                self.filter_length)

    def get_output_shape_for(self, input_shape):
        output_length = conv_output_length(input_shape[2],
                                           self.filter_length,
                                           self.stride,
                                           self.border_mode)

        return (input_shape[0], self.num_output_channels, output_length)

    def get_params(self):
        params = []
        if not self.shared_weights:
            params.append(self.W)
        if not self.shared_biases:
            params.extend(self.get_bias_params())
        return params


class SharedWeightsDenseLayer(DenseLayer):
    def get_params(self):
        return self.get_bias_params()


class PolygonOutputLayer(Layer):
    def __init__(self, incomming, num_units, seq_length,
                 W_scale=init.Uniform(),
                 W_time=init.Uniform(),
                 b_scale=init.Constant(0.),
                 b_time=init.Constant(0.),
                 nonlinearity_scale=nonlinearities.identity,
                 nonlinearity_time=nonlinearities.sigmoid,
                 **kwargs):
        """
        :parameters:
            - num_units : int
                Number of segments the layer can represent.

            - seq_length : int
                Number of segments the layer can represent.

            - nonlinearity_scale, nonlinearity_time : callable or None

            - W_scale, W_time, b_scale, b_time :
                Theano shared variable, numpy array or callable
        """
        super(PolygonOutputLayer, self).__init__(incomming, **kwargs)
        self.num_units = num_units
        self.seq_length = seq_length
        self.nonlinearity_scale = nonlinearity_scale
        self.nonlinearity_time = nonlinearity_time

        # params
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.W_scale = self.create_param(
            W_scale, (num_inputs, num_units), name='W_scale')
        self.b_scale = (self.create_param(
            b_scale, (num_units, ), name='b_scale')
            if b_scale is not None else None)

        if num_units > 1:
            self.W_time = self.create_param(
                W_time, (num_inputs, num_units-1), name='W_time')
            self.b_time = (self.create_param(
                b_time, (num_units-1, ), name='b_time')
                if b_time is not None else None)
        else:
            self.W_time = None
            self.b_time = None

    def get_output_for(self, input, *args, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        def forward_pass(param):
            W = getattr(self, 'W_' + param)
            b = getattr(self, 'b_' + param)
            nonlinearity = getattr(self, 'nonlinearity_' + param)
            activation = T.dot(input, W)
            if b is not None:
                activation += b.dimshuffle('x', 0)
            output = nonlinearity(activation)
            output.name = param
            return output

        scale_output = forward_pass('scale')[:, :, np.newaxis]

        if self.num_units == 1:
            output = scale_output.repeat(repeats=self.seq_length, axis=1)
        else:
            time_output = forward_pass('time')
            # TODO handle batches
            batch_i = 0
            segments = []
            remaining_length = self.seq_length
            for segment_i in range(self.num_units):
                segment_length = remaining_length * time_output[batch_i, segment_i]
                segment = scale_output.repeat(repeats=segment_length.eval(), axis=1)
                segments.append(segment)
                remaining_length -= segment_length
            output = T.concatenate(segments, axis=1)

        return output

    def get_params(self):
        weight_params = remove_nones(self.W_scale, self.W_time)
        return weight_params + self.get_bias_params()

    def get_bias_params(self):
        return remove_nones(self.b_scale, self.b_time)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.seq_length, 1)

"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/neuralnilm/layers.py /mnt/sshfs/imperial/workspace/python/neuralnilm/neuralnilm/"
End:
"""

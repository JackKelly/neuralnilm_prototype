from __future__ import print_function, division
import theano
import theano.tensor as T
import numpy as np


def remove_nones(*args):
    return [a for a in args if a is not None]


def sfloatX(data):
    """Convert scalar to floatX"""
    return getattr(np, theano.config.floatX)(data)


def none_to_dict(data):
    return {} if data is None else data


def ndim_tensor(name, ndim, dtype=theano.config.floatX):
    tensor_type = T.TensorType(dtype=dtype, broadcastable=((False,) * ndim))
    return tensor_type(name=name)

from __future__ import print_function, division
import theano
import numpy as np


def remove_nones(*args):
    return [a for a in args if a is not None]


def sfloatX(data):
    """Convert scalar to floatX"""
    return getattr(np, theano.config.floatX)(data)

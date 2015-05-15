from __future__ import print_function, division
import matplotlib
import logging
from sys import stdout
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from neuralnilm import (Net, RealApplianceSource,
                        BLSTMLayer, DimshuffleLayer,
                        BidirectionalRecurrentLayer)
from neuralnilm.source import (standardise, discretize, fdiff, power_and_fdiff,
                               RandomSegments, RandomSegmentsInMemory,
                               SameLocation)
from neuralnilm.experiment import run_experiment, init_experiment
from neuralnilm.net import TrainingError
from neuralnilm.layers import (MixtureDensityLayer, DeConv1DLayer, 
                               SharedWeightsDenseLayer)
from neuralnilm.objectives import (scaled_cost, mdn_nll, 
                                   scaled_cost_ignore_inactive, ignore_inactive,
                                   scaled_cost3)
from neuralnilm.plot import MDNPlotter, CentralOutputPlotter, Plotter
from neuralnilm.updates import clipped_nesterov_momentum
from neuralnilm.disaggregate import disaggregate

from lasagne.nonlinearities import sigmoid, rectify, tanh, identity
from lasagne.objectives import mse, binary_crossentropy
from lasagne.init import Uniform, Normal, Identity
from lasagne.layers import (LSTMLayer, DenseLayer, Conv1DLayer,
                            ReshapeLayer, FeaturePoolLayer, RecurrentLayer)
from lasagne.layers.batch_norm import BatchNormLayer
from lasagne.updates import nesterov_momentum, momentum
from functools import partial
import os
import __main__
from copy import deepcopy
from math import sqrt
import numpy as np
import theano.tensor as T
import gc

"""
447: first attempt at disaggregation
"""

NAME = os.path.splitext(os.path.split(__main__.__file__)[1])[0]
#PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"
PATH = "/data/dk3810/figures"
SAVE_PLOT_INTERVAL = 500

N_SEQ_PER_BATCH = 64

source_dict = dict(
    filename='/data/dk3810/ukdale.h5',
    window=("2013-03-18", None),
    train_buildings=[1],
    validation_buildings=[1],
    n_seq_per_batch=N_SEQ_PER_BATCH,
    standardise_input=True,
    standardise_targets=True,
    ignore_incomplete=True
#    offset_probability=0.5,
#    ignore_offset_activations=True
)


net_dict = dict(
    save_plot_interval=SAVE_PLOT_INTERVAL,
#    loss_function=partial(ignore_inactive, loss_func=mdn_nll, seq_length=SEQ_LENGTH),
#    loss_function=lambda x, t: mdn_nll(x, t).mean(),
#    loss_function=lambda x, t: (mse(x, t) * MASK).mean(),
    loss_function=lambda x, t: mse(x, t).mean(),
#    loss_function=lambda x, t: binary_crossentropy(x, t).mean(),
#    loss_function=partial(scaled_cost, loss_func=mse),
#    loss_function=ignore_inactive,
#    loss_function=partial(scaled_cost3, ignore_inactive=False),
#    updates_func=momentum,
    updates_func=clipped_nesterov_momentum,
    updates_kwargs={'clip_range': (0, 10)},
    learning_rate=1e-2,
    learning_rate_changes_by_iteration={
        1000: 1e-3,
        5000: 1e-4
    },
    do_save_activations=True,
    auto_reshape=False,
#    plotter=CentralOutputPlotter
    plotter=Plotter(n_seq_to_plot=32)
)


def exp_a(name, target_appliance, seq_length):
    global source
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy.update(dict(
        target_appliance=target_appliance,
        logger=logging.getLogger(name),
        seq_length=seq_length
    ))
    source = SameLocation(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    NUM_FILTERS = 8
    net_dict_copy['layers_config'] = [
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)  # (batch, features, time)
        },
        {
            'label': 'conv0',
            'type': Conv1DLayer,  # convolve over the time axis
            'num_filters': NUM_FILTERS,
            'filter_length': 4,
            'stride': 1,
            'nonlinearity': None,
            'border_mode': 'valid'
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)  # back to (batch, time, features)
        },
        {
            'label': 'dense0',
            'type': DenseLayer,
            'num_units': (seq_length - 3) * NUM_FILTERS,
            'nonlinearity': rectify
        },
        {
            'label': 'dense2',
            'type': DenseLayer,
            'num_units': 16,
            'nonlinearity': rectify
        },
        {
            'type': DenseLayer,
            'num_units': (seq_length - 3) * NUM_FILTERS,
            'nonlinearity': rectify
        },
        {
            'type': ReshapeLayer,
            'shape': (N_SEQ_PER_BATCH, seq_length - 3, NUM_FILTERS)
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)  # (batch, features, time)
        },
        {
            'type': DeConv1DLayer,
            'num_output_channels': 1,
            'filter_length': 4,
            'stride': 1,
            'nonlinearity': None,
            'border_mode': 'full'
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)  # back to (batch, time, features)
        }
    ]
    net = Net(**net_dict_copy)
    return net


def main():
    APPLIANCES = [
        ('a', 'fridge freezer', 512),
        ('b', 'coffee maker', 512),
        ('c', 'dish washer', 2000),
        ('d', 'hair dryer', 256),
        ('e', 'kettle', 256),
        ('f', 'oven', 2000),
        ('g', 'toaster', 256),
        ('h', 'light', 2000),
        ('i', 'washer dryer', 2000)
    ]
    for experiment, appliance, seq_length in APPLIANCES:
        full_exp_name = NAME + experiment
        func_call = init_experiment(PATH, 'a', full_exp_name)
        func_call = func_call[:-1] + ", '{}', {})".format(appliance, seq_length)
        logger = logging.getLogger(full_exp_name)
        try:
            net = eval(func_call)
            run_experiment(net, epochs=20000)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt")
            break
        except Exception as exception:
            logger.exception("Exception")
            # raise
        else:
            del net.source
            del net
            gc.collect()
        finally:
            logging.shutdown()


if __name__ == "__main__":
    main()


"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/scripts/e460.py /mnt/sshfs/imperial/workspace/python/neuralnilm/scripts/"
End:
"""

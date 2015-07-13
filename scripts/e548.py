from __future__ import print_function, division
import matplotlib
import logging
from sys import stdout
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from neuralnilm import (Net, RealApplianceSource)
from neuralnilm.source import (standardise, discretize, fdiff, power_and_fdiff,
                               RandomSegments, RandomSegmentsInMemory,
                               SameLocation, MultiSource)
from neuralnilm.experiment import run_experiment, init_experiment
from neuralnilm.net import TrainingError
from neuralnilm.layers import (MixtureDensityLayer, DeConv1DLayer,
                               SharedWeightsDenseLayer)
from neuralnilm.objectives import (scaled_cost, mdn_nll,
                                   scaled_cost_ignore_inactive, ignore_inactive,
                                   scaled_cost3)
from neuralnilm.plot import MDNPlotter, CentralOutputPlotter, Plotter, RectangularOutputPlotter, StartEndMeanPlotter
from neuralnilm.updates import clipped_nesterov_momentum
from neuralnilm.disaggregate import disaggregate
from neuralnilm.rectangulariser import rectangularise

from lasagne.nonlinearities import sigmoid, rectify, tanh, identity, softmax
from lasagne.objectives import squared_error, binary_crossentropy
from lasagne.init import Uniform, Normal
from lasagne.layers import (DenseLayer, Conv1DLayer,
                            ReshapeLayer, FeaturePoolLayer,
                            DimshuffleLayer, DropoutLayer, ConcatLayer, PadLayer)
from lasagne.updates import nesterov_momentum, momentum
from functools import partial
import os
import __main__
from copy import deepcopy
from math import sqrt
import numpy as np
import theano.tensor as T
import gc


NAME = os.path.splitext(os.path.split(__main__.__file__)[1])[0]
#PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"
PATH = "/data/dk3810/figures"
# PATH = "/home/jack/experiments/neuralnilm/figures"
SAVE_PLOT_INTERVAL = 25000

UKDALE_FILENAME = '/data/dk3810/ukdale.h5'

MAX_TARGET_POWER = 300

TARGET_APPLIANCE = ['fridge freezer', 'fridge', 'freezer']

SEQ_LENGTH = 512
N_SEQ_PER_BATCH = 64
TRAIN_BUILDINGS = [1, 2, 3, 4]
VALIDATION_BUILDINGS = [5]
SKIP_PROBABILITY_FOR_TARGET = 0.5

WINDOW_PER_BUILDING = {
    1: ("2013-03-17", "2014-12-01"),
    2: ("2013-05-22", "2013-10-01"),
    3: ("2013-02-27", "2013-04-01"),
    4: ("2013-03-09", "2013-09-20"),
    5: ("2014-06-29", "2014-08-27")
}

INPUT_STATS = {
    'mean': np.array([297.87216187], dtype=np.float32),
    'std': np.array([374.43884277], dtype=np.float32)
}

real_appliance_source1 = RealApplianceSource(
    filename=UKDALE_FILENAME,
    appliances=[
        TARGET_APPLIANCE,
        ['washer dryer', 'washing machine'],
        'dish washer',
        'kettle',
        'microwave'
    ],
    max_appliance_powers=[MAX_TARGET_POWER, 2400, 2500, 2600, 1500],
    on_power_thresholds=[5] * 5,
    min_on_durations=[60, 1800, 1800, 12, 12],
    min_off_durations=[12, 600, 1800, 12, 12],
    divide_input_by_max_input_power=False,
    window_per_building=WINDOW_PER_BUILDING,
    seq_length=SEQ_LENGTH,
    output_one_appliance=True,
    train_buildings=TRAIN_BUILDINGS,
    validation_buildings=VALIDATION_BUILDINGS,
    n_seq_per_batch=N_SEQ_PER_BATCH,
    skip_probability=0.75,
    skip_probability_for_first_appliance=SKIP_PROBABILITY_FOR_TARGET,
    target_is_start_and_end_and_mean=True,
    standardise_input=True,
    input_stats=INPUT_STATS
)

same_location_source1 = SameLocation(
    filename=UKDALE_FILENAME,
    target_appliance=TARGET_APPLIANCE,
    window_per_building=WINDOW_PER_BUILDING,
    seq_length=SEQ_LENGTH,
    train_buildings=TRAIN_BUILDINGS,
    validation_buildings=VALIDATION_BUILDINGS,
    n_seq_per_batch=N_SEQ_PER_BATCH,
    skip_probability=SKIP_PROBABILITY_FOR_TARGET,
    target_is_start_and_end_and_mean=True,
    standardise_input=True,
    offset_probability=1,
    divide_target_by=MAX_TARGET_POWER,
    input_stats=INPUT_STATS
)

multi_source = MultiSource(
    sources=[
        {
            'source': real_appliance_source1,
            'train_probability': 0.5,
            'validation_probability': 0
        },
        {
            'source': same_location_source1,
            'train_probability': 0.5,
            'validation_probability': 1
        }
    ],
    standardisation_source=same_location_source1
)


def only_train_on_real_data(net, iteration):
    net.logger.info(
        "Iteration {}: Now only training on real data.".format(iteration))
    net.source.sources[0]['train_probability'] = 0.0
    net.source.sources[1]['train_probability'] = 1.0


net_dict = dict(
    save_plot_interval=SAVE_PLOT_INTERVAL,
    loss_function=lambda x, t: squared_error(x, t).mean(),
    updates_func=nesterov_momentum,
    learning_rate=1e-4,
    learning_rate_changes_by_iteration={
        400000: 1e-5,
        500000: 1e-6
    },
    epoch_callbacks={
        350000: only_train_on_real_data
    },
    do_save_activations=True,
    auto_reshape=False,
    layers_config=[
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)  # (batch, features, time)
        },
        {
            'type': PadLayer,
            'width': 4
        },
        {
            'type': Conv1DLayer,  # convolve over the time axis
            'num_filters': 16,
            'filter_size': 4,
            'stride': 1,
            'nonlinearity': None,
            'border_mode': 'valid'
        },
        {
            'type': Conv1DLayer,  # convolve over the time axis
            'num_filters': 16,
            'filter_size': 4,
            'stride': 1,
            'nonlinearity': None,
            'border_mode': 'valid'
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1),  # back to (batch, time, features)
            'label': 'dimshuffle3'
        },
        {
            'type': DenseLayer,
            'num_units': 512 * 8,
            'nonlinearity': rectify,
            'label': 'dense0'
        },
        {
            'type': DenseLayer,
            'num_units': 512 * 6,
            'nonlinearity': rectify,
            'label': 'dense1'
        },
        {
            'type': DenseLayer,
            'num_units': 512 * 4,
            'nonlinearity': rectify,
            'label': 'dense2'
        },
        {
            'type': DenseLayer,
            'num_units': 512,
            'nonlinearity': rectify
        },
        {
            'type': DenseLayer,
            'num_units': 3,
            'nonlinearity': None
        }
    ]
)


def exp_a(name):
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=multi_source,
        plotter=StartEndMeanPlotter(
            n_seq_to_plot=32, max_target_power=MAX_TARGET_POWER)
    ))
    net = Net(**net_dict_copy)
    return net


def main():
    EXPERIMENTS = list('a')
    for experiment in EXPERIMENTS:
        full_exp_name = NAME + experiment
        func_call = init_experiment(PATH, experiment, full_exp_name)
        logger = logging.getLogger(full_exp_name)
        try:
            net = eval(func_call)
            run_experiment(net, epochs=None)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt")
            break
        except Exception as exception:
            logger.exception("Exception")
            # raise
        finally:
            logging.shutdown()


if __name__ == "__main__":
    main()


"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/scripts/e548.py /mnt/sshfs/imperial/workspace/python/neuralnilm/scripts/"
End:
"""

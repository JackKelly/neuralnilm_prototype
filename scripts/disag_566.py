from __future__ import print_function, division
#import matplotlib
import logging
from sys import stdout
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from neuralnilm import (Net, RealApplianceSource)
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
from neuralnilm.plot import (
    StartEndMeanPlotter, plot_disaggregate_start_stop_end)
from neuralnilm.disaggregate import (
    disaggregate_start_stop_end, rectangles_to_matrix,
    rectangles_matrix_to_vector, save_rectangles,
    disag_ae_or_rnn)
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
from os.path import expanduser, join
import __main__
from copy import deepcopy
from math import sqrt
import numpy as np
import theano.tensor as T
import gc

from e566 import (
    net_dict_rnn, net_dict_ae, net_dict_rectangles, get_source, INPUT_STATS)

EXPERIMENT = "e566"
NAME = 'e_disag_' + EXPERIMENT
OUTPUT_PATH = expanduser(
    "~/PhD/experiments/neural_nilm/data_for_BuildSys2015/disag_estimates")
NET_BASE_PATH = "/storage/experiments/neuralnilm/figures/"
GROUND_TRUTH_PATH = expanduser(
    "~/PhD/experiments/neural_nilm/data_for_BuildSys2015/ground_truth_and_mains")
UKDALE_FILENAME = '/data/mine/vadeec/merged/ukdale.h5'
PAD_WIDTH = 1536

print("Changing directory to", OUTPUT_PATH)
os.chdir(OUTPUT_PATH)

logger = logging.getLogger(NAME)
if not logger.handlers:
    fh = logging.FileHandler(NAME + '.log')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(stream=stdout))

logger.setLevel(logging.DEBUG)
logger.info("***********************************")
logger.info("Preparing " + NAME + "...")

# disag
STRIDE = 16


def get_net(appliance, architecture):
    """
    Parameters
    ----------
    appliance : string
    architecture : {'rnn', 'ae', 'rectangles'}
    """
    NET_DICTS = {
        'rnn': net_dict_rnn,
        'ae': net_dict_ae,
        'rectangles': net_dict_rectangles
    }
    net_dict_func = NET_DICTS[architecture]
    source = get_source(
        appliance,
        logger,
        target_is_start_and_end_and_mean=(architecture == 'rectangles'),
        is_rnn=(architecture == 'rnn'),
        window_per_building={  # just load a tiny bit of data. Won't be used.
            1: ("2013-04-12", "2013-05-12"),
            2: ("2013-05-22", "2013-06-22"),
            3: ("2013-02-27", "2013-03-27"),
            4: ("2013-03-09", "2013-04-09"),
            5: ("2014-06-29", "2014-07-29")
        },
        source_type='real_appliance_source',
        filename=UKDALE_FILENAME
    )
    seq_length = source.seq_length
    net_dict = net_dict_func(seq_length)
    epochs = net_dict.pop('epochs')
    net_dict_copy = deepcopy(net_dict)
    experiment_name = EXPERIMENT + "_" + appliance + "_" + architecture
    net_dict_copy.update(dict(
        source=source,
        logger=logger,
        experiment_name=experiment_name
    ))
    net = Net(**net_dict_copy)
    net.plotter.max_target_power = source.max_appliance_powers.values()[0]
    net.load_params(iteration=epochs,
                    path=join(NET_BASE_PATH, experiment_name))
    net.print_net()
    net.compile()
    return net


def disag_rectangles(net, mains, max_target_power, on_power_threshold):
    rectangles = disaggregate_start_stop_end(
        mains, net, stride=STRIDE, max_target_power=max_target_power)
    rectangles_matrix = rectangles_to_matrix(rectangles[0], max_target_power)
    disag_vector = rectangles_matrix_to_vector(
        rectangles_matrix,
        min_on_power=on_power_threshold,
        overlap_threshold=0.50
    )
    return disag_vector


def disaggregate(net, architecture, mains, appliance):
    max_target_power = net.source.max_appliance_powers.values()[0]
    on_power_threshold = net.source.on_power_thresholds[0]
    kwargs = dict(net=net, mains=mains, max_target_power=max_target_power)
    if architecture == 'rectangles':
        kwargs['on_power_threshold'] = on_power_threshold
    else:
        kwargs['stride'] = STRIDE
    DISAG_FUNCS = {
        'rnn': disag_ae_or_rnn,
        'ae': disag_ae_or_rnn,
        'rectangles': disag_rectangles
    }
    disag_func = DISAG_FUNCS[architecture]
    estimates = disag_func(**kwargs)

    estimates = estimates[PAD_WIDTH:-PAD_WIDTH]  # remove padding
    estimates = np.round(estimates).astype(int)
    return estimates


# list of tuples in the form (<appliance name>, <houses>)
APPLIANCES = [
    ('microwave', (1, 2, 3)),
    ('fridge', (1, 2, 4, 5)),
    ('dish washer', (1, 2, 5)),
    ('kettle', (1, 2, 4, 5)),
    ('washing machine', (1, 2, 5))
]


def get_mains(building_i):
        # Load mains
        mains_filename = "building_{:d}_mains.csv".format(building_i)
        mains_filename = join(GROUND_TRUTH_PATH, mains_filename)
        mains = np.loadtxt(mains_filename, delimiter=',')

        # Pad
        mains = np.pad(
            mains, pad_width=(PAD_WIDTH, PAD_WIDTH), mode='constant')

        # Standardise mains
        mains -= INPUT_STATS['mean']
        mains /= INPUT_STATS['std']

        return mains


for appliance, buildings in APPLIANCES:
    for architecture in ['ae', 'rectangles']:
        net = get_net(appliance, architecture)
        for building_i in buildings:
            logger.info("Starting disag for {}, {}, house {}..."
                        .format(appliance, architecture, building_i))
            mains = get_mains(building_i)
            estimates = disaggregate(net, architecture, mains, appliance)
            estimates_filename = (
                "{:s}_building_{:d}_estimates_{:s}.csv"
                .format(architecture, building_i, appliance))
            estimates_filename = join(OUTPUT_PATH, estimates_filename)
            np.savetxt(estimates_filename, estimates, delimiter=',', fmt='%.1d')
            logger.info("Finished disag for {}, {}, house {}."
                        .format(appliance, architecture, building_i))


"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/scripts/disag_566.py /mnt/sshfs/imperial/workspace/python/neuralnilm/scripts/"
End:
"""

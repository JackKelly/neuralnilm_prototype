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
    rectangles_matrix_to_vector, save_rectangles)
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


NAME = 'e566'
PATH = "/data/dk3810/figures"

full_exp_name = NAME + 'a' # TODO
path = os.path.join(PATH, full_exp_name)
print("Changing directory to", path)
os.chdir(path)

logger = logging.getLogger(full_exp_name)
if not logger.handlers:
    fh = logging.FileHandler(full_exp_name + '.log')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(stream=stdout))

logger.setLevel(logging.DEBUG)
logger.info("***********************************")
logger.info("Preparing " + full_exp_name + "...")

# TODO: Load input stats

# disag
STRIDE = 16


from .e566 import net_dict_rnn, net_dict_ae, net_dict_rectangles, get_source


def get_net(appliance, architecture):
    NET_DICTS = {'rnn': net_dict_rnn, 'ae': net_dict_ae, 'rectangles': net_dict_rectangles}
    net_dict_func = NET_DICTS[architecture]
    multi_source = get_source(
        appliance,
        logger,
        target_is_start_and_end_and_mean=(architecture == 'rectangles'),
        is_rnn=(architecture == 'rnn')
    )
    seq_length = multi_source.sources[0]['source'].seq_length
    net_dict = net_dict_func(seq_length)
    net_dict.pop('epochs')
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        source=multi_source
    ))
    net = Net(**net_dict_copy)
    net.plotter.max_target_power = multi_source.sources[1]['source'].divide_target_by
    # TODO: load params
    return net


def disag_rnn(net, mains):
    pass


def disag_ae(net, mains):
    pass


def disag_rectangles(net, mains):
    # TODO: figure out way to pass relevant parameters...
    rectangles = disaggregate_start_stop_end(
        mains, net, stride=STRIDE, max_target_power=MAX_TARGET_POWER)
    rectangles_matrix = rectangles_to_matrix(rectangles[0], MAX_TARGET_POWER)
    disag_vector = rectangles_matrix_to_vector(
        rectangles_matrix, min_on_power=100, overlap_threshold=0.50)
    return disag_vector


def disaggregate(architecture, mains, appliance):
    logger.info("Starting disag...")
    disag_funcs = {'rnn': disag_rnn, 'ae': disag_ae, 'rectangles': disag_rectangles}
    net = get_net(appliance, architecture)
    net.print_net()
    net.compile()
    estimates = disag_funcs[architecture](net, mains)
    return estimates


APPLIANCES = [
    ('microwave', (1, 2, 3)),
    ('fridge', (1, 2, 4, 5)),
    ('dish washer', (1, 2, 5)),
    ('kettle', (1, 2, 4, 5)),
    ('washing machine', (1, 2, 5))
]

for appliance, buildings in APPLIANCES:
    for building_i in buildings:
        filename = "building_{:d}_mains.csv".format(building_i)
        mains = np.loadtxt(filename, sep=',')
        # TODO standardise mains
        for architecture in ['rnn', 'ae', 'rectangles']:
            estimates = disaggregate(architecture, mains, appliance)
            estimates = np.round(estimates).astype(int)
            estimates_filename = "building_{:d}_estimates_{:s}.csv".format(building_i, appliance)
            np.savetxt(estimates_filename, estimates, sep=',')


"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/scripts/disag_566.py /mnt/sshfs/imperial/workspace/python/neuralnilm/scripts/"
End:
"""

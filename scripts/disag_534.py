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
from neuralnilm.plot import MDNPlotter, CentralOutputPlotter, Plotter, RectangularOutputPlotter, StartEndMeanPlotter
from neuralnilm.updates import clipped_nesterov_momentum
from neuralnilm.disaggregate import disaggregate, disaggregate_start_stop_end
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


NAME = 'e534'
PATH = "/data/dk3810/figures"
SAVE_PLOT_INTERVAL = 25000

N_SEQ_PER_BATCH = 64
MAX_TARGET_POWER = 300


source_dict = dict(
    filename='/data/dk3810/ukdale.h5',
    appliances=[
        ['fridge freezer', 'fridge', 'freezer'],
        ['washer dryer', 'washing machine'],
        'hair straighteners',
        'television',
        'dish washer'
    ],
    max_appliance_powers=[MAX_TARGET_POWER, 2400, 500, 200, 2500],
    on_power_thresholds=[5] * 5,
    min_on_durations=[60, 1800, 60, 60, 1800],
    min_off_durations=[12, 600, 12, 12, 1800],
    window=("2013-03-18", "2013-04-18"),
    seq_length=512,
    output_one_appliance=True,
    train_buildings=[1],
    validation_buildings=[1],
    n_seq_per_batch=N_SEQ_PER_BATCH,
    standardise_input=True,
    independently_center_inputs=False,
    skip_probability=0.75,
    skip_probability_for_first_appliance=0.5,
    target_is_start_and_end_and_mean=True,
    one_target_per_seq=False
)


net_dict = dict(
    save_plot_interval=SAVE_PLOT_INTERVAL,
    loss_function=lambda x, t: squared_error(x, t).mean(),
    updates_func=clipped_nesterov_momentum,
    updates_kwargs={'clip_range': (0, 10)},
    learning_rate=1e-3,
    do_save_activations=True,
    auto_reshape=False,
    plotter=StartEndMeanPlotter(
        n_seq_to_plot=32, max_target_power=MAX_TARGET_POWER)
)


def exp_a(name):
    global source
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy.update(dict(
        logger=logging.getLogger(name)
    ))
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    NUM_FILTERS = 16
    target_seq_length = source.output_shape_after_processing()[1]
    net_dict_copy['layers_config'] = [
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
            'num_filters': NUM_FILTERS,
            'filter_size': 4,
            'stride': 1,
            'nonlinearity': None,
            'border_mode': 'valid'
        },
        {
            'type': Conv1DLayer,  # convolve over the time axis
            'num_filters': NUM_FILTERS,
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
            'num_units': 512 * 16,
            'nonlinearity': rectify,
            'label': 'dense0'
        },
        {
            'type': DenseLayer,
            'num_units': 512 * 8,
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
            'num_units': target_seq_length,
            'nonlinearity': None
        }
    ]
    net = Net(**net_dict_copy)
    net.load_params(573215)
    return net


# Load neural net
full_exp_name = NAME + 'a'
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

net = exp_a(full_exp_name)
net.print_net()
net.compile()

# Generate mains data
# create new source, based on net's source,
# but with 5 outputs (so each seq includes entire appliance activation,
# and to make it easier to plot every appliance),
# and long seq length,
# then make one long mains by concatenating each seq
source_dict_copy = deepcopy(source_dict)
source_dict_copy.update(dict(
    logger=logger,
    seq_length=2000,
    output_one_appliance=False
))
mains_source = RealApplianceSource(**source_dict_copy)

N_BATCHES = 3
logger.info("Preparing synthetic mains data for {} batches.".format(N_BATCHES))
mains = None
for batch_i in range(N_BATCHES):
    mains_batch, targets_batch = mains_source._gen_data()
    mains_batch, targets_batch = mains_source._process_data(
        mains_batch, targets_batch)
    if mains is None:
        mains = mains_batch
    else:
        mains = np.concatenate((mains, mains_batch))
mains = mains.flatten()
logger.info("Done preparing synthetic mains data!")

logger.info("Starting disag...")
ax = disaggregate_start_stop_end(mains, net, stride=None)
plt.show()
logger.info("DONE!")

"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/scripts/disag_534.py /mnt/sshfs/imperial/workspace/python/neuralnilm/scripts/"
End:
"""

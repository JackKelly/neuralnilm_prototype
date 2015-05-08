from __future__ import print_function, division
import matplotlib
import logging
from sys import stdout
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from neuralnilm import (Net, RealApplianceSource, 
                        BLSTMLayer, DimshuffleLayer, 
                        BidirectionalRecurrentLayer)
from neuralnilm.source import standardise, discretize, fdiff, power_and_fdiff
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

from nilmtk import DataSet

"""
447: first attempt at disaggregation
"""

NAME = os.path.splitext(os.path.split(__main__.__file__)[1])[0]
#PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"
PATH = "/data/dk3810/figures"
SAVE_PLOT_INTERVAL = 5000

N_SEQ_PER_BATCH = 64
SEQ_LENGTH = 1024

source_dict = dict(
    filename='/data/dk3810/ukdale.h5',
    appliances=[
        ['washer dryer', 'washing machine'],
        'hair straighteners',
        'television',
        'dish washer',
        ['fridge freezer', 'fridge', 'freezer']
    ],
    max_appliance_powers=[2400, 500, 200, 2500, 200],
#    max_input_power=200,   = 5800
    max_diff=200,
    on_power_thresholds=[5] * 5,
    min_on_durations=[1800, 60, 60, 1800, 60],
    min_off_durations=[600, 12, 12, 1800, 12],
    window=("2013-06-01", "2014-07-01"),
    seq_length=SEQ_LENGTH,
#    random_window=64,
    output_one_appliance=True,
    boolean_targets=False,
    train_buildings=[1],
    validation_buildings=[1],
    skip_probability=0.75,
    skip_probability_for_first_appliance=0.2,
    one_target_per_seq=False,
    n_seq_per_batch=N_SEQ_PER_BATCH,
#    subsample_target=4,
    include_diff=False,
    include_power=True,
    clip_appliance_power=False,
    target_is_prediction=False,
#   independently_center_inputs=True,
    standardise_input=True,
    standardise_targets=True,
#    unit_variance_targets=False,
#    input_padding=2,
    lag=0,
    clip_input=False,
    # two_pass=True,
    # clock_type='ramp',
    # clock_period=SEQ_LENGTH
#    classification=True
#    reshape_target_to_2D=True
    # input_stats={'mean': np.array([ 0.05526326], dtype=np.float32),
    #              'std': np.array([ 0.12636775], dtype=np.float32)},
    # target_stats={
    #     'mean': np.array([ 0.04066789,  0.01881946,  
    #                        0.24639061,  0.17608672,  0.10273963], 
    #                      dtype=np.float32),
    #     'std': np.array([ 0.11449792,  0.07338708,  
    #                    0.26608968,  0.33463112,  0.21250485], 
    #                  dtype=np.float32)}

  # input_stats = 
  # {'std': array([ 0.17724811], dtype=float32), 'mean': array([ 0.13002439], dtype=float32)}

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
        20000: 1e-3,
        40000: 1e-4
    },
    do_save_activations=True,
    auto_reshape=False,
#    plotter=CentralOutputPlotter
    plotter=Plotter(n_seq_to_plot=32)
)


def exp_o(name):
    global source
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source,
        learning_rate=1e-1,
        learning_rate_changes_by_iteration={}
    ))
    NUM_FILTERS = 4
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
            'num_units': 1021 * NUM_FILTERS,
            'nonlinearity': rectify
        },
        {
            'label': 'dense1',
            'type': DenseLayer,
            'num_units': 1021,
            'nonlinearity': rectify
        },
        {
            'type': DenseLayer,
            'num_units': 1021 * NUM_FILTERS,
            'nonlinearity': rectify
        },
        {
            'type': ReshapeLayer,
            'shape': (N_SEQ_PER_BATCH, 1021, NUM_FILTERS)
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


os.chdir('/data/dk3810/figures/e446o/')
net = exp_o('e446o')
net.compile()
net.load_params(50000, '/data/dk3810/figures/e446o/e446o.hdf5')

dataset = DataSet('/data/dk3810/ukdale.h5')
dataset.set_window("2013-06-01", "2014-07-01")
elec = dataset.buildings[1].elec
elec.use_alternative_mains()
mains = elec.mains().power_series_all_data()
washer = elec['washer dryer'].power_series_all_data()

N = 131072
estimates = disaggregate(mains.values[:N], net)

fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(mains[:N].index, estimates)
axes[1].plot(mains[:N].index, mains[:N])
axes[2].plot(washer[:N].index, washer[:N])


"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/scripts/e447.py /mnt/sshfs/imperial/workspace/python/neuralnilm/scripts/"
End:
"""

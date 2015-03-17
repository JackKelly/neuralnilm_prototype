from __future__ import print_function, division
import matplotlib
import logging
from sys import stdout
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from neuralnilm import (Net, RealApplianceSource, 
                        BLSTMLayer, DimshuffleLayer, 
                        BidirectionalRecurrentLayer)
from neuralnilm.source import standardise, discretize, fdiff, power_and_fdiff
from neuralnilm.experiment import run_experiment, init_experiment
from neuralnilm.net import TrainingError
from neuralnilm.objectives import scaled_cost

from lasagne.nonlinearities import sigmoid, rectify, tanh
from lasagne.objectives import crossentropy, mse
from lasagne.init import Uniform, Normal
from lasagne.layers import LSTMLayer, DenseLayer, Conv1DLayer, ReshapeLayer, FeaturePoolLayer, RecurrentLayer
from lasagne.updates import nesterov_momentum
from functools import partial
import os
import __main__
from copy import deepcopy
from math import sqrt
import numpy as np

NAME = os.path.splitext(os.path.split(__main__.__file__)[1])[0]
PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"
SAVE_PLOT_INTERVAL = 500
GRADIENT_STEPS = 100

source_dict = dict(
    filename='/data/dk3810/ukdale.h5',
    appliances=[
        ['fridge freezer', 'fridge', 'freezer'], 
        'hair straighteners', 
        'television',
        'dish washer',
        ['washer dryer', 'washing machine']
    ],
    max_appliance_powers=[300, 500, 200, 2500, 2400],
    on_power_thresholds=[5] * 5,
    max_input_power=5900,
    min_on_durations=[60, 60, 60, 1800, 1800],
    min_off_durations=[12, 12, 12, 1800, 600],
    window=("2013-06-01", "2014-07-01"),
    seq_length=1500,
    output_one_appliance=False,
    boolean_targets=False,
    train_buildings=[1],
    validation_buildings=[1], 
    skip_probability=0.7,
    n_seq_per_batch=10,
    subsample_target=3,
    include_diff=False,
    clip_appliance_power=True,
    target_is_prediction=False,
    standardise_input=True,
    standardise_targets=True,
    input_padding=0,
    lag=0,
    input_stats={'mean': np.array([ 0.05526326], dtype=np.float32),
                 'std': np.array([ 0.12636775], dtype=np.float32)},
    target_stats={
        'mean': np.array([ 0.04066789,  0.01881946,  
                           0.24639061,  0.17608672,  0.10273963], 
                         dtype=np.float32),
        'std': np.array([ 0.11449792,  0.07338708,  
                       0.26608968,  0.33463112,  0.21250485], 
                     dtype=np.float32)}
)


def change_learning_rate(net, epoch):
    net.updates = partial(nesterov_momentum, learning_rate=0.001)
    net.compile()


def change_subsample(net, epoch):
    net.source.subsample_target = 3
    net.generate_validation_data_and_set_shapes()


net_dict = dict(        
    save_plot_interval=SAVE_PLOT_INTERVAL,
    loss_function=scaled_cost,
    updates=partial(nesterov_momentum, learning_rate=0.01),
    do_save_activations=True,
    epoch_callbacks={501: change_learning_rate}
)


def exp_a(name):
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net


def exp_b(name):
    # 4 layers
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net



def exp_c(name):
    # BLSTM as last layer
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['subsample_target'] = 3
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BLSTMLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_cell': Normal(std=1/sqrt(N)),
            'peepholes': False
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net



def exp_d(name):
    # 3 layers, 2x2x downsample
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['subsample_target'] = 4
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 2, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 2, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net


def exp_e(name):
    # single 2x downsample
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['subsample_target'] = 2
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 2, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net




def exp_f(name):
    # 2xBLSTM as last layer
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['subsample_target'] = 3
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BLSTMLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_cell': Normal(std=1/sqrt(N)),
            'peepholes': False
        },
        {
            'type': BLSTMLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_cell': Normal(std=1/sqrt(N)),
            'peepholes': False
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net


def exp_g(name):
    # 2xBLSTM (either side of the pool) as last layer
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['subsample_target'] = 3
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': BLSTMLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_cell': Normal(std=1/sqrt(N)),
            'peepholes': False
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BLSTMLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_cell': Normal(std=1/sqrt(N)),
            'peepholes': False
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net



def exp_h(name):
    # Conv AND pool, with 50 filters
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy.update(dict(
        subsample_target=3,
        input_padding=2
    ))
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)  # (batch, features, time)
        },
        {
            'type': Conv1DLayer, # convolve over the time axis
            'num_filters': 50,
            'filter_length': 3,
            'stride': 1,
            'nonlinearity': tanh,
            'W': Normal(std=1/sqrt(N))
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1) # back to (batch, time, features)
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net



def exp_i(name):
    # Conv AND pool, with 50 filters and BLSTM
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy.update(dict(
        subsample_target=3,
        input_padding=2
    ))
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BLSTMLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_cell': Normal(std=1.),
            'peepholes': False
        },
        {
            'type': BLSTMLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_cell': Normal(std=1/sqrt(N)),
            'peepholes': False
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)  # (batch, features, time)
        },
        {
            'type': Conv1DLayer, # convolve over the time axis
            'num_filters': 50,
            'filter_length': 3,
            'stride': 1,
            'nonlinearity': tanh,
            'W': Normal(std=1/sqrt(N))
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1) # back to (batch, time, features)
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BLSTMLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_cell': Normal(std=1/sqrt(N)),
            'peepholes': False
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net



def exp_j(name):
    # 3 BLSTM layers
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['subsample_target'] = 3
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BLSTMLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_cell': Normal(std=1.),
            'peepholes': False
        },
        {
            'type': BLSTMLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_cell': Normal(std=(1/sqrt(N))),
            'peepholes': False
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BLSTMLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_cell': Normal(std=(1/sqrt(N))),
            'peepholes': False
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net


def exp_k(name):
    # no downsampling
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['subsample_target'] = 1
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net


def exp_l(name):
    # 4 layers
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net

def exp_m(name):
    # lag = 5
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['lag'] = 5
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net


def exp_n(name):
    # lag = 25
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['lag'] = 25
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net



def exp_o(name):
    # layerwise pre-training
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['subsample_target'] = 1
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net_dict_copy['layer_changes'] = {
        1501: {
            'remove_from': -3,
            'new_layers': [
                {
                    'type': BidirectionalRecurrentLayer,
                    'num_units': N,
                    'gradient_steps': GRADIENT_STEPS,
                    'W_in_to_hid': Normal(std=1/sqrt(N)),
                    'nonlinearity': tanh
                },
                {
                    'type': DenseLayer,
                    'num_units': source.n_outputs,
                    'nonlinearity': None,
                    'W': Normal(std=(1/sqrt(N)))
                }
            ]
        },
        3001: {
            'remove_from': -3,
            'callback': change_subsample,
            'new_layers': [
                {
                    'type': FeaturePoolLayer,
                    'ds': 3, # number of feature maps to be pooled together
                    'axis': 1, # pool over the time axis
                    'pool_function': T.max
                },
                {
                    'type': BidirectionalRecurrentLayer,
                    'num_units': N,
                    'gradient_steps': GRADIENT_STEPS,
                    'W_in_to_hid': Normal(std=1/sqrt(N)),
                    'nonlinearity': tanh
                },
                {
                    'type': DenseLayer,
                    'num_units': source.n_outputs,
                    'nonlinearity': None,
                    'W': Normal(std=(1/sqrt(N)))
                }
            ]
        }
    }
    net = Net(**net_dict_copy)
    return net


def exp_p(name):
    # layerwise pre-training (pre-train with pool)
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['subsample_target'] = 1
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net_dict_copy['layer_changes'] = {
        1501: {
            'remove_from': -3,
            'callback': change_subsample,
            'new_layers': [
                {
                    'type': BidirectionalRecurrentLayer,
                    'num_units': N,
                    'gradient_steps': GRADIENT_STEPS,
                    'W_in_to_hid': Normal(std=1/sqrt(N)),
                    'nonlinearity': tanh
                },
                {
                    'type': DenseLayer,
                    'num_units': source.n_outputs,
                    'nonlinearity': None,
                    'W': Normal(std=(1/sqrt(N)))
                },
                {
                    'type': FeaturePoolLayer,
                    'ds': 3, # number of feature maps to be pooled together
                    'axis': 1, # pool over the time axis
                    'pool_function': T.max
                }
            ]
        },
        3001: {
            'remove_from': -3,
            'new_layers': [
                {
                    'type': BidirectionalRecurrentLayer,
                    'num_units': N,
                    'gradient_steps': GRADIENT_STEPS,
                    'W_in_to_hid': Normal(std=1/sqrt(N)),
                    'nonlinearity': tanh
                },
                {
                    'type': DenseLayer,
                    'num_units': source.n_outputs,
                    'nonlinearity': None,
                    'W': Normal(std=(1/sqrt(N)))
                }
            ]
        }
    }
    net = Net(**net_dict_copy)
    return net


def exp_r(name):
    # mse cost
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source,
        loss_function=mse
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net

def exp_s(name):
    # 75 neurons per layer
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 75
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net


def exp_t(name):
    # 100 neurons per layer
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 100
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net

def exp_u(name):
    # 50, 10, 50
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 10,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(10)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net

def exp_v(name):
    # N = 25, 5 layers (!), 2x2x subsampling
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['subsample_target'] = 4
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 25
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 2, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 2, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net

def exp_w(name):
    # mean
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 3, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.mean
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net

def exp_x(name):
    # 5x
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['subsample_target'] = 5
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 5, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net

def exp_y(name):
    # 5x and mean
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['subsample_target'] = 5
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 5, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.mean
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    return net


def exp_z(name):
    # N = 50, 5 layers (!), 2x2x subsampling
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['subsample_target'] = 4
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source,
        updates=partial(nesterov_momentum, learning_rate=0.001),
        epoch_callbacks={},
        do_save_activations=False
    ))
    N = 50
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 2, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 2, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.max
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': N,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(N)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(N)))
        }
    ]
    net = Net(**net_dict_copy)
    net.load_params('e277z.hdf5', 1500)
    return net


def main():
    # EXPERIMENTS = list('abcdefghijklmnopqrstuvwxyz')
    EXPERIMENTS = list('z')
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


if __name__ == "__main__":
    main()

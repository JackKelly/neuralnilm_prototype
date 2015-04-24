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
from neuralnilm.layers import MixtureDensityLayer
from neuralnilm.objectives import (scaled_cost, mdn_nll, 
                                   scaled_cost_ignore_inactive, ignore_inactive,
                                   scaled_cost3)
from neuralnilm.plot import MDNPlotter, CentralOutputPlotter
from neuralnilm.updates import clipped_nesterov_momentum

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

"""
e370
longer seq
"""

NAME = os.path.splitext(os.path.split(__main__.__file__)[1])[0]
#PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"
PATH = "/data/dk3810/figures"
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
    min_on_durations=[60, 60, 60, 1800, 1800],
    min_off_durations=[12, 12, 12, 1800, 600],
    window=("2013-06-01", "2014-07-01"),
    seq_length=512,
#    random_window=64,
    output_one_appliance=True,
    boolean_targets=False,
    train_buildings=[1],
    validation_buildings=[1], 
    skip_probability=0.9,
    one_target_per_seq=False,
    n_seq_per_batch=64,
    subsample_target=4,
    include_diff=False,
    include_power=True,
    clip_appliance_power=True,
    target_is_prediction=False,
    independently_center_inputs=True,
    standardise_input=True,
    unit_variance_targets=True,
    input_padding=2,
    lag=0
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
)

N = 50
net_dict = dict(        
    save_plot_interval=SAVE_PLOT_INTERVAL,
#    loss_function=partial(ignore_inactive, loss_func=mdn_nll, seq_length=SEQ_LENGTH),
#    loss_function=lambda x, t: mdn_nll(x, t).mean(),
    loss_function=lambda x, t: mse(x, t).mean(),
#    loss_function=lambda x, t: binary_crossentropy(x, t).mean(),
#    loss_function=partial(scaled_cost, loss_func=mse),
#    loss_function=ignore_inactive,
#    loss_function=partial(scaled_cost3, ignore_inactive=False),
#    updates_func=momentum,
    updates_func=clipped_nesterov_momentum,
    updates_kwargs={'clip_range': (0, 10)},
    learning_rate=1e-3,
    learning_rate_changes_by_iteration={
        500: 1e-4,
        700: 1e-5
        # 800: 1e-4
#        500: 1e-3
       #  4000: 1e-03,
       # 6000: 5e-06,
       # 7000: 1e-06
       # 2000: 5e-06
        # 3000: 1e-05
        # 7000: 5e-06,
        # 10000: 1e-06,
        # 15000: 5e-07,
        # 50000: 1e-07
    },  
    do_save_activations=True,
#    auto_reshape=False,
#    plotter=CentralOutputPlotter
#    plotter=MDNPlotter
)


def exp_a(name):
    global source
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    net_dict_copy['layers_config']= [
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 40,
            'filter_length': 2,
            'stride': 1,
            'nonlinearity': identity,
            'b_in_to_hid': None,
            'border_mode': 'same'
        },
        {
            'type': BatchNormLayer,
            'nonlinearity': rectify
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 40,
#            'gradient_steps': GRADIENT_STEPS,
            'nonlinearity': identity,
            'b_in_to_hid': None,
            'W_hid_to_hid': Identity(scale=0.5),
            'learn_init': True
        },
        {
            'type': BatchNormLayer,
            'axes': (0, 1),
            'nonlinearity': rectify
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 40,
            'filter_length': 4,
            'stride': 4,
            'nonlinearity': identity,
            'b_in_to_hid': None
        },
        {
            'type': BatchNormLayer,
            'nonlinearity': rectify
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 40,
#            'gradient_steps': GRADIENT_STEPS,
            'nonlinearity': identity,
            'b_in_to_hid': None,
            'W_hid_to_hid': Identity(scale=0.5),
            'learn_init': True
        },
        {
            'type': BatchNormLayer,
            'axes': (0, 1),
            'nonlinearity': rectify
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': T.nnet.softplus
        }
    ]
    net = Net(**net_dict_copy)
    return net



def exp_b(name):
    """
    tanh first layer, all others are rectify
    """
    global source
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    net_dict_copy['layers_config']= [
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 40,
            'filter_length': 2,
            'stride': 1,
            'nonlinearity': identity,
            'b_in_to_hid': None,
            'border_mode': 'same'
        },
        {
            'type': BatchNormLayer,
            'nonlinearity': tanh
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 40,
#            'gradient_steps': GRADIENT_STEPS,
            'nonlinearity': identity,
            'b_in_to_hid': None,
            'W_hid_to_hid': Identity(scale=0.5),
            'learn_init': True
        },
        {
            'type': BatchNormLayer,
            'axes': (0, 1),
            'nonlinearity': rectify
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 40,
            'filter_length': 4,
            'stride': 4,
            'nonlinearity': identity,
            'b_in_to_hid': None
        },
        {
            'type': BatchNormLayer,
            'nonlinearity': rectify
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 40,
#            'gradient_steps': GRADIENT_STEPS,
            'nonlinearity': identity,
            'b_in_to_hid': None,
            'W_hid_to_hid': Identity(scale=0.5),
            'learn_init': True
        },
        {
            'type': BatchNormLayer,
            'axes': (0, 1),
            'nonlinearity': rectify
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': T.nnet.softplus
        }
    ]
    net = Net(**net_dict_copy)
    return net



def exp_c(name):
    """
    tanh all the way through.  Identity init of RNNs
    """
    global source
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    net_dict_copy['layers_config']= [
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 40,
            'filter_length': 2,
            'stride': 1,
            'nonlinearity': identity,
            'b_in_to_hid': None,
            'border_mode': 'same'
        },
        {
            'type': BatchNormLayer,
            'nonlinearity': tanh
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 40,
#            'gradient_steps': GRADIENT_STEPS,
            'nonlinearity': identity,
            'b_in_to_hid': None,
            'W_hid_to_hid': Identity(scale=0.5),
            'learn_init': True
        },
        {
            'type': BatchNormLayer,
            'axes': (0, 1),
            'nonlinearity': tanh
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 40,
            'filter_length': 4,
            'stride': 4,
            'nonlinearity': identity,
            'b_in_to_hid': None
        },
        {
            'type': BatchNormLayer,
            'nonlinearity': tanh
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 40,
#            'gradient_steps': GRADIENT_STEPS,
            'nonlinearity': identity,
            'b_in_to_hid': None,
            'W_hid_to_hid': Identity(scale=0.5),
            'learn_init': True
        },
        {
            'type': BatchNormLayer,
            'axes': (0, 1),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': T.nnet.softplus
        }
    ]
    net = Net(**net_dict_copy)
    return net


def exp_d(name):
    """
    e380 (tanh all the way though, default inits) with batch norm
    """
    global source
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    net_dict_copy['layers_config']= [
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 40,
            'filter_length': 2,
            'stride': 1,
            'nonlinearity': identity,
            'b_in_to_hid': None,
            'border_mode': 'same'
        },
        {
            'type': BatchNormLayer,
            'nonlinearity': tanh
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 40,
#            'gradient_steps': GRADIENT_STEPS,
            'nonlinearity': identity,
            'b_in_to_hid': None,
            'learn_init': True
        },
        {
            'type': BatchNormLayer,
            'axes': (0, 1),
            'nonlinearity': tanh
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 40,
            'filter_length': 4,
            'stride': 4,
            'nonlinearity': identity,
            'b_in_to_hid': None
        },
        {
            'type': BatchNormLayer,
            'nonlinearity': tanh
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 40,
#            'gradient_steps': GRADIENT_STEPS,
            'nonlinearity': identity,
            'b_in_to_hid': None,
            'learn_init': True
        },
        {
            'type': BatchNormLayer,
            'axes': (0, 1),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': T.nnet.softplus
        }
    ]
    net = Net(**net_dict_copy)
    return net



def exp_e(name):
    """
    e380 again
    """
    global source
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=source
    ))
    net_dict_copy['layers_config']= [
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 40,
            'filter_length': 2,
            'stride': 1,
            'nonlinearity': tanh,
            'border_mode': 'same'
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 40,
#            'gradient_steps': GRADIENT_STEPS,
            'nonlinearity': tanh,
            'learn_init': True
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 40,
            'filter_length': 4,
            'stride': 4,
            'nonlinearity': tanh
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 40,
#            'gradient_steps': GRADIENT_STEPS,
            'nonlinearity': tanh,
            'learn_init': True
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': T.nnet.softplus
        }
    ]
    net = Net(**net_dict_copy)
    return net




def main():
    #     EXPERIMENTS = list('abcdefghijklmnopqrstuvwxyz')
    EXPERIMENTS = list('abcde')
    for experiment in EXPERIMENTS:
        full_exp_name = NAME + experiment
        func_call = init_experiment(PATH, experiment, full_exp_name)
        logger = logging.getLogger(full_exp_name)
        try:
            net = eval(func_call)
            run_experiment(net, epochs=1000)
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

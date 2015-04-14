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
from neuralnilm.plot import MDNPlotter

from lasagne.nonlinearities import sigmoid, rectify, tanh
from lasagne.objectives import mse
from lasagne.init import Uniform, Normal
from lasagne.layers import (LSTMLayer, DenseLayer, Conv1DLayer, 
                            ReshapeLayer, FeaturePoolLayer, RecurrentLayer)
from lasagne.updates import nesterov_momentum, momentum
from functools import partial
import os
import __main__
from copy import deepcopy
from math import sqrt
import numpy as np
import theano.tensor as T

NAME = os.path.splitext(os.path.split(__main__.__file__)[1])[0]
PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"
SAVE_PLOT_INTERVAL = 1000
GRADIENT_STEPS = 100
SEQ_LENGTH = 512

source_dict = dict(
    filename='/data/dk3810/ukdale.h5',
    appliances=[
        ['fridge freezer', 'fridge', 'freezer'], 
#        'hair straighteners', 
#        'television',
        'dish washer',
        ['washer dryer', 'washing machine']
    ],
    max_appliance_powers=[300, 2500, 2400],
    on_power_thresholds=[5] * 5,
    max_input_power=5900,
    min_on_durations=[60, 1800, 1800],
    min_off_durations=[12, 1800, 600],
    window=("2013-06-01", "2014-07-01"),
    seq_length=SEQ_LENGTH,
    output_one_appliance=False,
    boolean_targets=False,
    train_buildings=[1],
    validation_buildings=[1], 
#   skip_probability=0.5,
    one_target_per_seq=True,
    n_seq_per_batch=16,
    subsample_target=4,
    include_diff=False,
    clip_appliance_power=True,
    target_is_prediction=False,
#    independently_center_inputs = True,
    standardise_input=True,
    unit_variance_targets=True,
    input_padding=0,
    lag=0
#    reshape_target_to_2D=True,
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
#    loss_function=lambda x, t: mse(x, t).mean(),
#    loss_function=partial(scaled_cost, loss_func=mse),
#    loss_function=ignore_inactive,
    loss_function=partial(scaled_cost3, ignore_inactive=True),
    updates_func=momentum,
    learning_rate=1e-3,
    learning_rate_changes_by_iteration={
         25: 5e-4,
        100: 1e-4
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
    do_save_activations=True
#    plotter=MDNPlotter
)

def exp_a(name):
    global source
    # source_dict_copy = deepcopy(source_dict)
    # source = RealApplianceSource(**source_dict_copy)
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
        # {
        #     'type': FeaturePoolLayer,
        #     'ds': 2, # number of feature maps to be pooled together
        #     'axis': 1, # pool over the time axis
        #     'pool_function': T.max
        # },
        # {
        #     'type': BidirectionalRecurrentLayer,
        #     'num_units': N,
        #     'gradient_steps': GRADIENT_STEPS,
        #     'W_in_to_hid': Normal(std=1/sqrt(N)),
        #     'nonlinearity': tanh
        # },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'W': Normal(std=1/sqrt(N)),
            'nonlinearity': T.nnet.softplus
        }

        # {
        #     'type': MixtureDensityLayer,
        #     'num_units': source.n_outputs,
        #     'num_components': 1,
        #     'nonlinearity_mu': T.nnet.softplus
        # }
    ]
    net = Net(**net_dict_copy)
    return net


def main():
    #     EXPERIMENTS = list('abcdefghijklmnopqrstuvwxyz')
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
            raise
        finally:
            logging.shutdown()


if __name__ == "__main__":
    main()

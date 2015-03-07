from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from neuralnilm import Net, RealApplianceSource, BLSTMLayer, DimshuffleLayer
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.objectives import crossentropy, mse
from lasagne.init import Uniform, Normal
from lasagne.layers import LSTMLayer, DenseLayer, Conv1DLayer, ReshapeLayer, FeaturePoolLayer
from lasagne.updates import nesterov_momentum
from functools import partial
import os
from neuralnilm.source import standardise, discretize, fdiff, power_and_fdiff
from neuralnilm.experiment import run_experiment
from neuralnilm.net import TrainingError
import __main__
from copy import deepcopy

NAME = os.path.splitext(os.path.split(__main__.__file__)[1])[0]
PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"
SAVE_PLOT_INTERVAL = 1000
GRADIENT_STEPS = 100

"""
e233
based on e131c but with:
* lag=32
* pool

e234
* init final layer and conv layer

235
no lag

236
should be exactly as 131c: no pool, no lag, no init for final and conv layer

237
putting the pool back

238
seems pooling hurts us! disable pooling.
enable lag = 32

239
BLSTM
lag = 20

240
LSTM not BLSTM
various lags

ideas for next TODO:
* 3 LSTM layers with smaller conv between them
* why does pooling hurt us?
"""

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
    subsample_target=5,
    include_diff=False,
    clip_appliance_power=True,
    lag=0
)

net_dict = dict(        
    save_plot_interval=SAVE_PLOT_INTERVAL,
    loss_function=crossentropy,
    updates=partial(nesterov_momentum, learning_rate=1.0),
    layers_config=[
        {
            'type': DenseLayer,
            'num_units': 50,
            'nonlinearity': sigmoid,
            'W': Uniform(25),
            'b': Uniform(25)
        },
        {
            'type': DenseLayer,
            'num_units': 50,
            'nonlinearity': sigmoid,
            'W': Uniform(10),
            'b': Uniform(10)
        },
        {
            'type': LSTMLayer,
            'num_units': 40,
            'W_in_to_cell': Uniform(5),
            'gradient_steps': GRADIENT_STEPS,
            'peepholes': False
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 20,
            'filter_length': 5,
            'stride': 5,
            'nonlinearity': sigmoid
#           'W': Uniform(1)
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        # {
        #     'type': FeaturePoolLayer,
        #     'ds': 5, # number of feature maps to be pooled together
        #     'axis': 1 # pool over the time axis
        # },
        {
            'type': LSTMLayer,
            'num_units': 80,
            'W_in_to_cell': Uniform(5),
            'gradient_steps': GRADIENT_STEPS,
            'peepholes': False
        }
    ]
)

def exp_a(name):
    # like 239 but LSTM not BLSTM and no lag and clip appliance power
    source = RealApplianceSource(**source_dict)
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(experiment_name=name, source=source))
    net_dict_copy['layers_config'].append(
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': sigmoid
        }

    )
    net = Net(**net_dict_copy)
    return net


def exp_b(name):
    # as A but lag = 10
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['lag'] = 10
    source = RealApplianceSource(**source_dict_copy)

    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(experiment_name=name, source=source))
    net_dict_copy['layers_config'].append(
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': sigmoid
        }

    )
    net = Net(**net_dict_copy)
    return net


def exp_c(name):
    # as A but lag = 20
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['lag'] = 20
    source = RealApplianceSource(**source_dict_copy)

    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(experiment_name=name, source=source))
    net_dict_copy['layers_config'].append(
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': sigmoid
        }

    )
    net = Net(**net_dict_copy)
    return net


def exp_d(name):
    # as A but lag = 40
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['lag'] = 40
    source = RealApplianceSource(**source_dict_copy)

    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(experiment_name=name, source=source))
    net_dict_copy['layers_config'].append(
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': sigmoid
        }

    )
    net = Net(**net_dict_copy)
    return net


def exp_e(name):
    # as A but lag = 80
    source_dict_copy = deepcopy(source_dict)
    source_dict_copy['lag'] = 80
    source = RealApplianceSource(**source_dict_copy)

    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(experiment_name=name, source=source))
    net_dict_copy['layers_config'].append(
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': sigmoid
        }

    )
    net = Net(**net_dict_copy)
    return net



def init_experiment(experiment):
    full_exp_name = NAME + experiment
    func_call = 'exp_{:s}(full_exp_name)'.format(experiment)
    print("***********************************")
    print("Preparing", full_exp_name, "...")
    net = eval(func_call)
    return net


def main():
    for experiment in list('abcde'):
        full_exp_name = NAME + experiment
        path = os.path.join(PATH, full_exp_name)
        try:
            net = init_experiment(experiment)
            run_experiment(net, path, epochs=10000)
        except KeyboardInterrupt:
            break
        except TrainingError as exception:
            print("EXCEPTION:", exception)
        except Exception as exception:
            print("EXCEPTION:", exception)


if __name__ == "__main__":
    main()

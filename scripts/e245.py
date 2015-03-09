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
import numpy as np


NAME = os.path.splitext(os.path.split(__main__.__file__)[1])[0]
PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"
SAVE_PLOT_INTERVAL = 250
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

241
output is prediction

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
    # skip_probability=0.0,
    n_seq_per_batch=10,
    # subsample_target=5,
    include_diff=False,
    clip_appliance_power=True,
    target_is_prediction=True
    #lag=0
)

net_dict = dict(        
    save_plot_interval=SAVE_PLOT_INTERVAL,
    loss_function=crossentropy,
    updates=partial(nesterov_momentum, learning_rate=1.0),
    layers_config=[
        {
            'type': LSTMLayer,
            'num_units': 50,
            'gradient_steps': GRADIENT_STEPS,
            'peepholes': False,
            'W_in_to_cell': np.array([-1.4713676, -3.4655192, -3.290631, -0.2836367, -1.2835358, -4.309059, -0.8759276, -0.94482726, -0.49211252, -0.16434391, -3.8585103, -0.28171384, -1.3792592, 0.025873292, -0.70238507, -3.800167, -0.81310165, -0.292926, -0.16176611, -0.5409762, -0.8898347, -0.7629556, -0.9762615, -1.5364829, -0.63402534, -2.6672506, -0.581932, -1.132505, -0.1686768, -0.5464546, -0.7155229, -0.98150927, -2.487797, -1.6145508, -0.7313065, -0.10417248, -1.1441768, -2.7225356, -0.4825784, -0.35094073, -1.231792, -4.0353093, -0.120160125, -0.56850415, -0.9948139, -0.033999957, -0.6853825, -0.6664621, -1.7320231, -0.6675205], dtype=np.float32).reshape((1,50))
        }
    ]
)


def exp_a(name):
    # global source
    # source = RealApplianceSource(**source_dict)
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
    for experiment in list('a'):
        full_exp_name = NAME + experiment
        path = os.path.join(PATH, full_exp_name)
        try:
            net = init_experiment(experiment)
            run_experiment(net, path, epochs=1000)
        except KeyboardInterrupt:
            break
        except TrainingError as exception:
            print("EXCEPTION:", exception)
        except Exception as exception:
            print("EXCEPTION:", exception)


if __name__ == "__main__":
    main()

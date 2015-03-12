from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from neuralnilm import Net, RealApplianceSource, BLSTMLayer, DimshuffleLayer
from neuralnilm.net import BidirectionalRecurrentLayer
from lasagne.nonlinearities import sigmoid, rectify, tanh
from lasagne.objectives import crossentropy, mse
from lasagne.init import Uniform, Normal
from lasagne.layers import LSTMLayer, DenseLayer, Conv1DLayer, ReshapeLayer, FeaturePoolLayer, RecurrentLayer
from lasagne.updates import nesterov_momentum
from functools import partial
import os
from neuralnilm.source import standardise, discretize, fdiff, power_and_fdiff
from neuralnilm.experiment import run_experiment
from neuralnilm.net import TrainingError
import __main__
from copy import deepcopy
from math import sqrt

NAME = os.path.splitext(os.path.split(__main__.__file__)[1])[0]
PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"
SAVE_PLOT_INTERVAL = 500
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

260
standardise inputs and outputs.

261
trying just 3 appliances.  Standardisation

263
conv1d between layers

ideas for next TODO:
* 3 LSTM layers with smaller conv between them
* why does pooling hurt us?
"""

from theano.ifelse import ifelse
import theano.tensor as T

THRESHOLD = 0
def scaled_cost(x, t):
    sq_error = (x - t) ** 2
    def mask_and_mean_sq_error(mask):
        masked_sq_error = sq_error[mask.nonzero()]
        mean = masked_sq_error.mean()
        mean = ifelse(T.isnan(mean), 0.0, mean)
        return mean
    above_thresh_mean = mask_and_mean_sq_error(t > THRESHOLD)
    below_thresh_mean = mask_and_mean_sq_error(t <= THRESHOLD)
    return (above_thresh_mean + below_thresh_mean) / 2.0


source_dict = dict(
    filename='/data/dk3810/ukdale.h5',
    appliances=[
        ['fridge freezer', 'fridge', 'freezer'], 
        'hair straighteners', 
        'television'
        #'dish washer',
        #['washer dryer', 'washing machine']
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
    target_is_prediction=False,
    standardise_input=True,
    standardise_targets=True,
    input_padding=0,
    lag=0
)


def change_learning_rate(net, epoch):
    net.updates = partial(nesterov_momentum, learning_rate=0.001)
    net.compile()


def change_subsample(net, epoch):
    net.source.subsample_target = 5
    net.generate_validation_data_and_set_shapes()

net_dict = dict(        
    save_plot_interval=SAVE_PLOT_INTERVAL,
    loss_function=scaled_cost,
    updates=partial(nesterov_momentum, learning_rate=0.01),
    do_save_activations=True,
    epoch_callbacks={500: change_learning_rate}
)


def exp_a(name):
    global source
    source_dict_copy = deepcopy(source_dict)
    source = RealApplianceSource(**source_dict_copy)
    source.subsample_target = 4
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(experiment_name=name, source=source))
    net_dict_copy['layers_config'] = [
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 25,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1.),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 2, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.mean
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 10,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(25)),
            'nonlinearity': tanh
        },
        {
            'type': FeaturePoolLayer,
            'ds': 2, # number of feature maps to be pooled together
            'axis': 1, # pool over the time axis
            'pool_function': T.mean
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 5,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(10)),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 10,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(5)),
            'nonlinearity': tanh
        },
        {
            'type': BidirectionalRecurrentLayer,
            'num_units': 25,
            'gradient_steps': GRADIENT_STEPS,
            'W_in_to_hid': Normal(std=1/sqrt(10)),
            'nonlinearity': tanh
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': None,
            'W': Normal(std=(1/sqrt(25)))
        }
    ]
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
            run_experiment(net, path, epochs=None)
        except KeyboardInterrupt:
            break
        except TrainingError as exception:
            print("EXCEPTION:", exception)
        except Exception as exception:
            print("EXCEPTION:", exception)
            raise


if __name__ == "__main__":
    main()

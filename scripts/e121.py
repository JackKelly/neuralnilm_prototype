from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from neuralnilm import Net, RealApplianceSource, BLSTMLayer, SubsampleLayer, DimshuffleLayer
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.objectives import crossentropy, mse
from lasagne.init import Uniform, Normal
from lasagne.layers import LSTMLayer, DenseLayer, Conv1DLayer, ReshapeLayer
from lasagne.updates import adagrad, nesterov_momentum
from functools import partial
import os
from neuralnilm.source import standardise
from neuralnilm.experiment import run_experiment
from neuralnilm.net import TrainingError
import __main__

NAME = os.path.splitext(os.path.split(__main__.__file__)[1])[0]
PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"
SAVE_PLOT_INTERVAL = 250
GRADIENT_STEPS = 100

"""
e103
Discovered that bottom layer is hardly changing.  So will try
just a single lstm layer

e104
standard init
lower learning rate

e106
lower learning rate to 0.001

e108
is e107 but with batch size of 5

e109
Normal(1) for LSTM

e110
* Back to Uniform(5) for LSTM
* Using nntools eb17bd923ef9ff2cacde2e92d7323b4e51bb5f1f
RESULTS: Seems to run fine again!

e111
* Try with nntools head
* peepholes=False
RESULTS: appears to be working well.  Haven't seen a NaN, 
even with training rate of 0.1

e112
* n_seq_per_batch = 50

e114
* Trying looking at layer by layer training again.
* Start with single LSTM layer

e115
* Learning rate = 1

e116
* Standard inits

e117
* Uniform(1) init

e119
* Learning rate 10
# Result: didn't work well!

e120
* init: Normal(1)
* not as good as Uniform(5)

e121
* Uniform(25)

e122
* Just 10 cells
* Uniform(5)
"""

def exp_a(name):
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            ['fridge freezer', 'fridge', 'freezer'], 
            'hair straighteners', 
            'television'
            # 'dish washer',
            # ['washer dryer', 'washing machine']
        ],
        max_appliance_powers=[300, 500, 200], #, 2500, 2400],
        on_power_thresholds=[20, 20, 20], #, 20, 20],
        max_input_power=1000,
        min_on_durations=[60, 60, 60], #, 1800, 1800],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0,
        n_seq_per_batch=50
    )

    net = Net(
        experiment_name=name,
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=1.0),
        layers_config=[
            {
                'type': LSTMLayer,
                'num_units': 50,
                'W_in_to_cell': Uniform(25),
                'gradient_steps': GRADIENT_STEPS,
                'peepholes': False
            },
            {
                'type': DenseLayer,
                'num_units': source.n_outputs,
                'nonlinearity': sigmoid
            }
        ]
    )
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
            run_experiment(net, path, epochs=5000)
        except KeyboardInterrupt:
            break
        except TrainingError as e:
            print("EXCEPTION:", e)


if __name__ == "__main__":
    main()

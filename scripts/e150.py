from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from neuralnilm import Net, RealApplianceSource, BLSTMLayer, DimshuffleLayer
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.objectives import crossentropy, mse
from lasagne.init import Uniform, Normal
from lasagne.layers import LSTMLayer, DenseLayer, Conv1DLayer, ReshapeLayer
from neuralnilm.updates import nesterov_momentum
from functools import partial
import os
from neuralnilm.source import standardise, discretize, fdiff, power_and_fdiff
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

e125
* Pre-train lower layers

e128
* Add back all 5 appliances
* Seq length 1500
* skip_prob = 0.7

e129
* max_input_power = None
* 2nd layer has Uniform(5)
* pre-train bottom layer for 2000 epochs
* add third layer at 4000 epochs

e131

e138
* Trying to replicate e82 and then break it ;)

e140
diff

e141
conv1D layer has Uniform(1), as does 2nd LSTM layer

e142
diff AND power

e144
diff and power and max power is 5900

e145
Uniform(25) for first layer

e146
gradient clip and use peepholes

e147
* try again with new code

e148
* learning rate 0.1

e150
* Same as e149 but without peepholes and using LSTM not BLSTM
"""


def set_subsample_target(net, epoch):
    net.source.subsample_target = 5


def exp_a(name):
    # e148 but with larger net
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            ['fridge freezer', 'fridge', 'freezer'], 
            'hair straighteners', 
            'television'
#            'dish washer',
#            ['washer dryer', 'washing machine']
        ],
        max_appliance_powers=[300, 500, 200, 2500, 2400],
        on_power_thresholds=[5, 5, 5, 5, 5],
        max_input_power=5900,
        min_on_durations=[60, 60, 60, 1800, 1800],
        min_off_durations=[12, 12, 12, 1800, 600],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=False,
        boolean_targets=False,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0,
        n_seq_per_batch=50,
        subsample_target=5,
        include_diff=True
    )

    net = Net(
        experiment_name=name,
        source=source,
        save_plot_interval=250,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=.1, clip_range=(-1, 1)),
        layers_config=[
            {
                'type': LSTMLayer,
                'num_units': 60,
                'W_in_to_cell': Uniform(25),
                'gradient_steps': GRADIENT_STEPS,
                'peepholes': False
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 80,
                'filter_length': 5,
                'stride': 5,
                'nonlinearity': sigmoid,
                'W': Uniform(1)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': LSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(1),
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



def exp_b(name):
    # 'A' but with 4 appliances and seq length of 1500 and 25 seq per batch
    # (50 ran out of GPU memory)
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            ['fridge freezer', 'fridge', 'freezer'], 
            'hair straighteners', 
            'television',
            'dish washer'
#            ['washer dryer', 'washing machine']
        ],
        max_appliance_powers=[300, 500, 200, 2500, 2400],
        on_power_thresholds=[5, 5, 5, 5, 5],
        max_input_power=5900,
        min_on_durations=[60, 60, 60, 1800, 1800],
        min_off_durations=[12, 12, 12, 1800, 600],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1500,
        output_one_appliance=False,
        boolean_targets=False,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0,
        n_seq_per_batch=25,
        subsample_target=5,
        include_diff=True
    )

    net = Net(
        experiment_name=name,
        source=source,
        save_plot_interval=250,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=.1, clip_range=(-1, 1)),
        layers_config=[
            {
                'type': LSTMLayer,
                'num_units': 60,
                'W_in_to_cell': Uniform(25),
                'gradient_steps': GRADIENT_STEPS,
                'peepholes': False

            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 80,
                'filter_length': 5,
                'stride': 5,
                'nonlinearity': sigmoid,
                'W': Uniform(1)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': LSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(1),
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



def exp_c(name):
    # 'A' but with 5 appliances and seq length of 1500
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            ['fridge freezer', 'fridge', 'freezer'], 
            'hair straighteners', 
            'television',
            'dish washer',
            ['washer dryer', 'washing machine']
        ],
        max_appliance_powers=[300, 500, 200, 2500, 2400],
        on_power_thresholds=[5, 5, 5, 5, 5],
        max_input_power=5900,
        min_on_durations=[60, 60, 60, 1800, 1800],
        min_off_durations=[12, 12, 12, 1800, 600],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1500,
        output_one_appliance=False,
        boolean_targets=False,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0,
        n_seq_per_batch=25,
        subsample_target=5,
        include_diff=True
    )

    net = Net(
        experiment_name=name,
        source=source,
        save_plot_interval=250,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=.1, clip_range=(-1, 1)),
        layers_config=[
            {
                'type': LSTMLayer,
                'num_units': 60,
                'W_in_to_cell': Uniform(25),
                'gradient_steps': GRADIENT_STEPS,
                'peepholes': False

            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 80,
                'filter_length': 5,
                'stride': 5,
                'nonlinearity': sigmoid,
                'W': Uniform(1)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': LSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(1),
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



def exp_d(name):
    # 'C' but with pre-training
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            ['fridge freezer', 'fridge', 'freezer'], 
            'hair straighteners', 
            'television',
            'dish washer',
            ['washer dryer', 'washing machine']
        ],
        max_appliance_powers=[300, 500, 200, 2500, 2400],
        on_power_thresholds=[5, 5, 5, 5, 5],
        max_input_power=5900,
        min_on_durations=[60, 60, 60, 1800, 1800],
        min_off_durations=[12, 12, 12, 1800, 600],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1500,
        output_one_appliance=False,
        boolean_targets=False,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0,
        n_seq_per_batch=25,
        include_diff=True
    )

    net = Net(
        experiment_name=name,
        source=source,
        save_plot_interval=250,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=.1, clip_range=(-1, 1)),
        layers_config=[
            {
                'type': LSTMLayer,
                'num_units': 60,
                'W_in_to_cell': Uniform(25),
                'gradient_steps': GRADIENT_STEPS,
                'peepholes': False

            },
            {
                'type': DenseLayer,
                'num_units': source.n_outputs,
                'nonlinearity': sigmoid
            }            
        ],
        layer_changes={
            1000: {
                'remove_from': -3,
                'callback': set_subsample_target,
                'new_layers': 
                [
                    {
                        'type': DimshuffleLayer,
                        'pattern': (0, 2, 1)
                    },
                    {
                        'type': Conv1DLayer,
                        'num_filters': 80,
                        'filter_length': 5,
                        'stride': 5,
                        'nonlinearity': sigmoid,
                        'W': Uniform(1)
                    },
                    {
                        'type': DimshuffleLayer,
                        'pattern': (0, 2, 1)
                    },
                    {
                        'type': DenseLayer,
                        'num_units': source.n_outputs,
                        'nonlinearity': sigmoid
                    }
                ],
            2000: {
                'remove_from': -3,
                'new_layers': 
                [
                    {
                        'type': LSTMLayer,
                        'num_units': 80,
                        'W_in_to_cell': Uniform(1),
                        'gradient_steps': GRADIENT_STEPS,
                'peepholes': False

                    },
                    {
                        'type': DenseLayer,
                        'num_units': source.n_outputs,
                        'nonlinearity': sigmoid
                    }            
                ]
            }
        }
        }
    )
    return net



def exp_e(name):
    # 'C' but with stride-3 Conv
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            ['fridge freezer', 'fridge', 'freezer'], 
            'hair straighteners', 
            'television',
            'dish washer',
            ['washer dryer', 'washing machine']
        ],
        max_appliance_powers=[300, 500, 200, 2500, 2400],
        on_power_thresholds=[5, 5, 5, 5, 5],
        max_input_power=5900,
        min_on_durations=[60, 60, 60, 1800, 1800],
        min_off_durations=[12, 12, 12, 1800, 600],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1500,
        output_one_appliance=False,
        boolean_targets=False,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0,
        n_seq_per_batch=25,
        subsample_target=3,
        include_diff=True
    )

    net = Net(
        experiment_name=name,
        source=source,
        save_plot_interval=250,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=.1, clip_range=(-1, 1)),
        layers_config=[
            {
                'type': LSTMLayer,
                'num_units': 60,
                'W_in_to_cell': Uniform(25),
                'gradient_steps': GRADIENT_STEPS,
                'peepholes': False

            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 80,
                'filter_length': 3,
                'stride': 3,
                'nonlinearity': sigmoid,
                'W': Uniform(1)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': LSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(1),
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



def exp_f(name):
    # 'E' but with stride-3 Conv and 3 layers
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            ['fridge freezer', 'fridge', 'freezer'], 
            'hair straighteners', 
            'television',
            'dish washer',
            ['washer dryer', 'washing machine']
        ],
        max_appliance_powers=[300, 500, 200, 2500, 2400],
        on_power_thresholds=[5, 5, 5, 5, 5],
        max_input_power=5900,
        min_on_durations=[60, 60, 60, 1800, 1800],
        min_off_durations=[12, 12, 12, 1800, 600],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1800,
        output_one_appliance=False,
        boolean_targets=False,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0,
        n_seq_per_batch=25,
        subsample_target=9,
        include_diff=True
    )

    net = Net(
        experiment_name=name,
        source=source,
        save_plot_interval=250,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=.1, clip_range=(-1, 1)),
        layers_config=[
            {
                'type': LSTMLayer,
                'num_units': 60,
                'W_in_to_cell': Uniform(25),
                'gradient_steps': GRADIENT_STEPS,
                'peepholes': False

            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 80,
                'filter_length': 3,
                'stride': 3,
                'nonlinearity': sigmoid,
                'W': Uniform(1)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': LSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(1),
                'gradient_steps': GRADIENT_STEPS,
                'peepholes': False

            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 80,
                'filter_length': 3,
                'stride': 3,
                'nonlinearity': sigmoid,
                'W': Uniform(1)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': LSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(1),
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



def exp_g(name):
    # 'F' but with smaller layers
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            ['fridge freezer', 'fridge', 'freezer'], 
            'hair straighteners', 
            'television',
            'dish washer',
            ['washer dryer', 'washing machine']
        ],
        max_appliance_powers=[300, 500, 200, 2500, 2400],
        on_power_thresholds=[5, 5, 5, 5, 5],
        max_input_power=5900,
        min_on_durations=[60, 60, 60, 1800, 1800],
        min_off_durations=[12, 12, 12, 1800, 600],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1800,
        output_one_appliance=False,
        boolean_targets=False,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0,
        n_seq_per_batch=25,
        subsample_target=9,
        include_diff=True
    )

    net = Net(
        experiment_name=name,
        source=source,
        save_plot_interval=250,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=.1, clip_range=(-1, 1)),
        layers_config=[
            {
                'type': LSTMLayer,
                'num_units': 30,
                'W_in_to_cell': Uniform(25),
                'gradient_steps': GRADIENT_STEPS,
                'peepholes': False

            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 50,
                'filter_length': 3,
                'stride': 3,
                'nonlinearity': sigmoid,
                'W': Uniform(1)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': LSTMLayer,
                'num_units': 50,
                'W_in_to_cell': Uniform(1),
                'gradient_steps': GRADIENT_STEPS,
                'peepholes': False

            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 70,
                'filter_length': 3,
                'stride': 3,
                'nonlinearity': sigmoid,
                'W': Uniform(1)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': LSTMLayer,
                'num_units': 70,
                'W_in_to_cell': Uniform(1),
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
    for experiment in list('abcdefg'):
        full_exp_name = NAME + experiment
        path = os.path.join(PATH, full_exp_name)
        try:
            net = init_experiment(experiment)
            run_experiment(net, path, epochs=3000)
        except KeyboardInterrupt:
            break
        except TrainingError as e:
            print("EXCEPTION:", e)


if __name__ == "__main__":
    main()

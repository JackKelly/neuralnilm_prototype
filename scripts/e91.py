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

NAME = "e91"
PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"
SAVE_PLOT_INTERVAL=250



def exp_a(name):
    print("e82 with seq length 1000 and 5 appliances")
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
        on_power_thresholds=[80, 20, 20, 20, 600],
        min_on_durations=[60, 60, 60, 300, 300],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1]
    )

    net = Net(
        experiment_name=name + 'a',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.1),
        layers_config=[
            {
                'type': BLSTMLayer,
                'num_units': 60,
                'W_in_to_cell': Uniform(5)
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
                'nonlinearity': sigmoid
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': BLSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(5)
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
    print("e82 with seq length 1000 and 5 appliances and learning rate 0.01")
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
        on_power_thresholds=[80, 20, 20, 20, 600],
        min_on_durations=[60, 60, 60, 300, 300],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1]
    )

    net = Net(
        experiment_name=name + 'b',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.01),
        layers_config=[
            {
                'type': BLSTMLayer,
                'num_units': 60,
                'W_in_to_cell': Uniform(5)
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
                'nonlinearity': sigmoid
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': BLSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(5)
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
    print("e59 but with 5 appliances and 60 units in Conv1D")
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
        on_power_thresholds=[80, 20, 20, 20, 600],
        min_on_durations=[60, 60, 60, 300, 300],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1]
    )

    net = Net(
        experiment_name=name + 'c',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.1),
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
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 60,
                'filter_length': 5,
                'stride': 5,
                'nonlinearity': sigmoid
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': BLSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(5)
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
    print("e59 but with 5 appliances and learning rate 0.01")
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
        on_power_thresholds=[80, 20, 20, 20, 600],
        min_on_durations=[60, 60, 60, 300, 300],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1]
    )

    net = Net(
        experiment_name=name + 'd',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.01),
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
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 60,
                'filter_length': 5,
                'stride': 5,
                'nonlinearity': sigmoid
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': BLSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DenseLayer,
                'num_units': source.n_outputs,
                'nonlinearity': sigmoid
            }
        ]
    )
    return net



def exp_e(name):
    print("e59 but with 5 appliances and learning rate 0.01, linear outputs and MSE")
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
        on_power_thresholds=[80, 20, 20, 20, 600],
        min_on_durations=[60, 60, 60, 300, 300],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1]
    )

    net = Net(
        experiment_name=name + 'e',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=mse,
        updates=partial(nesterov_momentum, learning_rate=0.01),
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
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 60,
                'filter_length': 5,
                'stride': 5,
                'nonlinearity': sigmoid
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': BLSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DenseLayer,
                'num_units': source.n_outputs,
                'nonlinearity': None
            }
        ]
    )
    return net



def exp_f(name):
    print("e59 but with 5 appliances and learning rate 0.1, and single output (Fridge)")
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
        on_power_thresholds=[80, 20, 20, 20, 600],
        min_on_durations=[60, 60, 60, 300, 300],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=True,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1]
    )

    net = Net(
        experiment_name=name + 'f',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.1),
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
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 60,
                'filter_length': 5,
                'stride': 5,
                'nonlinearity': sigmoid
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': BLSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(5)
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
    print("e59 but with 5 appliances and learning rate 0.1, and single output (Fridge), linear output and MSE")
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
        on_power_thresholds=[80, 20, 20, 20, 600],
        min_on_durations=[60, 60, 60, 300, 300],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=True,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1]
    )

    net = Net(
        experiment_name=name + 'g',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=mse,
        updates=partial(nesterov_momentum, learning_rate=0.1),
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
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 60,
                'filter_length': 5,
                'stride': 5,
                'nonlinearity': sigmoid
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': BLSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DenseLayer,
                'num_units': source.n_outputs,
                'nonlinearity': None
            }
        ]
    )
    return net


def exp_g(name):
    print("e59 but with 5 appliances and learning rate 0.1, and single output (washer), linear output and MSE")
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            'dish washer',
            ['fridge freezer', 'fridge', 'freezer'], 
            'hair straighteners', 
            'television',
            ['washer dryer', 'washing machine']
        ],
        max_appliance_powers=[300, 500, 200, 2500, 2400],
        on_power_thresholds=[80, 20, 20, 20, 600],
        min_on_durations=[60, 60, 60, 300, 300],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=True,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1]
    )

    net = Net(
        experiment_name=name + 'g',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=mse,
        updates=partial(nesterov_momentum, learning_rate=0.1),
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
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 60,
                'filter_length': 5,
                'stride': 5,
                'nonlinearity': sigmoid
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': BLSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DenseLayer,
                'num_units': source.n_outputs,
                'nonlinearity': None
            }
        ]
    )
    return net



def exp_h(name):
    print("e59 but with 5 appliances and learning rate 0.1, and single output (washer)")
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            'dish washer',
            ['fridge freezer', 'fridge', 'freezer'], 
            'hair straighteners', 
            'television',
            ['washer dryer', 'washing machine']
        ],
        max_appliance_powers=[300, 500, 200, 2500, 2400],
        on_power_thresholds=[80, 20, 20, 20, 600],
        min_on_durations=[60, 60, 60, 300, 300],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=True,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1]
    )

    net = Net(
        experiment_name=name + 'h',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.1),
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
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 60,
                'filter_length': 5,
                'stride': 5,
                'nonlinearity': sigmoid
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': BLSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DenseLayer,
                'num_units': source.n_outputs,
                'nonlinearity': sigmoid
            }
        ]
    )
    return net



def exp_i(name):
    print("e59 but with 5 appliances and 60 units in Conv1D and seq length of 500")
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
        on_power_thresholds=[80, 20, 20, 20, 600],
        min_on_durations=[60, 60, 60, 300, 300],
        window=("2013-06-01", "2014-07-01"),
        seq_length=500,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1]
    )

    net = Net(
        experiment_name=name + 'i',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.1),
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
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': Conv1DLayer,
                'num_filters': 60,
                'filter_length': 5,
                'stride': 5,
                'nonlinearity': sigmoid
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)
            },
            {
                'type': BLSTMLayer,
                'num_units': 80,
                'W_in_to_cell': Uniform(5)
            },
            {
                'type': DenseLayer,
                'num_units': source.n_outputs,
                'nonlinearity': sigmoid
            }
        ]
    )
    return net




def run_experiment(experiment):
    exp_name = 'exp_{:s}(NAME)'.format(experiment)
    print("***********************************")
    print("Running", exp_name, "...")
    net = eval(exp_name)
    net.print_net()
    net.compile()
    path = os.path.join(PATH, NAME+experiment)
    os.mkdir(path)
    os.chdir(path)
    net.fit(1501)


def main():
    for experiment in list('abcdefghij'):
        try:
            run_experiment(experiment)
        except Exception as e:
            print("EXCEPTION:", e)


if __name__ == "__main__":
    main()

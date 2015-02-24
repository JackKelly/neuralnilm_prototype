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

NAME = "e99"
PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"
SAVE_PLOT_INTERVAL = 250
GRADIENT_STEPS = 100

"""
e92: Changes common to all these (compared to e91)
* on_power_thresh is now 20 watts for everything
* fixed max_appliance_powers and min_on_durations
* seq_length = 1500 (to fit in washer)
* min_on_durations increased for washer and dish washer
* skip_probability = 0.7

e94:
* Exactly the same experiments as e93 but using `gradient_steps=100` and 5000 epochs
* using `gradient_steps`
"""


def exp_a(name):
    """Results: Learning rate of 0.1 still NaNs."""

    """e91d but learning rate 0.01 
    and smaller inits (to try to capture 
    smaller changes) and larger first layer

    And e96 centres input data.  And I've fixed a problem where only the last
    instance of an appliance if multiple appliances occured within a batch would
    be shown in the targets.

    e98:
    Output just the fridge and use bool targets

    e99
    seq_length = 1000
    learning rate = 0.01 (tried 0.1 and 0.05 but both NaN'd)
    max_input_power = 500
    don't bother centering X
    only 50 units in first layer
    back to just 3 appliances
    skip prob = 0

    e100
    boolean_targets = False
    output_one_appliance=False
    """
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
        max_input_power=500,
        min_on_durations=[60, 60, 60], #, 1800, 1800],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0
    )

    net = Net(
        experiment_name=name + 'a',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.01),
        layers_config=[
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(10),
                'b': Uniform(10)
            },
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(5),
                'b': Uniform(5)
            },
            {
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
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
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
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
    """Same as above but with skip_prob=0.7"""
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
        max_input_power=500,
        min_on_durations=[60, 60, 60], #, 1800, 1800],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0.7
    )

    net = Net(
        experiment_name=name + 'b',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.01),
        layers_config=[
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(10),
                'b': Uniform(10)
            },
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(5),
                'b': Uniform(5)
            },
            {
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
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
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
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
    """Same as A but with seq_length=1500"""
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
        max_input_power=500,
        min_on_durations=[60, 60, 60], #, 1800, 1800],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1500,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0
    )

    net = Net(
        experiment_name=name + 'c',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.01),
        layers_config=[
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(10),
                'b': Uniform(10)
            },
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(5),
                'b': Uniform(5)
            },
            {
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
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
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
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
    """Same as A but with max_input_power=2000"""
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
        max_input_power=2000,
        min_on_durations=[60, 60, 60], #, 1800, 1800],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1500,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0
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
                'W': Uniform(10),
                'b': Uniform(10)
            },
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(5),
                'b': Uniform(5)
            },
            {
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
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
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
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
    """Same as A but with max_input_power=8000"""
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
        max_input_power=8000,
        min_on_durations=[60, 60, 60], #, 1800, 1800],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1500,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0
    )

    net = Net(
        experiment_name=name + 'e',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.01),
        layers_config=[
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(10),
                'b': Uniform(10)
            },
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(5),
                'b': Uniform(5)
            },
            {
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
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
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
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
    """Same as A but with 4 appliances"""
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            ['fridge freezer', 'fridge', 'freezer'], 
            'hair straighteners', 
            'television',
            'dish washer'
            # ['washer dryer', 'washing machine']
        ],
        max_appliance_powers=[300, 500, 200], #, 2500, 2400],
        on_power_thresholds=[20, 20, 20], #, 20, 20],
        max_input_power=8000,
        min_on_durations=[60, 60, 60], #, 1800, 1800],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1500,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0
    )

    net = Net(
        experiment_name=name + 'f',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.01),
        layers_config=[
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(10),
                'b': Uniform(10)
            },
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(5),
                'b': Uniform(5)
            },
            {
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
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
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
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
    """Same as A but with 5 appliances"""
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            ['fridge freezer', 'fridge', 'freezer'], 
            'hair straighteners', 
            'television',
            'dish washer',
            ['washer dryer', 'washing machine']
        ],
        max_appliance_powers=[300, 500, 200], #, 2500, 2400],
        on_power_thresholds=[20, 20, 20], #, 20, 20],
        max_input_power=8000,
        min_on_durations=[60, 60, 60], #, 1800, 1800],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1500,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0
    )

    net = Net(
        experiment_name=name + 'g',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.01),
        layers_config=[
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(10),
                'b': Uniform(10)
            },
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(5),
                'b': Uniform(5)
            },
            {
                'type': BLSTMLayer,
                'num_units': 40,
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
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
                'W_in_to_cell': Uniform(5),
                'gradient_steps': GRADIENT_STEPS
            },
            {
                'type': DenseLayer,
                'num_units': source.n_outputs,
                'nonlinearity': sigmoid
            }
        ]
    )
    return net



def exp_h(name):
    # Same as A but without gradient_steps = 100
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
        max_input_power=500,
        min_on_durations=[60, 60, 60], #, 1800, 1800],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1000,
        output_one_appliance=False,
        boolean_targets=False,
        min_off_duration=60,
        subsample_target=5,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0
    )

    net = Net(
        experiment_name=name + 'h',
        source=source,
        save_plot_interval=SAVE_PLOT_INTERVAL,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.01),
        layers_config=[
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(10),
                'b': Uniform(10)
            },
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(5),
                'b': Uniform(5)
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
    print("Preparing", NAME + experiment, "...")
    net = eval(exp_name)
    net.print_net()
    net.compile()
    path = os.path.join(PATH, NAME+experiment)
    try:
        os.mkdir(path)
    except OSError as exception:
        if exception.errno == 17:
            print(path, "already exists.  Reusing directory.")
        else:
            raise
    os.chdir(path)
    fit(net, experiment)


def fit(net, experiment, epochs=2998):
    print("Running net.fit for", NAME + experiment)
    save_plots = "y"
    continue_fit = "n"
    try:
        net.fit(epochs)
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
        enter_debugger = raw_input("Enter debugger [N/y]? ")
        if enter_debugger.lower() == 'y':
            import ipdb; ipdb.set_trace()
        save_plots = raw_input("Save latest data [Y/n]? ")
        stop_all = raw_input("Stop all experiments [Y/n]? ")
        if not stop_all or stop_all.lower() == "y":
            raise
        continue_fit = raw_input("Continue fitting this experiment [N/y]? ")
    finally:
        if not save_plots or save_plots.lower() == "y":
            print("Saving plots...")
            net.plot_estimates(save=True, all_sequences=True)
            net.plot_costs(save=True)
            print("Saving params...")
            net.save_params()
            print("Done saving.")

    if continue_fit == "y":
        new_epochs = raw_input("Change number of epochs [currently {}]? "
                               .format(epochs))
        if new_epochs:
            epochs = int(new_epochs)
        fit(net, experiment, epochs)


def main():
    for experiment in list('abcdefgh'):
        try:
            run_experiment(experiment)
        except Exception as e:
            print("EXCEPTION:", e)
            raise


if __name__ == "__main__":
    main()

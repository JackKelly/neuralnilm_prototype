from __future__ import print_function, division
import matplotlib
import logging
from sys import stdout
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from neuralnilm import (Net, RealApplianceSource)
from neuralnilm.source import (standardise, discretize, fdiff, power_and_fdiff,
                               RandomSegments, RandomSegmentsInMemory,
                               SameLocation, MultiSource)
from neuralnilm.experiment import (run_experiment, init_experiment,
                                   change_dir, configure_logger)
from neuralnilm.net import TrainingError
from neuralnilm.layers import (MixtureDensityLayer, DeConv1DLayer,
                               SharedWeightsDenseLayer, BLSTMLayer)
from neuralnilm.objectives import (scaled_cost, mdn_nll,
                                   scaled_cost_ignore_inactive, ignore_inactive,
                                   scaled_cost3)
from neuralnilm.plot import MDNPlotter, CentralOutputPlotter, Plotter, RectangularOutputPlotter, StartEndMeanPlotter
from neuralnilm.updates import clipped_nesterov_momentum
from neuralnilm.rectangulariser import rectangularise

from lasagne.nonlinearities import (sigmoid, rectify, tanh, identity, softmax)
from lasagne.objectives import squared_error, binary_crossentropy
from lasagne.init import Uniform, Normal
from lasagne.layers import (DenseLayer, Conv1DLayer,
                            ReshapeLayer, FeaturePoolLayer,
                            DimshuffleLayer, DropoutLayer, ConcatLayer, PadLayer)
from lasagne.updates import nesterov_momentum, momentum
from functools import partial
import os
import __main__
from copy import deepcopy
from math import sqrt
import numpy as np
import theano.tensor as T
import gc


NAME = os.path.splitext(os.path.split(__main__.__file__)[1])[0]
#PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"
PATH = "/data/dk3810/figures"
# PATH = "/home/jack/experiments/neuralnilm/figures"

UKDALE_FILENAME = '/data/dk3810/ukdale.h5'

SKIP_PROBABILITY_FOR_TARGET = 0.5
INDEPENDENTLY_CENTER_INPUTS = True

WINDOW_PER_BUILDING = {
    1: ("2013-04-12", "2014-12-15"),
    2: ("2013-05-22", "2013-10-03 06:16:00"),
    3: ("2013-02-27", "2013-04-01 06:15:05"),
    4: ("2013-03-09", "2013-09-24 06:15:14"),
    5: ("2014-06-29", "2014-09-01")
}

INPUT_STATS = {
    'mean': np.array([297.87216187], dtype=np.float32),
    'std': np.array([374.43884277], dtype=np.float32)
}


def get_source(appliance, logger, target_is_start_and_end_and_mean=False,
               is_rnn=False, window_per_building=WINDOW_PER_BUILDING,
               source_type='multisource',
               filename=UKDALE_FILENAME):
    """
    Parameters
    ----------
    source_type : {'multisource', 'real_appliance_source'}

    Returns
    -------
    Source
    """
    N_SEQ_PER_BATCH = 64
    TRAIN_BUILDINGS_REAL = None

    if appliance == 'microwave':
        SEQ_LENGTH = 288
        TRAIN_BUILDINGS = [1, 2]
        VALIDATION_BUILDINGS = [5]
        APPLIANCES = [
            'microwave',
            ['fridge freezer', 'fridge', 'freezer'],
            'dish washer',
            'kettle',
            ['washer dryer', 'washing machine']
        ]
        MAX_APPLIANCE_POWERS = [3000,  300, 2500, 3100, 2500]
        ON_POWER_THRESHOLDS  = [ 200,   50,   10, 2000,   20]
        MIN_ON_DURATIONS     = [  12,   60, 1800,   12, 1800]
        MIN_OFF_DURATIONS    = [  30,   12, 1800,    0,  160]

    elif appliance == 'washing machine':
        SEQ_LENGTH = 1536
        TRAIN_BUILDINGS = [1, 5]
        VALIDATION_BUILDINGS = [2]
        APPLIANCES = [
            ['washer dryer', 'washing machine'],
            ['fridge freezer', 'fridge', 'freezer'],
            'dish washer',
            'kettle',
            'microwave'
        ]
        MAX_APPLIANCE_POWERS = [2500,  300, 2500, 3100, 3000]
        ON_POWER_THRESHOLDS  = [  20,   50,   10, 2000,  200]
        MIN_ON_DURATIONS     = [1800,   60, 1800,   12,   12]
        MIN_OFF_DURATIONS    = [ 160,   12, 1800,    0,   30]
        if is_rnn:
            N_SEQ_PER_BATCH = 16

    elif appliance == 'fridge':
        SEQ_LENGTH = 1024
        TRAIN_BUILDINGS = [1, 2, 4]
        VALIDATION_BUILDINGS = [5]
        APPLIANCES = [
            ['fridge freezer', 'fridge', 'freezer'],
            ['washer dryer', 'washing machine'],
            'dish washer',
            'kettle',
            'microwave'
        ]
        MAX_APPLIANCE_POWERS = [ 300, 2500, 2500, 3100, 3000]
        ON_POWER_THRESHOLDS  = [  50,   20,   10, 2000,  200]
        MIN_ON_DURATIONS     = [  60, 1800, 1800,   12,   12]
        MIN_OFF_DURATIONS    = [  12,  160, 1800,    0,   30]
        if is_rnn:
            N_SEQ_PER_BATCH = 16
        
    elif appliance == 'kettle':
        SEQ_LENGTH = 128
        TRAIN_BUILDINGS = [1, 2, 3, 4]
        # House 3's mains often doesn't include kettle!
        TRAIN_BUILDINGS_REAL = [1, 2, 4]
        VALIDATION_BUILDINGS = [5]
        APPLIANCES = [
            'kettle',
            ['fridge freezer', 'fridge', 'freezer'],
            ['washer dryer', 'washing machine'],
            'dish washer',
            'microwave'
        ]
        MAX_APPLIANCE_POWERS = [3100,  300, 2500, 2500, 3000]
        ON_POWER_THRESHOLDS  = [2000,   50,   20,   10,  200]
        MIN_ON_DURATIONS     = [  12,   60, 1800, 1800,   12]
        MIN_OFF_DURATIONS    = [   0,   12,  160, 1800,   30]

    elif appliance == 'dish washer':
        SEQ_LENGTH = 1536
        TRAIN_BUILDINGS = [1, 2]
        VALIDATION_BUILDINGS = [5]
        APPLIANCES = [
            'dish washer',
            ['fridge freezer', 'fridge', 'freezer'],
            ['washer dryer', 'washing machine'],
            'kettle',
            'microwave'
        ]
        MAX_APPLIANCE_POWERS = [2500,  300, 2500, 3100, 3000]
        ON_POWER_THRESHOLDS  = [  10,   50,   20, 2000,  200]
        MIN_ON_DURATIONS     = [1800,   60, 1800,   12,   12]
        MIN_OFF_DURATIONS    = [1800,   12,  160,    0,   30]

        if is_rnn:
            N_SEQ_PER_BATCH = 16
        
    TARGET_APPLIANCE = APPLIANCES[0]
    MAX_TARGET_POWER = MAX_APPLIANCE_POWERS[0]
    ON_POWER_THRESHOLD = ON_POWER_THRESHOLDS[0]
    MIN_ON_DURATION = MIN_ON_DURATIONS[0]
    MIN_OFF_DURATION = MIN_OFF_DURATIONS[0]
    if TRAIN_BUILDINGS_REAL is None:
        TRAIN_BUILDINGS_REAL = TRAIN_BUILDINGS

    real_appliance_source1 = RealApplianceSource(
        logger=logger,
        filename=filename,
        appliances=APPLIANCES,
        max_appliance_powers=MAX_APPLIANCE_POWERS,
        on_power_thresholds=ON_POWER_THRESHOLDS,
        min_on_durations=MIN_ON_DURATIONS,
        min_off_durations=MIN_OFF_DURATIONS,
        divide_input_by_max_input_power=False,
        window_per_building=window_per_building,
        seq_length=SEQ_LENGTH,
        output_one_appliance=True,
        train_buildings=TRAIN_BUILDINGS,
        validation_buildings=VALIDATION_BUILDINGS,
        n_seq_per_batch=N_SEQ_PER_BATCH,
        skip_probability=0.75,
        skip_probability_for_first_appliance=SKIP_PROBABILITY_FOR_TARGET,
        standardise_input=True,
        input_stats=INPUT_STATS,
        independently_center_inputs=INDEPENDENTLY_CENTER_INPUTS,
        target_is_start_and_end_and_mean=target_is_start_and_end_and_mean
    )

    if source_type != 'multisource':
        return real_appliance_source1

    same_location_source1 = SameLocation(
        logger=logger,
        filename=filename,
        target_appliance=TARGET_APPLIANCE,
        window_per_building=window_per_building,
        seq_length=SEQ_LENGTH,
        train_buildings=TRAIN_BUILDINGS_REAL,
        validation_buildings=VALIDATION_BUILDINGS,
        n_seq_per_batch=N_SEQ_PER_BATCH,
        skip_probability=SKIP_PROBABILITY_FOR_TARGET,
        standardise_input=True,
        offset_probability=1,
        divide_target_by=MAX_TARGET_POWER,
        input_stats=INPUT_STATS,
        independently_center_inputs=INDEPENDENTLY_CENTER_INPUTS,
        on_power_threshold=ON_POWER_THRESHOLD,
        min_on_duration=MIN_ON_DURATION,
        min_off_duration=MIN_OFF_DURATION,
        include_all=not target_is_start_and_end_and_mean,
        allow_incomplete=not target_is_start_and_end_and_mean,
        target_is_start_and_end_and_mean=target_is_start_and_end_and_mean
    )

    multi_source = MultiSource(
        sources=[
            {
                'source': real_appliance_source1,
                'train_probability': 0.5,
                'validation_probability': 0
            },
            {
                'source': same_location_source1,
                'train_probability': 0.5,
                'validation_probability': 1
            }
        ],
        standardisation_source=same_location_source1
    )

    return multi_source


def only_train_on_real_data(net, iteration):
    net.logger.info(
        "Iteration {}: Now only training on real data.".format(iteration))
    net.source.sources[0]['train_probability'] = 0.0
    net.source.sources[1]['train_probability'] = 1.0


def net_dict_rectangles(seq_length):
    return dict(
        epochs=300000,
        save_plot_interval=25000,
        loss_function=lambda x, t: squared_error(x, t).mean(),
        updates_func=nesterov_momentum,
        learning_rate=1e-4,
        learning_rate_changes_by_iteration={
            200000: 1e-5,
            250000: 1e-6
        },
        epoch_callbacks={
            350000: only_train_on_real_data
        },
        do_save_activations=True,
        auto_reshape=False,
        plotter=StartEndMeanPlotter(
            n_seq_to_plot=32,
            n_training_examples_to_plot=16
        ),
        layers_config=[
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)  # (batch, features, time)
            },
            {
                'type': PadLayer,
                'width': 4
            },
            {
                'type': Conv1DLayer,  # convolve over the time axis
                'num_filters': 16,
                'filter_size': 4,
                'stride': 1,
                'nonlinearity': None,
                'border_mode': 'valid'
            },
            {
                'type': Conv1DLayer,  # convolve over the time axis
                'num_filters': 16,
                'filter_size': 4,
                'stride': 1,
                'nonlinearity': None,
                'border_mode': 'valid'
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)  # back to (batch, time, features)
            },
            {
                'type': DenseLayer,
                'num_units': 512 * 8,
                'nonlinearity': rectify
            },
            # {
            #     'type': DenseLayer,
            #     'num_units': 512 * 6,
            #     'nonlinearity': rectify
            # },
            {
                'type': DenseLayer,
                'num_units': 512 * 4,
                'nonlinearity': rectify
            },
            {
                'type': DenseLayer,
                'num_units': 512,
                'nonlinearity': rectify
            },
            {
                'type': DenseLayer,
                'num_units': 3,
                'nonlinearity': None
            }
        ]
    )
net_dict_rectangles.name = 'rectangles'


def net_dict_rnn(seq_length):
    if seq_length <= 300:
        learning_rate = 1e-2
        learning_rate_changes_by_iteration = {
            1000: 1e-3,
            10000: 1e-4
        }
    elif seq_length < 1500:
        learning_rate = 1e-4
        learning_rate_changes_by_iteration = {
            5000: 1e-5,
            9000: 1e-6
        }
    else:
        learning_rate = 1e-5
        learning_rate_changes_by_iteration = {
            5000: 1e-6,
            9000: 1e-7
        }
    return dict(
        epochs=10000,
        save_plot_interval=1000,
        loss_function=lambda x, t: squared_error(x, t).mean(),
        updates_func=nesterov_momentum,
        learning_rate=learning_rate,
        learning_rate_changes_by_iteration=learning_rate_changes_by_iteration,
        do_save_activations=True,
        auto_reshape=True,
        plotter=Plotter(
            n_seq_to_plot=32,
            n_training_examples_to_plot=16
        ),
        layers_config=[
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)  # (batch, features, time)
            },
            {
                'type': Conv1DLayer,  # convolve over the time axis
                'num_filters': 16,
                'filter_size': 4,
                'stride': 1,
                'nonlinearity': None,
                'border_mode': 'same'
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1),  # back to (batch, time, features)
                'label': 'dimshuffle3'
            },
            {
                'type': BLSTMLayer,
                'num_units': 128,
                'merge_mode': 'concatenate',
                'grad_clipping': 10.0,
                'gradient_steps': 500
            },
            {
                'type': BLSTMLayer,
                'num_units': 256,
                'merge_mode': 'concatenate',
                'grad_clipping': 10.0,
                'gradient_steps': 500
            },
            {
                'type': DenseLayer,
                'num_units': 128,
                'nonlinearity': tanh
            },
            {
                'type': DenseLayer,
                'num_units': 1,
                'nonlinearity': None
            }
        ]
    )
net_dict_rnn.name = 'rnn'


def net_dict_ae(seq_length):
    NUM_FILTERS = 4
    return dict(
        epochs=100000,
        save_plot_interval=25000,
        loss_function=lambda x, t: squared_error(x, t).mean(),
        updates_func=nesterov_momentum,
        learning_rate=1e-1,
        learning_rate_changes_by_iteration={
            50000: 1e-2,
            75000: 1e-3
        },
        do_save_activations=True,
        auto_reshape=False,
        plotter=Plotter(
            n_seq_to_plot=32,
            n_training_examples_to_plot=16
        ),
        layers_config=[
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)  # (batch, features, time)
            },
            {
                'label': 'conv0',
                'type': Conv1DLayer,  # convolve over the time axis
                'num_filters': NUM_FILTERS,
                'filter_size': 4,
                'stride': 1,
                'nonlinearity': None,
                'border_mode': 'valid'
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)  # back to (batch, time, features)
            },
            {
                'type': DenseLayer,
                'num_units': (seq_length - 3) * NUM_FILTERS,
                'nonlinearity': rectify
            },
            {
                'type': ReshapeLayer,
                'shape': (-1, (seq_length - 3), NUM_FILTERS)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)  # (batch, features, time)
            },
            {   # DeConv
                'type': Conv1DLayer,
                'num_filters': 1,
                'filter_size': 4,
                'stride': 1,
                'nonlinearity': None,
                'border_mode': 'full'
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)  # back to (batch, time, features)
            }
        ]
    )
net_dict_ae.name = 'ae'


def exp_a(name, net_dict, multi_source):
    net_dict_copy = deepcopy(net_dict)
    net_dict_copy.update(dict(
        experiment_name=name,
        source=multi_source,
    ))
    net = Net(**net_dict_copy)
    net.plotter.max_target_power = multi_source.sources[1]['source'].divide_target_by
    return net


def main():
#    for net_dict_func in [net_dict_ae, net_dict_rectangles, net_dict_rnn]:
    for net_dict_func in [net_dict_rnn]:
        # for appliance in ['microwave', 'washing machine',
        #                   'fridge', 'kettle', 'dish washer']:
        for appliance in ['washing machine',
                          'fridge', 'kettle', 'dish washer']:
            # REMOVE IF RUN FROM SCRATCH:
            if net_dict_func == net_dict_ae:
                if appliance in ['microwave', 'washing machine', 'dish washer']:
                    continue
            elif net_dict_func == net_dict_rectangles:
                if appliance == 'microwave':
                    continue

            full_exp_name = NAME + '_' + appliance + '_' + net_dict_func.name
            change_dir(PATH, full_exp_name)
            configure_logger(full_exp_name)
            logger = logging.getLogger(full_exp_name)
            global multi_source
            multi_source = get_source(
                appliance,
                logger,
                target_is_start_and_end_and_mean=(net_dict_func == net_dict_rectangles),
                is_rnn=(net_dict_func == net_dict_rnn)
            )
            seq_length = multi_source.sources[0]['source'].seq_length
            net_dict = net_dict_func(seq_length)
            epochs = net_dict.pop('epochs')
            try:
                net = exp_a(full_exp_name, net_dict, multi_source)
                # REMOVE IF RUN FROM SCRATCH:
                if (appliance == 'washing machine' and
                        net_dict_func == net_dict_rectangles):
                    net.load_params(85351)
                if (appliance == 'washing machine' and
                        net_dict_func == net_dict_rnn):
                    net.load_params(4000)
                run_experiment(net, epochs=epochs)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt")
                break
            except Exception:
                logger.exception("Exception")
                # raise
            finally:
                logging.shutdown()


if __name__ == "__main__":
    main()


"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/scripts/e566.py /mnt/sshfs/imperial/workspace/python/neuralnilm/scripts/"
End:
"""

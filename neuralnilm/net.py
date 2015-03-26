from __future__ import division, print_function
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import h5py
from datetime import datetime, timedelta
import logging
from numpy.random import rand
from time import time

import theano
import theano.tensor as T
theano.config.compute_test_value = 'raise'

import lasagne
from lasagne.layers import (InputLayer, LSTMLayer, ReshapeLayer, Layer,
                            ConcatLayer, ElemwiseSumLayer, DenseLayer,
                            get_all_layers, Conv1DLayer, FeaturePoolLayer, 
                            RecurrentLayer)
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.utils import floatX
from lasagne.updates import nesterov_momentum

from .source import quantize
from .layers import BLSTMLayer, DimshuffleLayer, MixtureDensityLayer
from .utils import sfloatX, none_to_dict, ndim_tensor
from .plot import Plotter

class ansi:
    # from dnouri/nolearn/nolearn/lasagne.py
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'


class TrainingError(Exception):
    pass
    

######################## Neural network class ########################
class Net(object):
    # Much of this code is adapted from craffel/nntools/examples/lstm.py

    def __init__(self, source, layers_config, 
                 updates_func=nesterov_momentum,
                 updates_kwargs=None,
                 learning_rate=0.1,
                 learning_rate_changes_by_iteration=None,
                 experiment_name="", 
                 validation_interval=10, 
                 save_plot_interval=100,
                 loss_function=lasagne.objectives.mse,
                 layer_changes=None,
                 seed=42,
                 epoch_callbacks=None,
                 do_save_activations=True,
                 plotter=Plotter
    ):
        """
        Parameters
        ----------
        layers_config : list of dicts.  Keys are:
            'type' : BLSTMLayer or a subclass of lasagne.layers.Layer
            'num_units' : int
        """
        self.logger = logging.getLogger(experiment_name)
        self.logger.info("Initialising network...")

        if seed is not None:
            np.random.seed(seed)
        self.source = source
        self.updates_func = updates_func
        self.learning_rate = theano.shared(
            sfloatX(learning_rate), name='learning_rate')
        self.learning_rate_changes_by_iteration = none_to_dict(
            learning_rate_changes_by_iteration)
        self.updates_kwargs = none_to_dict(updates_kwargs)
        self.experiment_name = experiment_name
        self.validation_interval = validation_interval
        self.save_plot_interval = save_plot_interval
        self.loss_function = loss_function
        self.layer_changes = none_to_dict(layer_changes)
        self.epoch_callbacks = none_to_dict(epoch_callbacks)
        self.do_save_activations = do_save_activations
        self.plotter = plotter(self)

        self.csv_filenames = {
            'training_costs': self.experiment_name + "_training_costs.csv",
            'validation_costs': self.experiment_name + "_validation_costs.csv",
            'best_costs': self.experiment_name + "_best_costs.txt"
        }

        self.generate_validation_data_and_set_shapes()

        self.validation_costs = []
        self.training_costs = []
        self.layers = []

        # Shape is (number of examples per batch,
        #           maximum number of time steps per example,
        #           number of features per example)
        self.layers.append(InputLayer(shape=self.input_shape))
        self.add_layers(layers_config)
        self.logger.info("Done initialising network for " + self.experiment_name)

    def generate_validation_data_and_set_shapes(self):
        # Generate a "validation" sequence whose cost we will compute
        self.X_val, self.y_val = self.source.validation_data()
        self.input_shape = self.X_val.shape
        self.n_seq_per_batch = self.input_shape[0]
        self.output_shape = self.y_val.shape
        self.n_outputs = self.output_shape[-1]

    def add_layers(self, layers_config):
        for layer_config in layers_config:
            layer_type = layer_config.pop('type')

            # Reshape if necessary
            prev_layer_output_shape = self.layers[-1].get_output_shape()
            n_dims = len(prev_layer_output_shape)
            n_features = prev_layer_output_shape[-1]
            if layer_type in [LSTMLayer, BLSTMLayer, DimshuffleLayer]:
                if n_dims == 2:
                    seq_length = int(prev_layer_output_shape[0] / 
                                     self.source.n_seq_per_batch)
                    shape = (self.source.n_seq_per_batch, 
                             seq_length,
                             n_features)
                    self.layers.append(ReshapeLayer(self.layers[-1], shape))
            elif layer_type in [DenseLayer, MixtureDensityLayer]:
                if n_dims == 3:
                    # The prev layer_config was a time-aware layer_config, so reshape to 2-dims.
                    seq_length = prev_layer_output_shape[1]
                    shape = (self.source.n_seq_per_batch * seq_length,
                             n_features)
                    self.layers.append(ReshapeLayer(self.layers[-1], shape))

            # Init new layer_config
            self.logger.info('Initialising layer_config : {}'.format(layer_type))
            self.layers.append(layer_type(self.layers[-1], **layer_config))

        # Reshape output if necessary...
        if (self.layers[-1].get_output_shape() != self.output_shape and 
            layer_type != MixtureDensityLayer):
            self.layers.append(ReshapeLayer(self.layers[-1], self.output_shape))

    def print_net(self):
        for layer in self.layers:
            self.logger.info(str(layer))
            try:
                input_shape = layer.input_shape
            except:
                pass
            else:
                self.logger.info(" Input shape: {}".format(input_shape))
            self.logger.info("Output shape: {}".format(layer.get_output_shape()))

    def compile(self):
        input = ndim_tensor(name='input', ndim=self.X_val.ndim)
        target_output = ndim_tensor(name='target_output', ndim=self.y_val.ndim)

        # add test values
        input.tag.test_value = floatX(rand(*self.input_shape))
        target_output.tag.test_value = floatX(rand(*self.output_shape))

        self.logger.info("Compiling Theano functions...")
        loss = self.loss_function(
            self.layers[-1].get_output(input), target_output)

        # Updates
        all_params = lasagne.layers.get_all_params(self.layers[-1])
        updates = self.updates_func(
            loss, all_params, learning_rate=self.learning_rate, 
            **self.updates_kwargs)

        # Theano functions for training, getting output, and computing loss
        self.train = theano.function(
            [input, target_output],
            loss, updates=updates, on_unused_input='warn',
            allow_input_downcast=True)

        self.y_pred = theano.function(
            [input], self.layers[-1].get_output(input), on_unused_input='warn',
            allow_input_downcast=True)

        self.compute_cost = theano.function(
            [input, target_output], loss, on_unused_input='warn',
            allow_input_downcast=True)

        self.logger.info("Done compiling Theano functions.")

    def fit(self, n_iterations=None):
        # Training loop. Need to wrap this in a try-except loop so
        # we can always call self.source.stop()
        self.source.start()
        try:
            self._training_loop(n_iterations)
        except:
            raise
        finally:
            self.source.stop()      

    def _change_layers(self, epoch):
        self.source.stop()
        self.source.empty_queue()
        self.logger.info("Changing layers...\nOld architecture:")
        self.print_net()        
        layer_changes = self.layer_changes[epoch]
        for layer_to_remove in range(layer_changes['remove_from'], 0):
            self.logger.info("Removed {}".format(self.layers.pop(layer_to_remove)))
        if 'callback' in layer_changes:
            layer_changes['callback'](self, epoch)
        self.add_layers(layer_changes['new_layers'])
        self.logger.info("New architecture:")
        self.print_net()
        self.compile()
        self.source.start()

    def print_and_save_training_progress(self, duration):
        iteration = self.n_iterations()
        train_cost = self.training_costs[-1]
        validation_cost = (self.validation_costs[-1] if self.validation_costs 
                           else None)
        _write_csv_row(self.csv_filenames['training_costs'],
                       [iteration, train_cost, duration])
        best_train_cost = min(self.training_costs)
        best_valid_cost = min(self.validation_costs)
        is_best_train = train_cost == best_train_cost
        is_best_valid = validation_cost == best_valid_cost

        # write bests to disk
        FMT = "{:14.10f}"
        N = 500
        K = 25
        txt = (
            "BEST COSTS\n" + 
            ("best train cost =" + FMT + " at iteration{:6d}\n").format(
                best_train_cost, self.training_costs.index(best_train_cost)) + 
            ("best valid cost =" + FMT + " at iteration{:6d}\n").format(
                best_valid_cost, 
                self.validation_costs.index(best_valid_cost) * 
                self.validation_interval) + 
            "\n" +
            "AVERAGE FOR THE TOP {:d} ITERATIONS\n".format(K) +
            (" avg train cost =" + FMT + "\n").format(
                np.mean(np.sort(self.training_costs)[:K])) +
            (" avg valid cost =" + FMT + "\n").format(
                np.mean(np.sort(self.validation_costs)[:K])) + 
            "\n" + 
            "AVERAGE COSTS FOR THE LAST {:d} ITERATIONS\n".format(N) +
            (" avg train cost =" + FMT + "\n").format(
                np.mean(self.training_costs[-N:])) +
            (" avg valid cost =" + FMT + "\n").format(
                np.mean(self.validation_costs[-N:]))
        )
        with open(self.csv_filenames['best_costs'], mode='w') as fh:
            fh.write(txt)

        # print bests to screen
        print("  {:>5} |  {}{:>10.6f}{}  |  {}{:>10.6f}{}  |"
              "  {:>11.6f}  |  {:>3.1f}s".format(
                  iteration,
                  ansi.BLUE if is_best_train else "",
                  train_cost,
                  ansi.ENDC if is_best_train else "",
                  ansi.GREEN if is_best_valid else "",
                  validation_cost,
                  ansi.ENDC if is_best_valid else "",
                  train_cost / validation_cost,
                  duration
        ))
        if np.isnan(train_cost):
            msg = "training cost is NaN at iteration {}!".format(iteration)
            self.logger.error(msg)
            raise TrainingError(msg)

    def _training_loop(self, n_iterations):
        # Adapted from dnouri/nolearn/nolearn/lasagne.py
        self.logger.info("Starting training for {} iterations."
                         .format(n_iterations))
        print("""
 Update |  Train cost  |  Valid cost  |  Train / Val  | Secs per update
--------|--------------|--------------|---------------|----------------\
""")
        iteration = self.n_iterations()
        if iteration == 0:
            # Header for CSV file
            _write_csv_row(
                self.csv_filenames['training_costs'],
                row=['iteration', 'train_cost', 'duration'], 
                mode='w')
            _write_csv_row(
                self.csv_filenames['validation_costs'],
                row=['iteration', 'validation_cost'], 
                mode='w')

        while iteration != n_iterations:
            t0 = time() # for calculating training duration
            iteration = len(self.training_costs)
            if iteration in self.learning_rate_changes_by_iteration:
                new_lr = self.learning_rate_changes_by_iteration[iteration]
                new_lr = sfloatX(new_lr)
                self.logger.info(
                    "Changing learning rate from {} to {}"
                    .format(self.learning_rate.get_value(), new_lr))
                self.learning_rate.set_value(new_lr)
            if iteration in self.layer_changes:
                self._change_layers(iteration)
            if iteration in self.epoch_callbacks:
                self.epoch_callbacks[iteration](self, iteration)
            X, y = self.source.queue.get(timeout=30)
            train_cost = self.train(X, y).flatten()[0]
            self.training_costs.append(train_cost)
            if not iteration % self.validation_interval:
                validation_cost = self.compute_cost(self.X_val, self.y_val).flatten()[0]
                self.validation_costs.append(validation_cost)
                _write_csv_row(
                    self.csv_filenames['validation_costs'],
                    row=[iteration, validation_cost])
            if not iteration % self.save_plot_interval:
                self.plotter.plot_all()
                self.save_params()
                self.save_activations()
            duration = time() - t0
            self.print_and_save_training_progress(duration)
        self.logger.info("Finished training")

    def n_iterations(self):
        return max(len(self.training_costs) - 1, 0)

    def save_params(self, filename=None):
        """
        Save it to HDF in the following format:
            /epoch<N>/L<I>_<type>/P<I>_<name>
        """
        if filename is None:
            filename = self.experiment_name + ".hdf5"

        mode = 'w' if self.n_iterations() == 0 else 'a'
        f = h5py.File(filename, mode=mode)
        epoch_name = 'epoch{:06d}'.format(self.n_iterations())
        try:
            epoch_group = f.create_group(epoch_name)
        except ValueError as exception:
            self.logger.exception("Cannot save params!")
            f.close()
            return
            
        layers = get_all_layers(self.layers[-1])
        layers.reverse()
        for layer_i, layer in enumerate(layers):
            params = layer.get_params()
            if not params:
                continue
            layer_name = 'L{:02d}_{}'.format(layer_i, layer.__class__.__name__)
            layer_group = epoch_group.create_group(layer_name)
            for param_i, param in enumerate(params):
                param_name = 'P{:02d}'.format(param_i)
                if param.name:
                    param_name += "_" + param.name
                data = param.get_value()
                layer_group.create_dataset(param_name, data=data, compression="gzip")
            
        f.close()

    def load_params(self, iteration, filename=None):
        """
        Load params from HDF in the following format:
            /epoch<N>/L<I>_<type>/P<I>_<name>
        """
        # Process function parameters
        if filename is None:
            filename = self.experiment_name + ".hdf5"
        self.logger.info('Loading params from ' + filename + '...')

        f = h5py.File(filename, mode='r')
        epoch_name = 'epoch{:06d}'.format(iteration)
        epoch_group = f[epoch_name]
            
        layers = get_all_layers(self.layers[-1])
        layers.reverse()
        for layer_i, layer in enumerate(layers):
            params = layer.get_params()
            if not params:
                continue
            layer_name = 'L{:02d}_{}'.format(layer_i, layer.__class__.__name__)
            layer_group = epoch_group[layer_name]
            for param_i, param in enumerate(params):
                param_name = 'P{:02d}'.format(param_i)
                if param.name:
                    param_name += "_" + param.name
                data = layer_group[param_name]
                param.set_value(data.value)
        f.close()
        self.logger.info('Done loading params from ' + filename + '.')
        def load_csv(key):
            filename = self.csv_filenames[key]
            return list(np.genfromtxt(filename, delimiter=',')[:,1])
        self.training_costs = load_csv('training_costs')
        self.validation_costs = load_csv('validation_costs')

    def save_activations(self):
        if not self.do_save_activations:
            return
        filename = self.experiment_name + "_activations.hdf5"
        mode = 'w' if self.n_iterations() == 0 else 'a'
        f = h5py.File(filename, mode=mode)
        epoch_name = 'epoch{:06d}'.format(self.n_iterations())
        try:
            epoch_group = f.create_group(epoch_name)
        except ValueError as exception:
            self.logger.exception("Cannot save params!")
            f.close()
            return

        layers = get_all_layers(self.layers[-1])
        layers.reverse()
        for layer_i, layer in enumerate(layers):
            # We only care about layers with params
            if not (layer.get_params() or isinstance(layer, FeaturePoolLayer)):
                continue

            output = layer.get_output(self.X_val).eval()
            n_features = output.shape[-1]
            seq_length = int(output.shape[0] / self.source.n_seq_per_batch)

            if isinstance(layer, DenseLayer):
                shape = (self.source.n_seq_per_batch, seq_length, n_features)
                output = output.reshape(shape)
            elif isinstance(layer, Conv1DLayer):
                output = output.transpose(0, 2, 1)

            layer_name = 'L{:02d}_{}'.format(layer_i, layer.__class__.__name__)
            epoch_group.create_dataset(layer_name, data=output, compression="gzip")

        # save validation data
        if self.n_iterations() == 0:
            f.create_dataset('validation_data', data=self.X_val, compression="gzip")

        f.close()


def _write_csv_row(filename, row, mode='a'):
    with open(filename, mode=mode) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(row)


"""
Emacs variables
Local Variables:
compile-command: "rsync -uvzr --progress --exclude '.git' --exclude '.ropeproject' --exclude 'ipynb_checkpoints' /home/jack/workspace/python/neuralnilm/ /mnt/sshfs/imperial/workspace/python/neuralnilm/"
End:
"""

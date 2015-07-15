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
from os.path import exists

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import (InputLayer, ReshapeLayer, Layer,
                            ConcatLayer, ElemwiseSumLayer, DenseLayer,
                            get_all_layers, Conv1DLayer, FeaturePoolLayer,
                            DimshuffleLayer, ConcatLayer)
# from lasagne.layers import LSTMLayer, RecurrentLayer
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.utils import floatX
from lasagne.updates import nesterov_momentum

from .source import quantize
# from .layers import BLSTMLayer, MixtureDensityLayer, BidirectionalRecurrentLayer
from .layers import MixtureDensityLayer
from .utils import sfloatX, none_to_dict, ndim_tensor
from .plot import Plotter
from .batch_norm import batch_norm


class ansi:
    # from dnouri/nolearn/nolearn/lasagne.py
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'


class TrainingError(Exception):
    pass


# ####################### Neural network class ########################
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
                 plotter=Plotter(),
                 auto_reshape=True):
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
        self._learning_rate = theano.shared(
            sfloatX(learning_rate), name='learning_rate')
        self.logger.info(
            "Learning rate initialised to {:.1E}".format(learning_rate))
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
        self.plotter = plotter
        self.plotter.net = self
        self.auto_reshape = auto_reshape

        self.csv_filenames = {
            'training_costs': self.experiment_name + "_training_costs.csv",
            'validation_costs': self.experiment_name + "_validation_costs.csv",
            'training_costs_metadata':
                self.experiment_name + "_training_costs_metadata.csv",
            'best_costs': self.experiment_name + "_best_costs.txt",
        }

        self.generate_validation_data_and_set_shapes()

        self.validation_costs = []
        self.training_costs = []
        self.training_costs_metadata = []
        self.layers = []
        self.layer_labels = {}

        # Shape is (number of examples per batch,
        #           maximum number of time steps per example,
        #           number of features per example)
        self.layers.append(InputLayer(shape=self.input_shape))
        self.add_layers(layers_config)
        self.logger.info(
            "Done initialising network for " + self.experiment_name)

    def generate_validation_data_and_set_shapes(self):
        # Generate a "validation" sequence whose cost we will compute
        self.validation_batch = self.source.validation_data()
        self.X_val, self.y_val = self.validation_batch.data
        self.input_shape = self.X_val.shape
        self.n_seq_per_batch = self.input_shape[0]
        self.output_shape = self.y_val.shape
        self.n_outputs = self.output_shape[-1]

    def add_layers(self, layers_config):
#        RECURRENT_LAYERS = [LSTMLayer, BLSTMLayer, DimshuffleLayer,
#                            RecurrentLayer, BidirectionalRecurrentLayer]
        RECURRENT_LAYERS = [DimshuffleLayer]

        for layer_config in layers_config:
            layer_type = layer_config.pop('type')
            layer_label = layer_config.pop('label', None)

            # Reshape if necessary
            if self.auto_reshape:
                prev_layer_output_shape = self.layers[-1].output_shape
                n_dims = len(prev_layer_output_shape)
                n_features = prev_layer_output_shape[-1]
                if layer_type in RECURRENT_LAYERS:
                    if n_dims == 2:
                        seq_length = int(prev_layer_output_shape[0] /
                                         self.source.n_seq_per_batch)
                        shape = (self.source.n_seq_per_batch,
                                 seq_length,
                                 n_features)
                        self.layers.append(ReshapeLayer(self.layers[-1], shape))
                elif layer_type in [DenseLayer, MixtureDensityLayer]:
                    if n_dims == 3:
                        # The prev layer_config was a time-aware layer_config,
                        # so reshape to 2-dims.
                        seq_length = prev_layer_output_shape[1]
                        shape = (self.source.n_seq_per_batch * seq_length,
                                 n_features)
                        self.layers.append(ReshapeLayer(self.layers[-1], shape))

            # Handle references:
            for k, v in layer_config.iteritems():
                if isinstance(v, basestring) and v.startswith("ref:"):
                    v = v[4:] # remove "ref:"
                    label, _, attr = v.partition('.')
                    target_layer = self.layer_labels[label]
#                    layer_config[k] = getattr(target_layer, attr)
                    layer_config[k] = eval("target_layer.{:s}".format(attr))
                    print(layer_config[k])
                    print(type(layer_config[k]))

            self.logger.info('Initialising layer_config : {}'.format(layer_type))
                    
            # Handle ConcatLayers
            if layer_type == ConcatLayer:
                incoming = [
                    self.layer_labels[ref]
                    for ref in layer_config.pop('incomings')]
            else:
                incoming = self.layers[-1]

            # Init new layer_config
            apply_batch_norm = layer_config.pop('batch_normalize', False)
            layer = layer_type(incoming, **layer_config)
            if apply_batch_norm:
                layer = batch_norm(layer)
            self.layers.append(layer)

            if layer_label is not None:
                self.layer_labels[layer_label] = layer

        # Reshape output if necessary...
        if (self.layers[-1].output_shape != self.output_shape and
            layer_type != MixtureDensityLayer):
            self.layers.append(ReshapeLayer(self.layers[-1], self.output_shape))

        self.logger.info("Total parameters = {}".format(
            sum([p.get_value().size for p in
                 lasagne.layers.get_all_params(self.layers[-1])])))

    def print_net(self):
        layers = get_all_layers(self.layers[-1])
        for layer in layers:
            self.logger.info(str(layer))
            try:
                input_shape = layer.input_shape
            except:
                pass
            else:
                self.logger.info(" Input shape: {}".format(input_shape))
            self.logger.info("Output shape: {}".format(layer.output_shape))

    def compile(self):
        self.logger.info("Compiling Theano functions...")
        target_output = ndim_tensor(name='target_output', ndim=self.y_val.ndim)
        network_input = ndim_tensor(name='network_input', ndim=self.X_val.ndim)
        output_layer = self.layers[-1]

        # Training
        network_output_train = lasagne.layers.get_output(
            output_layer, network_input)
        loss_train = self.loss_function(network_output_train, target_output)

        # Evaluation (test and validation)
        network_output_eval = lasagne.layers.get_output(
            output_layer, network_input, deterministic=True)
        loss_eval = self.loss_function(network_output_eval, target_output)

        # Updates
        all_params = lasagne.layers.get_all_params(
            output_layer, trainable=True)
        updates = self.updates_func(
            loss_train, all_params, learning_rate=self._learning_rate,
            **self.updates_kwargs)

        # Theano functions for training, getting output,
        # and computing loss_train
        self.train = theano.function(
            inputs=[network_input, target_output],
            outputs=loss_train,
            updates=updates,
            on_unused_input='warn',
            allow_input_downcast=True)

        deterministic_output = lasagne.layers.get_output(
            output_layer, network_input, deterministic=True)

        self.y_pred = theano.function(
            inputs=[network_input],
            outputs=deterministic_output,
            on_unused_input='warn',
            allow_input_downcast=True)

        self.compute_cost = theano.function(
            inputs=[network_input, target_output],
            outputs=[loss_eval, deterministic_output],
            on_unused_input='warn',
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
            self.logger.info(
                "Removed {}".format(self.layers.pop(layer_to_remove)))
        if 'callback' in layer_changes:
            layer_changes['callback'](self, epoch)
        self.add_layers(layer_changes['new_layers'])
        self.logger.info("New architecture:")
        self.print_net()
        self.compile()
        self.source.start()

    def _save_training_costs_metadata(self):
        if not self.training_costs_metadata:
            return
        keys = self.training_costs_metadata[-1].keys()
        n_iterations = self.n_iterations()
        if n_iterations == 0:
            mode = 'w'
        else:
            mode = 'a'
        with open(self.csv_filenames['training_costs_metadata'], mode) as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            if n_iterations == 0:
                writer.writeheader()
            writer.writerow(self.training_costs_metadata[-1])

    def print_and_save_training_progress(self, duration):
        iteration = self.n_iterations()
        train_cost = self.training_costs[-1]
        validation_cost = (self.validation_costs[-1] if self.validation_costs
                           else None)
        _write_csv_row(self.csv_filenames['training_costs'],
                       [iteration, train_cost, duration])
        self._save_training_costs_metadata()
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
              "  {:>11.6f}  |  {:>.3f}s".format(
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

    def _write_csv_headers(self, key='all'):
        if key in ['all', 'training_costs']:
            _write_csv_row(
                self.csv_filenames['training_costs'],
                row=['iteration', 'train_cost', 'duration'],
                mode='w')
        if key in ['all', 'validation_costs']:
            _write_csv_row(
                self.csv_filenames['validation_costs'],
                row=['iteration', 'validation_cost'],
                mode='w')

    @property
    def learning_rate(self):
        return self._learning_rate.get_value()

    @learning_rate.setter
    def learning_rate(self, rate):
        rate = sfloatX(rate)
        self.logger.info(
            "Iteration {:d}: Change learning rate to {:.1E}"
            .format(self.n_iterations(), rate))
        self._learning_rate.set_value(rate)

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
            self._write_csv_headers()

        while iteration != n_iterations:
            t0 = time()  # for calculating training duration
            iteration = len(self.training_costs)
            if iteration in self.learning_rate_changes_by_iteration:
                self.learning_rate = (
                    self.learning_rate_changes_by_iteration[iteration])
            if iteration in self.layer_changes:
                self._change_layers(iteration)
            if iteration in self.epoch_callbacks:
                self.epoch_callbacks[iteration](self, iteration)
            batch = self.source.queue.get(timeout=30)
            X, y = batch.data
            train_cost = self.train(X, y).flatten()[0]
            self.training_costs.append(train_cost)
            if batch.metadata:
                self.training_costs_metadata.append(batch.metadata)
            if not iteration % self.validation_interval:
                validation_cost = self.compute_cost(self.X_val, self.y_val)[0]
                validation_cost = validation_cost.flatten()[0]
                self.validation_costs.append(validation_cost)
                _write_csv_row(
                    self.csv_filenames['validation_costs'],
                    row=[iteration, validation_cost])
            if not iteration % self.save_plot_interval:
                self.save()
            duration = time() - t0
            self.print_and_save_training_progress(duration)
        self.logger.info("Finished training")

    def save(self):
        self.logger.info("Saving plots...")
        try:
            self.plotter.plot_all()
        except:
            self.logger.exception("")
        self.logger.info("Saving params...")
        try:
            self.save_params()
        except:
            self.logger.exception("")
        self.logger.info("Saving activations...")
        try:
            self.save_activations()
        except:
            self.logger.exception("")
        self.logger.info("Finished saving.")

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
        except ValueError:
            self.logger.exception("Cannot save params!")
            f.close()
            return

        layers = get_all_layers(self.layers[-1])
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
                layer_group.create_dataset(
                    param_name, data=data, compression="gzip")

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

        # LOAD COSTS
        def load_csv(key, limit):
            filename = self.csv_filenames[key]
            data = np.genfromtxt(filename, delimiter=',', skip_header=1)
            data = data[:limit, :]

            # overwrite costs file
            self._write_csv_headers(key)
            with open(filename, mode='a') as fh:
                np.savetxt(fh, data, delimiter=',')
            return list(data[:, 1])

        self.training_costs = load_csv('training_costs', iteration)
        self.validation_costs = load_csv(
            'validation_costs', iteration // self.validation_interval)

        # LOAD TRAINING COSTS METADATA
        metadata_fname = self.csv_filenames['training_costs_metadata']
        try:
            metadata_fh = open(metadata_fname, 'r')
        except IOError:
            pass
        else:
            reader = csv.DictReader(metadata_fh)
            training_costs_metadata = [row for row in reader]
            keys = training_costs_metadata[-1].keys()
            metadata_fh.close()
            self.training_costs_metadata = training_costs_metadata[:iteration]
            if len(training_costs_metadata) > iteration:
                # Overwrite old file
                with open(metadata_fname, 'w') as metadata_fh:
                    writer = csv.DictWriter(metadata_fh, keys)
                    writer.writeheader()
                    writer.writerows(self.training_costs_metadata)

        # set learning rate
        if self.learning_rate_changes_by_iteration:
            keys = self.learning_rate_changes_by_iteration.keys()
            keys.sort(reverse=True)
            for key in keys:
                if key < iteration:
                    self.learning_rate = (
                        self.learning_rate_changes_by_iteration[key])
                    break

    def save_activations(self):
        if not self.do_save_activations:
            return
        filename = self.experiment_name + "_activations.hdf5"
        mode = 'w' if self.n_iterations() == 0 else 'a'
        f = h5py.File(filename, mode=mode)
        epoch_name = 'epoch{:06d}'.format(self.n_iterations())
        try:
            epoch_group = f.create_group(epoch_name)
        except ValueError:
            self.logger.exception("Cannot save params!")
            f.close()
            return

        layers = get_all_layers(self.layers[-1])
        for layer_i, layer in enumerate(layers):
            # We only care about layers with params
            if not (layer.get_params() or isinstance(layer, FeaturePoolLayer)):
                continue

            output = lasagne.layers.get_output(layer, self.X_val).eval()
            n_features = output.shape[-1]
            seq_length = int(output.shape[0] / self.source.n_seq_per_batch)

            if isinstance(layer, DenseLayer):
                shape = (self.source.n_seq_per_batch, seq_length, n_features)
                output = output.reshape(shape)
            elif isinstance(layer, Conv1DLayer):
                output = output.transpose(0, 2, 1)

            layer_name = 'L{:02d}_{}'.format(layer_i, layer.__class__.__name__)
            epoch_group.create_dataset(
                layer_name, data=output, compression="gzip")

        # save validation data
        if self.n_iterations() == 0:
            f.create_dataset(
                'validation_data', data=self.X_val, compression="gzip")

        f.close()


def _write_csv_row(filename, row, mode='a'):
    with open(filename, mode=mode) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(row)


"""
Emacs variables
Local Variables:
compile-command: "rsync -uvzr --progress --exclude '.git' --exclude '.ropeproject' --exclude '*/.ipynb_checkpoints' --exclude '*/flycheck_*.py' /home/jack/workspace/python/neuralnilm/ /mnt/sshfs/imperial/workspace/python/neuralnilm/"
End:
"""

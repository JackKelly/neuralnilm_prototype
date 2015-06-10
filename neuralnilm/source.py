from __future__ import print_function, division
from Queue import Queue, Empty
import threading
import numpy as np
import pandas as pd
from nilmtk import DataSet, TimeFrame, MeterGroup
from datetime import timedelta
from sys import stdout
from collections import OrderedDict
from lasagne.utils import floatX
from warnings import warn
import gc
from nilmtk.electric import activation_series_for_chunk
from nilmtk.timeframegroup import TimeFrameGroup
import logging
from .rectangulariser import rectangularise

SECS_PER_DAY = 60 * 60 * 24


class Source(object):
    def __init__(self, seq_length, n_seq_per_batch, n_inputs, n_outputs,
                 logger=None,
                 X_processing_func=lambda X: X,
                 y_processing_func=lambda y: y,
                 reshape_target_to_2D=False,
                 independently_center_inputs=False,
                 standardise_input=False,
                 unit_variance_targets=False,
                 standardise_targets=False,
                 input_padding=0,
                 input_stats=None,
                 target_stats=None,
                 seed=42,
                 output_central_value=False,
                 classification=False,
                 random_window=0,
                 clock_period=None,
                 clock_type=None,
                 two_pass=False,
                 subsample_target=1,
                 n_rectangular_segments=None):
        """
        Parameters
        ----------
        clock_type : {'one_hot', 'ramp'}
        """
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            if not self.logger.handlers:
                self.logger.addHandler(logging.StreamHandler(stdout))
        else:
            self.logger = logger

        self.seq_length = seq_length
        self.n_seq_per_batch = n_seq_per_batch
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.input_padding = input_padding
        self.queue = Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = None
        self.X_processing_func = X_processing_func
        self.y_processing_func = y_processing_func
        self.reshape_target_to_2D = reshape_target_to_2D
        self.rng = np.random.RandomState(seed)
        self.output_central_value = output_central_value
        self.classification = classification
        self.random_window = random_window
        self.subsample_target = subsample_target
        self.n_rectangular_segments = n_rectangular_segments

        if self.seq_length % self.subsample_target:
            raise RuntimeError(
                "subsample_target must exactly divide seq_length.")

        if self.subsample_target == 0:
            raise RuntimeError("subsample_target must not be 0.")

        if self.random_window and self.subsample_target:
            if self.random_window % self.subsample_target:
                raise RuntimeError(
                    "subsample_target must exactly divide random_window")

        self.input_stats = input_stats
        self.target_stats = target_stats
        self.independently_center_inputs = independently_center_inputs
        self.standardise_input = standardise_input
        self.standardise_targets = standardise_targets
        self.unit_variance_targets = unit_variance_targets

        self._init_data()
        self._initialise_standardisation()

        if 'lag' in self.__dict__:
            self.clock_period = (self.lag if clock_period is None
                                 else clock_period)
        self.clock_type = clock_type
        self.two_pass = two_pass

    def _init_data():
        pass

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run)
        self._thread.start()

    def run(self):
        """Puts training data into a Queue"""
        while not self._stop.is_set():
            X, y = self._gen_data()
            X, y = self._process_data(X, y)
            self.queue.put((X, y))
        self.empty_queue()

    def _initialise_standardisation(self):
        if not (self.standardise_input or self.standardise_targets):
            return

        if self.input_stats is None:
            X, y = self._gen_data()
            X = X.reshape(
                self.n_seq_per_batch * (self.seq_length + self.input_padding),
                self.n_inputs)
            self.input_stats = {'mean': X.mean(axis=0), 'std': X.std(axis=0)}

        if self.target_stats is None:
            # Get targets.  Temporarily turn off skip probability
            if 'skip_probability' in self.__dict__:
                skip_prob = self.skip_probability
                self.skip_probability = 0
            if 'skip_probability_for_first_appliance' in self.__dict__:
                skip_prob_for_first_appliance = (
                    self.skip_probability_for_first_appliance)
                self.skip_probability_for_first_appliance = 0
            X, y = self._gen_data()
            if 'skip_probability' in self.__dict__:
                self.skip_probability = skip_prob
            if 'skip_probability_for_first_appliance' in self.__dict__:
                self.skip_probability_for_first_appliance = (
                    skip_prob_for_first_appliance)

            y = y.reshape(self.n_seq_per_batch * self.seq_length,
                          self.n_outputs)
            self.target_stats = {'mean': y.mean(axis=0), 'std': y.std(axis=0)}

        np.save('target_stats_mean', self.target_stats['mean'])
        np.save('target_stats_std', self.target_stats['std'])
        np.save('input_stats_mean', self.input_stats['mean'])
        np.save('input_stats_std', self.input_stats['std'])

    def stop(self):
        self.empty_queue()
        self._stop.set()
        self._thread.join()
        self._thread = None

    def empty_queue(self):
        while True:
            try:
                self.queue.get(block=False)
            except Empty:
                break

    def validation_data(self):
        X, y = self._gen_data(validation=True)
        return self._process_data(X, y)

    def _process_data(self, X, y):
        if self.subsample_target > 1:
            y = y.reshape(self.n_seq_per_batch, -1,
                          self.subsample_target, self.n_outputs)
            y = y.mean(axis=2)

        def _standardise(n, stats, data):
            for i in range(n):
                mean = stats['mean'][i]
                std = stats['std'][i]
                data[:, :, i] = standardise(
                    data[:, :, i], how='std=1', mean=mean, std=std)
            return data

        if self.independently_center_inputs:
            for seq_i in range(self.n_seq_per_batch):
                for input_i in range(self.n_inputs):
                    X[seq_i, :, input_i] = standardise(
                        X[seq_i, :, input_i], how='std=1',
                        std=self.input_stats['std'])

        elif self.standardise_input:
            X = _standardise(self.n_inputs, self.input_stats, X)

        if self.unit_variance_targets:
            for i in range(self.n_outputs):
                std = self.target_stats['std'][i]
                y[:, :, i] /= std
        elif self.standardise_targets:
            y = _standardise(self.n_outputs, self.target_stats, y)

        if self.random_window:
            y_seq_length = self.seq_length // self.subsample_target
            y_window_width = self.random_window // self.subsample_target
            latest_y_window_start = y_seq_length - y_window_width
            y_window_start = self.rng.randint(0, latest_y_window_start)
            y_window_end = y_window_start + y_window_width
            y = y[:, y_window_start:y_window_end, :]
            x_window_start = y_window_start * self.subsample_target
            x_window_end = y_window_end * self.subsample_target
            X = X[:, x_window_start:x_window_end, :]

        if self.reshape_target_to_2D:
            y = y.reshape(self.output_shape_after_processing())

        X = self.X_processing_func(X)
        y = self.y_processing_func(y)
        if self.n_rectangular_segments is not None:
            y = rectangularise(y, n_segments=self.n_rectangular_segments)

        if self.classification:
            y = (y > 0).max(axis=1).reshape(
                self.output_shape_after_processing())
        elif self.output_central_value:
            seq_length = y.shape[1]
            half_seq_length = seq_length // 2
            y = y[:, half_seq_length:half_seq_length+1, :]

        if self.clock_type == 'one_hot':
            clock = np.zeros(
                shape=(self.n_seq_per_batch, self.seq_length, self.n_inputs),
                dtype=np.float32)

            for i in range(self.clock_period):
                clock[:, i::self.clock_period, i] = 1

        elif self.clock_type == 'ramp':
            ramp = np.linspace(start=-1, stop=1, num=self.clock_period,
                               dtype=np.float32)
            n_ramps = np.ceil(self.seq_length / self.clock_period)
            ramp_for_one_seq = np.tile(ramp, n_ramps)[:self.seq_length]
            clock = np.tile(ramp_for_one_seq, (self.n_seq_per_batch, 1))
            clock = clock.reshape((self.n_seq_per_batch, self.seq_length, 1))

        if self.clock_type is not None:
            X = np.concatenate((X, clock), axis=2)

        if self.two_pass:
            X = np.tile(X, (1, 2, 1))
            shape = (self.n_seq_per_batch, self.seq_length * 2, 1)
            encode_flag = np.zeros(shape)
            encode_flag[:, :self.seq_length, :] = 0
            encode_flag[:, self.seq_length:, :] = 1
            X = np.concatenate((X, encode_flag), axis=2)
            X[:, self.seq_length:, 0] = 0
            y = np.tile(y, (1, 2, 1))

        X, y = floatX(X), floatX(y)
        self._check_data(X, y)
        return X, y

    def _gen_data(self, validation=False):
        raise NotImplementedError()

    def input_shape(self):
        if self.random_window:
            seq_length = self.random_window
        else:
            seq_length = self.seq_length

        return (self.n_seq_per_batch, seq_length, self.n_inputs)

    def input_shape_after_processing(self):
        n_seq_per_batch, seq_length, n_inputs = self.input_shape()
        if self.random_window:
            seq_length = self.random_window
        if self.two_pass:
            seq_length *= 2
        if self.clock_type == 'one_hot':
            n_inputs += self.clock_period
        elif self.clock_type == 'ramp':
            n_inputs += 1
        if self.two_pass:
            n_inputs += 1
        return (n_seq_per_batch, seq_length, n_inputs)

    def output_shape(self):
        return (self.n_seq_per_batch, self.seq_length, self.n_outputs)

    def output_shape_after_processing(self):
        n_seq_per_batch, seq_length, n_outputs = self.output_shape()
        if self.random_window:
            seq_length = self.random_window
        if self.two_pass:
            seq_length *= 2
        if self.subsample_target > 1:
            seq_length = seq_length // self.subsample_target
        if self.n_rectangular_segments:
            seq_length = self.n_rectangular_segments
            n_outputs = 1

        if self.reshape_target_to_2D:
            return (n_seq_per_batch * seq_length, n_outputs)
        elif self.output_central_value or self.classification:
            return (n_seq_per_batch, 1, n_outputs)
        else:
            return (n_seq_per_batch, seq_length, n_outputs)

    def _check_data(self, X, y):
        assert X.shape == self.input_shape_after_processing()
        assert not any(np.isnan(X.flatten()))
        if y is not None:
            assert y.shape == self.output_shape_after_processing()
            assert not any(np.isnan(y.flatten()))


def none_to_list(x):
    return [] if x is None else x


class ToySource(Source):
    def __init__(self, seq_length, n_seq_per_batch, n_inputs=1,
                 powers=None, on_durations=None, all_hot=True,
                 fdiff=False):
        """
        Parameters
        ----------
        n_inputs : int
            if > 1 then will quantize inputs
        powers : list of numbers
        on_durations : list of numbers
        """
        self.powers = [10, 40] if powers is None else powers
        self.on_durations = [3, 10] if on_durations is None else on_durations
        self.all_hot = all_hot
        self.fdiff = fdiff
        super(ToySource, self).__init__(
            seq_length=seq_length,
            n_seq_per_batch=n_seq_per_batch,
            n_inputs=n_inputs,
            n_outputs=1)

    def _gen_single_appliance(self, power, on_duration,
                              min_off_duration=20, p=0.2):
        length = self.seq_length + 1 if self.fdiff else self.seq_length
        appliance_power = np.zeros(length)
        i = 0
        while i < length:
            if self.rng.binomial(n=1, p=p):
                end = min(i + on_duration, length)
                appliance_power[i:end] = power
                i += on_duration + min_off_duration
            else:
                i += 1
        return np.diff(appliance_power) if self.fdiff else appliance_power

    def _gen_batches_of_single_appliance(self, *args, **kwargs):
        batches = np.empty(shape=(self.n_seq_per_batch, self.seq_length, 1))
        for i in range(self.n_seq_per_batch):
            single_appliance = self._gen_single_appliance(*args, **kwargs)
            batches[i, :, :] = single_appliance.reshape(self.seq_length, 1)
        return batches

    def _gen_unquantized_data(self, validation=False):
        y = self._gen_batches_of_single_appliance(
            power=self.powers[0], on_duration=self.on_durations[0])
        X = y.copy()
        for power, on_duration in zip(self.powers, self.on_durations)[1:]:
            X += self._gen_batches_of_single_appliance(
                power=power, on_duration=on_duration)

        return X / np.max(X), y / np.max(y)

    def _gen_data(self, *args, **kwargs):
        X, y = self._gen_unquantized_data(*args, **kwargs)
        if self.n_inputs > 1:
            X = quantize(X, self.n_inputs, self.all_hot)
        return X, y


class RealApplianceSource(Source):
    def __init__(self, filename, appliances,
                 min_on_durations,
                 max_appliance_powers=None,
                 min_off_durations=None,
                 on_power_thresholds=None,
                 max_input_power=None,
                 clip_input=True,
                 window=(None, None),
                 seq_length=1000,
                 n_seq_per_batch=5,
                 train_buildings=None,
                 validation_buildings=None,
                 output_one_appliance=True,
                 sample_period=6,
                 boolean_targets=False,
                 input_padding=0,
                 skip_probability=0,
                 skip_probability_for_first_appliance=None,
                 include_diff=False,
                 include_power=True,
                 target_is_diff=False,
                 max_diff=3000,
                 clip_appliance_power=True,
                 lag=None,
                 target_is_prediction=False,
                 one_target_per_seq=False,
                 ensure_all_appliances_represented=True,
                 **kwargs):
        """
        Parameters
        ----------
        filename : str
        appliances : list of strings
            The first one is the target appliance, if
            output_one_appliance is True.
            Use a list of lists for alternative names
            (e.g. ['fridge', 'fridge freezer']).
        subsample_target : int
            If > 1 then subsample the targets.
        skip_probability : float, [0, 1]
            If `skip_probability` == 0 then all appliances will be included in
            every sequence.  Else each appliance will be skipped with this
            probability but every appliance will be present in at least
            one sequence per batch.
        """
        self.dataset = DataSet(filename)
        self.appliances = appliances

        if max_input_power is None and max_appliance_powers is not None:
            self.max_input_power = np.sum(max_appliance_powers)
        else:
            self.max_input_power = max_input_power
        self.clip_input = clip_input

        if max_appliance_powers is None:
            max_appliance_powers = [None] * len(appliances)

        self.max_appliance_powers = {}
        for i, appliance in enumerate(appliances):
            if isinstance(appliance, list):
                appliance_name = appliance[0]
            else:
                appliance_name = appliance
            self.max_appliance_powers[appliance_name] = max_appliance_powers[i]

        self.dataset.set_window(*window)
        if train_buildings is None:
            train_buildings = [1]
        if validation_buildings is None:
            validation_buildings = [1]
        self.output_one_appliance = output_one_appliance
        self.sample_period = sample_period
        self.boolean_targets = boolean_targets
        self.skip_probability = skip_probability
        if skip_probability_for_first_appliance is None:
            self.skip_probability_for_first_appliance = skip_probability
        else:
            self.skip_probability_for_first_appliance = (
                skip_probability_for_first_appliance)
        self._tz = self.dataset.metadata['timezone']
        self.include_diff = include_diff
        self.include_power = include_power
        self.max_diff = max_diff
        self.clip_appliance_power = clip_appliance_power
        if lag is None:
            lag = -1 if target_is_prediction else 0
        elif lag == 0 and target_is_prediction:
            warn("lag is 0 and target_is_prediction==True."
                 "  Hence output will be identical to input.")
        self.lag = lag
        self.target_is_prediction = target_is_prediction
        self.target_is_diff = target_is_diff
        self.one_target_per_seq = one_target_per_seq
        self.ensure_all_appliances_represented = (
            ensure_all_appliances_represented)

        self.logger.info("Loading training activations...")
        if on_power_thresholds is None:
            on_power_thresholds = [None] * len(self.appliances)
        if min_on_durations is None:
            min_on_durations = [0] * len(self.appliances)

        self.train_activations = self._load_activations(
            train_buildings, min_on_durations, min_off_durations,
            on_power_thresholds)
        if train_buildings == validation_buildings:
            self.validation_activations = self.train_activations
        else:
            self.logger.info("Loading validation activations...")
            self.validation_activations = self._load_activations(
                validation_buildings, min_on_durations, min_off_durations,
                on_power_thresholds)
        self.dataset.store.close()

        super(RealApplianceSource, self).__init__(
            seq_length=seq_length,
            n_seq_per_batch=n_seq_per_batch,
            n_inputs=sum([include_diff, include_power]),
            n_outputs=(1 if output_one_appliance or target_is_prediction
                       else len(appliances)),
            input_padding=input_padding,
            **kwargs
        )
        assert not (self.input_padding and self.random_window)
        self.logger.info("Done loading activations.")

    def get_labels(self):
        return self.train_activations.keys()

    def _load_activations(self, buildings, min_on_durations, min_off_durations,
                          on_power_thresholds):
        activations = OrderedDict()
        for building_i in buildings:
            elec = self.dataset.buildings[building_i].elec
            meters = get_meters_for_appliances(elec, self.appliances)
            for appliance_i, meter in enumerate(meters):
                appliance = self.appliances[appliance_i]
                if isinstance(appliance, list):
                    appliance = appliance[0]
                self.logger.info(
                    "  Loading activations for {} from building {}..."
                    .format(appliance, building_i))
                activation_series = meter.activation_series(
                    on_power_threshold=on_power_thresholds[appliance_i],
                    min_on_duration=min_on_durations[appliance_i],
                    min_off_duration=min_off_durations[appliance_i])
                activations[appliance] = _preprocess_activations(
                    activation_series,
                    max_power=self.max_appliance_powers[appliance],
                    sample_period=self.sample_period,
                    clip_appliance_power=self.clip_appliance_power)
                self.logger.info(
                    "Loaded {:d} activations.".format(len(activation_series)))
        return activations

    def _gen_single_example(self, validation=False, appliances=None):
        if appliances is None:
            appliances = []
        X = np.zeros(shape=(self.seq_length, self.n_inputs), dtype=np.float32)
        y = np.zeros(shape=(self.seq_length, self.n_outputs), dtype=np.float32)
        POWER_THRESHOLD = 1
        BORDER = 5
        activations = (self.validation_activations if validation
                       else self.train_activations)

        if not self.one_target_per_seq:
            random_appliances = []
            appliance_names = activations.keys()
            while not random_appliances:
                if not self.rng.binomial(
                        n=1, p=self.skip_probability_for_first_appliance):
                    appliance_i = 0
                    appliance = appliance_names[0]
                    random_appliances.append((appliance_i, appliance))
                for appliance_i, appliance in enumerate(appliance_names[1:]):
                    if not self.rng.binomial(n=1, p=self.skip_probability):
                        random_appliances.append((appliance_i+1, appliance))

            appliances.extend(random_appliances)

        appliances = list(set(appliances))  # make unique

        for appliance_i, appliance in appliances:
            n_activations = len(activations[appliance])
            if n_activations == 0:
                continue
            activation_i = self.rng.randint(0, n_activations)
            activation = activations[appliance][activation_i]
            latest_start_i = ((self.seq_length - len(activation)) -
                              (BORDER + self.lag))
            latest_start_i = max(latest_start_i, BORDER)
            start_i = self.rng.randint(0, latest_start_i)
            end_i = start_i + len(activation)
            end_i = min(end_i, self.seq_length-(1+self.lag))
            target = activation.values[:end_i-start_i]
            X[start_i:end_i, 0] += target
            if (not self.target_is_prediction and
                    (appliance_i == 0 or not self.output_one_appliance)):
                target = np.copy(target)
                if self.boolean_targets:
                    target[target <= POWER_THRESHOLD] = 0
                    target[target > POWER_THRESHOLD] = 1
                else:
                    max_appliance_power = self.max_appliance_powers[appliance]
                    if max_appliance_power is not None:
                        target /= max_appliance_power
                if self.target_is_diff:
                    y[(start_i+self.lag):(end_i+self.lag-1), appliance_i] = (
                        np.diff(target))
                else:
                    y[(start_i+self.lag):(end_i+self.lag), appliance_i] = (
                        target)

        if self.clip_input:
            np.clip(X, 0, self.max_input_power, out=X)

        fdiff = np.diff(X[:, 0]) / self.max_diff
        if self.max_input_power is not None:
            X[:, 0] /= self.max_input_power

        if self.target_is_prediction:
            if self.target_is_diff:
                data = np.concatenate([fdiff, [0]]).reshape(self.seq_length, 1)
            else:
                data = np.copy(X)

            if self.lag > 0:
                y[self.lag:, :] = data[:-self.lag, :]
            elif self.lag == 0:
                y = data
            else:
                y[:self.lag, :] = data[-self.lag:, :]

        if self.include_diff:
            feature_i = int(self.include_power)
            X[:-1, feature_i] = fdiff

        return X, y

    def input_shape(self):
        return (self.n_seq_per_batch,
                self.seq_length + self.input_padding,
                self.n_inputs)

    def inside_padding(self):
        start = self.input_padding // 2
        end = -int(np.ceil(self.input_padding / 2))
        if end == 0:
            end = None
        return start, end

    def _appliances_for_sequence(self):
        """Returns a dict which maps from seq_i to a list of appliances which
        must be included in that sequence.  This is used to ensure that,
        if `skip_probability` > 0 then every appliance must be represented in
        at least one sequence.
        """
        if not self.ensure_all_appliances_represented:
            return {}
        all_appliances = list(enumerate(self.get_labels()))
        if self.one_target_per_seq:
            return {i: [all_appliances[i % len(all_appliances)]]
                    for i in range(self.n_seq_per_batch)}
        if self.skip_probability == 0:
            return {i: [] for i in range(self.n_seq_per_batch)}
        n_appliances = len(self.appliances)
        n_appliances_per_seq = n_appliances // self.n_seq_per_batch
        remainder = n_appliances % self.n_seq_per_batch
        appliances_for_sequence = {}

        for i in range(self.n_seq_per_batch):
            start = n_appliances_per_seq * i
            end = start + n_appliances_per_seq
            if remainder:
                end += 1
                remainder -= 1
            appliances = all_appliances[start:end]
            appliances_for_sequence[i] = appliances
        return appliances_for_sequence

    def _gen_data(self, validation=False):
        X = np.zeros(self.input_shape(), dtype=np.float32)
        y = np.zeros(self.output_shape(), dtype=np.float32)
        start, end = self.inside_padding()
        deterministic_appliances = self._appliances_for_sequence()
        for i in range(self.n_seq_per_batch):
            X[i, start:end, :], y[i, :, :] = self._gen_single_example(
                validation, deterministic_appliances.get(i))
        return X, y


class NILMTKSource(Source):
    def __init__(self, filename, appliances,
                 train_buildings, validation_buildings,
                 window=(None, None),
                 sample_period=6,
                 **kwargs):
        super(NILMTKSource, self).__init__(
            n_outputs=len(appliances),
            n_inputs=1,
            **kwargs)
        self.window = window
        self.dataset = DataSet(filename)
        self.dataset.set_window(*window)
        self.tz = self.dataset.metadata['timezone']
        self.window = [pd.Timestamp(ts, tz=self.tz) for ts in self.window]
        self.appliances = appliances
        self.train_buildings = train_buildings
        self.validation_buildings = validation_buildings
        self.sample_period = sample_period
        self._init_meter_groups()
        self._init_good_sections()

    def _init_meter_groups(self):
        self.metergroups = {}
        for building_i in self._all_buildings():
            elec = self.dataset.buildings[building_i].elec
            meters = get_meters_for_appliances(elec, self.appliances)
            self.metergroups[building_i] = MeterGroup(meters)

    def _all_buildings(self):
        buildings = self.train_buildings + self.validation_buildings
        buildings = list(set(buildings))
        return buildings

    def _init_good_sections(self):
        self.good_sections = {}
        min_duration_secs = self.sample_period * self.seq_length
        min_duration = timedelta(seconds=min_duration_secs)
        for building_i in self._all_buildings():
            self.logger.info(
                "init good sections for building {}".format(building_i))
            mains = self.dataset.buildings[building_i].elec.mains()
            self.good_sections[building_i] = [
                section for section in mains.good_sections()
                if section.timedelta > min_duration
            ]

    def _gen_single_example(self, validation=False):
        buildings = (self.validation_buildings if validation
                     else self.train_buildings)
        building_i = self.rng.choice(buildings)
        elec = self.dataset.buildings[building_i].elec
        section = self.rng.choice(self.good_sections[building_i])
        section_duration = section.timedelta.total_seconds()
        max_duration = self.sample_period * self.seq_length
        latest_start = section_duration - max_duration
        relative_start = self.rng.randint(0, latest_start)
        start = section.start + timedelta(seconds=relative_start)
        end = start + timedelta(seconds=max_duration)
        sections = [TimeFrame(start, end)]
        mains_power = elec.mains().power_series(
            sample_period=self.sample_period, sections=sections).next()
        appliances_power = self.metergroups[building_i].dataframe_of_meters(
            sample_period=self.sample_period, sections=sections)

        def truncate(data):
            n = len(data)
            assert n >= self.seq_length
            if n > self.seq_length:
                data = data[:self.seq_length]
            return data
        mains_power = truncate(mains_power)
        appliances_power = truncate(appliances_power)
        appliances_power.columns = elec.get_labels(appliances_power.columns)

        # time of day
        index = mains_power.index.tz_localize(None)
        secs_into_day = (index.astype(int) / 1E9) % SECS_PER_DAY
        time_of_day = ((secs_into_day / SECS_PER_DAY) * 2.) - 1.

        return appliances_power, mains_power, time_of_day


def timestamp_to_int(ts):
    ts = pd.Timestamp(ts)
    return ts.asm8.astype('datetime64[s]').astype(int)


class NILMTKSourceOld(Source):
    def __init__(self, filename, appliances, building=1):
        """
        Parameters
        ----------
        filename : str
        appliances : list of strings
            The first one is the target appliance
        building : int
        """
        super(NILMTKSource, self).__init__(
            seq_length=14400,
            n_seq_per_batch=5,
            n_inputs=1000,
            n_outputs=1)
        self.sample_period = 6
        self.min_power = 20
        self.max_power = 200
        self.dataset = DataSet(filename)
        self.appliances = appliances
        self._tz = self.dataset.metadata['timezone']
        self.metergroup = self.dataset.buildings[building].elec

    def _get_data_for_single_day(self, start):
        start = pd.Timestamp(start).date()
        end = start + timedelta(days=1)
        timeframe = TimeFrame(start, end, tz=self._tz)
        load_kwargs = dict(sample_period=self.sample_period,
                           sections=[timeframe])

        # Load output (target) data
        app = self.metergroup[self.appliances[0]]
        y = app.power_series_all_data(**load_kwargs)
        if y is None or y.max() < self.min_power:
            return None, None

        # Load input (aggregate) data
        app = y + self.metergroup[self.appliances[1]]
        X = app.power_series_all_data(**load_kwargs)
        for appliance in self.appliances[2:]:
            app = self.metergroup[appliance]
            X += app.power_series_all_data(**load_kwargs)

        freq = "{:d}S".format(self.sample_period)
        index = pd.date_range(start, end, freq=freq, tz=self._tz)

        def preprocess(data):
            data = data.fillna(0)
            data = data.clip(upper=self.max_power)
            data[data < self.min_power] = 0
            data = data.reindex(index, fill_value=0)
            data /= self.max_power
            return data

        def index_as_minus_one_to_plus_one(data):
            index = data.index.astype(np.int64)
            index -= np.min(index)
            index = index.astype(np.float32)
            index /= np.max(index)
            return np.vstack([index, data.values]).transpose()

        X = preprocess(X).diff().dropna().values
        y = preprocess(y).diff().dropna().values
        return X, y

    def _gen_unquantized_data(self, validation=False):
        X = np.empty(shape=(self.n_seq_per_batch, self.seq_length, 1))
        y = np.empty(shape=self.output_shape())
        N_DAYS = 600  # there are more like 632 days in the dataset
        FIRST_DAY = pd.Timestamp("2013-04-12")
        seq_i = 0
        while seq_i < self.n_seq_per_batch:
            if validation:
                low = N_DAYS
                high = N_DAYS + self.n_seq_per_batch
            else:
                low = 0
                high = N_DAYS
            days = self.rng.randint(low=low, high=high)
            start = FIRST_DAY + timedelta(days=days)
            X_one_seq, y_one_seq = self._get_data_for_single_day(start)

            if y_one_seq is not None:
                try:
                    X[seq_i, :, :] = X_one_seq.reshape(self.seq_length, 1)
                    y[seq_i, :, :] = y_one_seq.reshape(self.seq_length, 1)
                except ValueError as e:
                    self.logger.info(e)
                    self.logger.info("Skipping {}".format(start))
                else:
                    seq_i += 1
            else:
                self.logger.info("Skipping {}".format(start))
        return X, y

    def _gen_data(self, *args, **kwargs):
        X = kwargs.pop('X', None)
        if X is None:
            X, y = self._gen_unquantized_data(*args, **kwargs)
        else:
            y = None
        X_quantized = np.empty(shape=self.input_shape())
        for i in range(self.n_seq_per_batch):
            X_quantized[i, :, 0] = X[i, :, 0]  # time of day
            X_quantized[i, :, 1:] = quantize(X[i, :, 1], self.n_inputs)

        return X_quantized, y


class RandomSegments(Source):
    def __init__(self, filename, target_appliance,
                 train_buildings, validation_buildings,
                 window=(None, None),
                 sample_period=6,
                 ignore_incomplete=False,
                 on_power_threshold=50,
                 **kwargs):
        self.dataset = DataSet(filename)
        self.dataset.set_window(*window)
        self.window = self.dataset.store.window
        self.tz = self.dataset.metadata['timezone']
        self.target_appliance = target_appliance
        self.train_buildings = train_buildings
        self.validation_buildings = validation_buildings
        self.sample_period = sample_period
        self.ignore_incomplete = ignore_incomplete
        self.on_power_threshold = on_power_threshold
        super(RandomSegments, self).__init__(n_outputs=1, n_inputs=1, **kwargs)

    def _init_data(self):
        """Overridden by sub-classes."""
        self.good_sections = {}
        seq_duration_secs = self.seq_length * self.sample_period
        for building_i in self.get_all_buildings():
            elec = self.dataset.buildings[building_i].elec
            try:
                target_meter, _ = get_meter_for_appliance(
                    elec, self.target_appliance)
            except KeyError:
                self.logger.info(
                    "Building {} has no {}"
                    .format(building_i, self.target_appliance))
                self._remove_building(building_i)
                continue
            mains_good_sections = elec.mains().good_sections()
            target_good_sections = target_meter.good_sections()
            good_sections = mains_good_sections.intersection(
                target_good_sections)
            good_sections = good_sections.remove_shorter_than(
                seq_duration_secs)

            if len(good_sections) > 0:
                self.good_sections[building_i] = good_sections
            else:
                self.logger.info(
                    "Building {} has no good sections in window."
                    .format(building_i))
                self._remove_building(building_i)

    def _remove_building(self, building_i):
        def remove(lst):
            try:
                lst.remove(building_i)
            except ValueError:
                pass
        self.logger.info(
            "Removing building {} from validation_buildings and"
            " train_buildings.".format(building_i))
        remove(self.validation_buildings)
        remove(self.train_buildings)

    def get_all_buildings(self):
        return list(set(self.train_buildings + self.validation_buildings))

    def _gen_single_example(self, validation=False):
        # Select a building
        buildings = (self.validation_buildings if validation
                     else self.train_buildings)
        building_i = self.rng.choice(buildings, 1)[0]

        # Select a good section from building
        good_sections = self.good_sections[building_i]
        timeframe = self._select_section_and_time_and_load(
            validation, good_sections)
        mains = self._load_data('mains', building_i, timeframe)
        appliance = self._load_data('target', building_i, timeframe)
        return mains, appliance

    def _select_section_and_time_and_load(self, validation, good_sections):
        if set(self.train_buildings) == set(self.validation_buildings):
            if validation:
                good_sections = good_sections[-1:]
            else:
                good_sections = good_sections[:-1]
        if len(good_sections) > 1:
            good_section_i = self.rng.randint(low=0, high=len(good_sections)-1)
        else:
            good_section_i = 0
        good_section = good_sections[good_section_i]

        # Select a time window from within good section
        seq_duration_secs = self.seq_length * self.sample_period
        max_start_offset_secs = (good_section.timedelta.total_seconds() -
                                 seq_duration_secs)
        if max_start_offset_secs > 0:
            offset_secs = self.rng.randint(0, max_start_offset_secs)
        else:
            offset_secs = 0
        offset = timedelta(seconds=offset_secs)
        start = good_section.start + offset
        end = start + timedelta(seconds=seq_duration_secs)
        timeframe = TimeFrame(start, end)
        return timeframe

    def _load_data(self, mains_or_target, building_i, timeframe):
        elec = self.dataset.buildings[building_i].elec
        if mains_or_target == 'mains':
            electric = elec.mains()
        elif mains_or_target == 'target':
            electric = elec[self.target_appliance]
        else:
            raise RuntimeError("Unrecognised: '" + mains_or_target + "'")
        data = electric.power_series_all_data(
            sections=[timeframe], sample_period=self.sample_period)
        if mains_or_target == 'target' and self.ignore_incomplete:
            data = self._remove_incomplete(data)
        data.fillna(0, inplace=True)
        data = data.values[:self.seq_length]
        pad_width = self.seq_length - len(data)
        data = np.pad(data, (0, pad_width), 'constant')
        return data

    def _gen_data(self, validation=False):
        X = np.zeros(self.input_shape(), dtype=np.float32)
        y = np.zeros(self.output_shape(), dtype=np.float32)
        N_RETRIES = 10
        for i in range(self.n_seq_per_batch):
            for retry in range(N_RETRIES):
                single_X, single_y = self._gen_single_example(validation)
                if single_X is not None and single_y is not None:
                    X[i, :, 0], y[i, :, 0] = single_X, single_y
                    break
        return X, y

    def get_labels(self):
        return [self.target_appliance]

    def _remove_incomplete(self, data):
        activations = activation_series_for_chunk(
            data, on_power_threshold=self.on_power_threshold)
        new_data = pd.Series(0, index=data.index)
        for activation in activations:
            if activation is not None:
                new_data = new_data.add(activation, fill_value=0)
        return new_data


class RandomSegmentsInMemory(RandomSegments):
    def _init_data(self):
        super(RandomSegmentsInMemory, self)._init_data()
        self.data = OrderedDict()
        for building_i in self.get_all_buildings():
            elec = self.dataset.buildings[building_i].elec
            try:
                meter, target_app = get_meter_for_appliance(
                    elec, self.target_appliance)
            except KeyError:
                self.logger.info(
                    "Building {} has no {}"
                    .format(building_i, self.target_appliance))
                self._remove_building(building_i)
                continue
            target = meter.power_series_all_data(
                sample_period=self.sample_period)
            if target is None:
                self.logger.info(
                    "Building {} has no target data in time window."
                    .format(building_i))
                self._remove_building(building_i)
            else:
                target.fillna(0, inplace=True)
                self.data[building_i] = {
                    'mains': elec.mains().power_series_all_data(
                        sample_period=self.sample_period).dropna(),
                    'target': target
                }
            gc.collect()

    def _load_data(self, mains_or_target, building_i, timeframe):
        try:
            building_data = self.data[building_i]
        except KeyError:
            return
        data = building_data[mains_or_target][timeframe.start:timeframe.end]
        if mains_or_target == 'target' and self.ignore_incomplete:
            data = self._remove_incomplete(data)
        data = data.values[:self.seq_length]
        pad_width = self.seq_length - len(data)
        data = np.pad(data, (0, pad_width), 'constant')
        return data


class SameLocation(RandomSegments):
    def __init__(self,
                 offset_probability=0,
                 ignore_offset_activations=False,
                 clip_appliance_power=False,
                 max_appliance_power=1000,
                 skip_probability=0,
                 allow_incomplete=False,
                 ignore_incomplete=False,
                 include_all=False,
                 *args, **kwargs):
        self.offset_probability = offset_probability
        self.ignore_offset_activations = ignore_offset_activations
        self.clip_appliance_power = clip_appliance_power
        self.max_appliance_power = max_appliance_power
        self.skip_probability = skip_probability
        self.allow_incomplete = allow_incomplete
        self.include_all = include_all
        kwargs['ignore_incomplete'] = ignore_incomplete
        super(SameLocation, self).__init__(*args, **kwargs)

    def _init_data(self):
        """
        * Load all activations for target with nice sized border
        * Load all mains data into memory
        """
        self._load_activations()
        self._load_mains()
        if self.skip_probability:
            self._load_sections_without_target()

    def _load_sections_without_target(self):
        self.sections_without_target = {}
        for building_i in self.get_all_buildings():
            sections_without_target = TimeFrameGroup()
            prev_end = self.mains[building_i].index[0]
            for activation in self.activations[building_i]:
                sections_without_target.append(
                    TimeFrame(prev_end, activation.index[0]))
                prev_end = activation.index[-1]
            sections_without_target.append(
                TimeFrame(prev_end, self.mains[building_i].index[-1]))
            sections_without_target = sections_without_target.intersection(
                self.target_good_sections[building_i])
            sections_without_target = (
                sections_without_target.remove_shorter_than(
                    self.seq_length * self.sample_period))
            if len(sections_without_target) > 0:
                self.sections_without_target[building_i] = (
                    sections_without_target)

    def _gen_single_example(self, validation=False):
        N_RETRIES = 50
        for retry in range(N_RETRIES):
            X, y = self._gen_single_example_(validation)
            if X is None:
                continue
            if len(X) == self.seq_length:
                return X, y
        if X is None:
            raise RuntimeError("X is None")
        else:
            raise RuntimeError("X has wrong length: {}".format(len(X)))

    def _gen_single_example_(self, validation=False):
        """
        * pick building at random
        * pick an activation at random
        * data is that activation, padded at the end, plus mains at same time.
        """
        N_LEAD_IN = 50  # number of samples to lead in with

        # pick a building
        buildings = (self.validation_buildings if validation
                     else self.train_buildings)
        building_i = self.rng.choice(buildings, 1)[0]

        # pick an activation
        activations = self.activations[building_i]
        if set(self.validation_buildings) == set(self.train_buildings):
            n_validation_activations = (
                self.n_seq_per_batch // len(self.validation_buildings))
            if validation:
                activations = activations[:n_validation_activations]
            else:
                activations = activations[n_validation_activations:]
        if len(activations) == 0:
            raise RuntimeError(
                "No activations.  validation={}".format(validation))
        activation_i = self.rng.randint(low=0, high=len(activations)-1)
        activation = activations[activation_i]

        if self.rng.binomial(n=1, p=self.skip_probability):
            return self._seq_without_target(building_i, validation)

        y = np.pad(activation.values, (N_LEAD_IN, 0), 'constant')

        def zeros():
            return np.zeros(self.seq_length, dtype=np.float32)

        # random offset (number of samples)
        if self.rng.binomial(n=1, p=self.offset_probability):
            if self.rng.binomial(n=1, p=0.5):
                # shift backwards
                if self.allow_incomplete:
                    max_offset = N_LEAD_IN + len(activation)
                else:
                    max_offset = N_LEAD_IN
                offset = self.rng.randint(low=1, high=max_offset)
                if self.ignore_incomplete and offset >= N_LEAD_IN:
                    y = zeros()
                else:
                    y = y[offset:]
                offset_direction = 'backwards'
            else:
                # shift forwards
                if self.allow_incomplete:
                    max_offset = self.seq_length - N_LEAD_IN
                else:
                    max_offset = self.seq_length - N_LEAD_IN - len(activation)
                if max_offset > 1:
                    offset = self.rng.randint(low=1, high=max_offset)
                else:
                    offset = 0
                if self.ignore_incomplete and offset >= (
                        self.seq_length - N_LEAD_IN - len(activation)):
                    y = zeros()
                else:
                    y = np.pad(y, (offset, 0), 'constant')
                offset_direction = 'forwards'
            if self.ignore_offset_activations:
                y = zeros()
        else:
            offset = 0

        # clip or pad to get seq length to be exactly self.seq_length samples
        def clip_or_pad(data):
            data = data[:self.seq_length]
            n_zeros_to_pad = self.seq_length - len(data)
            data = np.pad(data, (0, n_zeros_to_pad), 'constant')
            return data

        y = clip_or_pad(y)

        # get mains
        mains = self.mains[building_i]
        start = activation.index[0] - timedelta(
            seconds=N_LEAD_IN * self.sample_period)
        if offset:
            offset_timedelta = timedelta(seconds=offset * self.sample_period)
            if offset_direction == 'backwards':
                start += offset_timedelta
            else:
                start -= offset_timedelta

        end = start + timedelta(
            seconds=(self.seq_length * self.sample_period) - 1)
        X = mains[start:end]

        if len(X) != self.seq_length:
            return None, None

        if self.include_all:
            y_series = pd.Series(y, index=X.index)
            # check activations backwards
            for i in range(activation_i-1, -1, -1):
                other_activation = activations[i]
                if (other_activation.index[-1] > start and
                    (not self.ignore_incomplete or
                     other_activation.index[0] > start)):
                    y_series = y_series.add(
                        other_activation[start:end], fill_value=0)
                else:
                    break

            # check activations forwards
            for i in range(activation_i+1, len(activations)):
                other_activation = activations[i]
                if (other_activation.index[0] < end and
                    (not self.ignore_incomplete or
                     other_activation.index[-1] < end)):
                    y_series = y_series.add(
                        other_activation[start:end], fill_value=0)
                else:
                    break

            y = y_series.values

        return X.values, y

    def _seq_without_target(self, building_i, validation):
        try:
            sections_without_target = self.sections_without_target[building_i]
        except KeyError:
            return None, None

        timeframe = self._select_section_and_time_and_load(
            validation, sections_without_target)

        y = np.zeros(self.seq_length, dtype=np.float32)
        X = self.mains[building_i][timeframe.start:timeframe.end]
        return X, y

    def _load_activations(self):
        activations = OrderedDict()
        self.target_good_sections = {}
        for building_i in self.get_all_buildings():
            elec = self.dataset.buildings[building_i].elec
            try:
                meter, target_app = get_meter_for_appliance(
                    elec, self.target_appliance)
            except KeyError:
                self.logger.info(
                    "Building {} has no {}"
                    .format(building_i, self.target_appliance))
                self._remove_building(building_i)
                continue
            self.target_good_sections[building_i] = meter.good_sections()
            activation_series = meter.activation_series(
                on_power_threshold=40)
            activations[building_i] = _preprocess_activations(
                activation_series,
                max_power=self.max_appliance_power,
                clip_appliance_power=self.clip_appliance_power)
            self.logger.info(
                "Loaded {:d} {:s} activations from house {:d}.".
                format(len(activation_series), target_app, building_i))
            gc.collect()
        self.activations = activations

    def _load_mains(self):
        mains = OrderedDict()
        for building_i in self.get_all_buildings():
            elec = self.dataset.buildings[building_i].elec
            meter = elec.mains()
            mains_data = meter.power_series_all_data(
                sample_period=self.sample_period)
            mains_data = mains_data.dropna()
            mains[building_i] = mains_data
            self.logger.info(
                "Loaded mains data for building {:d}.".format(building_i))
            gc.collect()
        self.mains = mains


def quantize(data, n_bins, all_hot=True, range=(-1, 1), length=None):
    midpoint = n_bins // 2
    if length is None:
        length = len(data)
    out = np.empty(shape=(length, n_bins))
    for i in np.arange(length):
        d = data[i]
        hist, _ = np.histogram(d, bins=n_bins, range=range)
        if all_hot:
            where = np.where(hist == 1)[0][0]
            if where > midpoint:
                hist[midpoint:where] = 1
            elif where < midpoint:
                hist[where:midpoint] = 1
        out[i, :] = hist
    return (out * 2) - 1


def standardise(X, how='range=2', mean=None, std=None, midrange=None,
                ptp=None):
    """Standardise.
    ftp://ftp.sas.com/pub/neural/FAQ2.html#A_std_in

    Parameters
    ----------
    X : matrix
        Each sample is in range [0, 1]
    how : str, {'range=2', 'std=1'}
        'range=2' sets midrange to 0 and enforces
        all values to be in the range [-1,1]
        'std=1' sets mean = 0 and std = 1

    Returns
    -------
    new_X : matrix
        Same shape as `X`.  Sample is in range [lower, upper]

    See also
    --------
    unstandardise
    """
    if how == 'std=1':
        if mean is None:
            mean = X.mean()
        if std is None:
            std = X.std()
        centered = X - mean
        if std == 0:
            return centered
        else:
            return centered / std
    elif how == 'range=2':
        if midrange is None:
            midrange = (X.max() + X.min()) / 2
        if ptp is None:
            ptp = X.ptp()
        return (X - midrange) / (ptp / 2)
    else:
        raise RuntimeError("unrecognised how '" + how + "'")


def discretize_scalar(X, n_bins=10, all_hot=False, boolean=True):
    output = np.zeros(n_bins)
    bin_i = int(X * n_bins)
    bin_i = min(bin_i, n_bins-1)
    output[bin_i] = 1 if boolean else ((X * n_bins) - bin_i)
    if all_hot:
        output[:bin_i] = 1
    return output


def discretize(X, n_bins=10, **kwargs):
    assert X.shape[2] == 1
    output = np.zeros((X.shape[0], X.shape[1], n_bins))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                output[i, j, :] = discretize_scalar(
                    X[i, j, k], n_bins, **kwargs)
    return output


def fdiff(X):
    assert X.shape[2] == 1
    output = np.zeros(X.shape)
    for i in range(X.shape[0]):
        output[i, :-1, 0] = np.diff(X[i, :, 0])
    return output


def power_and_fdiff(X):
    assert X.shape[2] == 1
    output = np.zeros((X.shape[0], X.shape[1], 2))
    for i in range(X.shape[0]):
        output[i, :, 0] = X[i, :, 0]
        output[i, :-1, 1] = np.diff(X[i, :, 0])
    return output


def get_meters_for_appliances(elec, appliances):
    meters = []
    for appliance_i, apps in enumerate(appliances):
        if not isinstance(apps, list):
            apps = [apps]
        for appliance in apps:
            try:
                meter = elec[appliance]
            except KeyError:
                pass
            else:
                meters.append(meter)
                break
        else:
            print("  No appliance matching", apps, "in building",
                  elec.building())
            continue
    return meters


def get_meter_for_appliance(elec, target_appliance):
    """
    Parameters
    ----------
    elec : nilmtk.MeterGroup
    target_appliance : list of strings or string

    Returns
    -------
    meter, target_app
    meter : nilmtk.ElecMeter or MeterGroup
    target_app : string
    """
    if isinstance(target_appliance, list):
        for app in target_appliance:
            try:
                meter = elec[app]
            except KeyError:
                continue
            else:
                target_app = app
                break
        else:
            raise KeyError()
    else:
        meter = elec[target_appliance]
        target_app = target_appliance

    return meter, target_app


def unstandardise(data, std, mean, maximum=None):
    unstandardised_data = (data * std) + mean
    if maximum is not None:
        unstandardised_data *= maximum
    return unstandardised_data


def _preprocess_activations(activations, max_power=None, sample_period=6,
                            clip_appliance_power=False):
    for i, activation in enumerate(activations):
        # tz_convert(None) is a workaround for Pandas bug #5172
        # (AmbiguousTimeError: Cannot infer dst time from Timestamp)
        tz = activation.index.tz.zone
        activation = activation.tz_convert('UTC')
        freq = "{:d}S".format(sample_period)
        activation = activation.resample(freq)
        activation.fillna(method='ffill', inplace=True)
        activation.fillna(method='bfill', inplace=True)
        if clip_appliance_power:
            activation = activation.clip(0, max_power)
        activation = activation.tz_convert(tz)
        activations[i] = activation.astype(np.float32)
    return activations


"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/neuralnilm/source.py /mnt/sshfs/imperial/workspace/python/neuralnilm/neuralnilm/"
End:
"""

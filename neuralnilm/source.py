from __future__ import print_function, division
from Queue import Queue, Empty
import threading
import numpy as np
from numpy.random import randint
import pandas as pd
from nilmtk import DataSet, TimeFrame
from datetime import timedelta
from sys import stdout
from collections import OrderedDict

class Source(object):
    def __init__(self, seq_length, n_seq_per_batch, n_inputs, n_outputs):
        super(Source, self).__init__()
        self.seq_length = seq_length
        self.n_seq_per_batch = n_seq_per_batch
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.queue = Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run)
        self._thread.start()    
        
    def run(self):
        """Puts training data into a Queue"""
        while not self._stop.is_set():
            self.queue.put(self._gen_data())
        self.empty_queue()
        self._thread = None
            
    def stop(self):
        self.empty_queue()
        self._stop.set()

    def empty_queue(self):
        while True:
            try:
                self.queue.get(block=False)
            except Empty:
                break
        
    def validation_data(self):
        return self._gen_data(validation=True)

    def _gen_data(self, validation=False):
        raise NotImplementedError()

    def input_shape(self):
        return (self.n_seq_per_batch, 
                self.seq_length, 
                self.n_inputs)

    def output_shape(self):
        return (self.n_seq_per_batch, 
                self.seq_length, 
                self.n_outputs)

    def _check_data(self, X, y):
        assert X.shape == self.input_shape()
        assert not any(np.isnan(X.flatten()))
        if y is not None:
            assert y.shape == self.output_shape()
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
        super(ToySource, self).__init__(
            seq_length=seq_length, 
            n_seq_per_batch=n_seq_per_batch,
            n_inputs=n_inputs, 
            n_outputs=1)
        self.powers = [10,40] if powers is None else powers
        self.on_durations = [3,10] if on_durations is None else on_durations
        self.all_hot = all_hot
        self.fdiff = fdiff

    def _gen_single_appliance(self, power, on_duration, 
                              min_off_duration=20, p=0.2):
        length = self.seq_length + 1 if self.fdiff else self.seq_length
        appliance_power = np.zeros(length)
        i = 0
        while i < length:
            if np.random.binomial(n=1, p=p):
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
                 max_appliance_powers, 
                 min_on_durations,
                 on_power_thresholds=None,
                 max_input_power=None,
                 window=(None, None), 
                 seq_length=1000,
                 train_buildings=None, 
                 validation_buildings=None,
                 output_one_appliance=True, 
                 sample_period=6,
                 boolean_targets=False,
                 subsample_target=1, 
                 input_padding=0,
                 min_off_duration=0,
                 skip_probability=0):
        """
        Parameters
        ----------
        filename : str
        appliances : list of strings
            The first one is the target appliance, if output_one_appliance is True.
            Use a list of lists for alternative names (e.g. ['fridge', 'fridge freezer'])
        building : int
        subsample_target : int
            If > 1 then subsample the targets.
        """
        super(RealApplianceSource, self).__init__(
            seq_length=seq_length, 
            n_seq_per_batch=5,
            n_inputs=1,
            n_outputs=1 if output_one_appliance else len(appliances)
        )
        self.dataset = DataSet(filename)
        self.appliances = appliances
        self.max_input_power = (np.sum(max_appliance_powers) 
                                if max_input_power is None else max_input_power)
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
        self.subsample_target = subsample_target
        self.input_padding = input_padding
        self.skip_probability = skip_probability
        self._tz = self.dataset.metadata['timezone']

        print("Loading training activations...")
        if on_power_thresholds is None:
            on_power_thresholds = [None] * len(self.appliances)
        self.train_activations = self._load_activations(
            train_buildings, min_on_durations, min_off_duration, on_power_thresholds)
        if train_buildings == validation_buildings:
            self.validation_activations = self.train_activations
        else:
            print("\nLoading validation activations...")
            self.validation_activations = self._load_activations(
                validation_buildings, min_on_durations, min_off_duration, on_power_thresholds)
        self.dataset.store.close()
        print("\nDone loading activations.")

    def get_labels(self):
        return self.train_activations.keys()

    def _load_activations(self, buildings, min_on_durations, min_off_duration, on_power_thresholds):
        activations = OrderedDict()
        for building_i in buildings:
            metergroup = self.dataset.buildings[building_i].elec
            for appliance_i, appliances in enumerate(self.appliances):
                if not isinstance(appliances, list):
                    appliances = [appliances]
                for appliance in appliances:
                    try:
                        meter = metergroup[appliance]
                    except KeyError:
                        pass
                    else:
                        break
                else:
                    print("  No appliance matching", appliances, "in building", building_i)
                    continue

                print("  Loading activations for", appliance, 
                      "from building", building_i, end="... ")
                stdout.flush()
                activation_series = meter.activation_series(
                    on_power_threshold=on_power_thresholds[appliance_i],
                    min_on_duration=min_on_durations[appliance_i], 
                    min_off_duration=min_off_duration)
                activations[appliances[0]] = self._preprocess_activations(
                    activation_series, self.max_appliance_powers[appliances[0]])
                print("Loaded", len(activation_series), "activations.")
        return activations

    def _preprocess_activations(self, activations, max_power):
        for i, activation in enumerate(activations):
            # tz_convert(None) is a workaround for Pandas bug #5172
            # (AmbiguousTimeError: Cannot infer dst time from Timestamp)
            activation = activation.tz_convert(None) 
            activation = activation.resample("{:d}S".format(self.sample_period))
            activation.fillna(method='ffill', inplace=True)
            activation.fillna(method='bfill', inplace=True)
            activation = activation.clip(0, max_power)
            activations[i] = activation
        return activations

    def _gen_single_example(self, validation=False):
        X = np.zeros(shape=(self.seq_length, self.n_inputs))
        y = np.zeros(shape=(self.seq_length, self.n_outputs))
        POWER_THRESHOLD = 5
        BORDER = 5
        activations = (self.validation_activations if validation 
                       else self.train_activations)

        appliances = []
        while not appliances:
            for appliance_i, appliance in enumerate(activations.keys()):
                if not np.random.binomial(n=1, p=self.skip_probability):
                    appliances.append((appliance_i, appliance))

        for appliance_i, appliance in appliances:
            n_activations = len(activations[appliance])
            if n_activations == 0:
                continue
            activation_i = randint(0, n_activations)
            activation = activations[appliance][activation_i]
            latest_start_i = (self.seq_length - len(activation)) - BORDER
            latest_start_i = max(latest_start_i, BORDER)
            start_i = randint(0, latest_start_i)
            end_i = start_i + len(activation)
            end_i = min(end_i, self.seq_length-1)
            target = activation.values[:end_i-start_i]
            X[start_i:end_i,0] += target 
            if appliance_i == 0 or not self.output_one_appliance:
                target = np.copy(target)
                if self.boolean_targets:
                    target[target <= POWER_THRESHOLD] = 0
                    target[target > POWER_THRESHOLD] = 1
                else:
                    max_appliance_power = self.max_appliance_powers[appliance]
                    target /= max_appliance_power
                y[start_i:end_i, appliance_i] = target
        np.clip(X, 0, self.max_input_power, out=X)
        if self.subsample_target > 1:
            shape = (int(self.seq_length / self.subsample_target), 
                     self.n_outputs)
            subsampled_y = np.empty(shape=shape)
            for output_i in range(self.n_outputs):
                subsampled_y[:,output_i] = np.mean(
                    y[:,output_i].reshape(-1, self.subsample_target), axis=-1)
            y = subsampled_y
        X /= self.max_input_power
        return X, y
    
    def input_shape(self):
        return (self.n_seq_per_batch, 
                self.seq_length + self.input_padding, 
                self.n_inputs)

    def output_shape(self):
        if self.seq_length % self.subsample_target:
            raise RuntimeError("subsample_target must exactly divide seq_length.")
        return (self.n_seq_per_batch, 
                int(self.seq_length / self.subsample_target),
                self.n_outputs)

    def inside_padding(self):
        start = self.input_padding // 2
        end = -int(np.ceil(self.input_padding / 2))
        if end == 0:
            end = None
        return start, end

    def _gen_data(self, validation=False):
        X = np.zeros(self.input_shape())
        y = np.zeros(self.output_shape())
        start, end = self.inside_padding()
        for i in range(self.n_seq_per_batch):
            X[i,start:end,:], y[i,:,:] = self._gen_single_example(validation)
        self._check_data(X, y)
        return X, y

class NILMTKSource(Source):
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
        self.min_power =  20
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
        y = self.metergroup[self.appliances[0]].power_series_all_data(**load_kwargs)
        if y is None or y.max() < self.min_power:
            return None, None

        # Load input (aggregate) data
        X = y + self.metergroup[self.appliances[1]].power_series_all_data(**load_kwargs)
        for appliance in self.appliances[2:]:
            X += self.metergroup[appliance].power_series_all_data(**load_kwargs)

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
        N_DAYS = 600 # there are more like 632 days in the dataset
        FIRST_DAY = pd.Timestamp("2013-04-12")
        seq_i = 0
        while seq_i < self.n_seq_per_batch:
            if validation:
                days = np.random.randint(low=N_DAYS, high=N_DAYS + self.n_seq_per_batch)
            else:
                days = np.random.randint(low=0, high=N_DAYS)
            start = FIRST_DAY + timedelta(days=days)
            X_one_seq, y_one_seq = self._get_data_for_single_day(start)

            if y_one_seq is not None:
                try:
                    X[seq_i,:,:] = X_one_seq.reshape(self.seq_length, 1)
                    y[seq_i,:,:] = y_one_seq.reshape(self.seq_length, 1)
                except ValueError as e:
                    print(e)
                    print("Skipping", start)
                else:
                    seq_i += 1
            else:
                print("Skipping", start)
        return X, y

    def _gen_data(self, *args, **kwargs):
        X = kwargs.pop('X', None)
        if X is None:
            X, y = self._gen_unquantized_data(*args, **kwargs)
        else:
            y = None
        X_quantized = np.empty(shape=self.input_shape())
        for i in range(self.n_seq_per_batch):
            X_quantized[i,:,0] = X[i,:,0] # time of day
            X_quantized[i,:,1:] = quantize(X[i,:,1], self.n_inputs)

        self._check_data(X_quantized, y)
        return X_quantized, y


def quantize(data, n_bins, all_hot=True, range=(-1, 1), length=None):
    midpoint = n_bins // 2
    if length is None:
        length = len(data)
    out = np.empty(shape=(length, n_bins))
    for i in np.arange(length):
        d = data[i]
        hist, _ = np.histogram(d, bins=n_bins, range=range)
        if all_hot:
            where = np.where(hist==1)[0][0]
            if where > midpoint:
                hist[midpoint:where] = 1
            elif where < midpoint:
                hist[where:midpoint] = 1
        out[i,:] = hist
    return (out * 2) - 1

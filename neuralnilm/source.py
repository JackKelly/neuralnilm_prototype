from __future__ import print_function, division
from Queue import Queue
import threading
import numpy as np
import pandas as pd
from nilmtk import DataSet, TimeFrame
from datetime import timedelta

class Source(threading.Thread):
    def __init__(self):
        super(Source, self).__init__()
        self.queue = Queue(maxsize=2)
        self._stop = threading.Event()
        
    def run(self):
        """Puts training data into a Queue"""
        self._stop.clear()
        while not self._stop.is_set():
            self.queue.put(self._gen_data())
            
    def stop(self):
        self.queue.get()
        self._stop.set()
        
    def validation_data(self):
        return self._gen_data(validation=True)

    def _gen_data(self, validation=False):
        raise NotImplementedError()

    def input_shape(self):
        raise NotImplementedError()
        
    def output_shape(self):
        raise NotImplementedError()

    def _check_data(self, X, y):
        assert X.shape == self.input_shape()
        if y is not None:
            assert y.shape == self.output_shape()


class NILMTKSource(Source):
    SEQ_LENGTH = 14400
    N_SEQ_PER_BATCH = 5    
    N_INPUTS = 1000
    N_OUTPUTS = 1
    FREQ = "6S"

    def __init__(self, filename, appliances, building=1):
        """
        Parameters
        ----------
        filename : str
        appliances : list of strings
            The first one is the target appliance
        building : int
        """
        super(NILMTKSource, self).__init__()
        self.dataset = DataSet(filename)
        self.appliances = appliances
        self._tz = self.dataset.metadata['timezone']
        self.metergroup = self.dataset.buildings[building].elec

    def input_shape(self):
        return (NILMTKSource.N_SEQ_PER_BATCH, 
                NILMTKSource.SEQ_LENGTH, 
                NILMTKSource.N_INPUTS)
        
    def output_shape(self):
        return (NILMTKSource.N_SEQ_PER_BATCH, 
                NILMTKSource.SEQ_LENGTH, 
                NILMTKSource.N_OUTPUTS)

    def _get_data_for_single_day(self, start):
        MAXIMUM = 200
        MINIMUM =  20
        start = pd.Timestamp(start).date()
        end = start + timedelta(days=1)
        timeframe = TimeFrame(start, end, tz=self._tz)
        load_kwargs = dict(sample_period=6, sections=[timeframe])

        # Load output (target) data
        y = self.metergroup[self.appliances[0]].power_series_all_data(**load_kwargs)
        if y is None or y.max() < MINIMUM:
            return None, None

        # Load input (aggregate) data
        X = y + self.metergroup[self.appliances[1]].power_series_all_data(**load_kwargs)
        for appliance in self.appliances[2:]:
            X += self.metergroup[appliance].power_series_all_data(**load_kwargs)

        index = pd.date_range(start, end, freq=NILMTKSource.FREQ, tz=self._tz)
        def preprocess(data):
            data = data.fillna(0)
            data = data.clip(upper=MAXIMUM)
            data[data < MINIMUM] = 0
            data = data.reindex(index, fill_value=0)
            data /= MAXIMUM
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
        X = np.empty(shape=(NILMTKSource.N_SEQ_PER_BATCH, NILMTKSource.SEQ_LENGTH, 1))
        y = np.empty(shape=self.output_shape())
        N_DAYS = 600 # there are more like 632 days in the dataset
        FIRST_DAY = pd.Timestamp("2013-04-12")
        seq_i = 0
        while seq_i < NILMTKSource.N_SEQ_PER_BATCH:
            if validation:
                days = np.random.randint(low=N_DAYS, high=N_DAYS + NILMTKSource.N_SEQ_PER_BATCH)
            else:
                days = np.random.randint(low=0, high=N_DAYS)
            start = FIRST_DAY + timedelta(days=days)
            X_one_seq, y_one_seq = self._get_data_for_single_day(start)

            if y_one_seq is not None:
                try:
                    X[seq_i,:,:] = X_one_seq.reshape(NILMTKSource.SEQ_LENGTH, 1)
                    y[seq_i,:,:] = y_one_seq.reshape(NILMTKSource.SEQ_LENGTH, 1)
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
        for i in range(NILMTKSource.N_SEQ_PER_BATCH):
            X_quantized[i,:,0] = X[i,:,0] # time of day
            X_quantized[i,:,1:] = quantize(X[i,:,1],
                                           NILMTKSource.N_INPUTS)

        self._check_data(X_quantized, y)
        return X_quantized, y



def quantize(data, n_bins, all_hot=True):
    midpoint = n_bins // 2
    out = np.empty(shape=(len(data), n_bins))
    for i, d in enumerate(data):
        hist, _ = np.histogram(d, bins=n_bins, range=(-1, 1))
        if all_hot:
            where = np.where(hist==1)[0][0]
            if where > midpoint:
                hist[midpoint:where] = 1
            elif where < midpoint:
                hist[where:midpoint] = 1
        out[i,:] = hist
    return (out * 2) - 1

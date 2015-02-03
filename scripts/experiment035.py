from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta
from numpy.random import rand
from time import time
from nilmtk import TimeFrame, DataSet, MeterGroup

"""
INPUT: quantized mains fdiff, all-hot, first bit is sign.
OUTPUT: appliance fdiff

Code taken from Lasagne and nolearn!

rsync command: 
rsync -uvz --progress /home/jack/workspace/python/neuralnilm/scripts/*.py /mnt/sshfs/imperial/workspace/python/neuralnilm/scripts
"""

SEQ_LENGTH = 14400
N_HIDDEN = 5
N_SEQ_PER_BATCH = 5 # Number of sequences per batch
LEARNING_RATE = 1e-1 # SGD learning rate
N_ITERATIONS = 1000  # Number of training iterations
N_INPUT_FEATURES = 1001 # 1 input for time of day + many-hot
N_OUTPUTS = 1
TZ = "Europe/London"
FILENAME = '/data/dk3810/ukdale.h5' # '/data/mine/vadeec/merged/ukdale.h5'

input_shape  = (N_SEQ_PER_BATCH, SEQ_LENGTH, N_INPUT_FEATURES)
output_shape = (N_SEQ_PER_BATCH, SEQ_LENGTH, N_OUTPUTS)


############### GENERATE DATA ##############################

def quantize(data):
    N_BINS = N_INPUT_FEATURES - 1
    MID = N_BINS // 2
    out = np.empty(shape=(len(data), N_BINS))
    for i, d in enumerate(data):
        hist, _ = np.histogram(d, bins=N_BINS, range=(-1, 1))
        where = np.where(hist==1)[0][0]
        if where > MID:
            hist[MID:where] = 1
        elif where < MID:
            hist[where:MID] = 1
        out[i,:] = hist
    return (out * 2) - 1


def get_data_for_single_day(metergroup, target_appliance, start):
    MAXIMUM = 200
    MINIMUM =  20
    start = pd.Timestamp(start).date()
    end = start + timedelta(days=1)
    timeframe = TimeFrame(start, end, tz=TZ)
    load_kwargs = dict(sample_period=6, sections=[timeframe])
    y = metergroup[target_appliance].power_series_all_data(**load_kwargs)
    if y is None or y.max() < MINIMUM:
        return None, None
    #X = metergroup.power_series_all_data(**load_kwargs)
    X = y + metergroup['boiler'].power_series_all_data(**load_kwargs)
    index = pd.date_range(start, end, freq="6S", tz=TZ)
    def get_diff(data):
        data = data.fillna(0)
        data = data.clip(upper=MAXIMUM)
        data[data < MINIMUM] = 0
        # data -= data.min()
        data = data.reindex(index, fill_value=0)
        data /= MAXIMUM
        return data.diff().dropna()
    
    def index_as_minus_one_to_plus_one(data):
        data = get_diff(data)
        index = data.index.astype(np.int64)
        index -= np.min(index)
        index = index.astype(np.float32)
        index /= np.max(index)
        return np.vstack([index, data.values]).transpose()
    
    return index_as_minus_one_to_plus_one(X), get_diff(y)


def gen_unquantized_data(metergroup, validation=False):
    '''Generate a simple energy disaggregation data.
    :returns:
        - X : np.ndarray, shape=(n_batch, length, 1)
            Input sequence
        - y : np.ndarray, shape=(n_batch, length, 1)
            Target sequence, appliance 1
    '''
    X = np.empty(shape=(N_SEQ_PER_BATCH, SEQ_LENGTH, 2))
    y = np.empty(output_shape)
    N_DAYS = 600 # there are more like 632 days in the dataset
    FIRST_DAY = pd.Timestamp("2013-04-12")
    seq_i = 0
    while seq_i < N_SEQ_PER_BATCH:
        if validation:
            days = np.random.randint(low=N_DAYS, high=N_DAYS + N_SEQ_PER_BATCH)
        else:
            days = np.random.randint(low=0, high=N_DAYS)
        start = FIRST_DAY + timedelta(days=days)
        X_one_seq, y_one_seq = get_data_for_single_day(metergroup, 
                                                       'television', start)
        if y_one_seq is not None:
            try:
                X[seq_i,:,:] = X_one_seq
                y[seq_i,:,:] = y_one_seq.reshape(SEQ_LENGTH, 1)
            except ValueError as e:
                print(e)
                print("Skipping", start)
            else:
                seq_i += 1
        else:
            print("Skipping", start)
    return X, y


def gen_data(X=None, *args, **kwargs):
    if X is None:
        X, y = gen_unquantized_data(*args, **kwargs)
    else:
        y = None
    X_quantized = np.empty(shape=input_shape)
    for i in range(N_SEQ_PER_BATCH):
        X_quantized[i,:,0] = X[i,:,0] # time of day
        X_quantized[i,:,1:] = quantize(X[i,:,1])
    return X_quantized, y


class ansi:
    # from dnouri/nolearn/nolearn/lasagne.py
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'

######################## Neural network class ########################
class Net(object):
    # Much of this code is adapted from craffel/nntools/examples/lstm.py

    def __init__(self):
        print("Initialising network...")
        import theano
        import theano.tensor as T
        import lasagne
        from lasagne.layers import (InputLayer, LSTMLayer, ReshapeLayer, 
                                    ConcatLayer, DenseLayer)
        theano.config.compute_test_value = 'raise'

        # Construct LSTM RNN: One LSTM layer and one dense output layer
        l_in = InputLayer(shape=input_shape)

        # setup fwd and bck LSTM layer.
        l_fwd = LSTMLayer(
            l_in, N_HIDDEN, backwards=False, learn_init=True, peepholes=True)
        l_bck = LSTMLayer(
            l_in, N_HIDDEN, backwards=True, learn_init=True, peepholes=True)

        # concatenate forward and backward LSTM layers
        concat_shape = (N_SEQ_PER_BATCH * SEQ_LENGTH, N_HIDDEN)
        l_fwd_reshape = ReshapeLayer(l_fwd, concat_shape)
        l_bck_reshape = ReshapeLayer(l_bck, concat_shape)
        l_concat = ConcatLayer([l_fwd_reshape, l_bck_reshape], axis=1)

        l_recurrent_out = DenseLayer(l_concat, num_units=N_OUTPUTS, 
                                     nonlinearity=None)
        l_out = ReshapeLayer(l_recurrent_out, output_shape)

        input = T.tensor3('input')
        target_output = T.tensor3('target_output')

        # add test values
        input.tag.test_value = rand(
            *input_shape).astype(theano.config.floatX)
        target_output.tag.test_value = rand(
            *output_shape).astype(theano.config.floatX)

        print("Compiling Theano functions...")
        # Cost = mean squared error
        cost = T.mean((l_out.get_output(input) - target_output)**2)

        # Use NAG for training
        all_params = lasagne.layers.get_all_params(l_out)
        updates = lasagne.updates.nesterov_momentum(cost, all_params, LEARNING_RATE)

        # Theano functions for training, getting output, and computing cost
        self.train = theano.function(
            [input, target_output],
            cost, updates=updates, on_unused_input='warn',
            allow_input_downcast=True)

        self.y_pred = theano.function(
            [input], l_out.get_output(input), on_unused_input='warn',
            allow_input_downcast=True)

        self.compute_cost = theano.function(
            [input, target_output], cost, on_unused_input='warn',
            allow_input_downcast=True)

        print("Done initialising network.")

    def training_loop(self):
        # column 0 = training cost
        # column 1 = validation cost
        self.costs = np.zeros(shape=(N_ITERATIONS, 2))
        self.costs[:,:] = np.nan

        from nilmtk import DataSet
        dataset = DataSet(FILENAME)
        elec = dataset.buildings[1].elec
        self.selected = elec
        # APPLIANCES = ['kettle', 'television']
        # selected_meters = [elec[appliance] for appliance in APPLIANCES]
        # self.selected = MeterGroup(selected_meters)

        # Generate a "validation" sequence whose cost we will compute
        X_val, y_val = gen_data(metergroup=self.selected, validation=True)
        assert X_val.shape == input_shape
        assert y_val.shape == output_shape

        # Adapted from dnouri/nolearn/nolearn/lasagne.py
        print("""
 Epoch  |  Train cost  |  Valid cost  |  Train / Val  | Dur per epoch
--------|--------------|--------------|---------------|---------------\
""")
        # Training loop
        for n in range(N_ITERATIONS):
            t0 = time() # for calculating training duration
            X, y = gen_data(metergroup=self.selected)
            train_cost = self.train(X, y).flatten()[0]
            validation_cost = self.compute_cost(X_val, y_val).flatten()[0]
            self.costs[n] = train_cost, validation_cost

            # Print progress
            duration = time() - t0
            is_best_train = train_cost == np.nanmin(self.costs[:,0])
            is_best_valid = validation_cost == np.nanmin(self.costs[:,1])
            print("  {:>5} |  {}{:>10.6f}{}  |  {}{:>10.6f}{}  |"
                  "  {:>11.6f}  |  {:>3.1f}s".format(
                      n,
                      ansi.BLUE if is_best_train else "",
                      train_cost,
                      ansi.ENDC if is_best_train else "",
                      ansi.GREEN if is_best_valid else "",
                      validation_cost,
                      ansi.ENDC if is_best_valid else "",
                      train_cost / validation_cost,
                      duration
            ))

    def plot_costs(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.costs[:,0], label='training')
        ax.plot(self.costs[:,1], label='validation')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.legend()
        plt.show()
        return ax

    def plot_estimates(self, axes=None):
        if axes is None:
            _, axes = plt.subplots(2, sharex=True)
        X, y = gen_unquantized_data(self.selected, validation=True)
        y_predictions = self.y_pred(gen_data(X=X)[0])
        axes[0].set_title('Appliance forward difference')
        axes[0].plot(y_predictions[0,:,0], label='Estimates')
        axes[0].plot(y[0,:,0], label='Appliance ground truth')
        axes[0].legend()
        axes[1].set_title('Aggregate')
        axes[1].plot(X[0,:,1], label='Fdiff')
        axes[1].plot(np.cumsum(X[0,:,1]), label='Cumsum')
        axes[1].legend()
        plt.show()

if __name__ == "__main__":
    net = Net()
    net.training_loop()
    net.plot_costs()
    net.plot_estimates()

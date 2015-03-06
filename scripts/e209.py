from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from neuralnilm import Net, RealApplianceSource, BLSTMLayer, DimshuffleLayer
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.objectives import crossentropy, mse
from lasagne.init import Uniform, Normal
from lasagne.layers import LSTMLayer, DenseLayer, Conv1DLayer, ReshapeLayer, FeaturePoolLayer
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
Normal(1) for BLSTM

e110
* Back to Uniform(5) for BLSTM
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
* Start with single BLSTM layer

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
conv1D layer has Uniform(1), as does 2nd BLSTM layer

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
* Same as e149 but without peepholes and using BLSTM not BBLSTM

e151
* Max pooling

171
lower learning rate

172
even lower learning rate

173
slightly higher learning rate!

175
same as 174 but with skip prob = 0, and LSTM not BLSTM, and only 4000 epochs

176
new cost function

177
another new cost func (this one avoids NaNs)
skip prob 0.7
10x higher learning rate

178
refactored cost func (functionally equiv to 177)
0.1x learning rate

e180
* mse

e181
* back to scaled cost
* different architecture:
  - convd1 at input (2x)
  - then 3 LSTM layers, each with a 2x conv in between
  - no diff input

e189
* divide dominant appliance power
* mse
"""


# def scaled_cost(x, t):
#     raw_cost = (x - t) ** 2
#     energy_per_seq = t.sum(axis=1)
#     energy_per_batch = energy_per_seq.sum(axis=1)
#     energy_per_batch = energy_per_batch.reshape((-1, 1))
#     normaliser = energy_per_seq / energy_per_batch
#     cost = raw_cost.mean(axis=1) * (1 - normaliser)
#     return cost.mean()

from theano.ifelse import ifelse
import theano.tensor as T

THRESHOLD = 0
def scaled_cost(x, t):
    sq_error = (x - t) ** 2
    def mask_and_mean_sq_error(mask):
        masked_sq_error = sq_error[mask.nonzero()]
        mean = masked_sq_error.mean()
        mean = ifelse(T.isnan(mean), 0.0, mean)
        return mean
    above_thresh_mean = mask_and_mean_sq_error(t > THRESHOLD)
    below_thresh_mean = mask_and_mean_sq_error(t <= THRESHOLD)
    return (above_thresh_mean + below_thresh_mean) / 2.0

def exp_a(name):
    global source
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            ['fridge freezer', 'fridge', 'freezer'], 
            'hair straighteners', 
            'television'
            # 'dish washer',
            # ['washer dryer', 'washing machine']
        ],
        max_appliance_powers=[2500] * 5,
        on_power_thresholds=[5] * 5,
        max_input_power=2500,
        min_on_durations=[60, 60, 60, 1800, 1800],
        min_off_durations=[12, 12, 12, 1800, 600],
        window=("2013-06-01", "2014-07-01"),
        seq_length=1520,
        output_one_appliance=False,
        boolean_targets=False,
        train_buildings=[1],
        validation_buildings=[1], 
        skip_probability=0.7,
        n_seq_per_batch=25,
        input_padding=4,
        include_diff=False,
        clip_appliance_power=False
    )

    net = Net(
        experiment_name=name,
        source=source,
        save_plot_interval=1000,
        loss_function=scaled_cost,
        updates=partial(nesterov_momentum, learning_rate=0.1, clip_range=(-1, 1)),
        layers_config=[
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)  # (batch, features, time)
            },
            {
                'type': Conv1DLayer, # convolve over the time axis
                'num_filters': 50,
                'filter_length': 5,
                'stride': 1,
                'nonlinearity': sigmoid,
                'W': Uniform(10)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1) # back to (batch, time, features)
            },
            {
                'type': DenseLayer,
                'num_units': 50,
                'nonlinearity': sigmoid,
                'W': Uniform(1)
            },
            {
                'type': DenseLayer,
                'num_units': source.n_outputs,
                'nonlinearity': None
#                'W': Uniform()
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
    for experiment in list('a'):
        full_exp_name = NAME + experiment
        path = os.path.join(PATH, full_exp_name)
        try:
            net = init_experiment(experiment)
            run_experiment(net, path, epochs=None)
        except KeyboardInterrupt:
            break
        except TrainingError as exception:
            print("EXCEPTION:", exception)
        except Exception as exception:
            raise
            print("EXCEPTION:", exception)
            import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()

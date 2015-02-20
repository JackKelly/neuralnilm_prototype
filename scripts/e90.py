from __future__ import print_function, division
from neuralnilm import Net, RealApplianceSource, BLSTMLayer, SubsampleLayer, DimshuffleLayer
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.objectives import crossentropy
from lasagne.init import Uniform, Normal
from lasagne.layers import LSTMLayer, DenseLayer, Conv1DLayer, ReshapeLayer
from lasagne.updates import adagrad, nesterov_momentum
from functools import partial
import os

"""
Setup:
* in_to_cell init weights are now Normal(1.0)
* output all appliances
* fix bug in RealApplianceSource
* use cross-entropy
* smaller network
* power targets
* trying without first two sigmoid layers.
* updated to craffel/nntools commit 097aca480d60fdfada513c20070f8132d71a26b0 
  which fixes LSTM bug.
  https://github.com/craffel/nntools/commit/097aca480d60fdfada513c20070f8132d71a26b0
* Subsampling *bidirectional* LSTM
* Output every sequence in the batch
* Change W_in_to_cell from Normal(1.0) to Uniform(5)
* put back the two sigmoid layers
* use Conv1D to create a hierarchical subsampling LSTM
* Using LSTM (not BLSTM) to speed up training while testing
* Use dimshuffle not reshape
* 2 dense layers back
* back to default init
* conv between LSTMs.
* More data
* BLSTM
* Try just using a 1D convnet on input
* add second Convnet layer (not sure this is correct thing to do?)
* third conv layer
* large inits
* back to 2 conv layers

e70
* Based on e65
* Using sigmoid instead of rectify in Conv1D layers

e71
* Larger layers
* More data

e72
* At a third conv layer

e73
* Add a dense layer after 3 conv layers

e74
* Removed dense layer after 3 conv layers (because it failed to learn anything)
* Trying standard inits for weights and biases throughout network.

e75
* Putting back large init for first layer

e76
* Removed 3rd conv layer

e77
* Try init Uniform(1)

e78
* Back to large inits for first layers
* Trying 3rd conv layer, also with large init

e79
* Trying to merge 1D conv on bottom layer with hierarchical subsampling 
  from e59a.
* Replace first LSTM with BLSTM
* Add second BLSTM layer
* Add conv1d between BLSTM layers.

e80
* Remove third 1d conv layer

e81
* Change num_filters in conv layer between BLSTMs from 20 to 80

e83
* Same net as e81
* Using different appliances, longer seq, and validation on house 5
  (unseen during training!)  Might be unfair because, for example,
  house 1 doesn't use its washer dryer in drying mode ever but it
  house 5 does.
* Using a seq_length of 4000 resulted in NaNs very quickly.
  Dropping to 2000 resulted in NaNs after 100 epochs
  1000 resulted in Nans after 4500 epochs

e83b
* Back to seq_length of 2000, modified net.py so it called IPDB
  if train error is NaN or > 1

e83c
* Changed inits to standard values to try to stop NaN train costs
Results: I let it run for a little over 100 epochs.  No Nans.  But 
wasn't learning anything very sane.

e83d
* Uniform(1)

e83e
* Try adagrad

e84
* Trying to find minimial example which gets NaNs
RESULT: Blows up after 19 epochs! Yay!

e85
* Try adagrad
RESULTS at different learning rates:
* 1 goes to NaN within 2 epochs ;(
* 0.01 went to NaN in 13 epochs
* 0.0001 doesn't go to NaN after 1000 epochs and may even be starting to learning something!
* 0.001 (e)85b  doesn't go to NaN about >140 epochs

e86
* Trying larger network again (with adagrad with learning rate 0.001)
* Doesn't go to NaN (after >770 epochs) and learns something very vaguely useful
  but not great.  At all.  Doesn't discriminate between appliances.

e87
* Network like e82.  Just LSTM -> Conv -> LSTM -> Dense.
* More data

e88
* Train and validate just on house 1

e89
* try nesterov_momentum but with learning rate 0.01

Results

"""
NAME = "e90"
PATH = "/homes/dk3810/workspace/python/neuralnilm/figures"

def main():
    for experiment in ['a']:
        try:
            run_experiment(experiment)
        except Exception as e:
            print("EXCEPTION:", e)

def run_experiment(experiment):
    net = eval('exp_{:s}(NAME)'.format(experiment))
    net.print_net()
    net.compile()
    path = os.path.join(PATH, NAME+experiment)
    os.mkdir(path)
    os.chdir(path)
    net.fit(5000)

"""
Things to try:
* try re-creating a previously successful approach like e82 or e59,
  completely re-create it
* try collection of experts (change Source so it accepts a 'target' parameter
  to specify the single target).
"""

def exp_a(name):
    source = RealApplianceSource(
        filename='/data/dk3810/ukdale.h5',
        appliances=[
            ['fridge freezer', 'fridge', 'freezer'], 
            'kettle', 
            'dish washer',
            ['washer dryer', 'washing machine'],
            'microwave'
        ],
        max_appliance_powers=[300, 3000, 2500, 2400, 2000],
        on_power_thresholds=[80, 200, 20, 600, 100],
        min_on_durations=[60, 10, 300, 300, 10],
        window=("2013-05-22", "2015-01-01"),
        seq_length=2000,
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
        save_plot_interval=50,
        loss_function=crossentropy,
        updates=partial(nesterov_momentum, learning_rate=0.001),
        layers_config=[
            {
                'type': LSTMLayer, # TODO change to BLSTM
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
                'type': LSTMLayer,
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

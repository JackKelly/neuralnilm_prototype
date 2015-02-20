from __future__ import print_function, division
from functools import partial
from neuralnilm import Net, RealApplianceSource, BLSTMLayer, SubsampleLayer, DimshuffleLayer
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.objectives import crossentropy
from lasagne.init import Uniform, Normal
from lasagne.layers import LSTMLayer, DenseLayer, Conv1DLayer, ReshapeLayer
from lasagne.updates import adagrad


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

TODO: 
* Try large inits for conv layer but Uniform(1) or standard for BLSTM

Results

"""

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
    input_padding=4,
    train_buildings=[1,2],
    validation_buildings=[5]
)

net = Net(
    experiment_name="e83e",
    source=source,
    save_plot_interval=50,
    loss_function=crossentropy,
    updates=partial(adagrad, learning_rate=0.001),
    layers_config=[
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 50,
            'filter_length': 3,
            'stride': 1,
            'nonlinearity': sigmoid,
            'W': Uniform(1),
            'b': Uniform(1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 50,
            'filter_length': 3,
            'stride': 1,
            'nonlinearity': sigmoid,
            'W': Uniform(1),
            'b': Uniform(1)
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': BLSTMLayer,
            'num_units': 50,
            'W_in_to_cell': Uniform(1)
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
            'type': BLSTMLayer,
            'num_units': 80,
            'W_in_to_cell': Uniform(1)
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': sigmoid
        }
    ]
)

net.print_net()
net.compile()
net.fit()

from __future__ import print_function, division
from neuralnilm import Net, RealApplianceSource, BLSTMLayer, SubsampleLayer, DimshuffleLayer
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.objectives import crossentropy
from lasagne.init import Uniform, Normal
from lasagne.layers import LSTMLayer, DenseLayer, Conv1DLayer, ReshapeLayer


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

Results

"""

source = RealApplianceSource(
    '/data/dk3810/ukdale.h5', 
    ['fridge freezer', 'hair straighteners', 'television'],
    max_input_power=1000, max_appliance_powers=[300, 500, 200],
    window=("2013-06-01", "2014-07-01"),
    output_one_appliance=False,
    boolean_targets=False,
    min_on_duration=60,
    input_padding=8,
    subsample_target=5
)

net = Net(
    experiment_name="e79",
    source=source,
    learning_rate=1e-1,
    save_plot_interval=250,
    loss_function=crossentropy,
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
            'W': Uniform(25),
            'b': Uniform(25)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 50,
            'filter_length': 3,
            'stride': 1,
            'nonlinearity': sigmoid,
            'W': Uniform(10),
            'b': Uniform(10)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 50,
            'filter_length': 5,
            'stride': 1,
            'nonlinearity': sigmoid,
            'W': Uniform(10),
            'b': Uniform(10)
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': BLSTMLayer,
            'num_units': 50,
            'W_in_to_cell': Uniform(5)
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 20,
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
            'W_in_to_cell': Uniform(5)
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


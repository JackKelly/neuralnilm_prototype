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

e69
* based on e59a (excellent performer)
* trying ReLU dense layer with standard inits
* using LSTM not BLSTM to speed up training

e69b
* Changed second denselayer to Uniform(10) as it was in e59a.

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
    subsample_target=5
)

net = Net(
    experiment_name="e69b",
    source=source,
    learning_rate=1e-1,
    save_plot_interval=50,
    loss_function=crossentropy,
    layers_config=[
        {
            'type': DenseLayer,
            'num_units': 50,
            'nonlinearity': rectify,
            'W': Uniform(25),
            'b': Uniform(25)
        },
        {
            'type': DenseLayer,
            'num_units': 50,
            'nonlinearity': rectify,
            'W': Uniform(10),
            'b': Uniform(10)
        },
        {
            'type': LSTMLayer,
            'num_units': 40,
            'W_in_to_cell': Uniform(5)
        },
        {
            'type': DimshuffleLayer,
            'pattern': (0, 2, 1)
        },
        {
            'type': Conv1DLayer,
            'num_filters': 20, # NEEDS INCREASING!
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

net.print_net()
net.compile()
net.fit()


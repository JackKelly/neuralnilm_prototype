from __future__ import print_function, division
from neuralnilm import Net, RealApplianceSource, BLSTMLayer, SubsampleLayer, DimshuffleLayer
from neuralnilm.net import QuantizeLayer
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
* 2 dense layers first.
* rectify for dense layers

Changes
* Quantize input

Results

"""

source = RealApplianceSource(
    '/data/dk3810/ukdale.h5', 
    ['fridge freezer', 'hair straighteners', 'television'],
    max_input_power=1000, max_appliance_powers=[300, 500, 200],
    window=("2013-06-01", "2013-07-01"),
    output_one_appliance=False,
    boolean_targets=False,
    min_on_duration=60,
    input_padding=4
)

net = Net(
    experiment_name="e68",
    source=source,
    learning_rate=1e-1,
    save_plot_interval=50,
    loss_function=crossentropy,
    layers_config=[
        {
            'type': QuantizeLayer,
            'n_bins': 50
        },
        {
            'type': DenseLayer,
            'num_units': 50,
            'nonlinearity': sigmoid
        },
        {
            'type': LSTMLayer,
            'num_units': 50,
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


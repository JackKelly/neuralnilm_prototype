from __future__ import print_function, division
from neuralnilm import Net, RealApplianceSource, BLSTMLayer
from lasagne.nonlinearities import sigmoid
from lasagne.objectives import crossentropy
from lasagne.init import Uniform, Normal
from lasagne.layers import LSTMLayer, DenseLayer


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

Changes:
* Dense layer, BLSTM, Dense, BLSTM, Dense, BLSTM, Output

"""

source = RealApplianceSource(
    '/data/dk3810/ukdale.h5', 
    ['fridge freezer', 'hair straighteners', 'television'],
    max_input_power=1000, max_appliance_powers=[300, 500, 200],
    window=("2013-06-01", "2014-07-01"),
    output_one_appliance=False,
    boolean_targets=False,
    min_on_duration=60
)

net = Net(
    experiment_name="e44a",
    source=source,
    learning_rate=1e-1,
    save_plot_interval=50,
    loss_function=crossentropy,
    layers=[
        {
            'type': DenseLayer, 
            'num_units': 50, 
            'nonlinearity': sigmoid,
            'b': Uniform(25), 
            'W': Uniform(25)
        },
        {
            'type': BLSTMLayer,
            'num_units': 50,
            'W_in_to_cell': Normal(1.0)
        },
        {
            'type': DenseLayer,
            'num_units': 50,
            'nonlinearity': sigmoid,
            'b': Uniform(1),
            'W': Uniform(1)
        },
        {
            'type': BLSTMLayer,
            'num_units': 50,
            'W_in_to_cell': Normal(1.0)
        },
        {
            'type': DenseLayer,
            'num_units': 50,
            'nonlinearity': sigmoid,
            'b': Uniform(1),
            'W': Uniform(1)
        },
        {
            'type': BLSTMLayer,
            'num_units': 50,
            'W_in_to_cell': Normal(1.0)
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': sigmoid,
            'b': Uniform(1),
            'W': Uniform(1)
        }
    ]
)

net.fit()


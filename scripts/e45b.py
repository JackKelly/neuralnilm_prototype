from __future__ import print_function, division
from neuralnilm import Net, RealApplianceSource, BLSTMLayer, SubsampleLayer
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
* Subsampling single-directional LSTM

"""

source = RealApplianceSource(
    '/data/dk3810/ukdale.h5', 
    ['fridge freezer', 'hair straighteners', 'television'],
    max_input_power=1000, max_appliance_powers=[300, 500, 200],
    window=("2013-06-01", "2014-07-01"),
    output_one_appliance=False,
    boolean_targets=False,
    min_on_duration=60,
    subsample_target=5*5
)

net = Net(
    experiment_name="e45b",
    source=source,
    learning_rate=1e-1,
    save_plot_interval=50,
    loss_function=crossentropy,
    layers=[
        {
            'type': LSTMLayer,
            'num_units': 20,
            'W_in_to_cell': Normal(1.0)
        },
        {
            'type': SubsampleLayer,
            'stride': 5
        },
        {
            'type': LSTMLayer,
            'num_units': 40,
            'W_in_to_cell': Normal(1.0)
        },
        {
            'type': SubsampleLayer,
            'stride': 5
        },
        {
            'type': LSTMLayer,
            'num_units': 80,
            'W_in_to_cell': Normal(1.0)
        },
        {
            'type': DenseLayer,
            'num_units': source.n_outputs,
            'nonlinearity': sigmoid
        }
    ]
)

net.fit()


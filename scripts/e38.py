from __future__ import print_function, division
from neuralnilm import Net, RealApplianceSource
from lasagne.nonlinearities import sigmoid

"""
Changes:
* in_to_cell init weights are now Normal(1.0)
* output all appliances
"""

source = RealApplianceSource(
    '/data/dk3810/ukdale.h5', 
    ['fridge freezer', 'hair straighteners', 'television'],
    max_input_power=1000, max_appliance_powers=[300, 500, 200],
    window=("2013-06-01", "2014-06-01"),
    output_one_appliance=False
#    sample_period=15, seq_length=400
)

net = Net(
    experiment_name="e38b",
    source=source,
    n_cells_per_hidden_layer=[50,50,50],
    output_nonlinearity=sigmoid,
    learning_rate=1e-1,
    n_dense_cells_per_layer=50,
    # validation_interval=2, 
    save_plot_interval=250
)

net.fit()

#net.plot_costs()
#net.plot_estimates()

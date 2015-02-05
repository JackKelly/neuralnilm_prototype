from __future__ import print_function, division
from neuralnilm import Net, RealApplianceSource
from lasagne.nonlinearities import sigmoid

source = RealApplianceSource(
    '/data/dk3810/ukdale.h5', 
    ['fridge freezer', 'hair straighteners', 'television'],
    max_input_power=1000, max_output_power=300,
    window=("2013-06-01", "2014-06-01")
)

net = Net(
    source=source,
    n_cells_per_hidden_layer=[50,50,50],
    output_nonlinearity=sigmoid,
    learning_rate=1e-1,
    n_dense_cells_per_layer=50
)

net.fit(n_iterations=2000)
net.plot_costs()
net.plot_estimates()

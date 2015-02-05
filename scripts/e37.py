from __future__ import print_function, division
from neuralnilm import Net, RealApplianceSource
from lasagne.nonlinearities import sigmoid

source = RealApplianceSource(
    '/data/mine/vadeec/merged/ukdale.h5', 
    ['fridge freezer', 'hair straighteners', 'television'],
    max_input_power=800, max_output_power=150
)

net = Net(
    source=source,
    n_cells_per_hidden_layer=[10],
    output_nonlinearity=sigmoid,
    learning_rate=1e-1
)

net.fit(n_iterations=1000)
net.plot_costs()
net.plot_estimates()

from __future__ import print_function, division
from neuralnilm import Net, ToySource
from lasagne.nonlinearities import sigmoid

source = ToySource(
    seq_length=100, 
    n_seq_per_batch=30
)

net = Net(
    source=source,
    n_cells_per_hidden_layer=[5],
    output_nonlinearity=sigmoid
)

net.fit(n_iterations=10)
net.plot_costs()
net.plot_estimates()

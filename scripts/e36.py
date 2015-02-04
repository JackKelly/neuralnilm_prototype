from __future__ import print_function, division
from neuralnilm import Net, ToySource
from lasagne.nonlinearities import sigmoid

source = ToySource(
    seq_length=300,
    n_seq_per_batch=30
)

net = Net(
    source=source,
    n_cells_per_hidden_layer=[10, 10, 10],
    output_nonlinearity=sigmoid,
    learning_rate=1e-2
)

net.fit(n_iterations=10000)
net.plot_costs()
net.plot_estimates()

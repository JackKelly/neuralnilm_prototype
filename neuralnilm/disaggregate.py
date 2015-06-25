from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

from neuralnilm.source import standardise
from neuralnilm.plot import plot_rectangles


def disaggregate(mains, net):
    """
    Parameters
    ----------
    mains : 1D np.ndarray
        Watts.
    net : neuralnilm.net.Net

    Returns
    -------
    appliance_estimates : 1D np.ndarray
    """
    n_seq_per_batch, seq_length = net.input_shape[:2]
    n_samples_per_batch = seq_length * n_seq_per_batch
    n_mains_samples = len(mains)
    n_batches = np.ceil(n_mains_samples / n_samples_per_batch).astype(int)
    n_output_samples = n_batches * n_samples_per_batch
    if n_mains_samples < n_output_samples:
        n_zeros_to_append = n_output_samples - n_mains_samples
        mains = np.pad(mains, (0, n_zeros_to_append), mode='constant')
    appliance_estimates = np.zeros(n_output_samples, dtype=np.float32)
    input_stats = net.source.input_stats
    mains = standardise(mains / net.source.max_input_power, how='std=1',
                        std=input_stats['std'], mean=input_stats['mean'])

    for batch_i in xrange(n_batches):
        start = batch_i * n_samples_per_batch
        end = start + n_samples_per_batch
        flat_batch = mains[start:end]
        batch = flat_batch.reshape((n_seq_per_batch, seq_length, 1))
        output = net.y_pred(batch)
        appliance_estimates[start:end] = output.flatten()

    return appliance_estimates


def disaggregate_start_stop_end(mains, net, stride=1, ax=None):
    """
    Parameters
    ----------
    mains : 1D np.ndarray
        Must already be standardised according to `net.source.input_stats`.
    net : neuralnilm.net.Net
    stride : int or None, optional
        if None then stide = seq_length
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    assert mains.ndim == 1
    n_seq_per_batch, seq_length = net.input_shape[:2]
    if stride is None:
        stride = seq_length
    if ax is None:
        ax = plt.gca()

    # Pad mains with zeros at both ends so we can slide
    # over the start and end of the mains data.
    pad_width = (seq_length, seq_length)
    mains_padded = np.pad(mains, pad_width, mode='constant')
    n_mains_samples = len(mains_padded)

    # Divide mains data into batches
    n_batches = (n_mains_samples / stride) / n_seq_per_batch
    n_batches = np.ceil(n_batches).astype(int)
    for batch_i in xrange(n_batches):
        net_input = np.zeros(net.input_shape, dtype=np.float32)
        batch_start = batch_i * n_seq_per_batch * stride
        for seq_i in xrange(n_seq_per_batch):
            start = batch_start + (seq_i * stride)
            end = start + seq_length
            seq = mains_padded[start:end]
            net_input[seq_i, :len(seq), 0] = seq

        net_output = net.y_pred(net_input)
        for seq_i in range(n_seq_per_batch):
            offset = batch_start + (seq_i * stride)
            plot_rectangles(ax, net_output, seq_i, offset=offset,
                            plot_seq_width=seq_length,
                            alpha=stride/seq_length)

    return ax


"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/neuralnilm/disaggregate.py /mnt/sshfs/imperial/workspace/python/neuralnilm/neuralnilm/"
End:
"""

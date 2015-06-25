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
        if None then stide = n_samples_per_batch
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    n_seq_per_batch, seq_length = net.input_shape[:2]
    n_samples_per_batch = seq_length * n_seq_per_batch
    if stride is None:
        stride = n_samples_per_batch
    if ax is None:
        ax = plt.gca()

    # Pad mains with zeros at both ends so we can slide
    # over the start and end of the mains data.
    pad_width = (seq_length, seq_length)
    mains_padded = np.pad(mains, pad_width, mode='constant')

    # Loop over the mains data, sliding the net over the data.
    n_mains_samples = len(mains_padded)
    last_mains_start_i = n_mains_samples - seq_length
    if last_mains_start_i < 1:
        raise RuntimeError("Not enough mains data!")

    for mains_start_i in xrange(0, last_mains_start_i, stride):
        mains_end_i = mains_start_i + n_samples_per_batch
        net_input_flat_batch = mains_padded[mains_start_i:mains_end_i]
        net_input = net_input_flat_batch.reshape(net.input_shape)
        net_output = net.y_pred(net_input)
        offset = mains_start_i / seq_length
        for seq_i in range(n_seq_per_batch):
            plot_rectangles(ax, net_output, seq_i, offset=offset)

    return ax


"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/neuralnilm/disaggregate.py /mnt/sshfs/imperial/workspace/python/neuralnilm/neuralnilm/"
End:
"""

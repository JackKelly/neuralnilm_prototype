from __future__ import division, print_function
import numpy as np
from collections import namedtuple
import csv
from os.path import join

from neuralnilm.source import standardise


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


Rectangle = namedtuple('Rectangle', ['left', 'right', 'height'])


def disaggregate_start_stop_end(mains, net, stride=1):
    """
    Parameters
    ----------
    mains : 1D np.ndarray
        Must already be standardised according to `net.source.input_stats`.
        And it is highly advisable to pad `mains` with `seq_length` elements
        at both ends so the net can slide over the very start and end.
    net : neuralnilm.net.Net
    stride : int or None, optional
        if None then stide = seq_length

    Returns
    -------
    rectangles : dict
        Each key is an output instance integer.
        Each value is a Rectangle namedtuple with fields:
        - 'start' : int, index into `mains`
        - 'stop' : int, index into `mains`
        - 'height' : float, raw network output
    """
    assert mains.ndim == 1
    n_seq_per_batch, seq_length = net.input_shape[:2]
    n_outputs = net.output_shape[2]
    if stride is None:
        stride = seq_length
    n_mains_samples = len(mains)

    # Divide mains data into batches
    n_batches = (n_mains_samples / stride) / n_seq_per_batch
    n_batches = np.ceil(n_batches).astype(int)

    rectangles = {output_i: [] for output_i in range(n_outputs)}

    # Iterate over each batch
    for batch_i in xrange(n_batches):
        net_input = np.zeros(net.input_shape, dtype=np.float32)
        batch_start = batch_i * n_seq_per_batch * stride
        for seq_i in xrange(n_seq_per_batch):
            mains_start_i = batch_start + (seq_i * stride)
            mains_end_i = mains_start_i + seq_length
            seq = mains[mains_start_i:mains_end_i]
            net_input[seq_i, :len(seq), 0] = seq

        net_output = net.y_pred(net_input)
        for seq_i in range(n_seq_per_batch):
            offset = batch_start + (seq_i * stride)
            for output_i in range(n_outputs):
                net_output_for_seq = net_output[seq_i, :, output_i]
                rect_left = (net_output_for_seq[0] * seq_length) + offset
                rect_left = int(round(rect_left))
                rect_right = (net_output_for_seq[1] * seq_length) + offset
                rect_right = int(round(rect_right))
                rect_height = net_output_for_seq[2]
                rect = Rectangle(
                    left=rect_left, right=rect_right, height=rect_height)
                rectangles[output_i].append(rect)

    return rectangles


def save_rectangles(rectangles, path=''):
    for output_i, rects in rectangles.iteritems():
        filename = 'disag_rectangles_output{:d}.csv'.format(output_i)
        filename = join(path, filename)
        print("Saving", filename)
        with open(filename, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(Rectangle._fields)
            writer.writerows(rects)
        print("Done saving", filename)

"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/neuralnilm/disaggregate.py /mnt/sshfs/imperial/workspace/python/neuralnilm/neuralnilm/"
End:
"""

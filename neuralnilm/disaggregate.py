from __future__ import division, print_function
import numpy as np
from collections import namedtuple
import csv
from os.path import join, expanduser

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


def rectangle_filename(output_i, path=''):
    """
    Parameters
    ----------
    output_i : int
    path : string

    Returns
    -------
    full_filename : string
    """
    path = expanduser(path)
    base_filename = 'disag_rectangles_output{:d}.csv'.format(output_i)
    full_filename = join(path, base_filename)
    return full_filename


def save_rectangles(rectangles, path=''):
    """
    Parameters
    ----------
    rectangles : dict
        Output from `disaggregate_start_stop_end()`
    path : string
    """
    for output_i, rects in rectangles.iteritems():
        filename = rectangle_filename(output_i, path)
        print("Saving", filename)
        with open(filename, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(Rectangle._fields)
            writer.writerows(rects)
        print("Done saving", filename)


def load_rectangles(path=''):
    rectangles = {}
    for output_i in range(256):
        filename = rectangle_filename(output_i, path)
        try:
            f = open(filename, 'rb')
        except IOError:
            if output_i == 0:
                raise IOError(
                    "No rectangle CSV files found in {}".format(path))
            else:
                break
        rects = []
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            left = int(row[0])
            right = int(row[1])
            height = float(row[2])
            rect = Rectangle(left=left, right=right, height=height)
            rects.append(rect)
        f.close()
        rectangles[output_i] = rects

    return rectangles


def rectangles_to_matrix(rectangles, max_appliance_power):
    """
    Parameters
    ----------
    rectangles : list of Rectangles
        Value of dict output from `disaggregate_start_stop_end()`
    max_appliance_power : int or float
        Watts

    Returns
    -------
    matrix : 2D numpy.ndarray
        Normalised to [0, 1]
    """
    n_samples = rectangles[-1].right
    matrix = np.zeros(shape=(max_appliance_power, n_samples))
    for rect in rectangles:
        height = int(round(rect.height * max_appliance_power))
        matrix[:height, rect.left:rect.right] += 1
    matrix /= matrix.max()
    return matrix


def plot_rectangles_matrix(matrix):
    import matplotlib.pyplot as plt
    plt.imshow(matrix, aspect='auto', interpolation='none', origin='lower')
    plt.show()


def rectangles_matrix_to_vector(matrix, overlap_threshold=0.5):
    """
    Parameters
    ----------
    matrix : 2D numpy.ndarray
        Output from `rectangles_to_matrix`
    overlap_threshold : float, [0, 1]

    Returns
    -------
    vector : 1D numpy.ndarray
        Watts
    """
    n_samples = matrix.shape[1]
    vector = np.zeros(n_samples)
    matrix[matrix < overlap_threshold] = 0
    for i in range(n_samples):
        row_indicies = np.nonzero(matrix[:, i])[0]
        if row_indicies.size > 0:
            power = np.max(row_indicies)
            vector[i] = power

    return vector

"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/neuralnilm/disaggregate.py /mnt/sshfs/imperial/workspace/python/neuralnilm/neuralnilm/"
End:
"""

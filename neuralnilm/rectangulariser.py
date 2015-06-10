from __future__ import print_function, division
import numpy as np
import sys


def rectangularise(data, n_segments, format='proportional'):
    if data.ndim == 1:
        return _rectangularise(data, n_segments=n_segments, format=format)
    else:
        n_seq_per_batch = data.shape[0]
        output = np.empty((n_seq_per_batch, n_segments, 1), dtype=np.float32)
        for batch_i in range(n_seq_per_batch):
            output[batch_i, :, 0] = _rectangularise(
                data[batch_i, :, 0], n_segments=n_segments, format=format)
        return output


def _rectangularise(data, n_segments, format='proportional'):
    changepoints = []
    for segment_i in range(n_segments-1):
        slice_with_highest_variance = _get_slice_with_highest_variance(
            changepoints, data)
        if slice_with_highest_variance is None:
            # fill changepoints with zeros
            n_zeros = n_segments - len(changepoints) - 1
            changepoints += [0] * n_zeros
            changepoints.sort()
            break
        chunk_with_highest_variance = data[slice_with_highest_variance]
        changepoint = (_get_changepoint(chunk_with_highest_variance)
                       + slice_with_highest_variance.start)
        changepoints.append(changepoint)
        changepoints.sort()
    segment_widths = [end - start for start, end in
                      zip([0] + changepoints, changepoints + [len(data)])]
    segment_widths = np.array(segment_widths)
    if format == 'proportional':
        total = segment_widths.sum()
        if total == 0:
            return segment_widths
        else:
            return segment_widths / total
    elif format == 'changepoints':
        return changepoints
    else:
        raise RuntimeError("Unknown format: '{}'".format(format))


def _get_changepoint(data):
    n_samples = len(data)
    best_error = sys.float_info.max
    best_i = 0
    for i in range(2, n_samples-2):
        chunk1 = data[:i]
        chunk2 = data[i:]
        error = (chunk1.var() * (len(chunk1) / n_samples) +
                 chunk2.var() * (len(chunk2) / n_samples))
        if error < best_error:
            best_error = error
            best_i = i
    return best_i


def _get_slice_with_highest_variance(changepoints, data):
    prev_changepoint = 0
    highest_variance = -1.0
    slice_with_highest_variance = None
    for changepoint in changepoints + [-1]:
        this_slice = slice(prev_changepoint, changepoint)
        chunk = data[this_slice]
        prev_changepoint = changepoint
        variance = chunk.var() * (len(chunk) / len(data))
        if variance > highest_variance:
            highest_variance = variance
            slice_with_highest_variance = this_slice
    if highest_variance == 0:
        return None
    else:
        return slice_with_highest_variance


"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/neuralnilm/rectangulariser.py /mnt/sshfs/imperial/workspace/python/neuralnilm/neuralnilm/"
End:
"""

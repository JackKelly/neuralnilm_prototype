from __future__ import division, print_function
import numpy as np

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

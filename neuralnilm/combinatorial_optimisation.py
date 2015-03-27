from __future__ import print_function, division
import theano
import numpy as np


def combinatorial_optimisation(network_input, 
                               network_output, 
                               input_normalisation_stats, 
                               output_normalisation_stats):
    """
    network_input : 
        shape = (n_seq_per_batch, seq_length, n_inputs)
    network_output : 
        shape = (n_seq_per_batch, seq_length, n_outputs, n_components, 3)
    input_normalisation_stats :
        dict with keys {'mean', 'std'}.  each is a 1D numpy array with 
        values for each appliance.
    """
    
    """
    For each time step:
      get modal values (means) from network output
      convert modal values to power in watts (I think Source scales twice! 
        Once to [0,1], then to standardised)
      go through each combination of modal power (watts) values:
        for each appliance, consider 3 states: off, mean1, mean2
        if the sum is above network input + margin of error then discard
        otherwise get mean negative log likelihood for this combination.
        if the NLL is lower than the previous lowest then store this combination
        and set the lowest NLL.
    """

    mu     = network_output[:, :, :, :, 0]
    sigma  = network_output[:, :, :, :, 1]
    mixing = network_output[:, :, :, :, 2]
    n_appliances = mu.shape[2]
    n_components = mu.shape[3]
    mu_watts = un_normalise(mu, output_normalisation_stats)


def un_normalise(normalised, stats):
    """
    To un-normalise: 
      1. multiply by stdev
      2. add mean

    Parameters
    ----------
        normalised : 
            shape = (n_seq_per_batch, seq_length, n_outputs, ...)
        stats :
            dict with keys {'mean', 'std'}.  each is a 1D numpy array with 
            values for each appliance.

    Returns
    -------
    watts
    """
    n_appliances = normalised.shape[2]
    watts = np.empty(shape=normalised.shape, dtype=theano.config.floatX)
    for appliance_i in range(n_appliances):
        watts_appliance = watts[:, :, appliance_i]
        watts_appliance *= stats['std'][appliance_i]
        watts_appliance += stats['mean'][appliance_i]
    return watts

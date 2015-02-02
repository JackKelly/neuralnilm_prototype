from __future__ import division
import numpy as np

"""
INPUT: quantized mains fdiff
OUTPUT: appliance power demand
"""

# Sequence length
LENGTH = 400
# Number of units in the hidden (recurrent) layer
N_BATCH = 30

def quantized(inp):
    n = 10
    n_batch, length, _ = inp.shape
    out = np.zeros(shape=(n_batch, length, n))
    for i_batch in range(n_batch):
        for i_element in range(length):
            out[i_batch,i_element,:], _ = np.histogram(inp[i_batch, i_element, 0], [-1,-.8,-.6,-.4,-.2,0.0,.2,.4,.6,.8,1])
    return (out * 2) - 1

def gen_single_appliance(length, power, on_duration, min_off_duration=20, 
                         fdiff=True):
    if fdiff:
        length += 1
    appliance_power = np.zeros(shape=(length))
    i = 0
    while i < length:
        if np.random.binomial(n=1, p=0.2):
            end = min(i + on_duration, length)
            appliance_power[i:end] = power
            i += on_duration + min_off_duration
        else:
            i += 1
    return np.diff(appliance_power) if fdiff else appliance_power

def gen_batches_of_single_appliance(length, n_batch, *args, **kwargs):
    batches = np.zeros(shape=(n_batch, length, 1))
    for i in range(n_batch):
        batches[i, :, :] = gen_single_appliance(length, *args, **kwargs).reshape(length, 1)
    return batches

def cumsum_seq(y):
    for i in range(y.shape[0]):
        y[i,:,:] = np.cumsum(y[i,:,:]).reshape(y.shape[1], y.shape[2])
    return y

def gen_data(length=LENGTH, n_batch=N_BATCH, n_appliances=2, 
             appliance_powers=[10,20], 
             appliance_on_durations=[10,2]):
    '''Generate a simple energy disaggregation data.

    :parameters:
        - length : int
            Length of sequences to generate
        - n_batch : int
            Number of training sequences per batch

    :returns:
        - X : np.ndarray, shape=(n_batch, length, 1)
            Input sequence
        - y : np.ndarray, shape=(n_batch, length, 1)
            Target sequence, appliance 1
    '''
    y = gen_batches_of_single_appliance(length, n_batch, 
                                        power=appliance_powers[0], 
                                        on_duration=appliance_on_durations[0])
    X = y.copy()
    for power, on_duration in zip(appliance_powers, appliance_on_durations)[1:]:
        X += gen_batches_of_single_appliance(length, n_batch, power=power, on_duration=on_duration)

    max_power = np.sum(appliance_powers)
    
    return quantized(X / max_power), ((cumsum_seq(y) * 2) / appliance_powers[0]) - 1

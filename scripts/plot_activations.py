from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py


def plot_activations(filename, epoch, seq_i=0, normalise=False):
    f = h5py.File(filename, mode='r')
    epoch_name = 'epoch{:06d}'.format(epoch)
    epoch_group = f[epoch_name]
    for layer_name, layer_activations in epoch_group.iteritems():
        print(layer_activations[seq_i, :, :].transpose().shape)
        activations = layer_activations[seq_i, :, :]
        if normalise:
            activations /= activations.max(axis=0)
        plt.imshow(activations.transpose(), aspect='auto', interpolation='none')
        break

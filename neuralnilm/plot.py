from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from scipy.stats import norm


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


class Plotter(object):
    def __init__(self, net):
        self.net = net
        self.n_seq_to_plot = 10
        self.linewidth = 0.2
        self.save = True
        self.seq_i = 0
        self.target_labels = (
            self.net.source.get_labels() if self.net is not None else [])
    
    def plot_all(self):
        self.plot_costs()
        self.plot_estimates()

    def plot_costs(self):
        fig, ax = plt.subplots(1)
        ax.plot(self.net.training_costs, label='Training')
        validation_x = np.arange(
            0, len(self.net.training_costs), self.net.validation_interval)
        n_validations = min(len(validation_x), len(self.net.validation_costs))
        ax.plot(validation_x[:n_validations], 
                self.net.validation_costs[:n_validations],
                label='Validation')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.legend()
        ax.grid(True)
        self._save_or_display_fig('costs', fig, include_epochs=False)
        return ax

    def plot_estimates(self):
        X, y = self.net.X_val, self.net.y_val
        output = self.net.y_pred(X)
        X, y, output = self._process(X, y, output)
        sequences = range(min(self.net.n_seq_per_batch, self.n_seq_to_plot))
        for seq_i in sequences:
            self.seq_i = seq_i
            fig, axes = self.create_estimates_fig(X, y, output)

    def _process(self, X, y, output):
        return X, y, output

    def create_estimates_fig(self, X, y, output):
        fig, axes = plt.subplots(3)
        self._plot_network_output(axes[0], output)
        self._plot_target(axes[1], y)
        self._plot_input(axes[2], X)
        for ax in axes:
            ax.grid(True)
        self._save_or_display_fig('estimates', fig, end_string=self.seq_i)
        return fig, axes

    def _save_or_display_fig(self, string, fig, 
                             include_epochs=True, end_string=""):
        fig.tight_layout()
        if not self.save:
            plt.show(block=True)
            return
        end_string = str(end_string)
        filename = (
            self.net.experiment_name + ("_" if self.net.experiment_name else "") + 
            string +
            ("_{:d}epochs".format(self.net.n_iterations()) if include_epochs else "") +
            ("_" if end_string else "") + end_string +
            ".pdf")
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    def _plot_network_output(self, ax, output):
        ax.set_title('Network output')
        ax.plot(output[self.seq_i, :, :], linewidth=self.linewidth)
        n = len(output[self.seq_i, :, :])
        ax.set_xlim([0, n])

    def _plot_target(self, ax, y):
        ax.set_title('Target')
        ax.plot(y[self.seq_i, :, :], linewidth=self.linewidth)
        # alpha: lower = more transparent
        ax.legend(self.target_labels, fancybox=True, 
                  framealpha=0.5, prop={'size': 6})
        n = len(y[self.seq_i, :, :])
        ax.set_xlim([0, n])

    def _plot_input(self, ax, X):
        ax.set_title('Network input')
        if self.net is None:
            data = X[self.seq_i, :, :]
        else:
            start, end = self.net.source.inside_padding()
            data = X[self.seq_i, start:end, :]
        ax.plot(data, linewidth=self.linewidth)
        ax.set_xlim([0, data.shape[0]])


class MDNPlotter(Plotter):
    def __init__(self, net=None, seq_length=None):
        super(MDNPlotter, self).__init__(net)
        self.seq_length = (self.net.source.output_shape()[1]
                           if seq_length is None else seq_length)

    def create_estimates_fig(self, X, y, output):
        fig, axes = plt.subplots(4)
        mu     = output[:,:,:,:,0]
        sigma  = output[:,:,:,:,1]
        mixing = output[:,:,:,:,2]
        self._plot_network_output(axes[0], mu, sigma, mixing)
        self._plot_network_output_means(axes[1], mu, mixing)
        self._plot_target(axes[2], y)
        self._plot_input(axes[3], X)
        for ax in axes:
            ax.grid(True)
        self._save_or_display_fig('estimates', fig, end_string=self.seq_i)
        return fig, axes

    def _plot_network_output(self, ax, mu, sigma, mixing):
        ax.set_title('Network output density')
        gmm_heatmap(ax,
            (mu[self.seq_i, :, 0, :],
             sigma[self.seq_i, :, 0, :], 
             mixing[self.seq_i, :, 0, :]))

    def _plot_network_output_means(self, ax, mu, mixing):
        n_components = mu.shape[-1]
        x = range(self.seq_length)
        for i in range(n_components):
            ax.scatter(
                x, mu[self.seq_i, :, 0, i], s=mixing[self.seq_i, :, 0, i] * 5)
        ax.set_xlim([0, self.seq_length])
        ax.set_title('Network output means')

    def _process(self, X, y, output, target_shape=None):
        if target_shape is None:
            target_shape = self.net.source.output_shape()
        y_reshaped = y.reshape(target_shape)
        output_reshaped = output.reshape(target_shape + output.shape[2:])
        return X, y_reshaped, output_reshaped


def gmm_pdf(theta, x):
    """
    Parameters
    ----------
    theta : tuple of (mu, sigma, mixing)
    """
    pdf = None
    for mu, sigma, mixing in zip(*theta):
        norm_pdf = norm.pdf(x=x, loc=mu, scale=sigma)
        norm_pdf *= mixing
        if pdf is None:
            pdf = norm_pdf
        else:
            pdf += norm_pdf
    return pdf


def gmm_heatmap(ax, thetas):
    """
    Parameters
    ----------
    thetas : tuple of (array of mus, array of sigmas, array of mixing)
    """
    N_X = 300
    UPPER_LIMIT = 5
    LOWER_LIMIT = -1
    n_y = len(thetas[0])
    x = np.linspace(UPPER_LIMIT, LOWER_LIMIT, N_X)
    img = np.zeros(shape=(N_X, n_y))
    i = 0
    for i, (mu, sigma, mixing) in enumerate(zip(*thetas)):
        img[:, i] = gmm_pdf((mu, sigma, mixing), x)
        img[:, i] /= np.max(img[:, i])
    EXTENT = (0, n_y, LOWER_LIMIT, UPPER_LIMIT) # left, right, bottom, top
    ax.imshow(img, interpolation='none', extent=EXTENT, aspect='auto')
    return ax    

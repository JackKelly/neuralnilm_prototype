from __future__ import division, print_function
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
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
    def __init__(self, n_seq_to_plot=10):
        self.n_seq_to_plot = n_seq_to_plot
        self.linewidth = 0.2
        self.save = True
        self.seq_i = 0
        self.plot_additional_seqs = 0
        self.net = None

    @property
    def target_labels(self):
        return self.net.source.get_labels() if self.net is not None else []

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
        self._save_or_display_fig(
            'costs', fig, include_epochs=False, suffix='png')
        return ax

    def plot_estimates(self):
        validation_batch = self.net.validation_batch
        X, y = validation_batch.data
        output = self.net.y_pred(X)
        X, y, output = self._process(X, y, output)
        sequences = range(min(self.net.n_seq_per_batch, self.n_seq_to_plot))
        for seq_i in sequences:
            self.seq_i = seq_i
            fig, axes = self.create_estimates_fig(
                X, y, output, validation_batch.target_power_timeseries)

    def _process(self, X, y, output):
        return X, y, output

    def create_estimates_fig(self, X, y, output, target_power_timeseries):
        fig, axes = plt.subplots(3)
        self._plot_network_output(axes[0], output)
        self._plot_target(axes[1], y, target_power_timeseries)
        self._plot_input(axes[2], X)
        for ax in axes:
            ax.grid(True)
        self._save_or_display_fig('estimates', fig, end_string=self.seq_i)
        return fig, axes

    def _save_or_display_fig(self, string, fig,
                             include_epochs=True, end_string="", suffix="pdf"):
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
            "." + suffix)
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    def _plot_network_output(self, ax, output):
        ax.set_title('Network output')
        ax.plot(output[self.seq_i, :, :], linewidth=self.linewidth)
        n = len(output[self.seq_i, :, :])
        ax.set_xlim([0, n])

    def _plot_target(self, ax, y, target_power_timeseries):
        ax.set_title('Target')
        ax.plot(y[self.seq_i, :, :], linewidth=self.linewidth)
        # alpha: lower = more transparent
        ax.legend(self.target_labels, fancybox=True,
                  framealpha=0.5, prop={'size': 6})
        n = len(y[self.seq_i, :, :])
        ax.set_xlim([0, n])

    def _plot_input(self, ax, X):
        ax.set_title('Network input')
        CHANNEL = 0
        if self.net is None:
            data = X[self.seq_i, :, CHANNEL]
        elif hasattr(self.net.source, 'inside_padding'):
            start, end = self.net.source.inside_padding()
            data = X[self.seq_i, start:end, CHANNEL]
        else:
            data = X[self.seq_i, :, CHANNEL]
        ax.plot(data, linewidth=self.linewidth)
        ax.set_xlim([0, data.shape[0]])


class MDNPlotter(Plotter):
    def __init__(self, net=None, seq_length=None):
        super(MDNPlotter, self).__init__(net)
        self.seq_length = (self.net.source.output_shape()[1]
                           if seq_length is None else seq_length)

    def create_estimates_fig(self, X, y, output):
        n_outputs = output.shape[2]
        fig, axes = plt.subplots(2 + n_outputs, figsize=(8, 11))
        self._plot_input(axes[0], X)
        self._plot_target(axes[1], y)
        for output_i in range(n_outputs):
            ax = axes[2 + output_i]
            self._plot_network_output(ax, output_i, output, y)
        for ax in axes:
            ax.grid(False)
        self._save_or_display_fig('estimates', fig, end_string=self.seq_i)
        return fig, axes

    def _plot_network_output(self, ax, output_i, output, target):
        title = 'Network output density'
        if self.target_labels:
            title += ' for {}'.format(self.target_labels[output_i])
        ax.set_title(title)
        output = output[self.seq_i, :, output_i, :, :]
        target = target[self.seq_i, :, output_i]
        mu     = output[:, :, 0]
        sigma  = output[:, :, 1]
        mixing = output[:, :, 2]

        y_extra = max(target.ptp() * 0.2, mu.ptp() * 0.2)
        y_lim = (min(target.min(), mu.min()) - y_extra,
                 max(target.max(), mu.max()) + y_extra)
        x_lim = (0, self.seq_length)
        gmm_heatmap(ax, (mu, sigma, mixing), x_lim, y_lim)

        # plot means
        n_components = mu.shape[-1]
        for component_i in range(n_components):
            ax.plot(mu[:, component_i], color='red', linewidth=0.5, alpha=0.5)

        # plot target
        ax.plot(target, color='green', linewidth=0.5, alpha=0.5)

        # set limits
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    def _process(self, X, y, output, target_shape=None):
        if target_shape is None:
            target_shape = self.net.source.output_shape()
        y_reshaped = y.reshape(target_shape)
        output_reshaped = output.reshape(target_shape + output.shape[2:])
        return X, y_reshaped, output_reshaped


class CentralOutputPlotter(Plotter):
    def _plot_network_output(self, ax, output):
        ax.set_title('Network output')
        n_outputs = output.shape[2]
        ax.bar(range(n_outputs), output[self.seq_i, 0, :])

    def _plot_target(self, ax, y, target_power_timeseries):
        ax.set_title('Target')
        n_outputs = y.shape[2]
        ax.bar(range(n_outputs), y[self.seq_i, 0, :])
        ax.set_xticklabels(self.target_labels)


class RectangularOutputPlotter(Plotter):
    def __init__(self, *args, **kwargs):
        self.cumsum = kwargs.pop('cumsum', False)
        super(RectangularOutputPlotter, self).__init__(*args, **kwargs)

    def _plot_network_output(self, ax, output):
        self._plot_scatter(ax, output, 'Network output')

    def _plot_target(self, ax, y, target_power_timeseries):
        self._plot_scatter(ax, y, 'Target')

    def _plot_scatter(self, ax, data, title):
        example = data[self.seq_i, :, 0]
        if self.cumsum:
            example = np.cumsum(example)
        y_values = [0] * len(example)
        ax.scatter(example, y_values)
        ax.set_xlim((0, 1))
        ax.set_title(title)


class StartEndMeanPlotter(Plotter):
    def __init__(self, *args, **kwargs):
        self.max_target_power = kwargs.pop('max_target_power', 100)
        super(StartEndMeanPlotter, self).__init__(*args, **kwargs)

    def _plot_target(self, ax, y, target_power_timeseries):
        ax.set_title('Target')
        # alpha: lower = more transparent
        ax.legend(self.target_labels, fancybox=True,
                  framealpha=0.5, prop={'size': 6})

        # plot time series
        target_power_timeseries = (
            target_power_timeseries[self.seq_i, :, :]) #  / self.max_target_power)
        n, n_outputs = target_power_timeseries.shape

        colours = colour_iter(n_outputs)
        for output_i in range(n_outputs):
            ax.plot(target_power_timeseries[:, output_i],
                    linewidth=self.linewidth,
                    c=colours[output_i],
                    label=self.target_labels[output_i])
        # alpha: lower = more transparent
        ax.legend(fancybox=True, framealpha=0.5, prop={'size': 6})

        # plot 'rectangle'
        for output_i in range(n_outputs):
            target = y[self.seq_i, :, output_i]
            left = target[0] * n
            width = (target[1] - target[0]) * n
            height = target[2]
            ax.bar(left, height, width, alpha=0.5,
                   color=colours[output_i],
                   edgecolor=colours[output_i])
        ax.set_xlim([0, n])

    def _plot_network_output(self, ax, output):
        n_outputs = output.shape[2]
        colours = colour_iter(n_outputs) 
        for output_i in range(n_outputs):
            single_output = output[self.seq_i, :, output_i]
            left = single_output[0]
            width = (single_output[1] - single_output[0])
            height = single_output[2]
            ax.bar(left, height, width,
                   color=colours[output_i],
                   edgecolor=colours[output_i])
        ax.set_xlim((0, 1))
        ax.set_title('Network output')


def colour_iter(n):
    return [c for c in cm.rainbow(np.linspace(0, 1, n))]


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


def gmm_heatmap(ax, thetas, x_lim, y_lim, normalise=False, 
                cmap=matplotlib.cm.Blues):
    """
    Parameters
    ----------
    thetas : tuple of (array of mus, array of sigmas, array of mixing)
    y_lim, x_lim : each is a 2-tuple of numbers
    """
    N_X = 200
    n_y = len(thetas[0])
    x_lim = (x_lim[0] - 0.5, x_lim[1] - 0.5)
    extent = x_lim + y_lim  # left, right, bottom, top
    x = np.linspace(y_lim[0], y_lim[1], N_X)
    img = np.zeros(shape=(N_X, n_y))
    i = 0
    for i, (mu, sigma, mixing) in enumerate(zip(*thetas)):
        img[:, i] = gmm_pdf((mu, sigma, mixing), x)
        if normalise:
            img[:, i] /= np.max(img[:, i])
    ax.imshow(img, interpolation='none', extent=extent, aspect='auto',
              origin='lower', cmap=cmap)
    return ax

"""
Emacs variables
Local Variables:
compile-command: "cp /home/jack/workspace/python/neuralnilm/neuralnilm/plot.py /mnt/sshfs/imperial/workspace/python/neuralnilm/neuralnilm/"
End:
"""

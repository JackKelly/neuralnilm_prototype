from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import yaml
from os.path import join
from copy import copy

BASE_DIRECTORY = '/home/jack/experiments/neuralnilm/figures/'

APPLIANCES = [
    'fridge freezer',
    'kettle',
    'HTPC',
    'washer dryer',
    'dish washer',
    'across all appliances'
]

METRICS = [
    'f1_score',
    'precision_score',
    'recall_score',
    'accuracy_score',
    'relative_error_in_total_energy',
    'total_energy_correctly_assigned',
    'mean_absolute_error'
]

# LOAD YAML
with open(join(BASE_DIRECTORY, 'benchmark_scores.yaml'), 'r') as fh:
    scores = yaml.load(fh)
with open(join(BASE_DIRECTORY, 'metric_scores.yaml'), 'r') as fh:
    scores['Neural NILM'] = yaml.load(fh)

algorithms = ['Always off', 'Mean', 'CO', 'FHMM', 'Neural NILM']
full_algorithm_names = ['Always off', 'Mean', 'Combinatorial Optimisation ', 'Factorial HMM', 'Neural NILM']
n_algorithms = len(algorithms)

# remove 'classification' / 'disaggregation' / 'regression' keys
for algo, appliances in scores.iteritems():
    for appliance, categories in appliances.iteritems():
        for category, metrics in copy(categories).iteritems():
            for metric, score in metrics.iteritems():
                scores[algo][appliance][metric] = score
            del scores[algo][appliance][category]

# Calc 'relative error in total energy' across all appliances
relative_error = np.zeros(n_algorithms)
i = 0
for algo, appliances in scores.iteritems():
    for appliance, metrics in appliances.iteritems():
        for metric, score in metrics.iteritems():
            if metric == 'relative_error_in_total_energy':
                relative_error[i] += score
    i += 1
relative_error /= n_algorithms
for i, algo in enumerate(scores):
    scores[algo]['across all appliances']['relative_error_in_total_energy'] = relative_error[i]

COLOR = ['#5F7343', '#99A63C', '#FEC06A', '#F25430', '#E61924']
nrows = len(METRICS)
ncols = len(APPLIANCES)
x = range(n_algorithms)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey='row', figsize=(10, 10))
fig.patch.set_facecolor('white')
for row_i, metric in enumerate(METRICS):
    for col_i, appliance in enumerate(APPLIANCES):
        ax = axes[row_i, col_i]
        scores_for_algorithms = []
        for algo in algorithms:
            try:
                scores_for_algorithms.append(scores[algo][appliance][metric])
            except:
                scores_for_algorithms.append(0)
        rects = ax.bar(
            x, scores_for_algorithms, color=COLOR, edgecolor=COLOR, zorder=3)

        # Numbers on the plot
        if row_i == 6:  # mean absolute error (watts)
            text_y = 150
            text_format = '{:3.0f}'
        else:
            text_y = 0.5
            text_format = '{:.2f}'

        # Draw text
        for i, rect in enumerate(rects):
            ax.text(
                rect.get_x() + rect.get_width() / 2.5,
                text_y,
                text_format.format(scores_for_algorithms[i]),
                va='center', rotation=90)

        # Formatting
        ax.set_xticks([])
        ax.tick_params(direction='out')
        ax.yaxis.grid(
            b=True, which='major', color='white', linestyle='-', zorder=0)
        ax.patch.set_facecolor((0.85, 0.85, 0.85))

        if row_i == 4:  # relative error in total energy
            ax.set_ylim((-1, 1))

        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

        if row_i == 0:
            if appliance == 'across all appliances':
                label = 'Across all\nappliances'
            else:
                label = appliance.replace(' ', '\n')
                label = label[0].capitalize() + label[1:]
            ax.set_title(label)
        if col_i == 0:
            label = metric.replace('_', '\n')
            if label == 'mean\nabsolute\nerror':
                label = label + '\n(watts)'
            elif label == 'total\nenergy\ncorrectly\nassigned':
                label = 'prop. of\n' + label
            elif label == 'relative\nerror\nin\ntotal\nenergy':
                label = 'relative\nerror in\ntotal\nenergy'
            label = label[0].capitalize() + label[1:]
            ylabel = ax.set_ylabel(label)
            ylabel.set_rotation('horizontal')
            ylabel.set_verticalalignment('center')
            ylabel.set_horizontalalignment('center')
            ax.yaxis.labelpad = 25
            ax.tick_params(axis='y', left='on', right='off')
        else:
            ax.tick_params(axis='y', left='off', right='off')

plt.subplots_adjust(hspace=0.3)
plt.legend(rects, full_algorithm_names, ncol=n_algorithms, loc=(-6, -0.8),
           frameon=False)

plt.show()

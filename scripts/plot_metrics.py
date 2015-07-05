from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import yaml
from os.path import join
from copy import copy

BASE_DIRECTORY = '/home/jack/experiments/neuralnilm/figures/'

APPLIANCES = [
    'fridge freezer',
    'washer dryer',
    'kettle',
    'HTPC',
    'dish washer',
    'across all appliances'
]

METRICS = [
    'accuracy_score',
    'f1_score',
    'precision_score',
    'recall_score',
    'relative_error_in_total_energy',
    'total_energy_correctly_assigned',
    'mean_absolute_error'
]

# LOAD YAML
with open(join(BASE_DIRECTORY, 'benchmark_scores.yaml'), 'r') as fh:
    scores = yaml.load(fh)
with open(join(BASE_DIRECTORY, 'metric_scores.yaml'), 'r') as fh:
    scores['Neural NILM'] = yaml.load(fh)

algorithms = scores.keys()

# remove 'classification' / 'disaggregation' / 'regression' keys
for algo, appliances in scores.iteritems():
    for appliance, categories in appliances.iteritems():
        for category, metrics in copy(categories).iteritems():
            for metric, score in metrics.iteritems():
                scores[algo][appliance][metric] = score
            del scores[algo][appliance][category]

# set up figure
# COLOR = ['#4F328C', '#A69B03', '#F25D50']
# COLOR = ['#FFF109', '#00DDFF', '#A758FF']
# COLOR = ['#', '#', '#']
# COLOR = ['#FFCF1D', '#750903', '#0E63B2']
# COLOR = ['#FEC06A', '#F25430', '#E61924']
COLOR = ['#5F7343', '#FEC06A', '#E61924']
nrows = len(METRICS)
ncols = len(APPLIANCES)
n_algorithms = len(algorithms)
x = range(n_algorithms)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey='row')
for row_i, metric in enumerate(METRICS):
    for col_i, appliance in enumerate(APPLIANCES):
        ax = axes[row_i, col_i]
        scores_for_algorithms = []
        for algo in algorithms:
            try:
                scores_for_algorithms.append(scores[algo][appliance][metric])
            except:
                scores_for_algorithms.append(0)
        print(row_i, col_i, scores_for_algorithms)
        ax.bar(x, scores_for_algorithms, color=COLOR)

        # Formatting
        ax.set_xticks([])
        if row_i == 0:
            ax.set_title(appliance.replace(' ', '\n'))
        if col_i == 0:
            ylabel = ax.set_ylabel(metric.replace('_', '\n'))
            ylabel.set_rotation('horizontal')
            ylabel.set_verticalalignment('center')
            ylabel.set_horizontalalignment('right')
        # if col_i != 0 and row_i != 0:
        #     ax.axis('off')

# set accuracy row ylim
axes[0, 0].set_ylim((0.75, 1))

plt.show()

"""
TODO
* across all appliances for relative error in total energy
* y ticks on right as well
* less cluttered y ticks
* legend for algorithms
"""

from __future__ import print_function, division
import numpy as np
import sklearn.metrics as metrics
from os.path import join
import matplotlib.pyplot as plt

BASE_DIRECTORY = '/storage/experiments/neuralnilm/figures/'
EXPERIMENT_DIRECTORIES = {'fridge': 'e544a'}

scores = {}
for appliance, exp_dir in EXPERIMENT_DIRECTORIES.iteritems():
    print("Metrics for", appliance)
    full_dir = join(BASE_DIRECTORY, exp_dir)

    def load(filename):
        return np.load(join(full_dir, filename + '.npy'))

    mains = load('mains')
    y_true = load('targets')
    y_pred = load('disag_vector')

    # Plot
    fig, ax = plt.subplots()
    ax.plot(y_true, label='y_true')
    ax.plot(y_pred, label='y_pred')
    ax.legend()
    plt.show()

    # Truncate
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    # Classification metrics
    ON_POWER = 10
    y_true_class = y_true > ON_POWER
    y_pred_class = y_pred > ON_POWER

    METRICS = {
        'classification': [
            'accuracy_score',
            'f1_score',
            'precision_score',
            'recall_score'
        ],
        'regression': [
            'explained_variance_score',
            'mean_absolute_error',
            'mean_squared_error'
        ]
    }

    scores[appliance] = {}
    for metric_type, metric_list in METRICS.iteritems():
        scores[appliance][metric_type] = {}
        for metric in metric_list:
            scores[appliance][metric_type][metric] = eval(
                'metrics.'+metric+'(y_true_class, y_pred_class)')

    # Total energy correctly assigned
    # See Eq(1) on p5 of Kolter & Johnson 2011
    abs_diff = np.fabs(y_pred - y_true)
    sum_abs_diff = np.sum(abs_diff)
    denominator = 2 * np.sum(mains)
    total_energy_correctly_assigned = 1 - (sum_abs_diff / denominator)

    scores[appliance]['disaggregation'] = {
        'total_energy_correctly_assigned': total_energy_correctly_assigned
    }

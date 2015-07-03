from __future__ import print_function, division
import numpy as np
import sklearn.metrics as metrics
from os.path import join
import matplotlib.pyplot as plt
import yaml  # for pretty-printing dict

# sklearn evokes warnings from numpy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

SHOW_PLOTS = False
BASE_DIRECTORY = '/storage/experiments/neuralnilm/figures/'
EXPERIMENT_DIRECTORIES = {
    'fridge freezer': 'e544a',
    'washer dryer': 'e545a',
    'kettle': 'e545b',
    'HTPC': 'e545c',
    'dish washer': 'e545d'
}

METRICS = {
    'classification': [
        'accuracy_score',
        'f1_score',
        'precision_score',
        'recall_score'
    ],
    'regression': [
        'explained_variance_score',
        'mean_absolute_error'
    ]
}

scores = {}
total_sum_abs_diff = 0.0
aggregate_predictions = None
for appliance, exp_dir in EXPERIMENT_DIRECTORIES.iteritems():
    full_dir = join(BASE_DIRECTORY, exp_dir)

    def load(filename):
        return np.load(join(full_dir, filename + '.npy'))

    y_true = load('targets')
    y_pred = load('disag_vector')

    # Plot
    if SHOW_PLOTS:
        fig, ax = plt.subplots()
        ax.plot(y_true, label='y_true')
        ax.plot(y_pred, label='y_pred')
        ax.legend()
        ax.set_title(appliance)
        plt.show()

    # Truncate
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    if aggregate_predictions is None:
        aggregate_predictions = y_pred
    else:
        n_agg = min(len(aggregate_predictions), len(y_pred))
        aggregate_predictions = aggregate_predictions[:n_agg]
        aggregate_predictions += y_pred[:n_agg]

    # Classification metrics
    ON_POWER = 10
    y_true[y_true <= ON_POWER] = 0
    y_true_class = y_true > ON_POWER
    y_pred_class = y_pred > ON_POWER

    ARGS = {
        'classification': '(y_true_class, y_pred_class)',
        'regression': '(y_true, y_pred)'
    }

    scores[appliance] = {}
    for metric_type, metric_list in METRICS.iteritems():
        scores[appliance][metric_type] = {}
        for metric in metric_list:
            score = eval('metrics.' + metric + ARGS[metric_type])
            scores[appliance][metric_type][metric] = float(score)

    sum_y_true = np.sum(y_true)
    sum_y_pred = np.sum(y_pred)
    # negative means underestimates
    relative_error_in_total_energy = float(
        (sum_y_pred - sum_y_true) / max(sum_y_true, sum_y_pred))

    # For total energy correctly assigned
    mains = load('mains')
    denominator = 2 * np.sum(mains)
    abs_diff = np.fabs(y_pred - y_true)
    sum_abs_diff = np.sum(abs_diff)
    total_energy_correctly_assigned = 1 - (sum_abs_diff / denominator)
    total_energy_correctly_assigned = float(total_energy_correctly_assigned)
    total_sum_abs_diff += sum_abs_diff

    scores[appliance]['disaggregation'] = {
        'relative_error_in_total_energy': relative_error_in_total_energy,
        'total_energy_correctly_assigned': total_energy_correctly_assigned
    }

# Total energy correctly assigned
# See Eq(1) on p5 of Kolter & Johnson 2011
denominator = 2 * np.sum(mains)
total_energy_correctly_assigned = 1 - (total_sum_abs_diff / denominator)
total_energy_correctly_assigned = float(total_energy_correctly_assigned)

# explained variance
n = min(len(mains), len(aggregate_predictions))
mains = mains[:n]
aggregate_predictions = aggregate_predictions[:n]

scores['across all appliances'] = {
    'disaggregation': {
        'total_energy_correctly_assigned': total_energy_correctly_assigned
    },
    'regression': {
        'explained_variance_score': float(
            metrics.explained_variance_score(mains, aggregate_predictions)),
        'mean_absolute_error': float(
            np.mean(
                [scores[app]['regression']['mean_absolute_error']
                 for app in scores]))
    },
    'classification': {
        metric: float(np.mean([scores[app]['classification'][metric]
                               for app in scores]))
        for metric in METRICS['classification']}
}

print()
print(yaml.dump(scores, default_flow_style=False))

metrics_filename = join(BASE_DIRECTORY, 'metric_scores.yaml')
print("Saving to", metrics_filename)
with open(metrics_filename, 'w') as fh:
    yaml.dump(scores, stream=fh, default_flow_style=False)

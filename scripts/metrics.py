from __future__ import print_function, division
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import yaml  # for pretty-printing dict
from neuralnilm.metrics import run_metrics, across_all_appliances

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


scores = {}
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

    mains = load('mains')
    scores[appliance] = run_metrics(y_true, y_pred, mains)

scores = across_all_appliances(scores, mains, aggregate_predictions)
print()
print(yaml.dump(scores, default_flow_style=False))

metrics_filename = join(BASE_DIRECTORY, 'metric_scores.yaml')
print("Saving to", metrics_filename)
with open(metrics_filename, 'w') as fh:
    yaml.dump(scores, stream=fh, default_flow_style=False)

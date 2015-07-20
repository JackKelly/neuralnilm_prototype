from __future__ import print_function, division
import numpy as np
from os.path import join, expanduser
import matplotlib.pyplot as plt
import yaml  # for pretty-printing dict
from neuralnilm.metrics import run_metrics, across_all_appliances

from disag_566 import APPLIANCES, OUTPUT_PATH

# sklearn evokes warnings from numpy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

ESTIMATES_PATH = OUTPUT_PATH
GROUND_TRUTH_PATH = expanduser(
    "~/PhD/experiments/neural_nilm/data_for_BuildSys2015/ground_truth_and_mains")


def load(architecture, building_i, appliance):
    # load estimates
    estimates_fname = "{}_building_{}_estimates_{}.csv".format(
        architecture, building_i, appliance)
    estimates_fname = join(ESTIMATES_PATH, estimates_fname)
    y_pred = np.loadtxt(estimates_fname, delimiter=',')

    # load ground truth
    y_true_fname = "building_{}_{}.csv".format(building_i, appliance)
    y_true_fname = join(GROUND_TRUTH_PATH, y_true_fname)
    y_true = np.loadtxt(y_true_fname, delimiter=',')

    # load mains
    mains_fname = "building_{}_mains.csv".format(building_i)
    mains_fname = join(GROUND_TRUTH_PATH, mains_fname)
    mains = np.loadtxt(mains_fname, delimiter=',')

    return y_true, y_pred, mains


def plot_all(y_true, y_pred, mains, title=None):
    fig, axes = plt.subplots(nrows=3, sharex=True)
    axes[0].plot(y_pred)
    axes[0].set_title('y_pred')
    axes[1].plot(y_true)
    axes[1].set_title('y_true')
    axes[2].plot(mains)
    axes[2].set_title('mains')
    if title:
        fig.set_title(title)
    plt.show()
    return fig, axes


def calculate_metrics():
    scores = {}
    for architecture in ['ae', 'rectangles']:
        scores[architecture] = {}
        for appliance, buildings in APPLIANCES:
            scores[architecture][appliance] = {}
            aggregate_predictions = None
            for building_i in buildings:
                y_true, y_pred, mains = load(
                    architecture, building_i, appliance)

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

                scores[architecture][appliance][building_i] = run_metrics(
                    y_true, y_pred, mains)

    return scores

            # scores = across_all_appliances(scores, mains, aggregate_predictions)
            # print()
            # print(yaml.dump(scores, default_flow_style=False))

            # metrics_filename = join(BASE_DIRECTORY, 'metric_scores.yaml')
            # print("Saving to", metrics_filename)
            # with open(metrics_filename, 'w') as fh:
            #     yaml.dump(scores, stream=fh, default_flow_style=False)

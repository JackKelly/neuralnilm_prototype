from __future__ import print_function, division
import numpy as np
import pandas as pd
from os.path import join
import nilmtk
from nilmtk.disaggregate import CombinatorialOptimisation, fhmm_exact
from neuralnilm.metrics import run_metrics, across_all_appliances
import yaml
import matplotlib.pyplot as plt

ukdale = nilmtk.DataSet('/data/mine/vadeec/merged/ukdale.h5')
#ukdale.set_window("2013-04-12", "2014-12-10")
ukdale.set_window("2013-04-12", "2013-05-12")
elec = ukdale.buildings[1].elec

BASE_DIRECTORY = '/home/jack/experiments/neuralnilm/figures/'
PLOT = True

EXPERIMENT_DIRECTORIES = {
    'fridge freezer': 'e544a',
    'washer dryer': 'e545a',
    'kettle': 'e545b',
    'HTPC': 'e545c',
    'dish washer': 'e545d'
}

APPLIANCES = [
    'fridge freezer',
    'washer dryer',
    'kettle',
    'HTPC',
    'dish washer'
]

meters = []
for appliance in APPLIANCES:
    meter = elec[appliance]
    meters.append(meter)
meters = nilmtk.MeterGroup(meters)

scores = {}

MAINS_APPLIANCES = ['washer dryer', 'HTPC', 'dish washer']
mains = np.load(join(BASE_DIRECTORY, 'e545a/mains.npy'))
mains = pd.DataFrame(mains)

fridge_mains = np.load(join(BASE_DIRECTORY, 'e544a/mains.npy'))
fridge_mains = pd.DataFrame(fridge_mains)

kettle_mains = np.load(join(BASE_DIRECTORY, 'e545b/mains.npy'))
kettle_mains = pd.DataFrame(kettle_mains)


def co_disag():
    # TRAIN
    disag = CombinatorialOptimisation()
    disag.train(meters)

    # TEST
    appliance_powers = disag.disaggregate_chunk(mains)
    for i, df in appliance_powers.iteritems():
        appliance = disag.model[i]['training_metadata'].dominant_appliance()
        appliance_type = appliance.identifier.type
        y_pred = df.values
        if appliance_type in MAINS_APPLIANCES:
            np.save(join(BASE_DIRECTORY, 'CO_' + appliance_type), y_pred)

    appliance_powers = disag.disaggregate_chunk(fridge_mains)
    for i, df in appliance_powers.iteritems():
        appliance = disag.model[i]['training_metadata'].dominant_appliance()
        appliance_type = appliance.identifier.type
        y_pred = df.values
        if appliance_type in ['fridge freezer']:
            np.save(join(BASE_DIRECTORY, 'CO_' + appliance_type), y_pred)

    appliance_powers = disag.disaggregate_chunk(kettle_mains)
    for i, df in appliance_powers.iteritems():
        appliance = disag.model[i]['training_metadata'].dominant_appliance()
        appliance_type = appliance.identifier.type
        y_pred = df.values
        if appliance_type in ['kettle']:
            np.save(join(BASE_DIRECTORY, 'CO_' + appliance_type), y_pred)


def co_metrics():
    scores['CO'] = {}
    aggregate_predictions = None
    for appliance_type in APPLIANCES:
        y_pred = np.load(join(BASE_DIRECTORY, 'CO_' + appliance_type + '.npy'))
        y_true_fname = join(
            BASE_DIRECTORY, EXPERIMENT_DIRECTORIES[appliance_type],
            'targets.npy')
        y_true = np.load(y_true_fname)
        n = min(len(y_true), len(y_pred))
        y_true = y_true[:n]
        y_pred = y_pred[:n]

        if appliance_type in ['fridge freezer']:
            m = fridge_mains
        elif appliance_type in ['kettle']:
            m = kettle_mains
        else:
            m = mains

        if PLOT:
            fig, axes = plt.subplots(nrows=2, sharex=True)
            axes[0].plot(y_true[:20000], label='y_true')
            axes[0].plot(y_pred[:20000], label='y_pred')
            axes[0].set_title(appliance_type)
            axes[0].legend()
            axes[1].plot(m.values[:20000], label='mains')
            axes[1].set_title('Mains')
            plt.show(block=True)

        scores['CO'][appliance_type] = run_metrics(
            y_true, y_pred, m.values)

        if aggregate_predictions is None:
            aggregate_predictions = y_pred
        else:
            n_agg = min(len(aggregate_predictions), len(y_pred))
            aggregate_predictions = aggregate_predictions[:n_agg]
            aggregate_predictions += y_pred[:n_agg]

    scores['CO'] = across_all_appliances(
        scores['CO'], mains, aggregate_predictions)


def fhmm_disag():
    # TRAIN
    disag = fhmm_exact.FHMM()
    disag.train(meters)

    # TEST
    appliance_powers = disag.disaggregate_chunk(mains)
    for meter, df in appliance_powers.iteritems():
        appliance = meter.dominant_appliance()
        appliance_type = appliance.identifier.type
        if appliance_type in MAINS_APPLIANCES:
            y_pred = df.values
            np.save(join(BASE_DIRECTORY, 'FHMM_' + appliance_type), y_pred)

    appliance_powers = disag.disaggregate_chunk(fridge_mains)
    for meter, df in appliance_powers.iteritems():
        appliance = meter.dominant_appliance()
        appliance_type = appliance.identifier.type
        if appliance_type in ['fridge freezer']:
            y_pred = df.values
            np.save(join(BASE_DIRECTORY, 'FHMM_' + appliance_type), y_pred)

    appliance_powers = disag.disaggregate_chunk(kettle_mains)
    for meter, df in appliance_powers.iteritems():
        appliance = meter.dominant_appliance()
        appliance_type = appliance.identifier.type
        if appliance_type in ['kettle']:
            y_pred = df.values
            np.save(join(BASE_DIRECTORY, 'FHMM_' + appliance_type), y_pred)


def fhmm_metrics():
    # METRICS
    scores['FHMM'] = {}
    aggregate_predictions = None
    for appliance_type in APPLIANCES:
        y_pred = np.load(
            join(BASE_DIRECTORY, 'FHMM_' + appliance_type + '.npy'))
        y_true_fname = join(
            BASE_DIRECTORY, EXPERIMENT_DIRECTORIES[appliance_type],
            'targets.npy')
        y_true = np.load(y_true_fname)
        n = min(len(y_true), len(y_pred))
        y_true = y_true[:n]
        y_pred = y_pred[:n]

        if appliance_type in ['fridge freezer']:
            m = fridge_mains
        elif appliance_type in ['kettle']:
            m = kettle_mains
        else:
            m = mains

        if PLOT:
            fig, axes = plt.subplots(nrows=2, sharex=True)
            axes[0].plot(y_true[:20000], label='y_true')
            axes[0].plot(y_pred[:20000], label='y_pred')
            axes[0].set_title(appliance_type)
            axes[0].legend()
            axes[1].plot(m.values[:20000], label='mains')
            axes[1].set_title('Mains')
            plt.show(block=True)

        scores['FHMM'][appliance_type] = run_metrics(
            y_true, y_pred, m.values)

        if aggregate_predictions is None:
            aggregate_predictions = y_pred
        else:
            n_agg = min(len(aggregate_predictions), len(y_pred))
            aggregate_predictions = aggregate_predictions[:n_agg]
            aggregate_predictions += y_pred[:n_agg]

    scores['FHMM'] = across_all_appliances(
        scores['FHMM'], mains, aggregate_predictions)


def always_off():
    # METRICS
    scores['Always off'] = {}
    aggregate_predictions = None
    for appliance in APPLIANCES:
        y_true_fname = join(
            BASE_DIRECTORY, EXPERIMENT_DIRECTORIES[appliance],
            'targets.npy')
        y_true = np.load(y_true_fname)
        n = len(y_true)
        y_pred = np.zeros(n)
        scores['Always off'][appliance] = run_metrics(
            y_true, y_pred, mains.values)

        if aggregate_predictions is None:
            aggregate_predictions = y_pred
        else:
            n_agg = min(len(aggregate_predictions), len(y_pred))
            aggregate_predictions = aggregate_predictions[:n_agg]
            aggregate_predictions += y_pred[:n_agg]

    scores['Always off'] = across_all_appliances(
        scores['Always off'], mains, aggregate_predictions)


def mean():
    # METRICS
    ALGO = 'Mean'
    scores[ALGO] = {}
    aggregate_predictions = None
    for appliance in APPLIANCES:
        y_true_fname = join(
            BASE_DIRECTORY, EXPERIMENT_DIRECTORIES[appliance],
            'targets.npy')
        y_true = np.load(y_true_fname)
        n = len(y_true)
        y_pred = np.zeros(n) + y_true.mean()
        print(appliance, y_true.mean())
        scores[ALGO][appliance] = run_metrics(
            y_true, y_pred, mains.values)

        if aggregate_predictions is None:
            aggregate_predictions = y_pred
        else:
            n_agg = min(len(aggregate_predictions), len(y_pred))
            aggregate_predictions = aggregate_predictions[:n_agg]
            aggregate_predictions += y_pred[:n_agg]

    scores[ALGO] = across_all_appliances(
        scores[ALGO], mains, aggregate_predictions)


# LOAD YAML IF NECESSARY
with open(join(BASE_DIRECTORY, 'benchmark_scores.yaml'), 'r') as fh:
    scores = yaml.load(fh)

# Run
# co_disag()
fhmm_disag()
# co_metrics()
fhmm_metrics()
always_off()
mean()


print()
print(yaml.dump(scores, default_flow_style=False))

metrics_filename = join(BASE_DIRECTORY, 'benchmark_scores.yaml')
print("Saving to", metrics_filename)
with open(metrics_filename, 'w') as fh:
    yaml.dump(scores, stream=fh, default_flow_style=False)

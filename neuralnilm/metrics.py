from __future__ import print_function, division
import numpy as np
import sklearn.metrics as metrics


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


def run_metrics(y_true, y_pred, mains, on_power_threshold=4):
    """
    Parameters
    ----------
    on_power_threshold : int
    """
    y_true[y_true <= on_power_threshold] = 0
    y_true_class = y_true > on_power_threshold
    y_pred_class = y_pred > on_power_threshold

    ARGS = {
        'classification': '(y_true_class, y_pred_class)',
        'regression': '(y_true, y_pred)'
    }

    scores = {}
    for metric_type, metric_list in METRICS.iteritems():
        args = ARGS[metric_type]
        for metric in metric_list:
            score = eval('metrics.' + metric + args)
            scores[metric] = float(score)

    sum_y_true = np.sum(y_true)
    sum_y_pred = np.sum(y_pred)
    # negative means underestimates
    relative_error_in_total_energy = float(
        (sum_y_pred - sum_y_true) / max(sum_y_true, sum_y_pred))

    # For total energy correctly assigned
    denominator = 2 * np.sum(mains)
    abs_diff = np.fabs(y_pred - y_true)
    sum_abs_diff = np.sum(abs_diff)
    total_energy_correctly_assigned = 1 - (sum_abs_diff / denominator)
    total_energy_correctly_assigned = float(total_energy_correctly_assigned)

    scores.update({
        'relative_error_in_total_energy': relative_error_in_total_energy,
        'total_energy_correctly_assigned': total_energy_correctly_assigned,
        'sum_abs_diff': float(sum_abs_diff)
    })

    return scores


def across_all_appliances(scores, mains, aggregate_predictions):
    total_sum_abs_diff = 0.0
    for appliance_scores in scores.values():
        total_sum_abs_diff += appliance_scores['sum_abs_diff']

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
        'total_energy_correctly_assigned': total_energy_correctly_assigned,
        'explained_variance_score': float(
            metrics.explained_variance_score(mains, aggregate_predictions)),
        'mean_absolute_error': float(
            np.mean(
                [scores[app]['mean_absolute_error']
                 for app in scores])),
        'relative_error_in_total_energy': float(
            np.mean(
                [scores[app]['relative_error_in_total_energy']
                 for app in scores])),
    }
    scores['across all appliances'].update({
        metric: float(np.mean([scores[app][metric] for app in scores]))
        for metric in METRICS['classification']
    })

    return scores

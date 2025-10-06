import numpy as np
from sklearn import metrics

SUPPORTED_METRICS = ['-log_loss', 'roc_auc']

def calc_metrics(metric_name, y_true, y_scores: list) -> list:
    return np.array(
        [calc_metric(metric_name, y_true, y_scores[i]) for i in range(len(y_scores))]
    )

def calc_metric(metric_name, y_true, y_score) -> float:
    if metric_name == '-log_loss':
        m = -metrics.log_loss(y_true=y_true, y_pred=y_score)
    elif metric_name == 'roc_auc':
        m = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
    else:
        raise ValueError('Metric not supported.')
    return m

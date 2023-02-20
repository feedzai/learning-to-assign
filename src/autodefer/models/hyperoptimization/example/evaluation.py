import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def calc_tpr_at_fpr(y_score, y_true, target_fpr):
    y_pred = predict_with_threshold(
        y_score=y_score,
        threshold=calc_threshold_at_fpr(
            y_true=y_true, y_score=y_score, fpr=target_fpr
        )
    )
    tn, fp, fn, tp = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=[0, 1]).ravel()

    fpr = fp / (tn + fp)
    tpr = tp / (tp + fn)

    if fpr > target_fpr:  # classifiers with uniform scores around the threshold ruin the thresholding
        tpr = 0

    return tpr

# UTILS
def predict_with_threshold(y_score: np.ndarray, threshold: float):
    """
    Thresholds predicted probability vector (positives above threshold, negatives below).
    Returns a numpy ndarray with dtype=int.
    """
    return (y_score >= threshold).astype(int)

def calc_threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr: float):
    """
    Calculates (positive) predicted probability threshold to achieve a given FPR on the test set.
    """

    temp_df = pd.DataFrame(
        {'y_true': y_true,
         'pred_proba': y_score,
         })
    temp_df = temp_df.sort_values(by='pred_proba', ascending=False, ignore_index=True)
    temp_df['pseudo_fpr'] = (temp_df['y_true']
                             .apply(lambda x: 1 if x == 0 else 0)
                             .cumsum()
                             .divide(sum(y_true == 0)))

    critical_threshold = float((
        temp_df
        .loc[temp_df['pseudo_fpr'] < fpr, 'pred_proba']  # < fpr ensures the fpr in case of several tied scores
        .iloc[-1]))

    return critical_threshold

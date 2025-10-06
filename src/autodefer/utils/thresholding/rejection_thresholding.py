"""
All thresholding extensions for rejection learning.
"""
import numpy as np

from autodefer.utils.thresholding import calc_threshold_at_fpr


def predict_and_reject_with_thresholds(
        y_score: np.ndarray,
        negat_threshold: float,
        posit_threshold: float):
    assert negat_threshold < posit_threshold

    pred_posit_ix = (y_score >= posit_threshold)
    pred_negat_ix = (y_score <= negat_threshold)
    y_rej = (~pred_posit_ix) & (~pred_negat_ix)

    y_pred = y_score.copy()
    y_pred[pred_posit_ix] = 1
    y_pred[pred_negat_ix] = 0
    y_pred[y_rej] = -1

    return y_pred, y_rej


def calc_rejection_threshold_at_coverage(
        confidence: np.ndarray,
        coverage: float):
    n = confidence.shape[0]
    n_covered = int(n * coverage) + 1

    # np.sort is ascending. Minus sign in arg makes it descending.
    # Second minus sign cancels it out:
    threshold = - np.sort(-confidence)[n_covered]

    return threshold


def calc_thresholds_at_cov_at_fpr(
        y_true: np.ndarray,
        y_score: np.ndarray,
        coverage: float,
        fpr: float):
    sorted_score = np.sort(y_score)  # ascending
    n_rej = int((1 - coverage) * y_true.shape[0])

    t_posit = calc_threshold_at_fpr(y_true, y_score, fpr)
    t_posit_ix = np.where(sorted_score >= t_posit)[0][0]
    t_negat = sorted_score[t_posit_ix - n_rej]

    return t_negat, t_posit

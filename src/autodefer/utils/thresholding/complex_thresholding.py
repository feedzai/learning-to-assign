"""
Helper functions to make predictions ensuring a given metric.
"""

import numpy as np

from autodefer.utils.thresholding import (
    calc_threshold_at_fpr,
    calc_threshold_at_pp,
    calc_thresholds_at_pp_by_group,
    calc_cost_at_threshold,
    predict_with_different_thresholds_by_group,
    predict_with_threshold,
)


def predict_at_fpr(
        y_true: np.ndarray, y_score: np.ndarray, fpr: float):
    threshold_at_fpr = calc_threshold_at_fpr(
        y_true=y_true, y_score=y_score, fpr=fpr)

    predictions = predict_with_threshold(
        y_score=y_score, threshold=threshold_at_fpr)

    return predictions


def predict_at_pp(y_score: np.ndarray, pp: float):
    threshold_at_pp = calc_threshold_at_pp(y_score=y_score, pp=pp)

    predictions = predict_with_threshold(
        y_score=y_score, threshold=threshold_at_pp)

    return predictions


def predict_at_pp_by_group(
        y_true: np.ndarray,
        y_score: np.ndarray,
        groups_arr: np.ndarray):
    thresholds_dict = calc_thresholds_at_pp_by_group(
        y_true=y_true,
        y_score=y_score,
        groups_arr=groups_arr
    )
    y_pred = predict_with_different_thresholds_by_group(
        y_score=y_score,
        groups_arr=groups_arr,
        thresholds_dict=thresholds_dict
    )

    return y_pred

def calc_cost_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr: float):
    return calc_cost_at_threshold(
        y_true=y_true,
        y_score=y_score,
        threshold=calc_threshold_at_fpr(y_true=y_true, y_score=y_score, fpr=fpr)
    )

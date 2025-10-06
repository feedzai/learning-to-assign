"""
Helper functions to:
- make predictions based on thresholds
- calculate thresholds that ensure a given metric
Note: functions that aggregate these two components are in .complex_thresholding
"""
import numpy as np
import pandas as pd


def predict_with_threshold(y_score: np.ndarray, threshold: float):
    """
    Thresholds predicted probability vector (positives above threshold, negatives below).
    Returns a numpy ndarray with dtype=int.
    """
    return (y_score >= threshold).astype(int)


def predict_with_different_thresholds_by_group(
        y_score: np.ndarray,
        groups_arr: np.ndarray,
        thresholds_dict: dict
):
    """
    Thresholds predicted probability vector according to different thresholds by specified keys.
    :param y_score: numpy array with model scores
    :param groups_arr: numpy array containing the keys to the thresholds dictionary
    :param thresholds_dict: dictionary matching keys to thresholds to be used
    :return:
    """
    get_threshold = np.vectorize(lambda x: thresholds_dict[x])
    thresholds_arr = get_threshold(groups_arr)
    y_pred = (y_score >= thresholds_arr).astype(int)

    return y_pred


def calc_threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr: float):
    """
    Calculates (positive) predicted probability threshold to achieve a given FPR on the test set.
    """

    temp_df = pd.DataFrame(
        {'y_true': y_true,
         'y_score': y_score,
         })
    temp_df = temp_df.sort_values(by='y_score', ascending=False, ignore_index=True)
    temp_df['pseudo_fpr'] = (temp_df['y_true']
                             .apply(lambda x: 1 if x == 0 else 0)
                             .cumsum()
                             .divide(sum(y_true == 0)))

    critical_threshold = float((
        temp_df
        .loc[temp_df['pseudo_fpr'] < fpr, 'y_score']  # < fpr ensures the fpr in case of several tied scores
        .iloc[-1]))

    return critical_threshold


def calc_threshold_at_pp(y_score: np.ndarray, pp: float):
    """
    Calculates (positive) predicted probability threshold to achieve a given predicted positive rate on the test set.
    """

    n_pp = int(y_score.shape[0] * (1 - pp))

    critical_threshold = np.sort(y_score)[n_pp]  # default is ascending

    return critical_threshold


def calc_thresholds_at_pp_by_group(
        y_true: np.ndarray,
        y_score: np.ndarray,
        groups_arr: np.ndarray,
        return_df: bool = False
):
    def calc_pp_and_threshold(x):
        pp = x['label'].sum() / x.shape[0]
        threshold = calc_threshold_at_pp(x['score'], pp=pp)
        return pd.Series({'pp': pp, 'threshold': threshold})

    data = pd.DataFrame({
        'label': y_true,
        'score': y_score,
        'group': groups_arr})

    pp_and_threshold = data.groupby('group').apply(lambda x: calc_pp_and_threshold(x))
    thresholds_dict = pp_and_threshold['threshold'].to_dict()
    if return_df:
        return thresholds_dict, pp_and_threshold
    else:
        return thresholds_dict

def calc_threshold_with_cost(y_true: np.ndarray, y_score: np.ndarray, fp_fn_cost_ratio: float):
    """
    Calculates (positive) predicted probability threshold to achieve a given FPR on the test set.
    """
    temp_df = pd.DataFrame(
        {'y_true': y_true,
         'y_score': y_score,
         })
    temp_df = temp_df.sort_values(by='y_score', ascending=False, ignore_index=True)
    temp_df['fp'] = (
        temp_df['y_true']
        .apply(lambda x: 1 if x == 0 else 0)
        .cumsum()
    )
    temp_df['fn'] = (
        temp_df['y_true'].sum()
        - temp_df['y_true'].cumsum()
    )
    temp_df['cost'] = fp_fn_cost_ratio * temp_df['fp'] + temp_df['fn']

    critical_threshold = temp_df.loc[temp_df['cost'].argmin(), 'y_score']

    return critical_threshold

def calc_cost_at_threshold(
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold,
        width=0.01,
):
    # (a.FP + FN)'=0 <=> aFP' + FN' = 0 <=> a = -FN'/FP' (derivative in order to threshold)
    temp_df = pd.DataFrame(
        {'y_true': y_true,
         'y_score': y_score,
         })
    temp_df = temp_df.sort_values(by='y_score', ascending=False, ignore_index=True)
    temp_df['fp'] = (
        temp_df['y_true']
        .apply(lambda x: 1 if x == 0 else 0)
        .cumsum()
    )
    temp_df['fn'] = (
        temp_df['y_true'].sum()
        - temp_df['y_true'].cumsum()
    )

    left = (temp_df['y_score'] - (threshold - width/2)).abs().argsort()[0]
    right = (temp_df['y_score'] - (threshold + width/2)).abs().argsort()[0]
    delta_fn = (
        (temp_df.loc[right, 'fn'] - temp_df.loc[left, 'fn'])
        / (temp_df.loc[right, 'y_score'] - temp_df.loc[left, 'y_score'])
    )
    delta_fp = (
        (temp_df.loc[right, 'fp'] - temp_df.loc[left, 'fp'])
        / (temp_df.loc[right, 'y_score'] - temp_df.loc[left, 'y_score'])
    )

    fp_fn_cost_ratio = - delta_fn / delta_fp

    return fp_fn_cost_ratio

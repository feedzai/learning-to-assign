# %% imports
import numpy as np
import pandas as pd
import yaml

from autodefer.utils.ClassificationEvaluator import ClassificationEvaluator  # noqa: E402
from autodefer.utils.RejectionEvaluator import RejectionEvaluator  # noqa: E402


def display_results(exp_name, short=False):

    set_df_display_settings()

    results_path = '~/results/'

    proj_rel_path = 'learning-to-defer/rejection_learning/'
    exp_rel_path = f'{proj_rel_path}{exp_name}/'

    # evaluation
    clf_val_eval = ClassificationEvaluator(
        filepath=results_path + exp_rel_path + 'clf_val_results.csv',
        exp_id_cols=['predictor'],
    )
    clf_eval = ClassificationEvaluator(
        filepath=results_path + exp_rel_path + 'clf_results.csv',
        exp_id_cols=['predictor'],
    )
    clf_rethr_eval = ClassificationEvaluator(  # guarantees constraints in test set
        filepath=results_path + exp_rel_path + 'clf_rethr_results.csv',
        exp_id_cols=['predictor'],
    )
    rej_val_eval = RejectionEvaluator(
        filepath=results_path + exp_rel_path + 'rej_val_results.csv',
        exp_id_cols=['predictor', 'rej_class', 'rej_params'],
    )
    rej_eval = RejectionEvaluator(
        filepath=results_path + exp_rel_path + 'rej_results.csv',
        exp_id_cols=['predictor', 'rej_class', 'rej_params'],
    )
    rej_rethr_eval = RejectionEvaluator(  # guarantees constraints in test set
        filepath=results_path + exp_rel_path + 'rej_rethr_results.csv',
        exp_id_cols=['predictor', 'rej_class', 'rej_params'],
    )

    print('\nFinal Results')

    print('\nClassification Task Results on Val Set')
    clf_val_eval.display_results(short=short)

    print('\nRejection Task Results on Val Set')
    rej_val_eval.display_results(short=short)

    print('\nClassification Task Results on Test Set')
    clf_eval.display_results(short=short)

    print('\nRejection Task Results on Test Set')
    rej_eval.display_results(short=short)

    print('\nClassification Task Results on Test Set (rethresholded)')
    clf_rethr_eval.display_results(short=short)

    print('\nRejection Task Results on Test Set (rethresholded)')
    rej_rethr_eval.display_results(short=short)


def set_df_display_settings():
    width = 450
    pd.set_option('display.width', width)
    np.set_printoptions(linewidth=width)
    pd.set_option('display.max_columns', 25)


if __name__ == '__main__':
    with open('configs/exp_config.yaml', 'r') as infile:
        exp_config = yaml.safe_load(infile)

    exp_name = exp_config['name']
    display_results(exp_name=exp_name, short=False)

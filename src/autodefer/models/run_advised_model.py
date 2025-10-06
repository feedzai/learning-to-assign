import os
from datetime import datetime

import numpy as np
import pandas as pd

from autodefer.models import outlier_models, run_lgbm


def make_advised_model_sets(
        X_train, X_train_val1, X_val2, X_test,
        weak_val1_score, weak_val2_score, weak_test_score,
        lof_val1_score, lof_val2_score, lof_test_score,
        iso_forest_val1_score, iso_forest_val2_score, iso_forest_test_score,
        without_score=False
):
    lof_val1_score_na_train = np.concatenate(
        (
            np.full(shape=(X_train.shape[0],), fill_value=-1),
            lof_val1_score
        ),
        axis=0
    )
    iso_forest_val1_score_na_train = np.concatenate(
        (
            np.full(shape=(X_train.shape[0],), fill_value=1),  # fill_value=1 since iso_score in [-1, 0]
            iso_forest_val1_score
        ),
        axis=0
    )
    weak_val1_score_na_train = np.concatenate(
        (
            np.full(shape=(X_train.shape[0],), fill_value=-1),
            weak_val1_score
        ),
        axis=0
    )

    adv_X_train_val1 = np.concatenate(  # advised's X_train
        (
            X_train_val1,
            weak_val1_score_na_train.reshape(-1, 1),
            lof_val1_score_na_train.reshape(-1, 1),
            iso_forest_val1_score_na_train.reshape(-1, 1)
        ),
        axis=1
    )
    adv_X_val2 = np.concatenate(  # advised's X_val
        (
            X_val2,
            weak_val2_score.reshape(-1, 1),
            lof_val2_score.reshape(-1, 1),
            iso_forest_val2_score.reshape(-1, 1)
        ),
        axis=1
    )
    adv_X_test = np.concatenate(  # advised's X_test
        (
            X_test,
            weak_test_score.reshape(-1, 1),
            lof_test_score.reshape(-1, 1),
            iso_forest_test_score.reshape(-1, 1)
        ),
        axis=1
    )

    arrs = (adv_X_train_val1, adv_X_val2, adv_X_test)

    if without_score:  # on the fly solution for a control; TODO refactor
        new_arrs_list = []
        for a in arrs:
            new_arrs_list.append(
                np.delete(a, -3, axis=1)  # removes score column
            )
        new_arrs = tuple(new_arrs_list)
        arrs = new_arrs

    return arrs


def run_alt_advised_model(
        data_path, models_path, exp_rel_path,
        X_train_val, y_train_val,
        X_val2, X_test,
        y_val2,
        exp_config,
        lgbm_param_grid,
        logs_path
):
    def timestamp_to_datetime(s: pd.Series):
        s_datetime = s.apply(
            lambda x: datetime.fromtimestamp(int(x) / 1000))
        return s_datetime

    train = pd.read_csv(data_path + 'wbe-data.joao.veiga.train.csv', index_col=0)

    train['datetime'] = timestamp_to_datetime(train['timestamp'])

    alt_train_ix = (train['datetime'] < '2016-05-01')
    alt_val1_ix = (train['datetime'] >= '2016-05-01') & (train['datetime'] < '2016-07-01')

    alt_X_train = X_train_val[alt_train_ix]
    alt_X_train_val1 = X_train_val[alt_train_ix | alt_val1_ix]
    alt_X_val1 = X_train_val[alt_val1_ix]

    alt_y_train_val1 = y_train_val[alt_train_ix | alt_val1_ix]
    alt_y_train = y_train_val[alt_train_ix]
    alt_y_val1 = y_train_val[alt_val1_ix]

    print(f'alt_X_train shape: {alt_X_train.shape}')
    print(f'alt_X_val1 shape: {alt_X_val1.shape}')

    # local outlier factor
    lof_name = 'alt_lof'
    lof_model_path = f'{models_path + exp_rel_path + lof_name}.pickle'
    lofs_dir = f'{models_path + exp_rel_path}scores/{lof_name}/'
    os.makedirs(lofs_dir, exist_ok=True)

    lof_model = outlier_models.ScaledLocalOutlierFactor(
        lof_model_path=lof_model_path,
        n_jobs=exp_config['n_jobs'],
        **exp_config['LOF'])  # TODO change from kwargs to params dict

    lof_model.fit(alt_X_train)

    lof_val1_score = lof_model.score_samples(
        X=alt_X_val1,
        scores_path=f'{lofs_dir}lof_val1_score.npy'
    )
    lof_val2_score = lof_model.score_samples(
        X=X_val2,
        scores_path=f'{lofs_dir}lof_val2_score.npy'
    )
    lof_test_score = lof_model.score_samples(
        X=X_test,
        scores_path=f'{lofs_dir}lof_test_score.npy'
    )

    # isolation forest
    iso_forest_name = 'alt_isolation_forest'
    iso_forest_model_path = f'{models_path + exp_rel_path}{iso_forest_name}.pickle'
    iso_forest_scores_dir = f'{models_path + exp_rel_path}scores/{iso_forest_name}/'
    os.makedirs(iso_forest_scores_dir, exist_ok=True)

    iso_forest = outlier_models.train_isolation_forest(
        model_path=iso_forest_model_path,
        X_train=alt_X_train,
        params=exp_config['isolation_forest'],
        n_jobs=exp_config['n_jobs'],
        random_seed=exp_config['random_seed'],
        logs_path=logs_path,
    )

    iso_forest_val1_score = outlier_models.score_with_isolation_forest(
        model=iso_forest,
        X=alt_X_val1,
        scores_path=f'{iso_forest_scores_dir}{iso_forest_name}_val1_score.npy'
    )
    iso_forest_val2_score = outlier_models.score_with_isolation_forest(
        model=iso_forest,
        X=X_val2,
        scores_path=f'{iso_forest_scores_dir}{iso_forest_name}_val2_score.npy'
    )
    iso_forest_test_score = outlier_models.score_with_isolation_forest(
        model=iso_forest,
        X=X_test,
        scores_path=f'{iso_forest_scores_dir}{iso_forest_name}_test_score.npy'
    )

    # weak classifier

    weak_name = 'alt_weak_light_gbm'
    weak_model_path = f'{models_path + exp_rel_path + weak_name}.pickle'
    weak_best_params_path = f'{models_path + exp_rel_path}params/{weak_name}_params.yaml'
    weak_optuna_trials_path = f'{models_path + exp_rel_path}optuna_trials/{weak_name}_trials.csv'

    weak_params = run_lgbm.tune_lgbm_params(
        best_params_path=weak_best_params_path,  # loads from if saved; saves to otherwise
        optuna_trials_path=weak_optuna_trials_path,  # saves performance by trial plot
        logs_path=logs_path,  # registers running times in
        X_train=alt_X_train, y_train=alt_y_train,
        X_val=alt_X_val1, y_val=alt_y_val1,
        n_trials=exp_config['n_trials'],
        param_grid_dict=lgbm_param_grid,
        target_metric='tpr@fpr',
        target_constraint=exp_config["target_fpr"],
        random_seed=exp_config['random_seed'],
        n_jobs=exp_config['n_jobs']
    )

    weak = run_lgbm.train_lgbm(
        model_path=weak_model_path,  # loads from if saved; saves to otherwise
        logs_path=logs_path,
        X_train=alt_X_train, y_train=alt_y_train,
        params=weak_params,
        random_seed=exp_config['random_seed'],
        deterministic=exp_config['deterministic'],
        n_jobs=exp_config['n_jobs']
    )
    weak_val1_score = weak.predict_proba(alt_X_val1)[:, weak.classes_ == 1].squeeze()
    weak_val2_score = weak.predict_proba(X_val2)[:, weak.classes_ == 1].squeeze()
    weak_test_score = weak.predict_proba(X_test)[:, weak.classes_ == 1].squeeze()

    alt_adv_X_train_val1, alt_adv_X_val2, alt_adv_X_test = make_advised_model_sets(
        alt_X_train, alt_X_train_val1, X_val2, X_test,
        weak_val1_score, weak_val2_score, weak_test_score,
        lof_val1_score, lof_val2_score, lof_test_score,
        iso_forest_val1_score, iso_forest_val2_score, iso_forest_test_score
    )

    alt_advised_name = 'alt_advised_light_gbm'
    alt_advised_model_path = f'{models_path + exp_rel_path + alt_advised_name}.pickle'
    alt_advised_best_params_path = f'{models_path + exp_rel_path}params/{alt_advised_name}_params.yaml'
    alt_advised_optuna_trials_path = f'{models_path + exp_rel_path}optuna_trials/{alt_advised_name}_trials.csv'

    alt_advised_params = run_lgbm.tune_lgbm_params(
        best_params_path=alt_advised_best_params_path,  # loads from if saved; saves to otherwise
        optuna_trials_path=alt_advised_optuna_trials_path,  # saves performance by trial plot
        logs_path=logs_path,  # registers running times in
        X_train=alt_adv_X_train_val1, y_train=alt_y_train_val1,
        X_val=alt_adv_X_val2, y_val=y_val2,
        n_trials=exp_config['n_trials'],
        param_grid_dict=lgbm_param_grid,
        target_metric='tpr@fpr',
        target_constraint=exp_config["target_fpr"],
        random_seed=exp_config['random_seed'],
        n_jobs=exp_config['n_jobs']
    )
    alt_advised = run_lgbm.train_lgbm(
        model_path=alt_advised_model_path,  # loads from if saved; saves to otherwise
        logs_path=logs_path,
        X_train=alt_adv_X_train_val1, y_train=alt_y_train_val1,
        params=alt_advised_params,
        random_seed=exp_config['random_seed'],
        deterministic=exp_config['deterministic'],
        n_jobs=exp_config['n_jobs']
    )

    return alt_advised, alt_adv_X_val2, alt_adv_X_test

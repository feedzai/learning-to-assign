import os
import pickle
from datetime import datetime

import numpy as np
import yaml
from lightgbm import LGBMClassifier

from autodefer.models import run_lgbm_optuna_tpe


def tune_lgbm_params(
        X_train, y_train, X_val, y_val,
        n_trials: int,
        param_grid_dict: dict,  # specific format as specified in run_lgbm_optuna_tpe
        target_metric: str,
        target_constraint: float,
        random_seed: int = None,
        n_jobs: int = 1,
        best_params_path: str = None,  # loads from if saved; saves to otherwise
        optuna_trials_path: str = None,  # saves performance by trial plot
        logs_path: str = None,  # registers running times in
        ):

    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as infile:
            best_params = yaml.safe_load(infile)
        print(f'Hyperparameters loaded from {best_params_path}')
        return best_params

    start_time = datetime.now()
    best_params = run_lgbm_optuna_tpe(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        n_trials=n_trials,
        param_grid_dict=param_grid_dict,
        target_metric=target_metric, target_constraint=target_constraint,
        n_jobs=n_jobs, random_seed=random_seed,
        trials_path=optuna_trials_path)
    finish_time = datetime.now()

    if best_params_path is not None:
        with open(best_params_path, 'w') as outfile:
            yaml.dump(best_params, outfile)

    # record hyperparameterization time
    duration = finish_time - start_time
    log_msg = (
        f'Hyperparameterization at {best_params_path}: ' +
        f'finish_time={str(finish_time)}, ' +
        f'duration=({str(duration)})'
    )
    with open(logs_path, 'a') as outfile:
        outfile.write(log_msg)

    return best_params


def train_lgbm(
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: dict,
        n_jobs: int = 1,
        random_seed: int = None,
        deterministic: bool = False,
        model_path: str = None,  # loads from if saved; saves to otherwise
        logs_path: str = None  # registers running times in
        ):

    # load model pickle and return it if it has already been saved to model_path
    if os.path.exists(model_path):
        with open(model_path, 'rb') as infile:
            model = pickle.load(infile)
        print(f'Model loaded from {model_path}')
        return model

    start_time = datetime.now()
    params.update({  # "operational" parameters must be reset
        'n_jobs': n_jobs,
        'random_state': random_seed,
        'deterministic': deterministic,
    })
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    finish_time = datetime.now()

    if model_path is not None:
        with open(model_path, 'wb') as outfile:
            pickle.dump(model, outfile)

    # record training time
    duration = finish_time - start_time
    log_msg = (
        f'Training at {model_path}: ' +
        f'finish_time={str(finish_time)}, ' +
        f'duration=({str(duration)})'
    )
    print(log_msg)
    with open(logs_path, 'a') as outfile:
        outfile.write(log_msg)
    
    return model

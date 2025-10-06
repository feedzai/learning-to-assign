import json
import os
import pickle

import pandas as pd
from fairautoml.backend import SklearnModelBackend
from fairautoml.tuners import TPE


def run_fair_automl(
        train: pd.DataFrame, val: pd.DataFrame,
        index_col: str, label_col: str,
        n_trials: int, param_grid: dict,
        alpha=1,
        retrain_on_val=False,
        random_seed=42, verbose=False,
        **kwargs
):
    search_kwargs = {
        'hyperparameter_space': param_grid,
        'n_trials': n_trials,
        'n_jobs': 1,  # total n_jobs = optuna.n_jobs * model.n_jobs
        'verbose': verbose
    }
    tuner_kwargs = {
        'train': train,
        'validation': val,
        'alpha': alpha,
        'seed': random_seed,
        **kwargs
    }
    backend = SklearnModelBackend(
        id_col=index_col,
        label_col=label_col,
    )
    
    tuner = TPE(
        model_backend=backend,
        **tuner_kwargs,
    )
    best_hyperparams = tuner.search(**search_kwargs)
    tuner_results = tuner.results.sort_values("metric_val", ascending=False)
    best_model = tuner.retrain_best_model()

    # current fairautoml version resorts columns in .retrain_best_model(), so we retrain it outside:
    if retrain_on_val:
        retrain_data = pd.concat((train, val), axis=0)
    else:
        retrain_data = train

    best_model.fit(retrain_data.drop(columns=[index_col, label_col]), retrain_data[label_col])

    return best_model, best_hyperparams, tuner_results

def get_fair_automl_hyperopt_model(
        model_path: str,
        best_hyperparams_path: str = None,
        tuner_results_path: str = None,
        **kwargs):
    """
    Loads best model if saved. Runs run_fair_automl and saves best model otherwise.
    :param model_path: path to save model to, or load model from if already saved.
    :param best_hyperparams_path: path to save json with the best hyperparameters.
    :param tuner_results_path: path to save csv with the tuner results.
    :param kwargs: run_fair_automl kwargs.
    :return: sklearn-type estimator of the best model returned from run_fair_automl.
    """
    if os.path.exists(model_path):
        with open(model_path, 'rb') as infile:
            best_model = pickle.load(infile)
    else:
        best_model, best_hyperparams, tuner_results = run_fair_automl(**kwargs)

        with open(model_path, 'wb') as outfile:
            pickle.dump(best_model, outfile)

        if best_hyperparams_path is not None:
            with open(best_hyperparams_path, 'w') as outfile:
                json.dump(best_hyperparams, outfile)

        if tuner_results_path is not None:
            tuner_results.to_csv(tuner_results_path)

    return best_model

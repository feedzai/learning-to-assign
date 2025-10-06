"""
Flexible OptunaTuner class.
"""
import os.path
import pickle
from typing import Union

import numpy as np
import yaml
from optuna import create_study, samplers

from .suggest import suggest_model
from .utils import load_hyperparameter_space


class OptunaTuner:

    def __init__(
            self,
            task: 'binary, multiclass or regression',
            sampler: 'tpe or random' = 'tpe',
            n_startup_trials=10,
            direction='maximize',
            random_seed=None,
            outputs_dir=None
    ):
        self.task = task
        self.sampler = self._get_sampler(sampler, n_startup_trials, random_seed)
        self.direction = direction
        self.outputs_dir = outputs_dir

        os.makedirs(self.outputs_dir, exist_ok=True)

        # records
        self.paths = {
            'y_val': self.outputs_dir + 'y_val.npy',
            'params_hist': self.outputs_dir + 'params_hist.yaml',
            'preds_hist': self.outputs_dir + 'preds_hist.pickle',
            'values_hist': self.outputs_dir + 'values_hist.yaml',
            'best_model': self.outputs_dir + 'best_model.pickle',
        }
        self.y_val = self._load(self.paths['y_val'], kind='npy', default=None)
        self.params_hist = self._load(self.paths['params_hist'], kind='yaml', default=list())
        self.preds_hist = self._load(self.paths['preds_hist'], kind='pickle', default=list())
        self.values_hist = self._load(self.paths['values_hist'], kind='yaml', default=list())
        self.best_model = self._load(self.paths['best_model'], kind='pickle', default=None)

        if not self.values_hist:
            self.study = create_study(direction=self.direction, sampler=self.sampler)
            self.best_value = None
            self.best_model_ix = None
        else:
            self.study = None
            self.best_value = self._get_best_value()
            self.best_model_ix = self._get_best_model_ix()

    def run(
        self,
        X_train, y_train, X_val, y_val,
        hyperparam_space: Union[str, dict],
        n_trials: int,
        evaluation_function=None,
    ):
        if self.values_hist:
            return self.best_model

        self.y_val = y_val  # for records
        self._save(self.y_val, self.paths['y_val'], kind='npy')

        hyperparam_space = load_hyperparameter_space(hyperparam_space)
        self.study.optimize(
            lambda trial: self._objective(
                trial=trial, hyperparam_space=hyperparam_space,
                X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                evaluation_function=evaluation_function,
                ),
            n_trials=n_trials,
        )

        if self.outputs_dir is not None:
            self._save(self.params_hist, self.paths['params_hist'], kind='yaml')
            self._save(self.preds_hist, self.paths['preds_hist'], kind='pickle')
            self._save(self.values_hist, self.paths['values_hist'], kind='yaml')
            self._save(self.best_model, self.paths['best_model'], kind='pickle')

        self.best_model_ix = self._get_best_model_ix()

        return self.best_model

    def _objective(
        self,
        trial, hyperparam_space,
        X_train, y_train, X_val, y_val,
        evaluation_function,
    ):
        params, model = suggest_model(trial=trial, hyperparam_space=hyperparam_space)
        model.fit(X_train, y_train)
        if self.task == 'binary':
            y_pred = model.predict_proba(X_val)[:, 1].squeeze()
        elif self.task == 'multiclass':
            y_pred = model.predict_proba(X_val)
        elif self.task == 'regression':
            y_pred = model.predict(X_val)

        if evaluation_function is not None:
            value = evaluation_function(y_true=y_val, y_pred=y_pred)
            self._record_if_best_model(model, value)
        else:
            value = 0

        self.params_hist.append(params)
        self.preds_hist.append(y_pred)
        self.values_hist.append(float(value))

        return value

    def _record_if_best_model(self, model, value):
        is_best_model = False
        if self.best_value is None:
            is_best_model = True
        else:
            if value > self.best_value:
                is_best_model = True
        
        if is_best_model:
            self.best_value = value
            self.best_model = model

    def _get_best_value(self):
        if self.direction == 'maximize':
            best_value = max(self.values_hist)
        elif self.direction == 'minimize':
            best_value = min(self.values_hist)

        return best_value

    def _get_best_model_ix(self):
        if self.direction == 'maximize':
            best_model_ix = np.argmax(self.values_hist)
        elif self.direction == 'minimize':
            best_model_ix = np.argmin(self.values_hist)

        return best_model_ix

    @staticmethod
    def _get_sampler(sampler, n_startup_trials, seed):
        # some available defaults
        if sampler == 'tpe':
            sampler = samplers.TPESampler(n_startup_trials=n_startup_trials, seed=seed)
        elif sampler == 'random':
            sampler = samplers.RandomSampler(seed=seed)

        return sampler

    @staticmethod
    def _load(path, kind, default=None):
        if os.path.exists(path):
            if kind == 'yaml':
                with open(path, 'r') as infile:
                    obj = yaml.safe_load(infile)
            elif kind == 'pickle':
                with open(path, 'rb') as infile:
                    obj = pickle.load(infile)
            elif kind == 'npy':
                obj = np.load(path, allow_pickle=True)
        else:
            obj = default

        return obj

    @staticmethod
    def _save(obj, path, kind):
        if kind == 'yaml':
            with open(path, 'w') as outfile:
                yaml.dump(obj, outfile)
        elif kind == 'pickle':
            with open(path, 'wb') as outfile:
                pickle.dump(obj, outfile)
        elif kind == 'npy':
            np.save(arr=obj, file=path, allow_pickle=True)

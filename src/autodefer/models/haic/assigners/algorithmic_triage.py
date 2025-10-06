import os
import pickle
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn import metrics

from autodefer.utils import plotting
from autodefer.models import hyperoptimization

from .base import AbstractAssigner
from .assignment_functions import (
    random_assignment,
    optimal_individual_assignment,
    case_by_case_assignment,
    optimal_batch_assignment,
)

class AlgorithmicTriage(AbstractAssigner):
    def __init__(self, expert_ids, outputs_dir, random_seed=None):
        super().__init__(expert_ids)
        self.human_expert_ids = expert_ids['human_ids']
        self.ml_model_ids = expert_ids['model_ids']
        self.outputs_dir = outputs_dir
        self.random_seed = random_seed

        os.makedirs(self.outputs_dir, exist_ok=True)

        # to be set in .fit()
        self.categorical_cols = None
        self.score_col = None
        self.assignment_col = None
        self.ordinal_encoder = None
        self.calibrator = None
        self.expert_model_tuner = None
        self.expert_model = None

    def fit(
            self,
            train: pd.DataFrame,
            categorical_cols: list, score_col: str,
            assignment_col: str, decision_col: str, ground_truth_col: str,
            hyperparam_space, val=None, val_size=0.25, n_trials=100,
            random_seed=42,
    ):
        self.categorical_cols = categorical_cols
        self.score_col = score_col
        self.assignment_col = assignment_col

        # ml model score calibration (unrelated from human expertise model)
        plotting.plot_calibration(
            y_true=train[ground_truth_col],
            y_score=train[score_col],
            bins=15
        )

        calibrator_path = (
            self.outputs_dir + 'calibrator.pickle' if self.outputs_dir is not None
            else None
        )
        if calibrator_path is not None and os.path.exists(calibrator_path):
            with open(calibrator_path, 'rb') as infile:
                self.calibrator = pickle.load(infile)
        else:
            self.calibrator = IsotonicRegression()
            calibrated_ml_model_score = self.calibrator.fit_transform(
                train[decision_col],
                train[ground_truth_col]
            )
            """
            plotting.plot_calibration(
                y_true=train[ground_truth_col],
                y_score=calibrated_ml_model_score,
                bins=15
            )
            """
            if self.outputs_dir is not None:
                with open(calibrator_path, 'wb') as outfile:
                    pickle.dump(self.calibrator, outfile)

        # expert modelling
        experts_train = train[train[self.assignment_col].isin(self.human_expert_ids)]
        experts_train['outcome'] = experts_train.apply(
            lambda x: self._get_outcome(label=x[ground_truth_col], pred=x[decision_col]),
            axis=1,
        )
        experts_train = experts_train.drop(columns=[ground_truth_col, assignment_col, decision_col])

        if val is None:
            experts_train, experts_val = train_test_split(
                experts_train, test_size=val_size, shuffle=False
            )
        else:
            experts_val = val[val[self.assignment_col].isin(self.human_expert_ids)]
            experts_val['outcome'] = experts_val.apply(
                lambda x: self._get_outcome(label=x[ground_truth_col], pred=x[decision_col]),
                axis=1,
            )
            experts_val = experts_val.drop(columns=[ground_truth_col, assignment_col, decision_col])

        # preprocessing
        experts_train = self._transform(experts_train)  # .fit_transform()
        experts_val = self._transform(experts_val)  # .transform()

        self.expert_model_tuner = hyperoptimization.OptunaTuner(
            task='multiclass',
            sampler='tpe',
            outputs_dir=self.outputs_dir,
            random_seed=random_seed,
        )
        self.expert_model = self.expert_model_tuner.run(
            X_train=experts_train.drop(columns='outcome'), y_train=experts_train['outcome'],
            X_val=experts_val.drop(columns='outcome'), y_val=experts_val['outcome'],
            hyperparam_space=hyperparam_space,
            evaluation_function=lambda y_true, y_pred: -metrics.log_loss(y_true=y_true,
                                                                         y_pred=y_pred),
            n_trials=n_trials,
        )

        # """ FOR DEBUGGING
        val_pred = self.expert_model.predict_proba(experts_val.drop(columns='outcome'))
        fn_roc_auc = metrics.roc_auc_score(
            y_true=(experts_val['outcome'] == 'fn').astype(int),
            y_score=val_pred[:, self.expert_model.classes_ == 'fn'],
        )
        fp_roc_auc = metrics.roc_auc_score(
            y_true=(experts_val['outcome'] == 'fp').astype(int),
            y_score=val_pred[:, self.expert_model.classes_ == 'fp'],
        )
        print(f'ROC_AUC(FN) = {fn_roc_auc:.4f}')
        print(f'ROC_AUC(FP) = {fp_roc_auc:.4f}')
        # """

    def assign(
            self,
            X,
            score_col,
            ml_model_threshold,
            fp_cost,
            calibration=True,
            batches=None, capacity=None,
            assignments_relative_path=None,
    ):
        assignments_path = (
            self.outputs_dir + assignments_relative_path
            if assignments_relative_path is not None
            else None
        )
        if os.path.exists(assignments_path):
            return self._load_assignments(assignments_path)

        if self.expert_model is None:
            raise ValueError('Assigner has  not been .fit().')

        assignments_list = list()
        for b in tqdm(np.sort(batches.iloc[:, 0].unique())):
            batch_filter = (batches.iloc[:, 0] == b)
            batch_X = X[batch_filter]
            all_possibilities = self._generate_possibilities(
                batch_X)
            all_outcome_probabilities = self.predict_outcome_probabilities(
                X=all_possibilities, score_col=score_col,
                ml_model_threshold=ml_model_threshold,
                calibration=calibration,
            )
            batch_loss_df = self._calculate_loss(
                all_outcome_probabilities,
                fp_cost=fp_cost,
            )
            batch_assignments = self._assign_in_batch(
                X=batch_X,
                loss_df=batch_loss_df,
                capacity=capacity[b],
            )
            assignments_list.append(batch_assignments)

        assignments = pd.Series(
            np.concatenate(assignments_list).squeeze(),
            index=X.index,
        )

        if assignments_path is not None:
            self._save_assignments(assignments, assignments_path)

        return assignments

    def predict_outcome_probabilities(
            self,
            X, score_col,
            ml_model_threshold,
            calibration,
    ):
        p = X.copy()
        expert_model_X = self._transform(X).drop(columns='index')
        outcomes = list(self.expert_model.classes_)
        p[outcomes] = pd.NA

        # ML MODEL PROBABILITIES
        is_model_ix = (p[self.assignment_col] == 0)
        y_score = p.loc[is_model_ix, score_col]
        y_pred_bool = (y_score >= ml_model_threshold)
        if calibration:
            y_score = self.calibrator.transform(y_score)
        p.loc[is_model_ix, 'fn'] = (~y_pred_bool).astype(int) * y_score
        p.loc[is_model_ix, 'fp'] = y_pred_bool.astype(int) * (1 - y_score)
        p.loc[is_model_ix, 'tn'] = (~y_pred_bool).astype(int) * (1 - y_score)
        p.loc[is_model_ix, 'tp'] = y_pred_bool.astype(int) * y_score

        # HUMAN PROBABILITIES
        is_human_ix = (p[self.assignment_col] == 1)
        expert_model_pred = self.expert_model.predict_proba(
            expert_model_X.loc[is_human_ix, :]
        )
        # order is guaranteed by the fact that outcomes=expert_model.classes_
        p.loc[is_human_ix, outcomes] = expert_model_pred

        return p[['index', self.assignment_col, *outcomes]]

    def _calculate_loss(
            self,
            outcome_probabilities,
            fp_cost,
    ):
        loss = outcome_probabilities.copy()
        loss['loss'] = (
                fp_cost * loss['fp']
                + loss['fn']
        )

        return loss[['index', self.assignment_col, 'loss']]

    def _assign_in_batch(
            self,
            X,
            loss_df,
            capacity,
    ):
        X_to_assign = X.copy()
        c = deepcopy(capacity)

        loss_df

        return assignments

    def _generate_possibilities(self, X):
        X_with_index_col = X.copy().assign(**{'index': X.index})
        assignments_series = pd.Series([0, 1], name=self.assignment_col)
        combos = X_with_index_col.merge(assignments_series, how='cross')
        combos[self.assignment_col] = combos[self.assignment_col].astype('category')

        return combos

    def _transform(self, X):
        X_enc = X.copy()
        to_encode = self.categorical_cols

        if self.assignment_col not in X.columns:
            X_enc[self.assignment_col] = 'null_category'

        if self.ordinal_encoder is None:
            self.ordinal_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            X_enc[to_encode] = self.ordinal_encoder.fit_transform(X_enc[to_encode])
        else:
            X_enc[to_encode] = self.ordinal_encoder.transform(X_enc[to_encode])

        X_enc[to_encode] = X_enc[to_encode].astype('category')  # for LGBM

        if self.assignment_col not in X.columns:
            X_enc = X_enc.drop(columns=self.assignment_col)

        return X_enc

    @staticmethod
    def _get_outcome(label, pred):
        if pred == 1:
            if label == 1:
                o = 'tp'
            elif label == 0:
                o = 'fp'
        elif pred == 0:
            if label == 1:
                o = 'fn'
            elif label == 0:
                o = 'tn'
        return o

    @staticmethod
    def _load_assignments(assignments_path):
        return pd.read_pickle(assignments_path)
        # return pd.read_parquet(assignments_path)

    @staticmethod
    def _save_assignments(assignments, assignments_path):
        assignments.to_pickle(assignments_path)
        # assignments.to_parquet(assignments_path

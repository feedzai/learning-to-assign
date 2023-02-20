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


class RiskMinimizingAssigner(AbstractAssigner):
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

        # to be temporarily set in .assign()
        self.fp_cost_hist = list()
        self.fp_protected_penalty_hist = list()
        self.predicted_fpr_hist = list()
        self.predicted_fpr_disparity_hist = list()

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
        experts_train = experts_train.drop(columns=[ground_truth_col, decision_col])

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
            experts_val = experts_val.drop(columns=[ground_truth_col, decision_col])

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

        """ FOR DEBUGGING
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
        """

    def assign(
            self,
            X,
            score_col,
            ml_model_threshold,
            fp_cost,
            calibration=True,
            fp_protected_penalty=0, protected_col=None, protected_group=None,
            batches=None, capacity=None,
            confidence_deferral=False, solver='individual',
            dynamic=False,
            target_fpr=None, target_fpr_disparity=None,
            fpr_learning_rate=None, fpr_disparity_learning_rate=None,
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

        if isinstance(protected_col, str):
            protected_arr = X[protected_col]
        else:
            protected_arr = protected_col

        Xs_list = list()
        protected_arr_list = list()
        assignments_list = list()
        for b in tqdm(np.sort(batches.iloc[:, 0].unique())):
            batch_filter = (batches.iloc[:, 0] == b)
            batch_ix = X.index[batch_filter]
            batch_X = X[batch_filter]
            all_possibilities = self._generate_possibilities(
                batch_X, experts=self.expert_ids)
            all_outcome_probabilities = self.predict_outcome_probabilities(
                X=all_possibilities, score_col=score_col,
                ml_model_threshold=ml_model_threshold,
                calibration=calibration,
            )
            batch_loss_df = self._calculate_loss(
                all_outcome_probabilities,
                fp_cost=fp_cost,
                fp_protected_penalty=fp_protected_penalty,
                protected_col=protected_arr[batch_ix].to_frame().merge(
                    all_possibilities,
                    left_index=True, right_on='index',
                    suffixes=('', '_other')
                )[protected_arr.name],
                protected_group=protected_group,
            )
            batch_assignments = self._assign_in_batch(
                X=batch_X, score_col=score_col, ml_model_threshold=ml_model_threshold,
                loss_df=batch_loss_df,
                capacity=capacity[b],
                confidence_deferral=confidence_deferral,
                solver=solver,
            )
            assignments_list.append(batch_assignments)

            if dynamic:
                Xs_list.append(batch_X)
                protected_arr_list.append(protected_arr[batch_ix])
                fp_cost, fp_protected_penalty, protected_group = self._update_constants(
                    past_Xs_list=Xs_list,
                    protected_arr_list=protected_arr_list,
                    past_assignments_list=assignments_list,
                    score_col=score_col,
                    ml_model_threshold=ml_model_threshold,
                    calibration=calibration,
                    current_fp_cost=fp_cost,
                    current_fp_protected_penalty=fp_protected_penalty,
                    target_fpr=target_fpr,
                    target_fpr_disparity=target_fpr_disparity,
                    fpr_learning_rate=fpr_learning_rate,
                    fpr_disparity_learning_rate=fpr_disparity_learning_rate,
                )

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
        is_model_ix = p[self.assignment_col].isin(self.ml_model_ids)
        y_score = p.loc[is_model_ix, score_col]
        y_pred_bool = (y_score >= ml_model_threshold)
        if calibration:
            y_score = self.calibrator.transform(y_score)
        p.loc[is_model_ix, 'fn'] = (~y_pred_bool).astype(int) * y_score
        p.loc[is_model_ix, 'fp'] = y_pred_bool.astype(int) * (1 - y_score)
        p.loc[is_model_ix, 'tn'] = (~y_pred_bool).astype(int) * (1 - y_score)
        p.loc[is_model_ix, 'tp'] = y_pred_bool.astype(int) * y_score

        # HUMAN PROBABILITIES
        is_human_ix = p[self.assignment_col].isin(self.human_expert_ids)
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
            fp_protected_penalty, protected_col, protected_group,
    ):
        loss = outcome_probabilities.copy()
        y_protected_attribute = (
            protected_col if protected_col is not None
            else np.full_like(loss.iloc[:, 0], fill_value=0)  # makes fairness loss term zero
        )

        y_is_protected_attribute = (
            (y_protected_attribute == protected_group)
            if protected_col is not None
            else np.full_like(loss.iloc[:, 0], fill_value=0)
        )

        loss['loss'] = (
                fp_cost * loss['fp']
                + loss['fn']
                + y_is_protected_attribute * fp_protected_penalty * loss['fp']
                - ~y_is_protected_attribute * fp_protected_penalty * loss['fp']
        )

        return loss[['index', self.assignment_col, 'loss']]

    def _assign_in_batch(
            self,
            X, score_col, ml_model_threshold,
            loss_df,
            capacity, confidence_deferral, solver
    ):
        X_to_assign = X.copy()
        c = deepcopy(capacity)

        if confidence_deferral:
            non_deferred_bool = self._defer(
                score_arr=X[score_col],
                ml_model_threshold=ml_model_threshold,
                model_capacity=c[self.ml_model_ids[0]],
            )
            c[self.ml_model_ids[0]] = 0
            loss_df = loss_df[loss_df['index'].isin(X_to_assign.index[non_deferred_bool])]
            X_to_assign = X_to_assign.drop(index=X_to_assign.index[non_deferred_bool])

        if solver == 'random':
            assignments = random_assignment(
                X=X_to_assign,
                capacity=c,
                random_seed=self.random_seed,
            )
        else:
            assignments = self._assign_with_costs(
                original_index=X_to_assign.index,
                loss_df=loss_df,
                index_col='index', assignment_col=self.assignment_col, cost_col='loss',
                capacity=c, solver=solver,
            )

        if confidence_deferral:
            human_assignments = assignments.copy()
            assignments = pd.Series(self.ml_model_ids[0], index=X.index)
            assignments[human_assignments.index] = human_assignments

        return assignments

    def _update_constants(
            self,
            past_Xs_list,
            protected_arr_list,
            past_assignments_list,
            score_col, ml_model_threshold, calibration,
            current_fp_cost,
            current_fp_protected_penalty,
            target_fpr,
            target_fpr_disparity,
            fpr_learning_rate,
            fpr_disparity_learning_rate,
    ):
        def _calc_fpr(x):
            return x['fp'].sum() / (x['fp'].sum() + x['tn'].sum())

        X = pd.concat(past_Xs_list)
        X['index'] = X.index
        assignments = pd.concat(past_assignments_list)
        protected_arr = pd.concat(protected_arr_list)
        pred_proba = self.predict_outcome_probabilities(
            X=X.assign(**{self.assignment_col: assignments}),
            score_col=score_col, ml_model_threshold=ml_model_threshold, calibration=calibration,
        )

        predicted_fpr = _calc_fpr(pred_proba)

        group_fprs = dict()
        for g in protected_arr.unique():
            g_subset = pred_proba[protected_arr == g]
            group_fprs[g] = _calc_fpr(g_subset)

        new_protected_group = max(group_fprs, key=group_fprs.get)  # dict argmax
        predicted_fpr_disparity = (
            max(list(group_fprs.values()))
            / min(list(group_fprs.values()))
        )

        new_fp_cost = (
            current_fp_cost
            * (1 + fpr_learning_rate * (predicted_fpr/target_fpr - 1))
        )
        new_fp_protected_penalty = (
            current_fp_protected_penalty
            * (1 + fpr_disparity_learning_rate *
               (predicted_fpr_disparity/target_fpr_disparity - 1))
        )

        print('')
        print('FPR')
        print(f'Pred = {predicted_fpr:.3f} (target = {target_fpr:.2f})')
        print(f'Current cost = {current_fp_cost:.4f}')
        print(f'New cost = {new_fp_cost:.4f}')
        print('')
        print('Disparity')
        print(f'Pred = {predicted_fpr_disparity:.3f} (target = {target_fpr_disparity:.1f})')
        print(f'Current cost = {current_fp_protected_penalty:.4f}')
        print(f'New cost = {new_fp_protected_penalty:.4f}')
        print('')

        self.fp_cost_hist.append(new_fp_cost)
        self.fp_protected_penalty_hist.append(new_fp_protected_penalty)
        self.predicted_fpr_hist.append(predicted_fpr)
        self.predicted_fpr_disparity_hist.append(predicted_fpr_disparity)

        return new_fp_cost, new_fp_protected_penalty, new_protected_group

    def _generate_possibilities(self, X, experts):
        X_with_index_col = X.copy().assign(**{'index': X.index})
        assignments_series = pd.Series(experts, name=self.assignment_col)
        combos = X_with_index_col.merge(assignments_series, how='cross')
        combos[self.assignment_col] = combos[self.assignment_col].astype('category')

        return combos

    def _transform(self, X):
        X_enc = X.copy()
        to_encode = [*self.categorical_cols, self.assignment_col]

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
    def _assign_with_costs(
            original_index,
            loss_df, index_col, assignment_col, cost_col,
            solver, capacity=None,
    ):
        if solver not in ['random', 'individual', 'scheduler']:
            raise ValueError('Solver not supported.')

        if capacity is None:
            assignments = optimal_individual_assignment(
                original_index=original_index,
                loss_df=loss_df,
                index_col=index_col, assignment_col=assignment_col, cost_col=cost_col,
            )
        else:
            if solver == 'individual':
                assignments = case_by_case_assignment(
                    original_index=original_index,
                    loss_df=loss_df,
                    index_col=index_col, assignment_col=assignment_col, cost_col=cost_col,
                    capacity=capacity,
                )

            elif solver == 'scheduler':
                assignments = optimal_batch_assignment(
                    original_index=original_index,
                    loss_df=loss_df,
                    index_col=index_col, assignment_col=assignment_col, cost_col=cost_col,
                    capacity=capacity,
                )
                if assignments is None:
                    assignments = case_by_case_assignment(
                        original_index=original_index,
                        loss_df=loss_df,
                        index_col=index_col, assignment_col=assignment_col, cost_col=cost_col,
                        capacity=capacity,
                    )

        return assignments

    @staticmethod
    def _defer(score_arr, ml_model_threshold, model_capacity):
        sorted_score_arr = score_arr.sort_values()
        non_deferred_bool = (
            (sorted_score_arr >= ml_model_threshold) |
            (sorted_score_arr <= sorted_score_arr.iloc[
                model_capacity - (sorted_score_arr >= ml_model_threshold).sum()])
        )
        if non_deferred_bool.sum() >= model_capacity:
            non_deferred_bool = (
                    (sorted_score_arr >= ml_model_threshold) |
                    (sorted_score_arr < sorted_score_arr.iloc[
                        model_capacity - (sorted_score_arr >= ml_model_threshold).sum()])
            )
        assert non_deferred_bool.sum() == model_capacity

        return non_deferred_bool

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

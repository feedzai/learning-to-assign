import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import confusion_matrix

from autodefer.models import RejectionAfterFPRThresholding
from autodefer.utils import evaluation
from autodefer.utils import thresholding as t


def run_lgbm_optuna_tpe(X_train, y_train, X_val, y_val,
                        n_trials: int, param_grid_dict: dict,
                        target_metric: str, target_constraint: float,
                        n_jobs=1, random_seed=None,
                        trials_path: str = None):
    """
    Runs optuna study for LightGBM hyperparameter optimization.
    """
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(n_startup_trials=int(0.1 * n_trials), seed=random_seed),
        direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, X_val, y_val, param_grid_dict, target_metric, target_constraint,
            n_jobs, random_seed),
        n_trials=n_trials,
        n_jobs=1)  # total n_jobs = optuna_n_jobs x model_n_jobs
    lgbm_best_trial = study.best_trial

    if trials_path is not None:
        study.trials_dataframe().to_csv(trials_path)

    print(f'Number of finished trials: {len(study.trials)}')

    print('Best trial:')
    print(f'  Value: {lgbm_best_trial.value}')
    print(f'  Params:')
    for k, v in lgbm_best_trial.params.items():
        print(f'    {k}: {v}')

    return lgbm_best_trial.params


def objective(
        trial, X_train, y_train, X_val, y_val,
        params_grid_config, target_metric, target_constraint,
        n_jobs, random_seed):
    """
    Optuna objective function for LightGBM hyperparameter optimization.
    Supports TPR@FPR & TPR@topk optimization (target & constraint).
    """
    params = suggest_params(trial, params_grid_config)
    params.update({
        'n_jobs': n_jobs,
        'random_state': random_seed
    })

    lgbm = lgb.LGBMClassifier(**params)
    lgbm.fit(X_train, y_train)
    lgbm_score = lgbm.predict_proba(X_val)[:, lgbm.classes_ == 1].squeeze()

    if target_metric == 'tpr@fpr':
        # metric: Recall @ FPR
        threshold_at_fpr = t.calc_threshold_at_fpr(
            y_true=y_val, y_score=lgbm_score, fpr=target_constraint)
        predictions = t.predict_with_threshold(
            y_score=lgbm_score, threshold=threshold_at_fpr)

        tn, fp, fn, tp = confusion_matrix(
            y_true=y_val,
            y_pred=predictions,
            labels=[0, 1]).ravel()

        fpr = fp / (tn + fp)
        tpr = tp / (tp + fn)

        if fpr <= target_constraint:  # classifiers with uniform scores around the threshold ruin the thresholding
            metric = tpr
        else:
            metric = -1

        return metric

    elif target_metric == 'tpr@topk':
        # metric: Recall @ Top-k
        top_k = int(y_val.shape[0] * target_constraint)
        sorted_lgbm_score_ix = np.argsort(lgbm_score)  # ascending
        top_k_ix = sorted_lgbm_score_ix[-top_k:]

        metric = (y_val[top_k_ix] == 1).sum() / (y_val == 1).sum()

    elif target_metric == 'opt_tpr@opt_fpr':
        try:
            metric = calc_opt_tpr_at_opt_fpr(
                y_val=y_val,
                val_score=lgbm_score,
                target_coverage=target_constraint['target_coverage'],
                target_fpr=target_constraint['target_fpr']
            )
        except:  # TODO this is a quick fix. make RejectionAfterFPRThresholding.threshold() robust
            metric = -1

    else:
        raise ValueError('supported targets are "tpr@fpr", "tpr@topk" and "opt_tpr@opt_fpr".')

    return metric


def suggest_params(trial, params_grid_config: dict):
    """
    Returns param dictionary for the LGBMClassifier.
    TODO Function is not flexible outside the assumed configuration structure.
    """

    suggester_type_mapping = {
        'float': trial.suggest_float,
        'int': trial.suggest_int
    }
    suggested_params = dict()

    for param, param_configs in params_grid_config.items():
        if isinstance(param_configs, dict):  # continuous suggesters
            suggester = suggester_type_mapping[param_configs['type']]  # function
            suggested_params[param] = suggester(
                name=param,
                low=param_configs['range'][0],
                high=param_configs['range'][1],
                log=param_configs['log']
            )
        else:  # finite options, assumed to be categorical
            suggested_params[param] = trial.suggest_categorical(
                name=param,
                choices=param_configs  # assumed to be a list
            )

    return suggested_params


def calc_opt_tpr_at_opt_fpr(y_val, val_score, target_coverage, target_fpr):
    double_thr = RejectionAfterFPRThresholding()
    double_thr.threshold(
        y_true=y_val,
        classifier_score=val_score,
        rejector_score=val_score,  # after FPR thres. the base score is still used
        coverage=target_coverage,
        fpr=target_fpr
    )
    double_thr_val_pred, double_thr_val_rej = double_thr.predict_and_reject(
        classifier_score=val_score,
        rejector_score=val_score
    )
    rej_eval = evaluation.RejectionEvaluator(exp_id_cols=['exp_id'])
    rej_eval.evaluate(
        exp_id=[0],
        y_true=y_val,
        y_pred=double_thr_val_pred,
        rejections=double_thr_val_rej
    )
    opt_tpr, opt_fpr = rej_eval.results.loc[0, ['opt_tpr', 'opt_fpr']]

    if opt_fpr <= target_fpr:  # classifiers with uniform scores around the threshold ruin the thresholding
        metric = opt_tpr
    else:
        metric = -1

    return metric

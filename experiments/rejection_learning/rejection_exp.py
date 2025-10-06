# %% imports
import copy
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from display_results import display_results
from scipy.stats import spearmanr

from autodefer.models import run_lgbm
from autodefer.models.LGBMEnsemble import LGBMEnsemble
from autodefer.models.make_advised_model_sets import make_advised_model_sets
from autodefer.models.outlier_models import ScaledLocalOutlierFactor, ScaledTrustScore
from autodefer.models.RejectionAfterFPRThresholding import RejectionAfterFPRThresholding
from autodefer.models.run_alt_advised_model import run_alt_advised_model
from autodefer.models.run_isolation_forest import score_with_isolation_forest, train_isolation_forest
from autodefer.utils import thresholding as t
from autodefer.utils.ClassificationEvaluator import ClassificationEvaluator
from autodefer.utils.RejectionEvaluator import RejectionEvaluator

# %% Configs
# dataframe display
width = 450
pd.set_option('display.width', width)
np.set_printoptions(linewidth=width)
pd.set_option('display.max_columns', 25)

# experiment config
with open('configs/exp_config.yaml', 'r') as infile:
    exp_config = yaml.safe_load(infile)

exp_name = exp_config['name']
data_path = '~/data/'
results_path = '~/results/'
models_path = '~/models/'
projects_path = '~/projects/'

proj_rel_path = 'learning-to-defer/rejection_learning/'
exp_rel_path = f'{proj_rel_path}{exp_name}/'

os.makedirs(results_path+exp_rel_path, exist_ok=True)
os.makedirs(models_path+exp_rel_path, exist_ok=True)
os.makedirs(models_path+exp_rel_path+'params/', exist_ok=True)
os.makedirs(models_path+exp_rel_path+'optuna_trials', exist_ok=True)

logs_path = results_path + exp_rel_path + 'logs.txt'

# LGBM hyperparameters grid
with open('configs/lgbm_param_grid.yaml', 'r') as infile:
    lgbm_param_grid = yaml.safe_load(infile)['LGBM']

# evaluation
clf_val_eval = ClassificationEvaluator(
    filepath=results_path + exp_rel_path + 'clf_val_results.csv',
    exp_id_cols=['predictor'],
    overwrite=exp_config['overwrite']
)
clf_eval = ClassificationEvaluator(
    filepath=results_path + exp_rel_path + 'clf_results.csv',
    exp_id_cols=['predictor'],
    overwrite=exp_config['overwrite']
)
clf_rethr_eval = ClassificationEvaluator(  # guarantees constraints in test set
    filepath=results_path + exp_rel_path + 'clf_rethr_results.csv',
    exp_id_cols=['predictor'],
    overwrite=exp_config['overwrite']
)
rej_val_eval = RejectionEvaluator(
    filepath=results_path + exp_rel_path + 'rej_val_results.csv',
    exp_id_cols=['predictor', 'rej_class', 'rej_params'],
    overwrite=exp_config['overwrite']
)
rej_eval = RejectionEvaluator(
    filepath=results_path + exp_rel_path + 'rej_results.csv',
    exp_id_cols=['predictor', 'rej_class', 'rej_params'],
    overwrite=exp_config['overwrite']
)
rej_rethr_eval = RejectionEvaluator(  # guarantees constraints in test set
    filepath=results_path + exp_rel_path + 'rej_rethr_results.csv',
    exp_id_cols=['predictor', 'rej_class', 'rej_params'],
    overwrite=exp_config['overwrite']
)
# %% Loading datasets
print('\nLoading datasets...')

train = pd.read_csv(data_path + 'wbe-data.joao.veiga.train.csv', index_col=0)
test = pd.read_csv(data_path + 'wbe-data.joao.veiga.validation.csv', index_col=0)

# target separation
target = 'fraud_bool'
# %% Time splits
print('\nTime splits')

def timestamp_to_datetime(s: pd.Series):
    s_datetime = s.apply(
        lambda x: datetime.fromtimestamp(int(x) / 1000))
    return s_datetime

train['datetime'] = timestamp_to_datetime(train['timestamp'])
train['datetime_strf'] = train['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
print('\nDatetime column (training set):\n', train['datetime_strf'])
test['datetime'] = timestamp_to_datetime(test['timestamp'])
test['datetime_strf'] = test['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
print('\nDatetime column (test set):\n', test['datetime_strf'])

print('\nCovered months (training set):',
      train['datetime'].dt.strftime('%Y-%m').unique().tolist(),
      '(last month will only 1 sample)')
print('Covered months (test set):',
      test['datetime'].dt.strftime('%Y-%m').unique().tolist(),
      '(last month will only 2 samples)\n')

train_ix = (train['datetime'] < '2016-06-01')
val1_ix = (train['datetime'] >= '2016-06-01') & (train['datetime'] < '2016-07-01')
val2_ix = (train['datetime'] >= '2016-07-01')
# %% EDA
print('\nEDA')

print(f'Training set shape: {train.shape}')
print(f'Test set shape: {test.shape}')

train.describe()
test.describe()

def prevalence(df, target_str):
    n_positives = (df[target_str] == 1).sum()
    n = df.shape[0]
    return n_positives/n

print(f'Prevalence (training set): {prevalence(train, target):.2%}')
print(f'Prevalence (testing set): {prevalence(test, target):.2%}')

if exp_config['show_plots']:
    months_datetime, prevalence_by_month = [], []
    for df_split in (train, test):
        months_datetime_df = list(df_split['datetime'].dt.month.unique())
        months_datetime_df.pop()  # last month without representation
        prevalence_by_month_df = [prevalence(df_split[df_split['datetime'].dt.month == m], target)
                                  for m in months_datetime_df]
        months_datetime.extend(months_datetime_df)
        prevalence_by_month.extend(prevalence_by_month_df)

    sns.barplot(x=months_datetime, y=prevalence_by_month, color='steelblue')
    plt.title('Prevalance by month')
    plt.show()
# %% X and y splits
train = train.drop(columns=['timestamp', 'datetime', 'datetime_strf']).reset_index(drop=True)
test = test.drop(columns=['timestamp', 'datetime', 'datetime_strf']).reset_index(drop=True)

features = list(train.drop(columns=[target]).columns)
X_train_val = train.drop(columns=[target]).values
y_train_val = train[target].values

X_test = test.drop(columns=[target]).values
y_test = test[target].values

X_train = X_train_val[train_ix]
X_train_val1 = X_train_val[train_ix | val1_ix]
X_val = X_train_val[val1_ix | val2_ix]
X_val1 = X_train_val[val1_ix]
X_val2 = X_train_val[val2_ix]

y_train = y_train_val[train_ix]
y_train_val1 = y_train_val[train_ix | val1_ix]
y_val = y_train_val[val1_ix | val2_ix]
y_val1 = y_train_val[val1_ix]
y_val2 = y_train_val[val2_ix]

print(f'X_train shape: {X_train.shape}')
print(f'X_val1 shape: {X_val1.shape}')
print(f'X_val2 shape: {X_val2.shape}')
# %% Predictor: LightGBM (hyperparameterized)
print('\nPredictor: LightGBM (hyperparameterized)')

print(f'FPR = {exp_config["target_fpr"]}')

base_name = 'light_gbm'
base_model_path = f'{models_path + exp_rel_path + base_name}.pickle'
base_best_params_path = f'{models_path + exp_rel_path}params/{base_name}_params.yaml'
base_optuna_trials_path = f'{models_path + exp_rel_path}optuna_trials/{base_name}_trials.csv'

base_params = run_lgbm.tune_lgbm_params(
    best_params_path=base_best_params_path,  # loads from if saved; saves to otherwise
    optuna_trials_path=base_optuna_trials_path,  # saves performance by trial plot
    logs_path=logs_path,  # registers running times in
    X_train=X_train_val1, y_train=y_train_val1,
    X_val=X_val2, y_val=y_val2,
    n_trials=exp_config['n_trials'],
    param_grid_dict=lgbm_param_grid,
    target_metric='tpr@fpr',
    target_constraint=exp_config["target_fpr"],
    random_seed=exp_config['random_seed'],
    n_jobs=exp_config['n_jobs']
)
base_optuna_trials = pd.read_csv(base_optuna_trials_path, index_col=0)
base = run_lgbm.train_lgbm(
    model_path=base_model_path,  # loads from if saved; saves to otherwise
    logs_path=logs_path,
    X_train=X_train_val1, y_train=y_train_val1,
    params=base_params,
    random_seed=exp_config['random_seed'],
    deterministic=exp_config['deterministic'],
    n_jobs=exp_config['n_jobs']
)

base_train_val1_score = base.predict_proba(X_train_val1)[:, base.classes_ == 1].squeeze()
base_val2_score = base.predict_proba(X_val2)[:, base.classes_ == 1].squeeze()
base_test_score = base.predict_proba(X_test)[:, base.classes_ == 1].squeeze()

base_val2_threshold_at_fpr = t.calc_threshold_at_fpr(
    y_true=y_val2,
    y_score=base_val2_score,
    fpr=exp_config["target_fpr"])

# evaluation in val set
base_val2_pred = t.predict_with_threshold(
    y_score=base_val2_score,
    threshold=base_val2_threshold_at_fpr)

print('Val set results:')
clf_val_eval.evaluate(
    exp_id=[base_name],
    y_true=y_val2,
    y_pred=base_val2_pred)

# evaluation in test set
base_test_pred = t.predict_with_threshold(
    y_score=base_test_score,
    threshold=base_val2_threshold_at_fpr)

print('Test set results:')
clf_eval.evaluate(
    exp_id=[base_name],
    y_true=y_test,
    y_pred=base_test_pred,
    display_results=True)

# evaluation in test set (rethresholded)
base_threshold_at_fpr_rethr = t.calc_threshold_at_fpr(
    y_true=y_test,
    y_score=base_test_score,
    fpr=exp_config["target_fpr"])
base_test_rethr_pred = t.predict_with_threshold(
    y_score=base_test_score,
    threshold=base_threshold_at_fpr_rethr)
clf_rethr_eval.evaluate(
    exp_id=[base_name],
    y_true=y_test,
    y_pred=base_test_rethr_pred
)
# %% Score exploration
print(f'\n{base_name} score exploration')

print(f'Threshold (@FPR{int(exp_config["target_fpr"] * 100)}) = {round(base_val2_threshold_at_fpr, 4)}')

if exp_config['show_plots']:
    sns.histplot(base_test_score[y_test == 1])
    plt.title(f'{base_name} score of label positives')
    plt.show()
# %% LEARNING WITH REJECTION @ cov=98, FPR=5%
print('\n\nLEARNING WITH REJECTION @ cov=98, FPR=5%')
print(f'FPR = {exp_config["target_fpr"]}')
print(f'Coverage = {exp_config["target_coverage"]}')
# %% LwR - Maximum Class Probability (naive baseline)
print('\nLearning to Reject - Maximum Class Probability (naive baseline)')
mcp_confidence_val2 = base.predict_proba(X_val2).max(axis=1)
mcp_confidence_test = base.predict_proba(X_test).max(axis=1)

val_threshold = t.calc_rejection_threshold_at_coverage(
    confidence=mcp_confidence_val2,
    coverage=exp_config["target_coverage"])
mcp_rej = (mcp_confidence_test <= val_threshold)

rej_eval.evaluate(
    exp_id=[base_name, 'MCP', 'cov & fpr'],
    y_true=y_test,
    y_pred=base_test_pred,
    rejections=mcp_rej
)  # TODO add evaluation in others val and rethresholded ?
# %% Double thresholding on baseline model
print('\nDouble thresholding on baseline model')
# rejection evaluation in val set
base_double_thr = RejectionAfterFPRThresholding()
base_double_thr.threshold(
    y_true=y_val2,
    classifier_score=base_val2_score,
    rejector_score=base_val2_score,  # after FPR thres. the base score is still used
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
base_double_thr_val_pred, base_double_thr_val_rej = base_double_thr.predict_and_reject(
    classifier_score=base_val2_score,
    rejector_score=base_val2_score
)
rej_val_eval.evaluate(
    exp_id=[base_name, 'double_thr', 'cov & fpr'],
    y_true=y_val2,
    y_pred=base_double_thr_val_pred,
    rejections=base_double_thr_val_rej
)

# rejection evaluation in test set
# rejector already thresholded on val2
base_double_thr_pred, base_double_thr_rej = base_double_thr.predict_and_reject(
    classifier_score=base_test_score,
    rejector_score=base_test_score
)
rej_eval.evaluate(
    exp_id=[base_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=base_double_thr_pred,
    rejections=base_double_thr_rej
)

# rejection evaluation in test set (rethresholded)
base_double_thr_rethr = RejectionAfterFPRThresholding()
base_double_thr_rethr.threshold(
    y_true=y_test,
    classifier_score=base_test_score,
    rejector_score=base_test_score,
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
base_double_thr_rethr_pred, base_double_thr_rethr_rej = base_double_thr_rethr.predict_and_reject(
    classifier_score=base_test_score,
    rejector_score=base_test_score
)
rej_rethr_eval.evaluate(
    exp_id=[base_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=base_double_thr_rethr_pred,
    rejections=base_double_thr_rethr_rej
)
# %% Baseline rejection exploration
sns.kdeplot(base_val2_score[y_val2 == 1], fill=True, bw_adjust=.6)
plt.title(f'Score of label positives on val set')
plt.xlim(xmin=0, xmax=1)
plt.xlabel('Score')
plt.show()
# %% Would a model trained only on non-rejected samples be different?
non_rej_ix = (
    (base_train_val1_score < base_double_thr.t_negat_rejector) |
    (base_train_val1_score > base_double_thr.t_posit_base)
)
X_train_val1_non_rej = X_train_val1[non_rej_ix]
y_train_val1_non_rej = y_train_val1[non_rej_ix]

print(f'Extrapolated non-rejected fraction in training set: {X_train_val1_non_rej.shape[0]/X_train_val1.shape[0]:0.4f}')

spec_non_rej_name = 'spec_non_rej'
spec_non_rej_model_path = f'{models_path + exp_rel_path + spec_non_rej_name}.pickle'
spec_non_rej_best_params_path = f'{models_path + exp_rel_path}params/{spec_non_rej_name}_params.yaml'
spec_non_rej_optuna_trials_path = f'{models_path + exp_rel_path}optuna_trials/{spec_non_rej_name}_trials.csv'

spec_non_rej_params = run_lgbm.tune_lgbm_params(
    best_params_path=spec_non_rej_best_params_path,  # loads from if saved; saves to otherwise
    optuna_trials_path=spec_non_rej_optuna_trials_path,  # saves performance by trial plot
    logs_path=logs_path,  # registers running times in
    X_train=X_train_val1_non_rej, y_train=y_train_val1_non_rej,
    X_val=X_val2, y_val=y_val2,
    n_trials=exp_config['n_trials'],
    param_grid_dict=lgbm_param_grid,
    target_metric='tpr@fpr',
    target_constraint=exp_config["target_fpr"],
    random_seed=exp_config['random_seed'],
    n_jobs=exp_config['n_jobs']
)
spec_non_rej_optuna_trials = pd.read_csv(spec_non_rej_optuna_trials_path, index_col=0)
spec_non_rej = run_lgbm.train_lgbm(
    model_path=spec_non_rej_model_path,  # loads from if saved; saves to otherwise
    logs_path=logs_path,
    X_train=X_train_val1_non_rej, y_train=y_train_val1_non_rej,
    params=spec_non_rej_params,
    random_seed=exp_config['random_seed'],
    deterministic=exp_config['deterministic'],
    n_jobs=exp_config['n_jobs']
)

spec_non_rej_val2_score = spec_non_rej.predict_proba(X_val2)[:, spec_non_rej.classes_ == 1].squeeze()
spec_non_rej_test_score = spec_non_rej.predict_proba(X_test)[:, spec_non_rej.classes_ == 1].squeeze()

sns.scatterplot(x=base_val2_score, y=spec_non_rej_val2_score)
plt.xlabel('Base model score')
plt.ylabel('Specialized model score')
plt.show()
print(f'Spearman(base, specialized) @ val = {spearmanr(base_val2_score, spec_non_rej_val2_score)}')

sns.scatterplot(x=base_test_score, y=spec_non_rej_test_score)
plt.xlabel('Base model score')
plt.ylabel('Specialized model score')
plt.show()
print(f'Spearman(base, specialized) @ val = {spearmanr(base_test_score, spec_non_rej_test_score)}')

print('\nDouble thresholding on specialized non-rejected samples model')
# rejection evaluation in val set
spec_non_rej_double_thr = RejectionAfterFPRThresholding()
spec_non_rej_double_thr.threshold(
    y_true=y_val2,
    classifier_score=spec_non_rej_val2_score,
    rejector_score=spec_non_rej_val2_score,  # after FPR thres. the spec_non_rej score is still used
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
spec_non_rej_double_thr_val_pred, spec_non_rej_double_thr_val_rej = spec_non_rej_double_thr.predict_and_reject(
    classifier_score=spec_non_rej_val2_score,
    rejector_score=spec_non_rej_val2_score
)
rej_val_eval.evaluate(
    exp_id=[spec_non_rej_name, 'double_thr', 'cov & fpr'],
    y_true=y_val2,
    y_pred=spec_non_rej_double_thr_val_pred,
    rejections=spec_non_rej_double_thr_val_rej
)

# rejection evaluation in test set
# rejector already thresholded on val2
spec_non_rej_double_thr_pred, spec_non_rej_double_thr_rej = spec_non_rej_double_thr.predict_and_reject(
    classifier_score=spec_non_rej_test_score,
    rejector_score=spec_non_rej_test_score
)
rej_eval.evaluate(
    exp_id=[spec_non_rej_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=spec_non_rej_double_thr_pred,
    rejections=spec_non_rej_double_thr_rej
)

# rejection evaluation in test set (rethresholded)
spec_non_rej_double_thr_rethr = RejectionAfterFPRThresholding()
spec_non_rej_double_thr_rethr.threshold(
    y_true=y_test,
    classifier_score=spec_non_rej_test_score,
    rejector_score=spec_non_rej_test_score,
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
spec_non_rej_double_thr_rethr_pred, spec_non_rej_double_thr_rethr_rej = spec_non_rej_double_thr_rethr.predict_and_reject(
    classifier_score=spec_non_rej_test_score,
    rejector_score=spec_non_rej_test_score
)
rej_rethr_eval.evaluate(
    exp_id=[spec_non_rej_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=spec_non_rej_double_thr_rethr_pred,
    rejections=spec_non_rej_double_thr_rethr_rej
)

# %% Double thresholding hyperparameterized at TPR@NegativesThresholdFPR
print('\nDouble thresholding hyperparameterized at TPR@NegativesThresholdFPR')
# what would be the FPR at the negative threshold on the base model?
higher_fpr_pred = t.predict_with_threshold(
    base_val2_score,
    threshold=base_double_thr.t_negat_rejector
)
higher_fpr = ClassificationEvaluator().calc_metrics(  # static methods
    **clf_val_eval.count_confusion_matrix(
        y_true=y_val2,
        y_pred=higher_fpr_pred)
)['fpr']
print('FPR if thresholded on negatives threshold on base model:', higher_fpr)
higher_fpr_model_name = 'higher_fpr_model_light_gbm'
higher_fpr_model_model_path = f'{models_path + exp_rel_path + higher_fpr_model_name}.pickle'
higher_fpr_model_best_params_path = f'{models_path + exp_rel_path}params/{higher_fpr_model_name}_params.yaml'
higher_fpr_model_optuna_trials_path = f'{models_path + exp_rel_path}optuna_trials/{higher_fpr_model_name}_trials.csv'

higher_fpr_model_params = run_lgbm.tune_lgbm_params(
    best_params_path=higher_fpr_model_best_params_path,  # loads from if saved; saves to otherwise
    optuna_trials_path=higher_fpr_model_optuna_trials_path,  # saves performance by trial plot
    logs_path=logs_path,  # registers running times in
    X_train=X_train_val1, y_train=y_train_val1,
    X_val=X_val2, y_val=y_val2,
    n_trials=exp_config['n_trials'],
    param_grid_dict=lgbm_param_grid,
    target_metric='tpr@fpr',
    target_constraint=higher_fpr,  # DIFFERENCE: higher target fpr
    random_seed=exp_config['random_seed'],
    n_jobs=exp_config['n_jobs']
)

higher_fpr_model = run_lgbm.train_lgbm(
    model_path=higher_fpr_model_model_path,  # loads from if saved; saves to otherwise
    logs_path=logs_path,
    X_train=X_train_val1, y_train=y_train_val1,
    params=higher_fpr_model_params,
    random_seed=exp_config['random_seed'],
    deterministic=exp_config['deterministic'],
    n_jobs=exp_config['n_jobs']
)

higher_fpr_model_val2_score = higher_fpr_model.predict_proba(X_val2)[:, higher_fpr_model.classes_ == 1].squeeze()
higher_fpr_model_test_score = higher_fpr_model.predict_proba(X_test)[:, higher_fpr_model.classes_ == 1].squeeze()

higher_fpr_model_val2_threshold_at_fpr = t.calc_threshold_at_fpr(
    y_true=y_val2,
    y_score=higher_fpr_model_val2_score,
    fpr=exp_config["target_fpr"])

# evaluation in val set
higher_fpr_model_val2_pred = t.predict_with_threshold(
    y_score=higher_fpr_model_val2_score,
    threshold=higher_fpr_model_val2_threshold_at_fpr)

print('Val set results:')
clf_val_eval.evaluate(
    exp_id=[higher_fpr_model_name],
    y_true=y_val2,
    y_pred=higher_fpr_model_val2_pred)

# evaluation in test set
higher_fpr_model_test_pred = t.predict_with_threshold(
    y_score=higher_fpr_model_test_score,
    threshold=higher_fpr_model_val2_threshold_at_fpr)

print('Test set results:')
clf_eval.evaluate(
    exp_id=[higher_fpr_model_name],
    y_true=y_test,
    y_pred=higher_fpr_model_test_pred,
    display_results=True)

# evaluation in test set (rethresholded)
higher_fpr_model_threshold_at_fpr_rethr = t.calc_threshold_at_fpr(
    y_true=y_test,
    y_score=higher_fpr_model_test_score,
    fpr=exp_config["target_fpr"])
higher_fpr_model_test_rethr_pred = t.predict_with_threshold(
    y_score=higher_fpr_model_test_score,
    threshold=higher_fpr_model_threshold_at_fpr_rethr)
clf_rethr_eval.evaluate(
    exp_id=[higher_fpr_model_name],
    y_true=y_test,
    y_pred=higher_fpr_model_test_rethr_pred
)
# rejection evaluation in val set
higher_fpr_model_double_thr = RejectionAfterFPRThresholding()
higher_fpr_model_double_thr.threshold(
    y_true=y_val2,
    classifier_score=higher_fpr_model_val2_score,
    rejector_score=higher_fpr_model_val2_score,  # after FPR thres. the higher_fpr_model score is still used
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
higher_fpr_model_double_thr_val_pred, higher_fpr_model_double_thr_val_rej = higher_fpr_model_double_thr.predict_and_reject(
    classifier_score=higher_fpr_model_val2_score,
    rejector_score=higher_fpr_model_val2_score
)
rej_val_eval.evaluate(
    exp_id=[higher_fpr_model_name, 'double_thr', 'cov & fpr'],
    y_true=y_val2,
    y_pred=higher_fpr_model_double_thr_val_pred,
    rejections=higher_fpr_model_double_thr_val_rej
)

# rejection evaluation in test set
# rejector already thresholded on val2
higher_fpr_model_double_thr_pred, higher_fpr_model_double_thr_rej = higher_fpr_model_double_thr.predict_and_reject(
    classifier_score=higher_fpr_model_test_score,
    rejector_score=higher_fpr_model_test_score
)
rej_eval.evaluate(
    exp_id=[higher_fpr_model_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=higher_fpr_model_double_thr_pred,
    rejections=higher_fpr_model_double_thr_rej
)

# rejection evaluation in test set (rethresholded)
higher_fpr_model_double_thr_rethr = RejectionAfterFPRThresholding()
higher_fpr_model_double_thr_rethr.threshold(
    y_true=y_test,
    classifier_score=higher_fpr_model_test_score,
    rejector_score=higher_fpr_model_test_score,
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
higher_fpr_model_double_thr_rethr_pred, higher_fpr_model_double_thr_rethr_rej = higher_fpr_model_double_thr_rethr.predict_and_reject(
    classifier_score=higher_fpr_model_test_score,
    rejector_score=higher_fpr_model_test_score
)
rej_rethr_eval.evaluate(
    exp_id=[higher_fpr_model_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=higher_fpr_model_double_thr_rethr_pred,
    rejections=higher_fpr_model_double_thr_rethr_rej
)
# %% Rejection optimized model
print('\nRejection optimized model')
# hyperparameterization target: opt_TPR @ opt_FPR

rej_opt_name = 'rej_opt_light_gbm'
rej_opt_model_path = f'{models_path + exp_rel_path + rej_opt_name}.pickle'
rej_opt_best_params_path = f'{models_path + exp_rel_path}params/{rej_opt_name}_params.yaml'
rej_opt_optuna_trials_path = f'{models_path + exp_rel_path}optuna_trials/{rej_opt_name}_trials.csv'

rej_opt_params = run_lgbm.tune_lgbm_params(
    best_params_path=rej_opt_best_params_path,  # loads from if saved; saves to otherwise
    optuna_trials_path=rej_opt_optuna_trials_path,  # saves performance by trial plot
    logs_path=logs_path,  # registers running times in
    X_train=X_train_val1, y_train=y_train_val1,
    X_val=X_val2, y_val=y_val2,
    n_trials=exp_config['n_trials'],
    param_grid_dict=lgbm_param_grid,
    target_metric='opt_tpr@opt_fpr',
    target_constraint={
        'target_coverage': exp_config['target_coverage'],
        'target_fpr': exp_config['target_fpr']
    },
    random_seed=exp_config['random_seed'],
    n_jobs=exp_config['n_jobs']
)

rej_opt = run_lgbm.train_lgbm(
    model_path=rej_opt_model_path,  # loads from if saved; saves to otherwise
    logs_path=logs_path,
    X_train=X_train_val1, y_train=y_train_val1,
    params=rej_opt_params,
    random_seed=exp_config['random_seed'],
    deterministic=exp_config['deterministic'],
    n_jobs=exp_config['n_jobs']
)

rej_opt_val2_score = rej_opt.predict_proba(X_val2)[:, rej_opt.classes_ == 1].squeeze()
rej_opt_test_score = rej_opt.predict_proba(X_test)[:, rej_opt.classes_ == 1].squeeze()

rej_opt_val2_threshold_at_fpr = t.calc_threshold_at_fpr(
    y_true=y_val2,
    y_score=rej_opt_val2_score,
    fpr=exp_config["target_fpr"])

# evaluation in val set
rej_opt_val2_pred = t.predict_with_threshold(
    y_score=rej_opt_val2_score,
    threshold=rej_opt_val2_threshold_at_fpr)

print('Val set results:')
clf_val_eval.evaluate(
    exp_id=[rej_opt_name],
    y_true=y_val2,
    y_pred=rej_opt_val2_pred)

# evaluation in test set
rej_opt_test_pred = t.predict_with_threshold(
    y_score=rej_opt_test_score,
    threshold=rej_opt_val2_threshold_at_fpr)

print('Test set results:')
clf_eval.evaluate(
    exp_id=[rej_opt_name],
    y_true=y_test,
    y_pred=rej_opt_test_pred,
    display_results=True)

# evaluation in test set (rethresholded)
rej_opt_threshold_at_fpr_rethr = t.calc_threshold_at_fpr(
    y_true=y_test,
    y_score=rej_opt_test_score,
    fpr=exp_config["target_fpr"])
rej_opt_test_rethr_pred = t.predict_with_threshold(
    y_score=rej_opt_test_score,
    threshold=rej_opt_threshold_at_fpr_rethr)
clf_rethr_eval.evaluate(
    exp_id=[rej_opt_name],
    y_true=y_test,
    y_pred=rej_opt_test_rethr_pred
)
# rejection evaluation in val set
rej_opt_double_thr = RejectionAfterFPRThresholding()
rej_opt_double_thr.threshold(
    y_true=y_val2,
    classifier_score=rej_opt_val2_score,
    rejector_score=rej_opt_val2_score,  # after FPR thres. the rej_opt score is still used
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
rej_opt_double_thr_val_pred, rej_opt_double_thr_val_rej = rej_opt_double_thr.predict_and_reject(
    classifier_score=rej_opt_val2_score,
    rejector_score=rej_opt_val2_score
)
rej_val_eval.evaluate(
    exp_id=[rej_opt_name, 'double_thr', 'cov & fpr'],
    y_true=y_val2,
    y_pred=rej_opt_double_thr_val_pred,
    rejections=rej_opt_double_thr_val_rej
)

# rejection evaluation in test set
# rejector already thresholded on val2
rej_opt_double_thr_pred, rej_opt_double_thr_rej = rej_opt_double_thr.predict_and_reject(
    classifier_score=rej_opt_test_score,
    rejector_score=rej_opt_test_score
)
rej_eval.evaluate(
    exp_id=[rej_opt_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=rej_opt_double_thr_pred,
    rejections=rej_opt_double_thr_rej
)

# rejection evaluation in test set (rethresholded)
rej_opt_double_thr_rethr = RejectionAfterFPRThresholding()
rej_opt_double_thr_rethr.threshold(
    y_true=y_test,
    classifier_score=rej_opt_test_score,
    rejector_score=rej_opt_test_score,
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
rej_opt_double_thr_rethr_pred, rej_opt_double_thr_rethr_rej = rej_opt_double_thr_rethr.predict_and_reject(
    classifier_score=rej_opt_test_score,
    rejector_score=rej_opt_test_score
)
rej_rethr_eval.evaluate(
    exp_id=[rej_opt_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=rej_opt_double_thr_rethr_pred,
    rejections=rej_opt_double_thr_rethr_rej
)
# %% Trust Score (Google)
print("\nTrust Score (Google)")  # TODO refactor into a single pipeline (?)

trust_score_name = 'trust_score'
trust_score_model_path = f'{models_path + exp_rel_path + trust_score_name}.pickle'
trust_scores_dir = f'{models_path + exp_rel_path}scores/{trust_score_name}/'
os.makedirs(trust_scores_dir, exist_ok=True)

trust_score_model = ScaledTrustScore(trust_score_model_path=trust_score_model_path)

trust_score_model.fit(X_train, y_train)

# trust score uses predictions, not scores
base_val2_pred = t.predict_with_threshold(
    y_score=base_val2_score,
    threshold=base_val2_threshold_at_fpr)

trust_score_val2 = trust_score_model.score_samples(
    X=X_val2,
    y_pred=base_val2_pred,
    scores_path=f'{trust_scores_dir}trust_score_val2.npy')
trust_score_test = trust_score_model.score_samples(
    X=X_test,
    y_pred=base_test_pred,
    scores_path=f'{trust_scores_dir}trust_score_test.npy')
trust_score_test_rethr = trust_score_model.score_samples(
    X=X_test,
    y_pred=base_test_rethr_pred,
    scores_path=f'{trust_scores_dir}trust_score_test_rethr.npy')

if exp_config['show_plots']:
    sns.histplot(x=trust_score_val2[trust_score_val2 < 20])
    plt.title('Trust Score')
    plt.show()

    sns.histplot(
        x=trust_score_val2[(trust_score_val2 < 20)],
        hue=(base_val2_pred != y_val2)[(trust_score_val2 < 20)])
    plt.title('Trust Score colored by error')
    plt.show()

# NOTE: uncertainty score is minus trust score, as trust score is a measure of confidence

# rejection evaluation on val set
trust_score_after_fpr_thresholder = RejectionAfterFPRThresholding()
trust_score_after_fpr_thresholder.threshold(
    y_true=y_val2,
    classifier_score=base_val2_score,
    rejector_score=-trust_score_val2,  # negative trust score
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
trust_score_val2_pred, trust_score_val2_rej = trust_score_after_fpr_thresholder.predict_and_reject(
    classifier_score=base_val2_score,
    rejector_score=-trust_score_val2  # negative trust score
)
rej_val_eval.evaluate(
    exp_id=[base_name, 'trust_score', 'cov & fpr'],
    y_true=y_val2,
    y_pred=trust_score_val2_pred,
    rejections=trust_score_val2_rej
)
# rejection evaluation on test set
# rejector already thresholded on val2 set
trust_score_test_pred, trust_score_test_rej = trust_score_after_fpr_thresholder.predict_and_reject(
    classifier_score=base_test_score,
    rejector_score=-trust_score_test  # negative trust score
)
rej_eval.evaluate(
    exp_id=[base_name, 'trust_score', 'cov & fpr'],
    y_true=y_test,
    y_pred=trust_score_test_pred,
    rejections=trust_score_test_rej
)
# rejection evaluation on test set (rethresholded)
trust_score_after_fpr_thresholder_rethr = RejectionAfterFPRThresholding()
trust_score_after_fpr_thresholder_rethr.threshold(
    y_true=y_test,
    classifier_score=base_test_score,
    rejector_score=-trust_score_test,  # negative trust score
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
trust_score_test_rethr_pred, trust_score_test_rethr_rej = trust_score_after_fpr_thresholder_rethr.predict_and_reject(
    classifier_score=base_test_score,
    rejector_score=-trust_score_test_rethr  # negative trust score
)
rej_rethr_eval.evaluate(
    exp_id=[base_name, 'trust_score', 'cov & fpr'],
    y_true=y_test,
    y_pred=trust_score_test_rethr_pred,
    rejections=trust_score_test_rethr_rej
)
# %% Specialized Model (after thresholding positives by FPR)
print('\nSpecialized Model (after thresholding positives by FPR)')
# targets the label itself using X in the training set after filtering
t_posit_base = t.calc_threshold_at_fpr(
    y_true=y_val2,
    y_score=base_val2_score,
    fpr=exp_config["target_fpr"]
)

base_train_val1_score = base.predict_proba(X_train_val1)[:, base.classes_ == 1].squeeze()
X_train_val1_fpr5 = X_train_val1[base_train_val1_score < t_posit_base]
y_train_val1_fpr5 = y_train_val1[base_train_val1_score < t_posit_base]

specialized_name = 'specialized_light_gbm'
specialized_model_path = f'{models_path + exp_rel_path + specialized_name}.pickle'
specialized_best_params_path = f'{models_path + exp_rel_path}params/{specialized_name}_params.yaml'
specialized_optuna_trials_path = f'{models_path + exp_rel_path}optuna_trials/{specialized_name}_trials.csv'


specialized_params = run_lgbm.tune_lgbm_params(
    best_params_path=specialized_best_params_path,  # loads from if saved; saves to otherwise
    optuna_trials_path=specialized_optuna_trials_path,  # saves performance by trial plot
    logs_path=logs_path,  # registers running times in
    X_train=X_train_val1_fpr5, y_train=y_train_val1_fpr5,
    X_val=X_val2, y_val=y_val2,
    n_trials=exp_config['n_trials'],
    param_grid_dict=lgbm_param_grid,
    target_metric='tpr@topk',
    target_constraint=(1-exp_config['target_coverage']),
    random_seed=exp_config['random_seed'],
    n_jobs=exp_config['n_jobs']
)

specialized = run_lgbm.train_lgbm(
    model_path=specialized_model_path,  # loads from if saved; saves to otherwise
    logs_path=logs_path,
    X_train=X_train_val1_fpr5, y_train=y_train_val1_fpr5,
    params=specialized_params,
    random_seed=exp_config['random_seed'],
    deterministic=exp_config['deterministic'],
    n_jobs=exp_config['n_jobs']
)

specialized_val2_score = specialized.predict_proba(X_val2)[:, specialized.classes_ == 1].squeeze()
specialized_test_score = specialized.predict_proba(X_test)[:, specialized.classes_ == 1].squeeze()

# rejection evaluation on val set
specialized_after_fpr_thresholder = RejectionAfterFPRThresholding()
specialized_after_fpr_thresholder.threshold(
    y_true=y_val2,
    classifier_score=base_val2_score,
    rejector_score=specialized_val2_score,
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
spec_val2_pred, spec_val2_rej = specialized_after_fpr_thresholder.predict_and_reject(
    classifier_score=base_val2_score,
    rejector_score=specialized_val2_score
)
rej_val_eval.evaluate(
    exp_id=[specialized_name, 'specialized', 'cov & fpr'],
    y_true=y_val2,
    y_pred=spec_val2_pred,
    rejections=spec_val2_rej
)
# rejection evaluation on test set
# rejector already thresholded on val2 set
spec_test_pred, spec_test_rej = specialized_after_fpr_thresholder.predict_and_reject(
    classifier_score=base_test_score,
    rejector_score=specialized_test_score
)
rej_eval.evaluate(
    exp_id=[specialized_name, 'specialized', 'cov & fpr'],
    y_true=y_test,
    y_pred=spec_test_pred,
    rejections=spec_test_rej
)
# rejection evaluation on test set (rethresholded)
specialized_after_fpr_thresholder_rethr = RejectionAfterFPRThresholding()
specialized_after_fpr_thresholder_rethr.threshold(
    y_true=y_test,
    classifier_score=base_test_score,
    rejector_score=specialized_test_score,
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
spec_test_rethr_pred, spec_test_rethr_rej = specialized_after_fpr_thresholder_rethr.predict_and_reject(
    classifier_score=base_test_score,
    rejector_score=specialized_test_score
)
rej_rethr_eval.evaluate(
    exp_id=[specialized_name, 'specialized', 'cov & fpr'],
    y_true=y_test,
    y_pred=spec_test_rethr_pred,
    rejections=spec_test_rethr_rej
)
# %% OUTLIER CALIBRATING MODELS
# The next models will use information from a weaker model trained on less data and from outlier metrics
# to calibrate the confidence of outliers
# %% Weak classifier
print('\nWeak classifier')
# base ("weak") estimator trained only on X_train (not X_train + X_val1)

weak_name = 'weak_light_gbm'
weak_model_path = f'{models_path + exp_rel_path + weak_name}.pickle'
weak_best_params_path = f'{models_path + exp_rel_path}params/{weak_name}_params.yaml'
weak_optuna_trials_path = f'{models_path + exp_rel_path}optuna_trials/{weak_name}_trials.csv'

weak_params = run_lgbm.tune_lgbm_params(
    best_params_path=weak_best_params_path,  # loads from if saved; saves to otherwise
    optuna_trials_path=weak_optuna_trials_path,  # saves performance by trial plot
    logs_path=logs_path,  # registers running times in
    X_train=X_train, y_train=y_train,
    X_val=X_val1, y_val=y_val1,
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
    X_train=X_train, y_train=y_train,
    params=weak_params,
    random_seed=exp_config['random_seed'],
    deterministic=exp_config['deterministic'],
    n_jobs=exp_config['n_jobs']
)
weak_val1_score = weak.predict_proba(X_val1)[:, weak.classes_ == 1].squeeze()
weak_val2_score = weak.predict_proba(X_val2)[:, weak.classes_ == 1].squeeze()
weak_test_score = weak.predict_proba(X_test)[:, weak.classes_ == 1].squeeze()

weak_threshold_at_fpr = t.calc_threshold_at_fpr(
    y_true=y_val2,
    y_score=weak_val2_score,
    fpr=exp_config["target_fpr"])

# evaluation in val set
weak_val2_pred = t.predict_with_threshold(
    y_score=weak_val2_score,
    threshold=weak_threshold_at_fpr)
clf_val_eval.evaluate(
    exp_id=[weak_name],
    y_true=y_val2,
    y_pred=weak_val2_pred
)
# evaluation in test set
weak_test_pred = t.predict_with_threshold(
    y_score=weak_test_score,
    threshold=weak_threshold_at_fpr)
clf_eval.evaluate(
    exp_id=[weak_name],
    y_true=y_test,
    y_pred=weak_test_pred
)
# evaluation in test set (rethresholded)
weak_threshold_at_fpr_rethr = t.calc_threshold_at_fpr(
    y_true=y_test,
    y_score=weak_test_score,
    fpr=exp_config["target_fpr"])
weak_test_rethr_pred = t.predict_with_threshold(
    y_score=weak_test_score,
    threshold=weak_threshold_at_fpr_rethr)
clf_rethr_eval.evaluate(
    exp_id=[weak_name],
    y_true=y_test,
    y_pred=weak_test_rethr_pred
)
# %% Local Outlier Factor
print('\nLocal Outlier Factor')
# TODO brief explanation
lof_name = 'lof'
lof_model_path = f'{models_path + exp_rel_path + lof_name}.pickle'
lofs_dir = f'{models_path + exp_rel_path}scores/{lof_name}/'
os.makedirs(lofs_dir, exist_ok=True)

lof_model = ScaledLocalOutlierFactor(
    lof_model_path=lof_model_path,
    n_jobs=exp_config['n_jobs'],
    **exp_config['LOF'])  # TODO change from kwargs to params dict

lof_model.fit(X_train)

lof_val1_score = lof_model.score_samples(
    X=X_val1,
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
# %% Isolation Forest
print('\nIsolation Forest')
# TODO brief explanation
iso_forest_name = 'isolation_forest'
iso_forest_model_path = f'{models_path + exp_rel_path}{iso_forest_name}.pickle'
iso_forest_scores_dir = f'{models_path + exp_rel_path}scores/{iso_forest_name}/'
os.makedirs(iso_forest_scores_dir, exist_ok=True)

iso_forest = train_isolation_forest(
    model_path=iso_forest_model_path,
    X_train=X_train,
    params=exp_config['isolation_forest'],
    n_jobs=exp_config['n_jobs'],
    random_seed=exp_config['random_seed'],
    logs_path=logs_path,
)

iso_forest_val1_score = score_with_isolation_forest(
    model=iso_forest,
    X=X_val1,
    scores_path=f'{iso_forest_scores_dir}{iso_forest_name}_val1_score.npy'
)
iso_forest_val2_score = score_with_isolation_forest(
    model=iso_forest,
    X=X_val2,
    scores_path=f'{iso_forest_scores_dir}{iso_forest_name}_val2_score.npy'
)
iso_forest_test_score = score_with_isolation_forest(
    model=iso_forest,
    X=X_test,
    scores_path=f'{iso_forest_scores_dir}{iso_forest_name}_test_score.npy'
)

if exp_config['show_plots']:
    sns.histplot(iso_forest_val1_score)
    plt.title('Isolation Forest Score Distribution on val1')
    plt.show()
# %% Calibrated Model
print('\nCalibrated Model (after thresholding positives by FPR)')
# targets the label itself using X + model scores + outlier metrics in the training set TODO define better
cal_X_val1 = np.concatenate(
    (
        X_val1,
        weak_val1_score.reshape(-1, 1),
        lof_val1_score.reshape(-1, 1),
        iso_forest_val1_score.reshape(-1, 1)
    ),
    axis=1
)
cal_X_val2 = np.concatenate(
    (
        X_val2,
        weak_val2_score.reshape(-1, 1),
        lof_val2_score.reshape(-1, 1),
        iso_forest_val2_score.reshape(-1, 1)
    ),
    axis=1
)
cal_X_test = np.concatenate(
    (
        X_test,
        weak_test_score.reshape(-1, 1),
        lof_test_score.reshape(-1, 1),
        iso_forest_test_score.reshape(-1, 1)
    ),
    axis=1
)

# calibrator training
calibrated_name = 'calibrated_light_gbm'
calibrated_model_path = f'{models_path + exp_rel_path + calibrated_name}.pickle'
calibrated_best_params_path = f'{models_path + exp_rel_path}params/{calibrated_name}_params.yaml'
calibrated_optuna_trials_path = f'{models_path + exp_rel_path}optuna_trials/{calibrated_name}_trials.csv'

calibrated_params = run_lgbm.tune_lgbm_params(
    best_params_path=calibrated_best_params_path,  # loads from if saved; saves to otherwise
    optuna_trials_path=calibrated_optuna_trials_path,  # saves performance by trial plot
    logs_path=logs_path,  # registers running times in
    X_train=cal_X_val1, y_train=y_val1,
    X_val=cal_X_val2, y_val=y_val2,
    n_trials=exp_config['n_trials'],
    param_grid_dict=lgbm_param_grid,
    target_metric='tpr@fpr',
    target_constraint=exp_config["target_fpr"],
    random_seed=exp_config['random_seed'],
    n_jobs=exp_config['n_jobs']
)

calibrated = run_lgbm.train_lgbm(
    model_path=calibrated_model_path,  # loads from if saved; saves to otherwise
    logs_path=logs_path,
    X_train=cal_X_val1, y_train=y_val1,
    params=calibrated_params,
    random_seed=exp_config['random_seed'],
    deterministic=exp_config['deterministic'],
    n_jobs=exp_config['n_jobs']
)

calibrated_val2_score = calibrated.predict_proba(cal_X_val2)[:, calibrated.classes_ == 1].squeeze()
calibrated_test_score = calibrated.predict_proba(cal_X_test)[:, calibrated.classes_ == 1].squeeze()

print('\nDouble Thresholding with calibrated model')
# rejection evaluation in val set
calibrated_double_thr = RejectionAfterFPRThresholding()
calibrated_double_thr.threshold(
    y_true=y_val2,
    classifier_score=calibrated_val2_score,
    rejector_score=calibrated_val2_score,  # after FPR thres. the calibrated score is still used
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
calibrated_double_thr_val_pred, calibrated_double_thr_val_rej = calibrated_double_thr.predict_and_reject(
    classifier_score=calibrated_val2_score,
    rejector_score=calibrated_val2_score
)
rej_val_eval.evaluate(
    exp_id=[calibrated_name, 'double_thr', 'cov & fpr'],
    y_true=y_val2,
    y_pred=calibrated_double_thr_val_pred,
    rejections=calibrated_double_thr_val_rej
)
# rejection evaluation in test set
# rejector already thresholded on val2 set
calibrated_double_thr_pred, calibrated_double_thr_rej = calibrated_double_thr.predict_and_reject(
    classifier_score=calibrated_test_score,
    rejector_score=calibrated_test_score
)
rej_eval.evaluate(
    exp_id=[calibrated_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=calibrated_double_thr_pred,
    rejections=calibrated_double_thr_rej
)
# rejection evaluation in test set (rethresholded)
calibrated_double_thr_rethr = RejectionAfterFPRThresholding()
calibrated_double_thr_rethr.threshold(
    y_true=y_test,
    classifier_score=calibrated_test_score,
    rejector_score=calibrated_test_score,
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
calibrated_double_thr_rethr_pred, calibrated_double_thr_rethr_rej = calibrated_double_thr_rethr.predict_and_reject(
    classifier_score=calibrated_test_score,
    rejector_score=calibrated_test_score
)
rej_rethr_eval.evaluate(
    exp_id=[calibrated_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=calibrated_double_thr_rethr_pred,
    rejections=calibrated_double_thr_rethr_rej
)
# %% Advised Model
# missing values on weak score and lof_score on train

adv_X_train_val1, adv_X_val2, adv_X_test = make_advised_model_sets(
    X_train, X_train_val1, X_val2, X_test,
    weak_val1_score, weak_val2_score, weak_test_score,
    lof_val1_score, lof_val2_score, lof_test_score,
    iso_forest_val1_score, iso_forest_val2_score, iso_forest_test_score
)

advised_name = 'advised_light_gbm'
advised_model_path = f'{models_path + exp_rel_path + advised_name}.pickle'
advised_best_params_path = f'{models_path + exp_rel_path}params/{advised_name}_params.yaml'
advised_optuna_trials_path = f'{models_path + exp_rel_path}optuna_trials/{advised_name}_trials.csv'

advised_params = run_lgbm.tune_lgbm_params(
    best_params_path=advised_best_params_path,  # loads from if saved; saves to otherwise
    optuna_trials_path=advised_optuna_trials_path,  # saves performance by trial plot
    logs_path=logs_path,  # registers running times in
    X_train=adv_X_train_val1, y_train=y_train_val1,
    X_val=adv_X_val2, y_val=y_val2,
    n_trials=exp_config['n_trials'],
    param_grid_dict=lgbm_param_grid,
    target_metric='tpr@fpr',
    target_constraint=exp_config["target_fpr"],
    random_seed=exp_config['random_seed'],
    n_jobs=exp_config['n_jobs']
)
advised = run_lgbm.train_lgbm(
    model_path=advised_model_path,  # loads from if saved; saves to otherwise
    logs_path=logs_path,
    X_train=adv_X_train_val1, y_train=y_train_val1,
    params=advised_params,
    random_seed=exp_config['random_seed'],
    deterministic=exp_config['deterministic'],
    n_jobs=exp_config['n_jobs']
)

if exp_config['show_plots']:
    advised_importance_data_dict = {
        'feature_name': features + ['weak_score', 'lof', 'iso_forest'],
        'split_importance': advised.booster_.feature_importance(importance_type='split'),
        'gain_importance': advised.booster_.feature_importance(importance_type='gain'),
        'extra_feature': [False for i in range(len(features))] + [True, True, True]
    }
    importance_data_df = pd.DataFrame.from_dict(advised_importance_data_dict, orient='columns')
    
    cmap = plt.get_cmap("Paired")

    split_importance_sort = importance_data_df.sort_values(by='split_importance')
    print(split_importance_sort[split_importance_sort['extra_feature']])
    plt.bar(
        x="feature_name",
        height="split_importance",
        data=split_importance_sort,
        color=cmap(split_importance_sort['extra_feature']))
    # plt.title('Advised model split importance by feature\n(extra features highlighted)')
    plt.xlabel('Features', fontsize=13)
    plt.ylabel('Split Importance', fontsize=13)
    plt.xticks(ticks=[])
    plt.tight_layout()
    plt.show()

    gain_importance_sort = importance_data_df.sort_values(by='gain_importance')
    plt.bar(
        x="feature_name",
        height="gain_importance",
        data=gain_importance_sort,
        color=cmap(gain_importance_sort['extra_feature']))
    # plt.title('Advised model gain importance by feature\n(extra features highlighted)')
    plt.xlabel('Features', fontsize=13)
    plt.ylabel('Gain Importance', fontsize=13)
    plt.xticks(ticks=[])
    plt.tight_layout()
    plt.show()

advised_val2_score = advised.predict_proba(adv_X_val2)[:, advised.classes_ == 1].squeeze()
advised_test_score = advised.predict_proba(adv_X_test)[:, advised.classes_ == 1].squeeze()

print('\nDouble Thresholding with advised model')
# rejection evaluation in val set
advised_double_thr = RejectionAfterFPRThresholding()
advised_double_thr.threshold(
    y_true=y_val2,
    classifier_score=advised_val2_score,
    rejector_score=advised_val2_score,  # after FPR thres. the advised score is still used
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
advised_double_thr_val_pred, advised_double_thr_val_rej = advised_double_thr.predict_and_reject(
    classifier_score=advised_val2_score,
    rejector_score=advised_val2_score
)
rej_val_eval.evaluate(
    exp_id=[advised_name, 'double_thr', 'cov & fpr'],
    y_true=y_val2,
    y_pred=advised_double_thr_val_pred,
    rejections=advised_double_thr_val_rej
)
# rejection evaluation in test set
# rejector already thresholded on val2 set
advised_double_thr_pred, advised_double_thr_rej = advised_double_thr.predict_and_reject(
    classifier_score=advised_test_score,
    rejector_score=advised_test_score
)
rej_eval.evaluate(
    exp_id=[advised_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=advised_double_thr_pred,
    rejections=advised_double_thr_rej
)
# rejection evaluation in test set (rethresholded)
advised_double_thr_rethr = RejectionAfterFPRThresholding()
advised_double_thr_rethr.threshold(
    y_true=y_test,
    classifier_score=advised_test_score,
    rejector_score=advised_test_score,
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
advised_double_thr_rethr_pred, advised_double_thr_rethr_rej = advised_double_thr_rethr.predict_and_reject(
    classifier_score=advised_test_score,
    rejector_score=advised_test_score
)
rej_rethr_eval.evaluate(
    exp_id=[advised_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=advised_double_thr_rethr_pred,
    rejections=advised_double_thr_rethr_rej
)
# %% Alt Advised Model
# bigger train, smaller val1 

alt_advised_name = 'alt_advised_light_gbm'
alt_advised, alt_adv_X_val2, alt_adv_X_test = run_alt_advised_model(
    data_path, models_path, exp_rel_path,
    X_train_val, y_train_val,
    X_val2, X_test,
    y_val2,
    exp_config,
    lgbm_param_grid,
    logs_path
)

alt_advised_val2_score = alt_advised.predict_proba(alt_adv_X_val2)[:, alt_advised.classes_ == 1].squeeze()
alt_advised_test_score = alt_advised.predict_proba(alt_adv_X_test)[:, alt_advised.classes_ == 1].squeeze()

print('\nDouble Thresholding with alt_advised model')
# rejection evaluation in val set
alt_advised_double_thr = RejectionAfterFPRThresholding()
alt_advised_double_thr.threshold(
    y_true=y_val2,
    classifier_score=alt_advised_val2_score,
    rejector_score=alt_advised_val2_score,  # after FPR thres. the alt_advised score is still used
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
alt_advised_double_thr_val_pred, alt_advised_double_thr_val_rej = alt_advised_double_thr.predict_and_reject(
    classifier_score=alt_advised_val2_score,
    rejector_score=alt_advised_val2_score
)
rej_val_eval.evaluate(
    exp_id=[alt_advised_name, 'double_thr', 'cov & fpr'],
    y_true=y_val2,
    y_pred=alt_advised_double_thr_val_pred,
    rejections=alt_advised_double_thr_val_rej
)
# rejection evaluation in test set
# rejector already thresholded on val2 set
alt_advised_double_thr_pred, alt_advised_double_thr_rej = alt_advised_double_thr.predict_and_reject(
    classifier_score=alt_advised_test_score,
    rejector_score=alt_advised_test_score
)
rej_eval.evaluate(
    exp_id=[alt_advised_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=alt_advised_double_thr_pred,
    rejections=alt_advised_double_thr_rej
)
# rejection evaluation in test set (rethresholded)
alt_advised_double_thr_rethr = RejectionAfterFPRThresholding()
alt_advised_double_thr_rethr.threshold(
    y_true=y_test,
    classifier_score=alt_advised_test_score,
    rejector_score=alt_advised_test_score,
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
alt_advised_double_thr_rethr_pred, alt_advised_double_thr_rethr_rej = alt_advised_double_thr_rethr.predict_and_reject(
    classifier_score=alt_advised_test_score,
    rejector_score=alt_advised_test_score
)
rej_rethr_eval.evaluate(
    exp_id=[alt_advised_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=alt_advised_double_thr_rethr_pred,
    rejections=alt_advised_double_thr_rethr_rej
)
# %% LightGBM Ensemble (ensemble of gradient boosted tree ensembles)
lgbm_ensemble_name = 'lgbm_ensemble'
lgbm_ensemble_models_dir = f'{models_path + exp_rel_path + lgbm_ensemble_name}/'
os.makedirs(lgbm_ensemble_models_dir, exist_ok=True)
lgbm_ensemble_individual_params_path = f'{models_path + exp_rel_path}params/{lgbm_ensemble_name}_params.yaml'
lgbm_ensemble_optuna_trials_path = f'{models_path + exp_rel_path}optuna_trials/{lgbm_ensemble_name}_trials.csv'

lgbm_ensemble = LGBMEnsemble(
    models_dir=lgbm_ensemble_models_dir,  # loads from if saved; saves to otherwise
    n_estimators=exp_config['LGBMEnsemble']['n_estimators'])

lgbm_ensemble_param_grid = copy.deepcopy(lgbm_param_grid)
lgbm_ensemble_param_grid.update(exp_config['LGBMEnsemble']['lgbm_param_grid'])
lgbm_ensemble_best_individual_params = lgbm_ensemble.tune_individual_lgbm_params(
    best_params_path=lgbm_ensemble_individual_params_path,  # loads from if saved; saves to otherwise
    optuna_trials_path=lgbm_ensemble_optuna_trials_path,  # saves performance by trial plot
    logs_path=logs_path,  # registers running times in
    X_train=X_train_val1, y_train=y_train_val1,
    X_val=X_val2, y_val=y_val2,
    n_trials=exp_config['n_trials'],
    param_grid_dict=lgbm_ensemble_param_grid,
    target_metric='tpr@fpr',
    target_constraint=exp_config["target_fpr"],
    n_jobs=exp_config['n_jobs'],
    random_seed=exp_config['random_seed']
)
lgbm_ensemble.fit(
    logs_path=logs_path,
    X_train=X_train_val1, y_train=y_train_val1,
    params=lgbm_ensemble_best_individual_params,
    n_jobs=exp_config['n_jobs'],
    random_seed=exp_config['random_seed'],
    deterministic=exp_config['deterministic']
)
lgbm_ensemble_val2_score = lgbm_ensemble.predict_proba_posit(X_val2)
lgbm_ensemble_test_score = lgbm_ensemble.predict_proba_posit(X_test)

lgbm_ensemble_val2_threshold_at_fpr = t.calc_threshold_at_fpr(
    y_true=y_val2,
    y_score=lgbm_ensemble_val2_score,
    fpr=exp_config["target_fpr"])

# pure classification evaluation in val set
lgbm_ensemble_val2_pred = t.predict_with_threshold(
    y_score=lgbm_ensemble_val2_score,
    threshold=lgbm_ensemble_val2_threshold_at_fpr)

clf_val_eval.evaluate(
    exp_id=[lgbm_ensemble_name],
    y_true=y_val2,
    y_pred=lgbm_ensemble_val2_pred)

# pure classification evaluation in test set
lgbm_ensemble_test_pred = t.predict_with_threshold(
    y_score=lgbm_ensemble_test_score,
    threshold=lgbm_ensemble_val2_threshold_at_fpr)

clf_eval.evaluate(
    exp_id=[lgbm_ensemble_name],
    y_true=y_test,
    y_pred=lgbm_ensemble_test_pred)

# pure classification evaluation in test set (rethresholded)
lgbm_ensemble_threshold_at_fpr_rethr = t.calc_threshold_at_fpr(
    y_true=y_test,
    y_score=lgbm_ensemble_test_score,
    fpr=exp_config["target_fpr"])
lgbm_ensemble_test_rethr_pred = t.predict_with_threshold(
    y_score=lgbm_ensemble_test_score,
    threshold=lgbm_ensemble_threshold_at_fpr_rethr)
clf_rethr_eval.evaluate(
    exp_id=[lgbm_ensemble_name],
    y_true=y_test,
    y_pred=lgbm_ensemble_test_rethr_pred
)

# rejection evaluation in val set
lgbm_ensemble_double_thr = RejectionAfterFPRThresholding()
lgbm_ensemble_double_thr.threshold(
    y_true=y_val2,
    classifier_score=lgbm_ensemble_val2_score,
    rejector_score=lgbm_ensemble_val2_score,  # after FPR thres. the lgbm_ensemble score is still used
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
lgbm_ensemble_double_thr_val_pred, lgbm_ensemble_double_thr_val_rej = lgbm_ensemble_double_thr.predict_and_reject(
    classifier_score=lgbm_ensemble_val2_score,
    rejector_score=lgbm_ensemble_val2_score
)
rej_val_eval.evaluate(
    exp_id=[lgbm_ensemble_name, 'double_thr', 'cov & fpr'],
    y_true=y_val2,
    y_pred=lgbm_ensemble_double_thr_val_pred,
    rejections=lgbm_ensemble_double_thr_val_rej
)
# rejection evaluation in test set
# rejector already thresholded on val2 set
lgbm_ensemble_double_thr_pred, lgbm_ensemble_double_thr_rej = lgbm_ensemble_double_thr.predict_and_reject(
    classifier_score=lgbm_ensemble_test_score,
    rejector_score=lgbm_ensemble_test_score
)
rej_eval.evaluate(
    exp_id=[lgbm_ensemble_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=lgbm_ensemble_double_thr_pred,
    rejections=lgbm_ensemble_double_thr_rej
)
# rejection evaluation in test set (rethresholded)
lgbm_ensemble_double_thr_rethr = RejectionAfterFPRThresholding()
lgbm_ensemble_double_thr_rethr.threshold(
    y_true=y_test,
    classifier_score=lgbm_ensemble_test_score,
    rejector_score=lgbm_ensemble_test_score,
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
lgbm_ensemble_double_thr_rethr_pred, lgbm_ensemble_double_thr_rethr_rej = lgbm_ensemble_double_thr_rethr.predict_and_reject(
    classifier_score=lgbm_ensemble_test_score,
    rejector_score=lgbm_ensemble_test_score
)
rej_rethr_eval.evaluate(
    exp_id=[lgbm_ensemble_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=lgbm_ensemble_double_thr_rethr_pred,
    rejections=lgbm_ensemble_double_thr_rethr_rej
)
# %% LightGBM Ensemble using GOSS but no subsampling
lgbm_ensemble_goss_name = 'lgbm_ensemble_goss'
lgbm_ensemble_goss_models_dir = f'{models_path + exp_rel_path + lgbm_ensemble_goss_name}/'
os.makedirs(lgbm_ensemble_goss_models_dir, exist_ok=True)
lgbm_ensemble_goss_individual_params_path = f'{models_path + exp_rel_path}params/{lgbm_ensemble_goss_name}_params.yaml'
lgbm_ensemble_goss_optuna_trials_path = f'{models_path + exp_rel_path}optuna_trials/{lgbm_ensemble_goss_name}_trials.csv'

lgbm_ensemble_goss = LGBMEnsemble(
    models_dir=lgbm_ensemble_goss_models_dir,  # loads from if saved; saves to otherwise
    n_estimators=exp_config['LGBMEnsembleGOSS']['n_estimators'])

lgbm_ensemble_goss_param_grid = copy.deepcopy(lgbm_param_grid)
lgbm_ensemble_goss_param_grid.update(exp_config['LGBMEnsembleGOSS']['lgbm_param_grid'])
lgbm_ensemble_goss_best_individual_params = lgbm_ensemble_goss.tune_individual_lgbm_params(
    best_params_path=lgbm_ensemble_goss_individual_params_path,  # loads from if saved; saves to otherwise
    optuna_trials_path=lgbm_ensemble_goss_optuna_trials_path,  # saves performance by trial plot
    logs_path=logs_path,  # registers running times in
    X_train=X_train_val1, y_train=y_train_val1,
    X_val=X_val2, y_val=y_val2,
    n_trials=exp_config['n_trials'],
    param_grid_dict=lgbm_ensemble_goss_param_grid,
    target_metric='tpr@fpr',
    target_constraint=exp_config["target_fpr"],
    n_jobs=exp_config['n_jobs'],
    random_seed=exp_config['random_seed']
)
lgbm_ensemble_goss.fit(
    logs_path=logs_path,
    X_train=X_train_val1, y_train=y_train_val1,
    params=lgbm_ensemble_goss_best_individual_params,
    n_jobs=exp_config['n_jobs'],
    random_seed=exp_config['random_seed'],
    deterministic=exp_config['deterministic']
)
lgbm_ensemble_goss_val2_score = lgbm_ensemble_goss.predict_proba_posit(X_val2)
lgbm_ensemble_goss_test_score = lgbm_ensemble_goss.predict_proba_posit(X_test)

lgbm_ensemble_goss_val2_threshold_at_fpr = t.calc_threshold_at_fpr(
    y_true=y_val2,
    y_score=lgbm_ensemble_goss_val2_score,
    fpr=exp_config["target_fpr"])

# pure classification evaluation in val set
lgbm_ensemble_goss_val2_pred = t.predict_with_threshold(
    y_score=lgbm_ensemble_goss_val2_score,
    threshold=lgbm_ensemble_goss_val2_threshold_at_fpr)

clf_val_eval.evaluate(
    exp_id=[lgbm_ensemble_goss_name],
    y_true=y_val2,
    y_pred=lgbm_ensemble_goss_val2_pred)

# pure classification evaluation in test set
lgbm_ensemble_goss_test_pred = t.predict_with_threshold(
    y_score=lgbm_ensemble_goss_test_score,
    threshold=lgbm_ensemble_goss_val2_threshold_at_fpr)

clf_eval.evaluate(
    exp_id=[lgbm_ensemble_goss_name],
    y_true=y_test,
    y_pred=lgbm_ensemble_goss_test_pred)

# pure classification evaluation in test set (rethresholded)
lgbm_ensemble_goss_threshold_at_fpr_rethr = t.calc_threshold_at_fpr(
    y_true=y_test,
    y_score=lgbm_ensemble_goss_test_score,
    fpr=exp_config["target_fpr"])
lgbm_ensemble_goss_test_rethr_pred = t.predict_with_threshold(
    y_score=lgbm_ensemble_goss_test_score,
    threshold=lgbm_ensemble_goss_threshold_at_fpr_rethr)
clf_rethr_eval.evaluate(
    exp_id=[lgbm_ensemble_goss_name],
    y_true=y_test,
    y_pred=lgbm_ensemble_goss_test_rethr_pred
)

# rejection evaluation in val set
lgbm_ensemble_goss_double_thr = RejectionAfterFPRThresholding()
lgbm_ensemble_goss_double_thr.threshold(
    y_true=y_val2,
    classifier_score=lgbm_ensemble_goss_val2_score,
    rejector_score=lgbm_ensemble_goss_val2_score,  # after FPR thres. the lgbm_ensemble_goss score is still used
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
lgbm_ensemble_goss_double_thr_val_pred, lgbm_ensemble_goss_double_thr_val_rej = lgbm_ensemble_goss_double_thr.predict_and_reject(
    classifier_score=lgbm_ensemble_goss_val2_score,
    rejector_score=lgbm_ensemble_goss_val2_score
)
rej_val_eval.evaluate(
    exp_id=[lgbm_ensemble_goss_name, 'double_thr', 'cov & fpr'],
    y_true=y_val2,
    y_pred=lgbm_ensemble_goss_double_thr_val_pred,
    rejections=lgbm_ensemble_goss_double_thr_val_rej
)
# rejection evaluation in test set
# rejector already thresholded on val2 set
lgbm_ensemble_goss_double_thr_pred, lgbm_ensemble_goss_double_thr_rej = lgbm_ensemble_goss_double_thr.predict_and_reject(
    classifier_score=lgbm_ensemble_goss_test_score,
    rejector_score=lgbm_ensemble_goss_test_score
)
rej_eval.evaluate(
    exp_id=[lgbm_ensemble_goss_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=lgbm_ensemble_goss_double_thr_pred,
    rejections=lgbm_ensemble_goss_double_thr_rej
)
# rejection evaluation in test set (rethresholded)
lgbm_ensemble_goss_double_thr_rethr = RejectionAfterFPRThresholding()
lgbm_ensemble_goss_double_thr_rethr.threshold(
    y_true=y_test,
    classifier_score=lgbm_ensemble_goss_test_score,
    rejector_score=lgbm_ensemble_goss_test_score,
    coverage=exp_config["target_coverage"],
    fpr=exp_config["target_fpr"]
)
lgbm_ensemble_goss_double_thr_rethr_pred, lgbm_ensemble_goss_double_thr_rethr_rej = lgbm_ensemble_goss_double_thr_rethr.predict_and_reject(
    classifier_score=lgbm_ensemble_goss_test_score,
    rejector_score=lgbm_ensemble_goss_test_score
)
rej_rethr_eval.evaluate(
    exp_id=[lgbm_ensemble_goss_name, 'double_thr', 'cov & fpr'],
    y_true=y_test,
    y_pred=lgbm_ensemble_goss_double_thr_rethr_pred,
    rejections=lgbm_ensemble_goss_double_thr_rethr_rej
)
# %% Final Results
display_results(exp_name=exp_name, short=False)

script_end = True  # for debugger

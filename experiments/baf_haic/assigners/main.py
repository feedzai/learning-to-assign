# %%
import os
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from joblib import Parallel, delayed
from sklearn import metrics
from aequitas.group import Group

from autodefer.models import haic
from autodefer.utils import thresholding as t, plotting

sns.set_style('whitegrid')

root_path = '~'
cfg_path = root_path + 'projects/learning-to-defer/experiments/baf_haic/assigners/cfg.yaml'
with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)

RESULTS_PATH = cfg['results_path'] + cfg['exp_name'] + '/'
MODELS_PATH = cfg['models_path'] + cfg['exp_name'] + '/'

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# import matplotlib; matplotlib.use('Agg')
width = 450
pd.set_option('display.width', width)
np.set_printoptions(linewidth=width)
pd.set_option('display.max_columns', 25)

# DATA LOADING -------------------------------------------------------------------------------------
with open(cfg['metadata'], 'r') as infile:
    metadata = yaml.safe_load(infile)

LABEL_COL = metadata['data_cols']['label']
PROTECTED_COL = metadata['data_cols']['protected']
CATEGORICAL_COLS = metadata['data_cols']['categorical']
TIMESTAMP_COL = metadata['data_cols']['timestamp']

SCORE_COL = metadata['data_cols']['score']
BATCH_COL = metadata['data_cols']['batch']
ASSIGNMENT_COL = metadata['data_cols']['assignment']
DECISION_COL = metadata['data_cols']['decision']

EXPERT_IDS = metadata['expert_ids']

# train
TRAIN_ENVS = {
    tuple(exp_dir.split('#')): {
        'train': pd.read_parquet(cfg['train_paths']['environments'] + exp_dir + '/train.parquet'),
        'batches': pd.read_parquet(cfg['train_paths']['environments'] + exp_dir + '/batches.parquet'),
        'capacity': pd.read_parquet(cfg['train_paths']['environments'] + exp_dir + '/capacity.parquet'),
    }
    for exp_dir in os.listdir(cfg['train_paths']['environments'])
    if os.path.isdir(cfg['train_paths']['environments']+exp_dir)
}

# test
test = pd.read_parquet(cfg['test_paths']['data'])
test_experts_pred = pd.read_parquet(cfg['test_paths']['experts_pred'])
TEST_ENVS = {
    tuple(exp_dir.split('#')): {
        'batches': pd.read_parquet(cfg['test_paths']['environments']+exp_dir+'/batches.parquet'),
        'capacity': pd.read_parquet(cfg['test_paths']['environments']+exp_dir+'/capacity.parquet'),
    }
    for exp_dir in os.listdir(cfg['test_paths']['environments'])
    if os.path.isdir(cfg['test_paths']['environments']+exp_dir)
}

# DEFINING FP COST ---------------------------------------------------------------------------------
temp_train = TRAIN_ENVS[('large', 'regular')]['train'].copy()
temp_train = temp_train[temp_train[TIMESTAMP_COL] != 6].drop(columns=TIMESTAMP_COL)
ML_MODEL_THRESHOLD = t.calc_threshold_at_fpr(
    y_true=temp_train[LABEL_COL],
    y_score=temp_train[DECISION_COL],
    fpr=cfg['fpr']
)
tn, fp, fn, tp = metrics.confusion_matrix(
    y_true=temp_train[LABEL_COL],
    y_pred=(temp_train[DECISION_COL] >= ML_MODEL_THRESHOLD).astype(int),
    labels=[0, 1]
).ravel()
print(f'FPR w/ full automation (train) = {fp/(fp+tn):.3f}')

""" from derivatives
fp_cost = t.calc_cost_at_threshold(
    y_true=temp_train[LABEL_COL],
    y_score=temp_train[DECISION_COL],
    threshold=ml_model_threshold,
    width=0.01
)
"""

# theoretical cost
# t = fp_protected_penalty / (fp_protected_penalty + 1) <=> t.fp_protected_penalty + t = fp_protected_penalty <=> fp_protected_penalty(t-1) = -t <=> fp_protected_penalty= -t/t-1
THEORETICAL_FP_COST = -ML_MODEL_THRESHOLD / (ML_MODEL_THRESHOLD - 1)

tn, fp, fn, tp = metrics.confusion_matrix(
    y_true=temp_train[LABEL_COL],
    y_pred=(temp_train[DECISION_COL] >= t.calc_threshold_with_cost(
        y_true=temp_train[LABEL_COL],
        y_score=temp_train[DECISION_COL],
        fp_fn_cost_ratio=THEORETICAL_FP_COST)
    ),
    labels=[0, 1]
).ravel()
print(f'FPR at cost = {fp/(fp+tn):.3f}')
# Risk Minimizing Assigners & Validation Set Construction ------------------------------------------
VAL_ENVS = dict()
VAL_X = None
RMAs = dict()
for env_id in TRAIN_ENVS:
    print(f'Loading {env_id} models')
    batch_id, capacity_id = env_id
    models_dir = f'{MODELS_PATH}{batch_id}_{capacity_id}/'
    os.makedirs(models_dir, exist_ok=True)

    train_with_val = TRAIN_ENVS[env_id]['train']
    train_with_val = train_with_val.copy().drop(columns=BATCH_COL)  # not needed
    is_val = (train_with_val[TIMESTAMP_COL] == 6)
    train_with_val = train_with_val.drop(columns=TIMESTAMP_COL)
    train = train_with_val[~is_val].copy()
    val = train_with_val[is_val].copy()

    RMAs[env_id] = haic.assigners.RiskMinimizingAssigner(
        expert_ids=EXPERT_IDS,
        outputs_dir=f'{models_dir}human_expertise_model/',
    )

    RMAs[env_id].fit(
        train=train,
        val=val,
        categorical_cols=CATEGORICAL_COLS, score_col=SCORE_COL,
        decision_col=DECISION_COL, ground_truth_col=LABEL_COL, assignment_col=ASSIGNMENT_COL,
        hyperparam_space=cfg['human_expertise_model']['hyperparam_space'],
        n_trials=cfg['human_expertise_model']['n_trials'],
        random_seed=cfg['human_expertise_model']['random_seed'],
    )

    VAL_ENVS[env_id] = dict()
    if VAL_X is None:  # does not change w/ env
        VAL_X_COMPLETE = val.copy()
        VAL_X = VAL_X_COMPLETE.copy().drop(columns=[ASSIGNMENT_COL, DECISION_COL, LABEL_COL])
    VAL_ENVS[env_id]['batches'] = (
        TRAIN_ENVS[env_id]['batches']
        .loc[val.index, ]
        .copy()
    )
    VAL_ENVS[env_id]['capacity'] = (
        TRAIN_ENVS[env_id]['capacity']
        .loc[VAL_ENVS[env_id]['batches']['batch'].unique(), ]
        .copy()
    )

# Evaluate Human Expertise Models ------------------------------------------------------------------
def get_outcome(label, pred):
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

OUTCOME_COL = 'error'
expert_val_X = VAL_X_COMPLETE.copy()
expert_val_X = expert_val_X[expert_val_X[ASSIGNMENT_COL] != EXPERT_IDS['model_ids'][0]]
expert_val_X[OUTCOME_COL] = expert_val_X.apply(
    lambda x: get_outcome(label=x[LABEL_COL], pred=x[DECISION_COL]),
    axis=1,
)
expert_val_X = expert_val_X.drop(columns=[DECISION_COL, LABEL_COL])

expert_model_results = dict()
for env_id in TRAIN_ENVS:
    model = RMAs[env_id]
    pred_proba = model.expert_model.predict_proba(expert_val_X.drop(columns=OUTCOME_COL))

    expert_model_results[env_id] = dict()
    expert_model_results[env_id] = {
        'training_set_size': (
            (TRAIN_ENVS[env_id]['train'][ASSIGNMENT_COL] != EXPERT_IDS['model_ids'][0]).sum()
        ),
        'cross_entropy': metrics.log_loss(
            y_true=expert_val_X[OUTCOME_COL], y_pred=pred_proba
        ),
        'avg_roc_auc': metrics.roc_auc_score(
            y_true=expert_val_X[OUTCOME_COL], y_score=pred_proba,
            multi_class='ovr',
            average='macro'  # unweighted
        ),
        'fp_cross_entropy': metrics.log_loss(
            y_true=(expert_val_X[OUTCOME_COL] == 'fp').astype(int),
            y_pred=pred_proba[:, model.expert_model.classes_ == 'fp'].squeeze()
        ),
        'fp_roc_auc': metrics.roc_auc_score(
            y_true=(expert_val_X[OUTCOME_COL] == 'fp').astype(int),
            y_score=pred_proba[:, model.expert_model.classes_ == 'fp'].squeeze(),
        ),
        'fn_cross_entropy': metrics.log_loss(
            y_true=(expert_val_X[OUTCOME_COL] == 'fn').astype(int),
            y_pred=pred_proba[:, model.expert_model.classes_ == 'fn'].squeeze()
        ),
        'fn_roc_auc': metrics.roc_auc_score(
            y_true=(expert_val_X[OUTCOME_COL] == 'fn').astype(int),
            y_score=pred_proba[:, model.expert_model.classes_ == 'fn'].squeeze(),
        ),
    }
    if env_id == ('large', 'regular'):
        assignments = expert_val_X[ASSIGNMENT_COL]
        plotting.plot_independent_roc_curves(
            labels_dict={
                expert_id: (expert_val_X[OUTCOME_COL] == 'fp').astype(int)[assignments == expert_id]
                for expert_id in EXPERT_IDS['human_ids']
            },
            scores_dict={
                expert_id: pred_proba[:, model.expert_model.classes_ == 'fp'].squeeze()[assignments == expert_id]
                for expert_id in EXPERT_IDS['human_ids']
            },
            alpha=0.3,
            title='False Positives'
        )
        plotting.plot_independent_roc_curves(
            labels_dict={
                expert_id: (expert_val_X[OUTCOME_COL] == 'fp').astype(int)[assignments == expert_id]
                for expert_id in EXPERT_IDS['human_ids']
            },
            scores_dict={
                expert_id: pred_proba[:, model.expert_model.classes_ == 'fn'].squeeze()[
                    assignments == expert_id]
                for expert_id in EXPERT_IDS['human_ids']
            },
            alpha=0.3,
            title='False Negatives'
        )
expert_model_results = pd.DataFrame(expert_model_results).T.reset_index(drop=False)
expert_model_results.columns = ['batch', 'capacity'] + list(expert_model_results.columns[2:])
expert_model_results

# EVALUATION FUNCTIONS -----------------------------------------------------------------------------
def make_id_str(tpl):
    printables = list()
    for i in tpl:
        if i == '':
            continue
        elif isinstance(i, (bool, int, float)):
            printables.append(str(i))
        else:
            printables.append(i)

    return '_'.join(printables)

def evaluate(exp_id, exp_batches, exp_capacity, assignments, evaluator):
    test_experts_pred_thresholded = test_experts_pred.copy()
    test_experts_pred_thresholded[EXPERT_IDS['model_ids'][0]] = (
            test_experts_pred_thresholded[EXPERT_IDS['model_ids'][0]] >= ML_MODEL_THRESHOLD
    ).astype(int)
    _decisions = haic.query_experts(
        pred=test_experts_pred_thresholded,
        assignments=assignments
    )

    evaluator.evaluate(
        exp_id=exp_id,
        assignments=assignments,
        decisions=_decisions,
        batches=exp_batches,
        capacity=exp_capacity.T.to_dict(),
        assert_capacity_constraints=False
    )

def product_dict(**kwargs):  # aux
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def make_params_combos(params_cfg):
    params_list = list()
    if not isinstance(params_cfg, list):
        params_cfg = [params_cfg]

    for cartesian_product_set in params_cfg:
        for k, v in cartesian_product_set.items():
            if isinstance(v, str):
                cartesian_product_set[k] = [v]
        for p in product_dict(**cartesian_product_set):
            p_params = {**BASE_CFG, **p}
            if p_params['fp_cost'] == 'theoretical':
                p_params['fp_cost'] = THEORETICAL_FP_COST
            if not (
                p_params['calibration'] and  # useless to calibrate in these cases
                (p_params['confidence_deferral'] or p_params['solver'] == 'random')
            ):
                params_list.append(p_params)

    return params_list

def make_assignments(X, envs, rma, exp_params):
    env_id = (exp_params['batch'], exp_params['capacity'])
    assigner_params = {k: v for k, v in exp_params.items() if k not in ['batch', 'capacity']}
    params_to_record = {k: exp_params[k] for k in FIELDS}
    exp_id = tuple([v for k, v in params_to_record.items()])
    print(exp_id)
    a = rma.assign(
        X=X, score_col=SCORE_COL,
        batches=envs[env_id]['batches'],
        capacity=envs[env_id]['capacity'].T.to_dict(),
        ml_model_threshold=ML_MODEL_THRESHOLD,
        protected_col=(X[PROTECTED_COL] >= 50).map({True: 'Older', False: 'Younger'}),
        protected_group='Older',
        assignments_relative_path=make_id_str(exp_id),
        **assigner_params
    )

    return exp_id, assigner_params, a

def predicted_evaluation(X, assignments, rma, fp_cost):
    X = X.copy().assign(**{ASSIGNMENT_COL: assignments})
    X['index'] = X.index

    pred_out_proba = rma.predict_outcome_probabilities(
        X=X, score_col=SCORE_COL,
        ml_model_threshold=ML_MODEL_THRESHOLD,
        calibration=True
    )
    loss = fp_cost * pred_out_proba['fp'].sum() + pred_out_proba['fn'].sum()
    tpr = pred_out_proba['tp'].sum() / (pred_out_proba['tp'].sum() + pred_out_proba['fn'].sum())
    fpr = pred_out_proba['fp'].sum() / (pred_out_proba['tn'].sum() + pred_out_proba['fp'].sum())

    protected_col = (X[PROTECTED_COL] >= 50).map({True: 'Older', False: 'Younger'})
    is_protected_bool = (protected_col == 'Older')
    fpr_disparity = (
        (pred_out_proba[~is_protected_bool]['fp'].sum()
           / (pred_out_proba[~is_protected_bool]['tn'].sum()
              + pred_out_proba[~is_protected_bool]['fp'].sum()))
        / (pred_out_proba[is_protected_bool]['fp'].sum()
           / (pred_out_proba[is_protected_bool]['tn'].sum()
              + pred_out_proba[is_protected_bool]['fp'].sum()))
    )
    return loss, tpr, fpr, fpr_disparity

def make_assignments_and_predict_evaluate(X, envs, rma, exp_params):
    exp_id, assigner_params, a = make_assignments(X=X, envs=envs, rma=rma, exp_params=exp_params)
    pred_loss, pred_tpr, pred_fpr, pred_fpr_disparity = predicted_evaluation(
        X=X, assignments=a, rma=rma, fp_cost=assigner_params['fp_cost'],
    )
    return exp_id, pred_loss, pred_tpr, pred_fpr, pred_fpr_disparity


ENV_FIELDS = ['batch', 'capacity']
ASSIGNER_FIELDS = [
    'confidence_deferral', 'solver', 'calibration', 'fp_cost', 'fp_protected_penalty',
    'dynamic', 'target_fpr_disparity', 'fpr_learning_rate', 'fpr_disparity_learning_rate'
]
FIELDS = ENV_FIELDS + ASSIGNER_FIELDS
print(tuple(FIELDS))

BASE_CFG = cfg['base_cfg']

# EXPERIMENTS --------------------------------------------------------------------------------------
val_results_dict = dict()
if cfg['n_jobs'] > 1:
    Parallel(n_jobs=cfg['n_jobs'])(
        delayed(make_assignments)(
            X=VAL_X,
            envs=VAL_ENVS,
            rma=RMAs[(exp_params['batch'], exp_params['capacity'])],
            exp_params=exp_params
        )
        for exp_params in make_params_combos(cfg['experiments'])
    )

for exp_params in make_params_combos(cfg['experiments']):
    exp_id, pred_loss, pred_tpr, pred_fpr, pred_fpr_disparity = (
        make_assignments_and_predict_evaluate(
            X=VAL_X,
            envs=VAL_ENVS,
            rma=RMAs[(exp_params['batch'], exp_params['capacity'])],
            exp_params=exp_params)
    )
    val_results_dict[exp_id] = dict(
        pred_loss=pred_loss, pred_tpr=pred_tpr, pred_fpr=pred_fpr,
        pred_fpr_disparity=pred_fpr_disparity
    )

val_results = pd.DataFrame(val_results_dict).T.reset_index(drop=False)
val_results.columns = FIELDS + ['pred_loss', 'pred_tpr', 'pred_fpr', 'pred_fpr_disparity']
val_results

# %%
val_results = val_results.drop(
    columns=['dynamic', 'target_fpr_disparity', 'fpr_learning_rate', 'fpr_disparity_learning_rate']
)
# RENAME FOR PLOTS
col_renamings = {
    'batch': 'Batch',
    'capacity': 'Capacity',
    'confidence_deferral': 'Confidence Deferral',
    'calibration': 'Calibration',
    'solver': 'Solver',
    'fp_cost': 'lambda',
    'fp_protected_penalty': 'alpha',
    'pred_loss': 'Loss',
    'pred_fpr': 'Predicted FPR',
    'pred_tpr': 'Predicted TPR',
    'pred_fpr_disparity': 'Predicted FPR Parity'
}

# %%
architecture_results = val_results[
    (val_results['fp_cost'] == THEORETICAL_FP_COST) &
    (val_results['fp_protected_penalty'] == 0)
]
architecture_results = architecture_results[
    ((architecture_results['confidence_deferral']) & (architecture_results['solver'] == 'random'))
    | ((architecture_results['confidence_deferral'] == False) & (architecture_results['solver'] != 'random'))
]
(
    architecture_results
    .groupby(['confidence_deferral', 'solver', 'calibration'])
    .mean()
    .sort_values(by='pred_loss')
    .reset_index()
)

# %%
architecture_results['Method'] = (
    architecture_results['confidence_deferral'].map({
        True: 'Model-Confidence Deferral',
        False: 'Learning to Assign'
    })
)
plot_data = architecture_results[architecture_results['solver'] != 'random'].copy()
plot_data = plot_data.rename(columns={'calibration': 'Calibration'})
plot_data = (
    plot_data
    .replace('individual', 'Greedy \n (instance-based)')
    .replace('scheduler', 'Linear Programming \n (batch-based)')
)
sns.stripplot(
    data=plot_data, x='solver', y='pred_loss',
    hue='Calibration'
)
plt.ylim(bottom=0)
plt.xlabel('')
plt.ylabel('Predicted Loss')
plt.show()
"""
sns.scatterplot(
    data=architecture_results, x='Method', y='pred_loss',
    hue='calibration', style='solver',
    alpha=0
)
handles, labels = plt.gca().get_legend_handles_labels()
greedy = architecture_results[architecture_results['solver'] == 'greedy']
m = sns.stripplot(
    data=greedy, x='solver', y='pred_loss', hue='calibration',
    marker='o', edgecolor='grey', jitter=1,
)

scheduler = architecture_results[architecture_results['solver'] == 'scheduler']
n = sns.stripplot(
    data=scheduler, x='solver', y='pred_loss', hue='calibration',
    marker='X', edgecolor='grey', jitter=1,
)
plt.legend(handles, labels)
plt.show()
"""

# %%
fp_cost_results = val_results[
    (val_results['confidence_deferral'] == False) &
    (val_results['solver'] == 'scheduler') &
    (val_results['calibration'] == True) &
    (val_results['fp_protected_penalty'] == 0)
]

(
    fp_cost_results
    .pivot(index='fp_cost', columns=['batch', 'capacity'], values='pred_fpr')
    .T.reset_index()
)
sns.lineplot(
    data=fp_cost_results[fp_cost_results['fp_cost'].isin([THEORETICAL_FP_COST, 0.05, 1, 2])],
    x='fp_cost', y='pred_fpr', markers=True,
    hue='capacity', style='batch',
    palette='colorblind'
)
plt.show()

plot_data = fp_cost_results[fp_cost_results['fp_cost'] < 1]
plot_data = plot_data.rename(columns={'capacity': 'Capacity', 'batch': 'Batch'})
plot_data = (
    plot_data
    .replace('regular', 'Regular')
    .replace('inconstant', 'Inconstant')
    .replace('model_dominant', 'Human-Scarce')
    .replace('irregular', 'Disparate')
    .replace('small', 'Small')
    .replace('large', 'Large')
)
sns.lineplot(
    data=plot_data,
    x='fp_cost', y='pred_fpr', markers=True,
    hue='Capacity', style='Batch',
    palette='colorblind',
)
plt.xlabel(r'$\lambda$')
plt.ylabel('Predicted FPR')
plt.axhline(cfg['fpr'], linestyle='dashed', color='grey')
plt.show()

# %%
fairness_results = val_results[
    (val_results['confidence_deferral'] == False)
    & (val_results['solver'] == 'scheduler')
    & (val_results['calibration'] == True)
]
sns.scatterplot(
    data=fairness_results,
    x='pred_fpr',
    y='pred_tpr',
    hue='fp_protected_penalty'
)
plt.show()

# fairness_results['violation'] = (fairness_results['pred_fpr'] - cfg['fpr']).abs()
fairness_results_below_fpr = fairness_results[fairness_results['pred_fpr'] <= cfg['fpr']]
fairness_results_below_fpr[
    (fairness_results_below_fpr['batch'] == 'large')
    & (fairness_results_below_fpr['capacity'] == 'inconstant')
].sort_values(by=['fp_protected_penalty', 'pred_fpr'])
fairness_results_at_fpr = (
    fairness_results_below_fpr
    # .sort_values(by='violation', ascending=True)
    .sort_values(by='pred_fpr', ascending=False)
    .groupby(['batch', 'capacity', 'fp_protected_penalty'])
    .head(1)
    .sort_values(by=['batch', 'capacity', 'fp_protected_penalty'])
)
plot_data = fairness_results_at_fpr
plot_data = plot_data.rename(columns={'capacity': 'Capacity', 'batch': 'Batch'})
plot_data = (
    plot_data
    .replace('regular', 'Regular')
    .replace('inconstant', 'Inconstant')
    .replace('model_dominant', 'Human-Scarce')
    .replace('irregular', 'Disparate')
    .replace('small', 'Small')
    .replace('large', 'Large')
)
sns.lineplot(
    data=plot_data,
    x='pred_tpr', y='pred_fpr_disparity', markers=True,
    hue='Capacity', style='Batch',
    palette='colorblind',
    sort=False,
)
plt.xlabel('Predicted TPR')
plt.ylabel('Predicted FPR Parity')
plt.xlim(0.5, 0.7)
plt.ylim(0, 1)
plt.show()

# TEST SET EVALUATION ------------------------------------------------------------------------------
TEST_X = test.drop(columns=[TIMESTAMP_COL, LABEL_COL])
test_experts_pred_thresholded = test_experts_pred.copy()
test_experts_pred_thresholded[EXPERT_IDS['model_ids'][0]] = (
        test_experts_pred_thresholded[EXPERT_IDS['model_ids'][0]] >= ML_MODEL_THRESHOLD
).astype(int)
test_eval = haic.HAICEvaluator(
    y_true=test[LABEL_COL],
    experts_pred=test_experts_pred,
    exp_id_cols=FIELDS
)

# L2A
for env_id, rma in RMAs.items():
    if 'test' not in rma.outputs_dir:  # avoid double change
        test_path = rma.outputs_dir[:-1] + '_test/'
        os.makedirs(test_path, exist_ok=True)
        rma.outputs_dir = test_path

to_test = list()

confidence_deferral_cfgs = val_results[
    (val_results['confidence_deferral'] == True) &
    (val_results['solver'] == 'random') &
    (val_results['calibration'] == False) &
    (val_results['fp_cost'] == THEORETICAL_FP_COST) &
    (val_results['fp_protected_penalty'] == 0)
]
for ix, row in confidence_deferral_cfgs.iterrows():
    exp_params = {k: v for k, v in dict(row).items() if k in FIELDS}
    to_test.append({**BASE_CFG, **exp_params})

cost_sensitive_cfgs = val_results[
    (val_results['confidence_deferral'] == False) &
    (val_results['solver'] == 'scheduler') &
    (val_results['calibration'] == True) &
    (val_results['fp_cost'] == THEORETICAL_FP_COST) &
    (val_results['fp_protected_penalty'] == 0)
]
for ix, row in cost_sensitive_cfgs.iterrows():
    exp_params = {k: v for k, v in dict(row).items() if k in FIELDS}
    to_test.append({**BASE_CFG, **exp_params})

for ix, row in fairness_results_at_fpr.iterrows():
    exp_params = {k: v for k, v in dict(row).items() if k in FIELDS}
    to_test.append({**BASE_CFG, **exp_params})

if cfg['n_jobs'] > 1:
    Parallel(n_jobs=cfg['n_jobs'])(
        delayed(make_assignments)(
            X=TEST_X,
            envs=TEST_ENVS,
            rma=RMAs[(exp_params['batch'], exp_params['capacity'])],
            exp_params=exp_params
        )
        for exp_params in to_test
    )
else:
    for exp_params in to_test:
        exp_id, assigner_params, a = make_assignments(
            X=TEST_X,
            envs=TEST_ENVS,
            rma=RMAs[(exp_params['batch'], exp_params['capacity'])],
            exp_params=exp_params
        )
        d = haic.query_experts(
            pred=test_experts_pred_thresholded,
            assignments=a
        )
        test_eval.evaluate(
            exp_id=exp_id,
            assignments=a,
            decisions=d,
            assert_capacity_constraints=False
        )

test_results = test_eval.get_results(short=False)
test_results['loss'] = (THEORETICAL_FP_COST * test_results['fp'] + test_results['fn']).astype('float')

g = Group()
test_results['fpr_parity'] = 0
for i, d in enumerate(test_eval.decisions.values()):
    aequitas_df = pd.DataFrame({
        'protected_attr': (test[PROTECTED_COL] >= 50).map({True: 'Older', False: 'Younger'}),
        'score': d,
        'label_value': test[LABEL_COL],
    })
    aequitas_results = g.get_crosstabs(aequitas_df, attr_cols=['protected_attr'])[0]
    fpr_parity = (
        aequitas_results[aequitas_results['attribute_value'] == 'Younger']['fpr'].item()
        / aequitas_results[aequitas_results['attribute_value'] == 'Older']['fpr'].item()
    )
    test_results.iloc[i, -1] = fpr_parity

architecture_results_test = test_results[
    (test_results['fp_cost'] == THEORETICAL_FP_COST) &
    (test_results['fp_protected_penalty'] == 0)
]
(
    architecture_results_test
    .groupby(['solver'])
    .mean()
    .reset_index()
)

fp_cost_test_results = test_results[
    (test_results['fp_cost'] != THEORETICAL_FP_COST)
]

fp_cost_test_results.groupby('fp_protected_penalty').mean()

tn, fp, fn, tp = metrics.confusion_matrix(
    y_true=test[LABEL_COL],
    y_pred=(test[SCORE_COL] >= ML_MODEL_THRESHOLD).astype(int),
    labels=[0, 1]
).ravel()
print(THEORETICAL_FP_COST * fp + fn)
print(tp/(tp+fn))
print(fp/(fp+tn))
aequitas_df = pd.DataFrame({
    'protected_attr': (test[PROTECTED_COL] >= 50).map({True: 'Older', False: 'Younger'}),
    'score': (test[SCORE_COL] >= ML_MODEL_THRESHOLD).astype(int),
    'label_value': test[LABEL_COL],
})
aequitas_results = g.get_crosstabs(aequitas_df, attr_cols=['protected_attr'])[0]
fpr_parity = (
        aequitas_results[aequitas_results['attribute_value'] == 'Younger']['fpr'].item()
        / aequitas_results[aequitas_results['attribute_value'] == 'Older']['fpr'].item()
)
print(fpr_parity)

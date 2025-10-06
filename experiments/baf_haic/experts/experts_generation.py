import numpy as np
import pandas as pd
import yaml
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler
from aequitas.group import Group

from autodefer.models import haic, hyperoptimization
from autodefer.utils import evaluation, preprocessing
from autodefer.utils import thresholding as t

cfg_path = './cfg.yaml'
with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)
    
with open(cfg['data_cfg_path'], 'r') as infile:
    data_cfg = yaml.safe_load(infile)

np_rng = np.random.default_rng(cfg['random_seed'])

# DATA LOADING -------------------------------------------------------------------------------------
data = pd.read_parquet(data_cfg['data_path'])
LABEL_COL = data_cfg['data_cols']['label']
TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
PROTECTED_COL = data_cfg['data_cols']['protected']
CATEGORICAL_COLS = data_cfg['data_cols']['categorical']
data[CATEGORICAL_COLS] = data[CATEGORICAL_COLS].astype('category')
del data_cfg

def splitter(df, timestamp_col, beginning: int, end: int):
    return df[
        (df[timestamp_col] >= beginning) &
        (df[timestamp_col] < end)].copy()

train = splitter(data, TIMESTAMP_COL, *cfg['splits']['train']).drop(columns=TIMESTAMP_COL)
ml_val = splitter(data, TIMESTAMP_COL, *cfg['splits']['ml_val']).drop(columns=TIMESTAMP_COL)
deployment = splitter(data, TIMESTAMP_COL, *cfg['splits']['deployment']).drop(columns=TIMESTAMP_COL)

# EXPERTS ------------------------------------------------------------------------------------------
expert_team = haic.experts.ExpertTeam()
EXPERT_IDS = dict(model_ids=list(), human_ids=list())
THRESHOLDS = dict()

# 1. ML MODEL --------------------------------------------------------------------------------------
ml_train = train.copy()
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ml_train[CATEGORICAL_COLS] = oe.fit_transform(ml_train[CATEGORICAL_COLS])
ml_train[CATEGORICAL_COLS] = ml_train[CATEGORICAL_COLS].astype('category')  # for LGBM
ml_val[CATEGORICAL_COLS] = oe.transform(ml_val[CATEGORICAL_COLS])
ml_val[CATEGORICAL_COLS] = ml_val[CATEGORICAL_COLS].astype('category')  # for LGBM

tuner = hyperoptimization.BinaryClassTuner(
    sampler=cfg['ml_model']['sampler'],
    outputs_dir=cfg['ml_model']['outputs_dir'])
ml_model = tuner.run(
    X_train=ml_train.drop(columns=LABEL_COL), y_train=ml_train[LABEL_COL],
    X_val=ml_val.drop(columns=LABEL_COL), y_val=ml_val[LABEL_COL],
    hyperparam_space=cfg['ml_model']['param_space_path'],
    evaluation_function=lambda y_true, y_pred:
        evaluation.calc_tpr_at_fpr(y_true=y_true, y_score=y_pred, target_fpr=cfg['ml_model']['fpr']),
    n_trials=cfg['ml_model']['n_trials'],
)

# threshold at arbitrary, exogeneous value
ml_model_threshold = t.calc_threshold_at_fpr(
    y_true=ml_val[LABEL_COL],
    y_score=ml_model.predict_proba(ml_val.drop(columns=LABEL_COL))[:, 1].squeeze(),
    fpr=cfg['ml_model']['fpr'],
)
ml_model_score = ml_model.predict_proba(ml_val.drop(columns=LABEL_COL))[:, 1].squeeze()
ml_model_pred = t.predict_with_threshold(
    y_score=ml_model_score, threshold=ml_model_threshold)

# get recall for expert properties
ml_model_recall = metrics.recall_score(
    y_true=ml_val[LABEL_COL],
    y_pred=ml_model_pred
)
print(f'Recall @ 5% FPR(val) = {ml_model_recall}')

# get FPR disparity for expert properties
# fairness
aequitas_df = pd.DataFrame({
    'protected_attr': (ml_val[PROTECTED_COL] >= 50).map({True: 'Older', False: 'Younger'}),
    'score': ml_model_pred,
    'label_value': ml_val[LABEL_COL],
})
g = Group()
aequitas_results = g.get_crosstabs(aequitas_df, attr_cols=['protected_attr'])[0]
ml_model_fpr_diff = (
    aequitas_results[aequitas_results['attribute_value'] == 'Older']['fpr'].item()
    - aequitas_results[aequitas_results['attribute_value'] == 'Younger']['fpr'].item()
)
print(f'FPR difference = {ml_model_fpr_diff}')

expert_team['model#0'] = haic.experts.MLModelExpert(fitted_model=ml_model, threshold=None)
EXPERT_IDS['model_ids'].append('model#0')
THRESHOLDS['model#0'] = ml_model_threshold

# 2. SYNTHETIC EXPERTS -----------------------------------------------------------------------------

# 2.1 PREPROCESSING --------------------------------------------------------------------------------
# fitting on train
experts_train_X = train.copy().drop(columns=LABEL_COL)
experts_train_X['score'] = expert_team[EXPERT_IDS['model_ids'][0]].predict(
    train.drop(columns=LABEL_COL))
experts_train_X[PROTECTED_COL] = (experts_train_X[PROTECTED_COL] >= 50).astype(int)

cols_to_quantile = experts_train_X.drop(columns=CATEGORICAL_COLS).columns.tolist()
qt = QuantileTransformer()
experts_train_X[cols_to_quantile] = (
    qt.fit_transform(experts_train_X[cols_to_quantile])
    - 0.5  # centered on 0
)

if cfg['experts']['encoding'] == 'one_hot_encode':
    ohe = preprocessing.InplaceOneHotEncoder(
        categorical_cols=CATEGORICAL_COLS,
        drop='first',
        handle_unknown='ignore',
    )
    experts_train_X = ohe.fit_transform(experts_train_X)
elif cfg['experts']['encoding'] == 'label_encode':
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    experts_train_X[CATEGORICAL_COLS] = oe.fit_transform(experts_train_X[CATEGORICAL_COLS])

ss = StandardScaler(with_std=False)
experts_train_X[:] = ss.fit_transform(experts_train_X)

cols_to_scale = [c for c in experts_train_X.columns if c not in cols_to_quantile]
desired_range = 1
scaling_factors = (
    desired_range /
    (experts_train_X[cols_to_scale].max() - experts_train_X[cols_to_scale].min())
)
experts_train_X[cols_to_scale] *= scaling_factors

# preprocess other splits
def preprocess(df):
    processed_X = df.copy()
    processed_X[cols_to_quantile] = qt.transform(processed_X[cols_to_quantile]) - 0.5  # centered on 0

    if cfg['experts']['encoding'] == 'one_hot_encode':
        processed_X = ohe.transform(processed_X)
    elif cfg['experts']['encoding'] == 'label_encode':
        processed_X[CATEGORICAL_COLS] = oe.transform(processed_X[CATEGORICAL_COLS])

    processed_X[:] = ss.transform(processed_X)
    processed_X[cols_to_scale] *= scaling_factors

    return processed_X

experts_deployment_X = deployment.copy().drop(columns=LABEL_COL)
experts_deployment_X['score'] = expert_team[EXPERT_IDS['model_ids'][0]].predict(
    deployment.drop(columns=LABEL_COL))
experts_deployment_X[PROTECTED_COL] = (experts_deployment_X[PROTECTED_COL] >= 50).astype(int)
experts_deployment_X = preprocess(experts_deployment_X)

# 2.2 GENERATION -----------------------------------------------------------------------------------

def process_groups_cfg(groups_cfg, baseline_name='regular'):
    full_groups_cfg = dict()
    for g_name in groups_cfg:
        if g_name == baseline_name:
            full_groups_cfg[g_name] = groups_cfg[g_name]
        else:
            full_groups_cfg[g_name] = dict()
            for k in groups_cfg[baseline_name]:
                if k not in list(groups_cfg[g_name].keys()):
                    full_groups_cfg[g_name][k] = full_groups_cfg[baseline_name][k]
                elif isinstance(groups_cfg[g_name][k], dict):
                    full_groups_cfg[g_name][k] = {  # update baseline cfg
                        **groups_cfg[baseline_name][k],
                        **groups_cfg[g_name][k]
                    }
                else:
                    full_groups_cfg[g_name][k] = groups_cfg[g_name][k]

    return full_groups_cfg

ensemble_cfg = process_groups_cfg(cfg['experts']['groups'])
expert_properties_list = list()

for group_name, group_cfg in ensemble_cfg.items():
    # substitute anchored values by actual values
    if group_cfg['fnr']['intercept_mean'] == 'model - stdev':
        group_cfg['fnr']['intercept_mean'] = (
            (1 - ml_model_recall)
            - group_cfg['fnr']['intercept_stdev']
        )
    if group_cfg['fpr']['intercept_mean'] == 'model - stdev':
        group_cfg['fpr']['intercept_mean'] = (
            cfg['ml_model']['fpr']
            - group_cfg['fpr']['intercept_stdev']
        )

    if group_cfg['fpr']['protected_mean'] == 'model':
        group_cfg['fpr']['protected_mean'] = ml_model_fpr_diff
    elif group_cfg['fpr']['protected_mean'] == 'model + 3stdev':
        group_cfg['fpr']['protected_mean'] = ml_model_fpr_diff + 3 * group_cfg['fpr']['protected_stdev']

    coefs = dict()
    for eq in ['fnr', 'fpr']:
        coefs[eq] = dict()
        for coef in ['intercept', 'score', 'protected']:
            coefs[eq][coef] = np_rng.normal(
                loc=group_cfg[eq][f'{coef}_mean'],
                scale=group_cfg[eq][f'{coef}_stdev'],
                size=group_cfg['n']
            )

    for i in range(group_cfg['n']):
        expert_name = f'{group_name}#{i}'
        expert_args = dict(
            fnr=coefs['fnr']['intercept'][i],
            fpr=coefs['fpr']['intercept'][i],
            fnr_beta_score=coefs['fnr']['score'][i],
            fpr_beta_score=coefs['fpr']['score'][i],
            fnr_beta_protected=coefs['fnr']['protected'][i],
            fpr_beta_protected=coefs['fpr']['protected'][i],
            fnr_betas_stdev=group_cfg['fnr']['betas_stdev'],
            fpr_betas_stdev=group_cfg['fpr']['betas_stdev'],
            fnr_betas_min=group_cfg['fnr']['betas_min'],
            fpr_betas_min=group_cfg['fpr']['betas_min'],
            fnr_noise_stdev=group_cfg['fnr']['noise_stdev'],
            fpr_noise_stdev=group_cfg['fpr']['noise_stdev'],
        )
        expert_team[expert_name] = haic.experts.LinearlyAccurateBinaryExpert(**expert_args)
        expert_properties_list.append({**{'expert': expert_name}, **expert_args})
        EXPERT_IDS['human_ids'].append(expert_name)

with open(cfg['output_paths']['ids'], 'w') as outfile:
    yaml.dump(EXPERT_IDS, outfile)

with open(cfg['output_paths']['thresholds'], 'w') as outfile:
    yaml.dump(THRESHOLDS, outfile)

# fitting
expert_team.fit(
    X=experts_train_X,
    y=train[LABEL_COL],
    score_col='score',
    protected_col=PROTECTED_COL,
)

# properties
expert_properties = pd.DataFrame(expert_properties_list)
expert_properties.to_parquet(cfg['output_paths']['properties'])

# 2.2 PREDICTIONS ----------------------------------------------------------------------------------
train_expert_pred = expert_team.predict(
    index=train.index,
    predict_kwargs={
        haic.experts.LinearlyAccurateBinaryExpert: {
            'X': experts_train_X,
            'y': train[LABEL_COL]
        },
        haic.experts.MLModelExpert: {
            'X': ml_train.drop(columns=[LABEL_COL])
        }}
)
train_expert_pred.to_parquet(cfg['output_paths']['train'])

deployment_expert_pred = expert_team.predict(
    index=deployment.index,
    predict_kwargs={
        haic.experts.LinearlyAccurateBinaryExpert: {
            'X': experts_deployment_X,
            'y': deployment[LABEL_COL]
        },
        haic.experts.MLModelExpert: {
            'X': deployment.drop(columns=[LABEL_COL])
        }
    },
)
deployment_expert_pred.to_parquet(cfg['output_paths']['deployment'])

print('Experts generated.')

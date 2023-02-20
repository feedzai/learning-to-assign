import pandas as pd
from sklearn import metrics
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm


def param_lister(param_grid: dict, n_trials=100, random_state=42) -> list:
    """Util parameter lister

    Returns a list of parameter grids for each algorithm, sampled
    from example grids
    """
    param_grids = list(ParameterSampler(param_distributions=param_grid, n_iter=n_trials, random_state=random_state))

    return param_grids


def models_builder(model, param_grid_list: list) -> list:
    """Util model builder

    From a list of parameter grids, generates a list of instances of models,
    for a given algorithm. Parameter model is an sklearn estimator
    """
    models = []
    for grid in param_grid_list:
        models.append(model(**grid))

    return models

def get_best_model_from_list(model_list, X_train, y_train, X_val, y_val, retrain=False):
    best_model, best_acc = None, 0
    for model in tqdm(model_list):
        print(model)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        acc = metrics.accuracy_score(y_true=y_val, y_pred=pred)
        if acc > best_acc:
            best_model = model
            best_acc = acc

    print(f'Best accuracy = {best_acc}')

    if retrain:
        model.fit(
            pd.concat((X_train, X_val), axis=0),
            pd.concat((y_train, y_val))
        )

    return best_model

def get_best_model(X_train, y_train, X_val, y_val, model_class, param_grid, n_trials=100, retrain=False, random_state=42):
    params_list = param_lister(param_grid, n_trials=n_trials, random_state=random_state)
    models = models_builder(model_class, params_list)
    best_model = get_best_model_from_list(models, X_train, y_train, X_val, y_val, retrain=retrain)

    return best_model

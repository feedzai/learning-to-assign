from copy import deepcopy
from typing import Iterable

from optuna.trial import BaseTrial

from .utils import instantiate_model


def suggest_model(trial, hyperparam_space: dict):
    params = suggest_callable_hyperparams(
        trial=trial,
        hyperparameter_space=hyperparam_space,
    )
    model = instantiate_model(**params)

    return params, model

def suggest_callable_hyperparams(
        trial: BaseTrial,
        hyperparameter_space: dict,
        param_prefix: str = "learner",
    ) -> dict:
    """Suggests the top-level hyperparameters for a class instantiation, or
    for parameterizing any other callable.

    This includes the classpath/importpath, and the conditional
    hyperparameters to use as kwargs for the class.

    Parameters
    ----------
    trial : BaseTrial
        The trial object to interact with the hyperparameter sampler.
    hyperparameter_space : dict
        A dictionary representing a hyperparameter space.
    param_prefix : str
        The prefix to attach to all parameters' names in order to uniquely
        identify them.

    Returns
    -------
    Dict
        The suggested callable + its suggested key-word arguments.
    """
    # Suggest callable type (top-level callable choice)
    callable_type = trial.suggest_categorical(
        f"{param_prefix}_type", list(hyperparameter_space.keys()),
    )

    hyperparam_subspace = hyperparameter_space[callable_type]
    callable_classpath = hyperparam_subspace['classpath']

    # Suggest callable kwargs (on conditional sub-space)
    callable_kwargs = suggest_hyperparams(
        trial, hyperparam_subspace.get('kwargs', {}),
        param_prefix=f"{param_prefix}_{callable_type}"
    )

    hyperparams = {'classpath': callable_classpath, **callable_kwargs}
    return hyperparams


def suggest_hyperparams(
        trial: BaseTrial,
        hyperparameter_space: dict,
        param_prefix: str = '',
    ) -> dict:
    """Uses the provided hyperparameter space to suggest specific
    configurations using the given Trial object.

    Parameters
    ----------
    trial : BaseTrial
        The trial object to interact with the hyperparameter sampler.
    hyperparameter_space : dict
        A dictionary representing a hyperparameter space.
    param_prefix : str
        The prefix to attach to all parameters' names in order to uniquely
        identify them.

    Returns
    -------
    An instantiation of the given hyperparameter space.
    """
    params = dict()
    for key, value in hyperparameter_space.items():
        param_id = f'{param_prefix}_{key}'

        # Fixed value
        if isinstance(value, (str, int, float)) or not isinstance(value, Iterable):
            params[key] = value

        # Categorical parameter
        elif isinstance(value, (list, tuple)):
            # Categorical parameters must be of one of the following types
            valid_categ_types = (bool, int, float, str)

            # If not, map values to strings for optuna, then remap them back
            if not all(el is None or isinstance(el, valid_categ_types) for el in value):
                categ_map = {str(el): el for el in value}
                suggested_categ_encoded = trial.suggest_categorical(
                    param_id, sorted(categ_map.keys()),
                )
                params[key] = categ_map[str(suggested_categ_encoded)]

            # If categorical values have valid types, use them directly
            else:
                params[key] = trial.suggest_categorical(
                    param_id, value,
                )

        # Numerical parameter
        elif isinstance(value, dict) and 'type' in value:
            params[key] = suggest_numerical_hyperparam(
                trial, value, param_id,
            )

        # Nested parameter
        elif isinstance(value, dict):
            params[key] = suggest_hyperparams(
                trial, value, f"{key}_{param_prefix}",
            )

        else:
            raise ValueError(
                f"Invalid hyperparameter configuration: {key}={value} "
                f"({type(key)} -> {type(value)})."
            )

    return params


def suggest_numerical_hyperparam(
        trial: BaseTrial,
        config: dict,
        param_id: str,
    ) -> float:
    """Helper function to suggest a numerical hyperparameter.

    Parameters
    ----------
    trial : BaseTrial
        The trial object to interact with the hyperparameter sampler.
    config : dict
        The distribution to sample the parameter from.
    param_id : str
        The parameter's name.

    Returns
    -------
    The sampled value.
    """
    # Parameter's type
    config = deepcopy(config)
    param_type = config.pop('type')
    valid_param_types = ('int', 'float', 'uniform', 'discrete_uniform', 'loguniform')
    assert param_type in valid_param_types, \
        f'Invalid parameter type {param_type}, choose one of {valid_param_types}'

    # Parameter's range
    assert 'range' in config and isinstance(config['range'], (list, tuple)), \
        'Must provide a range when searching a numerical parameter'
    low, high = config.pop('range')

    # Remaining suggestion parameters are used as is (e.g., log scale)
    suggest_param_func = getattr(trial, f'suggest_{param_type}')
    return suggest_param_func(
        param_id, low, high, **config,
    )

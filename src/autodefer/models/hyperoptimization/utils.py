import importlib
from typing import Callable, Union

import yaml


def load_hyperparameter_space(path_or_dict: Union[str, dict]) -> dict:
    # Read hyperparameter space from the YAML file (if given)
    if isinstance(path_or_dict, str):
        with open(path_or_dict, "r") as infile:
            hyperparameter_space = yaml.safe_load(infile)
    elif isinstance(path_or_dict, dict):
        hyperparameter_space = path_or_dict

    return hyperparameter_space

def instantiate_model(classpath, **hyperparams):
    return import_object(classpath)(**hyperparams)

def import_object(import_path: str) -> Union[object, Callable]:
    """Imports the object at the given module/class path.

    Parameters
    ----------
    import_path : str
        The import path for the object to import.

    Returns
    -------
    The imported object (this can be a class, a callable, a variable).
    """
    separator_idx = import_path.rindex('.')
    module_path = import_path[: separator_idx]
    obj_name = import_path[separator_idx + 1:]

    module = importlib.import_module(module_path)
    return getattr(module, obj_name)

""" Tools """
import inspect
import numpy as np
from omegaconf import OmegaConf, DictConfig


def get_class(mod_name, base_path, BaseClass):
    """ Get the class object which inherits from BaseClass and is defined in 
    the module named mod_name, child of base_path. 
    """

    mod_path = '{}.{}'.format(base_path, mod_name)
    mod = __import__(mod_path, fromlist=[''])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]


def print_conf(conf: DictConfig, prefix: str = ''):
    """ Print configurations. """
    print(f"{' Configurations ':=^80}")
    for k, v in conf.items():
        if isinstance(v, DictConfig):
            # Recursively iterate through nested configs
            print_conf(v, prefix=f"{prefix}{k}.")
        else:
            print(f"{prefix}{k}: {v}")

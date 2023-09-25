""" Tools """

import time
import inspect
import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf, DictConfig


class Timer(object):

    def __init__(self):
        self.reset()
    
    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1

    def total(self):
        """ Return the total amount of time. """
        return self.total_time
    
    def avg(self):
        """ Return the average amount of time. """
        return self.total_time / float(self.calls)
    
    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.

    
class Timers(object):

    def __init__(self):
        self.timers = defaultdict(Timer)

    def tic(self, key):
        self.timers[key].tic()

    def toc(self, key):
        self.timers[key].toc()

    def print(self, key=None):
        if key is None:
            # print all time 
            for k, v in self.timers.items():
                print("Average time for {:}: {:}".format(k, v.avg()))
        else:
            print("Average time for {:}: {:}".format(key, self.timers[key].avg()))
    
    def get_avg(self, key):
        return self.timers[key].avg()
    

class NullTimer(object):
    """ NullTimer does nothing when its methods are called. """
    def __init__(self):
        pass

    def tic(self, *args):
        pass

    def toc(self, *args):
        pass

    def print(self, *args):
        pass
    

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
    for k, v in conf.items():
        if isinstance(v, DictConfig):
            # Recursively iterate through nested configs
            print_conf(v, prefix=f"{prefix}{k}.")
        else:
            print(f"{prefix}{k}: {v}")
